"""SGD 的 Python 实现：源码随机顺序、惰性 L2、累计 L1、平均与增量训练。"""

import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

try:
    from .random_utils import RAND_R_MAX, shuffle_indices
    from .common import (
        LinearRegressor, iteration_count, positive_number, sample_weights, validate_xy,
    )
except ImportError:
    from LR.sklearn_impl.random_utils import RAND_R_MAX, shuffle_indices
    from LR.sklearn_impl.common import (
        LinearRegressor, iteration_count, positive_number, sample_weights, validate_xy,
    )


def loss_value_and_gradient(loss, y, prediction, epsilon):
    error = prediction - y
    absolute_error = abs(error)
    if loss == "squared_error":
        return 0.5 * error * error, error
    if loss == "huber":
        if absolute_error <= epsilon:
            return 0.5 * error * error, error
        return epsilon * absolute_error - 0.5 * epsilon ** 2, np.copysign(epsilon, error)
    outside = absolute_error - epsilon
    if outside <= 0:
        return 0.0, 0.0
    if loss == "epsilon_insensitive":
        return outside, np.copysign(1.0, error)
    return outside * outside, np.copysign(2 * outside, error)


class WeightVector:
    """按 _weight_vector.pyx.tp 保存未缩放系数与平均缓冲。"""

    def __init__(self, coefficients, average_coefficients=None):
        self.raw = coefficients
        self.average = average_coefficients
        self.scale_factor = 1.0
        self.squared_norm = float(coefficients @ coefficients)
        self.l1_norm = float(np.abs(coefficients).sum())
        self.average_a, self.average_b = 0.0, 1.0

    def dot(self, values, indices):
        return float(self.raw[indices] @ values) * self.scale_factor

    def add(self, values, indices, update):
        self.raw[indices] += values * (update / self.scale_factor)
        # 1.9.1 的 add 缓存确实只累计当前存储坐标并覆盖旧缓存。
        # L1 截断不刷新这些缓存；停止条件也保持同一版本的行为。
        touched = self.raw[indices]
        self.squared_norm = float(touched @ touched) * self.scale_factor ** 2
        self.l1_norm = float(np.abs(touched).sum()) * self.scale_factor

    def scale(self, factor):
        self.scale_factor *= factor
        self.squared_norm *= factor * factor
        self.l1_norm *= abs(factor)
        if self.scale_factor < 1e-9:
            self.reset_scale()

    def add_average(self, values, indices, update, count):
        self.average[indices] += self.average_a * values * (-update / self.scale_factor)
        mu = 1.0 / count
        if count > 1:
            self.average_b /= 1.0 - mu
        self.average_a += mu * self.average_b * self.scale_factor

    def reset_scale(self):
        if self.average is not None:
            self.average += self.average_a * self.raw
            self.average *= 1.0 / self.average_b
            self.average_a, self.average_b = 0.0, 1.0
        self.raw *= self.scale_factor
        self.scale_factor = 1.0

    def truncate_l1(self, indices, q, accumulated_penalty):
        old = self.raw[indices].copy()
        physical = self.scale_factor * old
        new = old.copy()
        positive, negative = physical > 0, physical < 0
        new[positive] = np.maximum(
            0.0, old[positive] - (accumulated_penalty + q[indices[positive]]) / self.scale_factor,
        )
        new[negative] = np.minimum(
            0.0, old[negative] + (accumulated_penalty - q[indices[negative]]) / self.scale_factor,
        )
        self.raw[indices] = new
        q[indices] += self.scale_factor * (new - old)


class SGDRegressor(LinearRegressor):
    def __init__(
        self, loss="squared_error", *, penalty="l2", alpha=0.0001,
        l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
        shuffle=True, epsilon=0.1, random_state=None, learning_rate="invscaling",
        eta0=0.01, power_t=0.25, n_iter_no_change=5, warm_start=False,
        average=False, early_stopping=False,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average
        self.early_stopping = early_stopping

    def _check_params(self):
        if self.loss not in {"squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"}:
            raise ValueError("不支持此 loss")
        if self.penalty not in {None, "l1", "l2", "elasticnet"}:
            raise ValueError("penalty 应为 None/l1/l2/elasticnet")
        if self.learning_rate not in {"constant", "invscaling", "optimal", "adaptive"}:
            raise ValueError("支持的学习率为 constant/invscaling/optimal/adaptive")
        if self.early_stopping:
            raise ValueError("本实现不包含验证集 early_stopping，请设为 False")
        positive_number(self.alpha, "alpha")
        positive_number(self.eta0, "eta0", allow_zero=False)
        positive_number(self.epsilon, "epsilon")
        positive_number(self.power_t, "power_t")
        if self.tol is not None:
            positive_number(self.tol, "tol")
        if self.learning_rate == "optimal" and self.alpha == 0:
            raise ValueError("optimal 学习率要求 alpha > 0")
        if self.penalty == "elasticnet" and (
            self.l1_ratio is None or not 0 <= self.l1_ratio <= 1
        ):
            raise ValueError("elasticnet 要求 l1_ratio 在 [0, 1] 内")
        iteration_count(self.max_iter)
        iteration_count(self.n_iter_no_change, "n_iter_no_change")
        if not isinstance(self.average, (bool, int, np.integer)) or self.average < 0:
            raise ValueError("average 应为布尔值或非负整数")

    def _initialize(self, n_features, coef_init, intercept_init):
        coefficients = np.zeros(n_features) if coef_init is None else np.asarray(coef_init, dtype=np.float64).ravel().copy()
        intercept = np.zeros(1) if intercept_init is None else np.asarray(intercept_init, dtype=np.float64).reshape(-1).copy()
        if coefficients.shape != (n_features,) or intercept.shape != (1,):
            raise ValueError("初始化系数或截距形状与数据不一致")
        self.coef_, self.intercept_ = coefficients, intercept
        self.__dict__.pop("_average_coef", None)
        self.__dict__.pop("_average_intercept", None)
        if self.average:
            self._standard_coef, self._standard_intercept = self.coef_, self.intercept_
            self._average_coef = np.zeros(n_features)
            self._average_intercept = np.zeros(1)

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        self._check_params()
        first_call = not (self.warm_start and hasattr(self, "coef_"))
        X, y = validate_xy(self, X, y, multi_output=False, sparse_format="csr", reset=first_call)
        if first_call:
            self._initialize(X.shape[1], coef_init, intercept_init)
        # 与 BaseSGDRegressor._fit 一致：fit 总是重置步数，partial_fit 延续步数。
        self.t_ = 1.0
        self._train(X, y, sample_weight, self.max_iter)
        if self.tol is not None and self.n_iter_ == self.max_iter:
            warnings.warn("到达 max_iter，建议增加训练轮数", ConvergenceWarning)
        return self

    def partial_fit(self, X, y, sample_weight=None):
        self._check_params()
        first_call = not hasattr(self, "coef_")
        X, y = validate_xy(self, X, y, multi_output=False, sparse_format="csr", reset=first_call)
        if first_call:
            self._initialize(X.shape[1], None, None)
            self.t_ = 1.0
        self._train(X, y, sample_weight, 1)
        return self

    def _train(self, X, y, sample_weight, max_iter):
        n_samples, n_features = X.shape
        weights = sample_weights(sample_weight, n_samples)
        if weights is None:
            weights = np.ones(n_samples)
        is_sparse = sparse.issparse(X)
        if is_sparse:
            X.sum_duplicates()
        rng = check_random_state(self.random_state)
        seed = int(rng.randint(0, RAND_R_MAX))
        rng.randint(1, RAND_R_MAX)  # make_dataset 消耗的 seed，shuffle 不使用它。
        order = np.arange(n_samples)
        dense_indices = np.arange(n_features)
        if self.average:
            vector = WeightVector(self._standard_coef, self._average_coef)
            intercept = float(self._standard_intercept[0])
            average_intercept = float(self._average_intercept[0])
        else:
            vector = WeightVector(self.coef_)
            intercept = float(self.intercept_[0])
            average_intercept = 0.0
        l1_ratio = 1.0 if self.penalty == "l1" else 0.0
        if self.penalty == "elasticnet":
            l1_ratio = self.l1_ratio
        q = np.zeros(n_features) if l1_ratio > 0 else None
        accumulated_penalty = 0.0
        t, eta = self.t_, self.eta0
        best_objective, no_improvement = np.inf, 0
        intercept_decay = 0.01 if is_sparse else 1.0
        optimal_init = 0.0
        if self.learning_rate == "optimal":
            typical_weight = np.sqrt(1.0 / np.sqrt(self.alpha))
            _, gradient = loss_value_and_gradient(self.loss, 1.0, -typical_weight, self.epsilon)
            initial_eta = typical_weight / max(1.0, gradient)
            optimal_init = 1.0 / (initial_eta * self.alpha)

        for epoch in range(max_iter):
            if self.shuffle:
                shuffle_indices(order, seed)
            objective_sum = 0.0
            for i in order:
                if is_sparse:
                    start, end = X.indptr[i:i + 2]
                    indices, values = X.indices[start:end], X.data[start:end]
                else:
                    indices, values = dense_indices, X[i]
                prediction = vector.dot(values, indices) + intercept
                if self.learning_rate == "optimal":
                    eta = 1.0 / (self.alpha * (optimal_init + t - 1))
                elif self.learning_rate == "invscaling":
                    eta = self.eta0 / t ** self.power_t
                loss, gradient = loss_value_and_gradient(self.loss, y[i], prediction, self.epsilon)
                if self.tol is not None:
                    # 1.9.1 的目标累计不乘样本权重，正则项读取 WeightVector 缓存。
                    objective_sum += loss
                    if self.penalty is not None:
                        objective_sum += self.alpha * (
                            (1 - l1_ratio) * 0.5 * vector.squared_norm
                            + l1_ratio * vector.l1_norm
                        )
                gradient = max(-1e12, min(1e12, gradient))
                update = -eta * gradient * weights[i]
                if self.penalty in {"l2", "elasticnet"}:
                    vector.scale(max(0.0, 1.0 - (1 - l1_ratio) * eta * self.alpha))
                if update != 0:
                    vector.add(values, indices, update)
                if self.fit_intercept:
                    intercept += update * intercept_decay
                if 0 < self.average <= t:
                    count = t - self.average + 1
                    vector.add_average(values, indices, update, count)
                    average_intercept += (intercept - average_intercept) / count
                if q is not None:
                    accumulated_penalty += l1_ratio * eta * self.alpha
                    vector.truncate_l1(indices, q, accumulated_penalty)
                t += 1

            if not np.isfinite(intercept) or not np.isfinite(vector.raw).all():
                raise ValueError("SGD 出现非有限参数，请缩放输入或调整学习率")
            if self.tol is not None:
                objective = objective_sum / n_samples
                no_improvement = no_improvement + 1 if objective > best_objective - self.tol else 0
                best_objective = min(best_objective, objective)
                if no_improvement >= self.n_iter_no_change:
                    if self.learning_rate == "adaptive" and eta > 1e-6:
                        eta /= 5.0
                        no_improvement = 0
                    else:
                        break
        vector.reset_scale()
        self.n_iter_ = epoch + 1
        self.t_ += self.n_iter_ * n_samples
        if self.average:
            self._standard_intercept = np.asarray([intercept])
            self._average_intercept = np.asarray([average_intercept])
            if self.average <= self.t_ - 1:
                self.coef_, self.intercept_ = self._average_coef, self._average_intercept
            else:
                self.coef_, self.intercept_ = self._standard_coef, self._standard_intercept
        else:
            self.intercept_ = np.asarray([intercept])
