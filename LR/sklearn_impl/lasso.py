"""手写稠密、CSC、Gram 坐标下降；不调用 sklearn 坐标下降内核。"""

import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

try:
    from .random_utils import RAND_R_MAX, our_rand_r
    from .common import (
        LinearRegressor, iteration_count, positive_number, preprocess,
        sample_weights, set_intercept, validate_xy,
    )
except ImportError:
    from LR.sklearn_impl.random_utils import RAND_R_MAX, our_rand_r
    from LR.sklearn_impl.common import (
        LinearRegressor, iteration_count, positive_number, preprocess,
        sample_weights, set_intercept, validate_xy,
    )


def lasso_gap(w, alpha, correlations, residual_norm, residual_y, positive):
    """_cd_fast 的 formulation A；alpha=0 改用一阶条件。"""
    if alpha == 0:
        return float(correlations @ correlations)
    dual_norm = correlations.max() if positive else np.abs(correlations).max()
    scale = alpha / dual_norm if dual_norm > alpha else 1.0
    primal = 0.5 * residual_norm + alpha * np.abs(w).sum()
    dual = -0.5 * scale * scale * residual_norm + scale * residual_y
    return float(primal - dual)


def coordinate_descent(X, y, w, *, alpha, tol, max_iter, positive, selection,
                       random_state, offset=None, weights=None, gram=None):
    """残差增量更新与相对变化/对偶间隙停止；省略 Gap Safe Screening。"""
    n_samples, n_features = X.shape
    state = int(check_random_state(random_state).randint(0, RAND_R_MAX))
    is_sparse = sparse.issparse(X)
    if is_sparse:
        sw = np.ones(n_samples) if weights is None else weights
        squared_y = y @ (sw * y)
        norms = np.empty(n_features)
        for j in range(n_features):
            start, end = X.indptr[j:j + 2]
            rows, values = X.indices[start:end], X.data[start:end]
            norms[j] = max(
                0.0, sw[rows] @ ((values - offset[j]) ** 2)
                + (sw.sum() - sw[rows].sum()) * offset[j] ** 2,
            )
        residual = sw * (y - X @ w + offset @ w)

        def gap():
            correlations = np.asarray(X.T @ residual).ravel() - offset * residual.sum()
            nonzero = sw > 0
            norm = np.sum(residual[nonzero] ** 2 / sw[nonzero])
            return lasso_gap(w, alpha, correlations, norm, residual @ y, positive)

    elif gram is not None:
        norms = np.diag(gram)
        Xy = X.T @ y
        correlations = Xy - gram @ w
        squared_y = y @ y

        def gap():
            norm = squared_y - 2 * (w @ Xy) + w @ (gram @ w)
            return lasso_gap(w, alpha, correlations, norm, squared_y - w @ Xy, positive)

    else:
        norms = np.einsum("ij,ij->j", X, X)
        squared_y = y @ y
        residual = y - X @ w

        def gap():
            return lasso_gap(w, alpha, X.T @ residual, residual @ residual, residual @ y, positive)

    tolerance = tol * squared_y
    current_gap = gap()
    if current_gap <= tolerance:
        return w, current_gap, 0

    for epoch in range(max_iter):
        max_change, max_weight = 0.0, 0.0
        for position in range(n_features):
            if selection == "random":
                state, value = our_rand_r(state)
                j = value % n_features
            else:
                j = position
            if norms[j] == 0:
                # 常量列/零列在中心化后没有可辨识的系数。
                w[j] = 0.0
                continue
            old = w[j]
            if is_sparse:
                start, end = X.indptr[j:j + 2]
                rows, values = X.indices[start:end], X.data[start:end]
                rho = values @ residual[rows] - offset[j] * residual.sum() + old * norms[j]
            elif gram is not None:
                rho = correlations[j] + old * norms[j]
            else:
                rho = X[:, j] @ residual + old * norms[j]
            if positive and rho < 0:
                new = 0.0
            else:
                new = np.sign(rho) * max(abs(rho) - alpha, 0.0) / norms[j]
            w[j] = new
            delta = old - new
            if delta != 0:
                if is_sparse:
                    residual[rows] += delta * sw[rows] * values
                    residual -= delta * offset[j] * sw
                elif gram is not None:
                    correlations += delta * gram[:, j]
                else:
                    residual += delta * X[:, j]
            max_change = max(max_change, abs(delta))
            max_weight = max(max_weight, abs(new))
        if max_weight == 0 or max_change <= tol * max_weight or epoch == max_iter - 1:
            current_gap = gap()
            if current_gap <= tolerance:
                break
    if current_gap > tolerance:
        warnings.warn("坐标下降未达到指定对偶间隙容差", ConvergenceWarning)
    return w, current_gap, epoch + 1


class Lasso(LinearRegressor):
    def __init__(
        self, alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True,
        max_iter=1000, tol=1e-4, warm_start=False, positive=False,
        random_state=None, selection="cyclic",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    @property
    def sparse_coef_(self):
        return sparse.csr_matrix(self.coef_)

    def fit(self, X, y, sample_weight=None):
        positive_number(self.alpha, "alpha")
        positive_number(self.tol, "tol")
        iteration_count(self.max_iter)
        if self.selection not in {"cyclic", "random"}:
            raise ValueError("selection 应为 cyclic 或 random")
        X, y = validate_xy(self, X, y, sparse_format="csc")
        if sparse.issparse(X):
            X.sum_duplicates()
        # sklearn Lasso 中标量权重不改变目标。
        weights = None if np.isscalar(sample_weight) else sample_weights(sample_weight, X.shape[0])
        if weights is not None:
            weights = weights * (X.shape[0] / weights.sum())
        X, y, offset, y_offset, _ = preprocess(
            X, y, fit_intercept=self.fit_intercept, copy_X=self.copy_X,
            weights=weights, rescale=not sparse.issparse(X),
        )
        targets = y[:, None] if y.ndim == 1 else y
        n_targets, n_features = targets.shape[1], X.shape[1]
        if self.warm_start and hasattr(self, "coef_"):
            coefficients = np.atleast_2d(self.coef_).copy()
            if coefficients.shape != (n_targets, n_features):
                raise ValueError("warm_start 的系数形状与新数据不一致")
        else:
            coefficients = np.zeros((n_targets, n_features))
        gram = None
        if not sparse.issparse(X):
            if isinstance(self.precompute, (bool, np.bool_)):
                if self.precompute:
                    gram = X.T @ X
            else:
                gram = np.asarray(self.precompute, dtype=np.float64)
                if gram.shape != (n_features, n_features):
                    raise ValueError("预计算 Gram 的形状应为 (n_features, n_features)")
                if weights is not None or (self.fit_intercept and not np.allclose(offset, 0)):
                    warnings.warn("中心化或加权改变了 X，重新计算 Gram 矩阵")
                    gram = X.T @ X
        gaps, iterations = [], []
        for k in range(n_targets):
            coefficients[k], gap, count = coordinate_descent(
                X, targets[:, k], coefficients[k], alpha=self.alpha * X.shape[0],
                tol=self.tol, max_iter=self.max_iter, positive=self.positive,
                selection=self.selection, random_state=self.random_state,
                offset=offset, weights=weights, gram=gram,
            )
            gaps.append(gap / X.shape[0])
            iterations.append(count)
        self.coef_ = coefficients[0] if n_targets == 1 else coefficients
        self.dual_gap_ = gaps[0] if n_targets == 1 else np.asarray(gaps)
        self.n_iter_ = iterations[0] if n_targets == 1 else iterations
        set_intercept(self, offset, y_offset)
        return self
