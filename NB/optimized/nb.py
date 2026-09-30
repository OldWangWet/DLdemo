"""参考 sklearn 1.9.1 naive_bayes.py 的算法与数学优化。

模型计算仅使用 NumPy/SciPy；保留本阶段的标量平滑和精简接口。
"""

from numbers import Real
import warnings

import numpy as np
from scipy import sparse
from scipy.special import logsumexp


def _finite_scalar(value, name, minimum=0, strict=False):
    if (not isinstance(value, Real) or not np.isfinite(value)
            or (value <= minimum if strict else value < minimum)):
        relation = "大于" if strict else "不小于"
        raise ValueError(f"{name} 必须为有限标量且{relation} {minimum}")


class _BaseNB:
    """公共校验、类别映射及按需归一化，类别顺序与 np.unique 一致。"""

    def _prepare_X(self, X):
        if sparse.issparse(X):
            raise ValueError("本阶段仅支持稠密输入")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X 必须是有限数值组成的二维特征矩阵")
        return X

    def _prepare_training(self, X, y):
        X = self._prepare_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != X.shape[0] or len(y) == 0:
            raise ValueError("y 必须是一维标签，且与 X 的非零样本数一致")
        if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
            raise ValueError("标签必须有限")
        return X, y

    def fit(self, X, y):
        X, y = self._prepare_training(X, y)
        self._is_fitted = False
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._fit(X, y)
        self._is_fitted = True
        return self

    def _validate_predict_X(self, X):
        if not getattr(self, "_is_fitted", False):
            raise ValueError("请先调用 fit")
        X = self._prepare_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("预测特征数必须与训练特征数一致")
        return X

    def joint_log_likelihood(self, X):
        """返回 (样本数, 类别数) 得分；补集模型返回分类得分。"""
        return self._joint_log_likelihood(self._validate_predict_X(X))

    def predict(self, X):
        scores = self.joint_log_likelihood(X)
        return self.classes_[scores.argmax(axis=1)]

    def predict_log_proba(self, X):
        """归一化对数得分；补集输出不表示生成模型的后验概率。"""
        scores = self.joint_log_likelihood(X)
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))


class GaussianNB(_BaseNB):
    """中心化方差、逐类预测和 sklearn 的批次均值方差合并。"""

    def __init__(self, var_smoothing=1e-9):
        _finite_scalar(var_smoothing, "var_smoothing")
        self.var_smoothing = var_smoothing

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X):
        """合并未平滑的最大似然统计量，对应源码同名方法。"""
        if X.shape[0] == 0:
            return mu, var
        n_new = X.shape[0]
        new_var = X.var(axis=0)
        new_mu = X.mean(axis=0)
        if n_past == 0:
            return new_mu, new_var
        n_total = float(n_past + n_new)
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        total_ssd = n_past * var + n_new * new_var
        total_ssd += (n_new * n_past / n_total) * (mu - new_mu) ** 2
        return total_mu, total_ssd / n_total

    def _fit(self, X, y):
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        shape = (len(self.classes_), self.n_features_in_)
        self.theta_ = np.zeros(shape)
        self.var_ = np.zeros(shape)
        self.class_count_ = np.zeros(len(self.classes_))
        for i, label in enumerate(self.classes_):
            group = X[y == label]
            self.theta_[i], self.var_[i] = self._update_mean_variance(
                self.class_count_[i], self.theta_[i], self.var_[i], group
            )
            self.class_count_[i] += len(group)
        self.var_ += self.epsilon_
        if not np.isfinite(self.var_).all() or (self.var_ <= 0).any():
            raise ValueError("高斯方差必须大于 0，请使用非恒定数据和正的方差平滑")
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

    def _joint_log_likelihood(self, X):
        scores = []
        for i in range(len(self.classes_)):
            n_ij = -0.5 * np.log(2 * np.pi * self.var_[i]).sum()
            n_ij -= 0.5 * (((X - self.theta_[i]) ** 2) / self.var_[i]).sum(axis=1)
            scores.append(np.log(self.class_prior_[i]) + n_ij)
        return np.column_stack(scores)


class _BaseDiscreteNB(_BaseNB):
    def __init__(self, alpha=1.0, force_alpha=True):
        _finite_scalar(alpha, "alpha", strict=True)
        if not isinstance(force_alpha, (bool, np.bool_)):
            raise ValueError("force_alpha 必须为布尔值")
        self.alpha = alpha
        self.force_alpha = force_alpha

    def _check_alpha(self):
        """保留源码的可选数值下限；不改变构造参数 alpha。"""
        alpha_lower_bound = 1e-10
        if self.alpha < alpha_lower_bound and not self.force_alpha:
            warnings.warn(
                "alpha too small will result in numeric errors, setting alpha ="
                f" {alpha_lower_bound:.1e}. Use `force_alpha=True` to keep alpha"
                " unchanged."
            )
            return np.maximum(self.alpha, alpha_lower_bound)
        return self.alpha

    def _prepare_X(self, X):
        X = super()._prepare_X(X)
        if (X < 0).any():
            raise ValueError("离散 NB 的输入必须非负")
        return X

    def _fit(self, X, y):
        self._init_counters()
        # 等价于源码 LabelBinarizer 的多类、二类及单类指示矩阵。
        Y = (y[:, None] == self.classes_[None, :]).astype(np.int64)
        self._count(X, Y)
        self._update_feature_log_prob(self._check_alpha())
        self.class_log_prior_ = (
            np.log(self.class_count_) - np.log(self.class_count_.sum())
        )

    def _init_counters(self):
        self.class_count_ = np.zeros(len(self.classes_))
        self.feature_count_ = np.zeros((len(self.classes_), self.n_features_in_))

    def _count(self, X, Y):
        self.feature_count_ += Y.T @ X
        self.class_count_ += Y.sum(axis=0)


class MultinomialNB(_BaseDiscreteNB):
    def _update_feature_log_prob(self, alpha):
        counts = self.feature_count_ + alpha
        self.feature_log_prob_ = np.log(counts) - np.log(counts.sum(axis=1))[:, None]

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_


class BernoulliNB(_BaseDiscreteNB):
    def __init__(self, alpha=1.0, binarize=0.0, force_alpha=True):
        super().__init__(alpha, force_alpha)
        if binarize is not None and (
            not isinstance(binarize, Real) or not np.isfinite(binarize)
        ):
            raise ValueError("binarize 必须是有限标量或 None")
        self.binarize = binarize

    def _prepare_X(self, X):
        X = _BaseNB._prepare_X(self, X)
        if self.binarize is not None:
            return (X > self.binarize).astype(float)
        if ((X != 0) & (X != 1)).any():
            raise ValueError("binarize=None 时输入必须为 0 或 1")
        return X

    def _update_feature_log_prob(self, alpha):
        self.feature_log_prob_ = np.log(self.feature_count_ + alpha)
        self.feature_log_prob_ -= np.log(self.class_count_ + 2 * alpha)[:, None]

    def _joint_log_likelihood(self, X):
        # 按源码在预测时计算未出现概率、权重差和偏置。
        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        scores = X @ (self.feature_log_prob_ - neg_prob).T
        return scores + self.class_log_prior_ + neg_prob.sum(axis=1)


class CategoricalNB(_BaseDiscreteNB):
    def __init__(self, alpha=1.0, min_categories=None, force_alpha=True):
        super().__init__(alpha, force_alpha)
        if min_categories is not None:
            _finite_scalar(min_categories, "min_categories", minimum=1)
            if min_categories != int(min_categories):
                raise ValueError("min_categories 必须为正整数")
        self.min_categories = min_categories

    def _prepare_X(self, X):
        X = super()._prepare_X(X)
        if (X != np.floor(X)).any() or (X >= np.iinfo(np.int64).max).any():
            raise ValueError("类别特征必须是可表示的非负整数编码")
        return X.astype(np.int64)

    def _init_counters(self):
        self.class_count_ = np.zeros(len(self.classes_))
        self.category_count_ = [
            np.zeros((len(self.classes_), 0)) for _ in range(self.n_features_in_)
        ]

    def _count(self, X, Y):
        self.class_count_ += Y.sum(axis=0)
        self.n_categories_ = np.maximum(X.max(axis=0) + 1, self.min_categories or 1)
        for j, size in enumerate(self.n_categories_):
            counts = self.category_count_[j]
            if size > counts.shape[1]:
                counts = np.pad(counts, ((0, 0), (0, size - counts.shape[1])))
            for c in range(len(self.classes_)):
                hist = np.bincount(X[Y[:, c].astype(bool), j])
                indices = np.nonzero(hist)[0]
                counts[c, indices] += hist[indices]
            self.category_count_[j] = counts

    def _update_feature_log_prob(self, alpha):
        self.feature_log_prob_ = []
        for table in self.category_count_:
            counts = table + alpha
            self.feature_log_prob_.append(
                np.log(counts) - np.log(counts.sum(axis=1))[:, None]
            )

    def _joint_log_likelihood(self, X):
        if (X >= self.n_categories_).any():
            raise ValueError("预测中出现了超出已设定范围的类别编码")
        scores = np.zeros((len(X), len(self.classes_)))
        for j, log_prob in enumerate(self.feature_log_prob_):
            scores += log_prob[:, X[:, j]].T
        return scores + self.class_log_prior_


class ComplementNB(_BaseDiscreteNB):
    """补集计数直接生成权重；至少两类，预测不加类别先验。"""

    def __init__(self, alpha=1.0, norm=False, force_alpha=True):
        super().__init__(alpha, force_alpha)
        if not isinstance(norm, (bool, np.bool_)):
            raise ValueError("norm 必须为布尔值")
        self.norm = norm

    def _init_counters(self):
        if len(self.classes_) < 2:
            raise ValueError("补集 NB 要求至少两个类别")
        super()._init_counters()

    def _count(self, X, Y):
        super()._count(X, Y)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        counts = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(counts / counts.sum(axis=1, keepdims=True))
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            if not np.isfinite(logged).all() or (summed == 0).any():
                self._is_fitted = False
                raise ValueError("norm=True 要求有限补集权重且每行权重和非零")
            self.feature_log_prob_ = logged / summed
        else:
            self.feature_log_prob_ = -logged

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T
