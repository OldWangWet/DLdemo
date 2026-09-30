"""按 NB/readme.md 的公式实现五种 NB，仅依赖 NumPy。"""

import numpy as np


class _BaseNB:
    """共享类别统计和预测流程，子类负责条件分布。"""

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X 必须是有限数值组成的二维特征矩阵")
        return X

    def fit(self, X, y):
        X = self._prepare_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError("y 必须是一维标签，且与 X 的非零样本数一致")
        self.n_features_in_ = X.shape[1]
        self.classes_, self.class_count_ = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(self.class_count_ / len(y))
        self._fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError("请先调用 fit")
        X = self._prepare_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("预测特征数必须与训练特征数一致")
        scores = np.column_stack([
            self._log_score(X, i) for i in range(len(self.classes_))
        ])
        return self.classes_[scores.argmax(axis=1)]


class GaussianNB(_BaseNB):
    def __init__(self, var_smoothing=1e-9):
        if var_smoothing < 0:
            raise ValueError("var_smoothing 必须非负")
        self.var_smoothing = var_smoothing

    def _fit(self, X, y):
        groups = [X[y == c] for c in self.classes_]
        self.theta_ = np.array([group.mean(axis=0) for group in groups])
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        self.var_ = np.array([group.var(axis=0) for group in groups])
        self.var_ += self.epsilon_
        if (self.var_ <= 0).any():
            raise ValueError("高斯方差必须大于 0，请使用非恒定数据和正的方差平滑")

    def _log_score(self, X, i):
        # ln P(c) - 1/2 * Σ[ln(2πσ²) + (x-μ)²/σ²]
        terms = np.log(2 * np.pi * self.var_[i])
        terms = terms + (X - self.theta_[i]) ** 2 / self.var_[i]
        return self.class_log_prior_[i] - 0.5 * terms.sum(axis=1)


class _DiscreteNB(_BaseNB):
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("基础实现要求 alpha > 0")
        self.alpha = alpha

    def _prepare_X(self, X):
        X = super()._prepare_X(X)
        if (X < 0).any():
            raise ValueError("离散 NB 的输入必须非负")
        return X


class MultinomialNB(_DiscreteNB):
    def _fit(self, X, y):
        self.feature_count_ = np.array([
            X[y == c].sum(axis=0) for c in self.classes_
        ])
        counts = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(counts / counts.sum(axis=1, keepdims=True))

    def _log_score(self, X, i):
        return self.class_log_prior_[i] + (X * self.feature_log_prob_[i]).sum(axis=1)


class CategoricalNB(_DiscreteNB):
    def __init__(self, alpha=1.0, min_categories=None):
        super().__init__(alpha)
        if min_categories is not None and (
            min_categories < 1 or min_categories != int(min_categories)
        ):
            raise ValueError("min_categories 必须为正整数")
        self.min_categories = min_categories

    def _prepare_X(self, X):
        X = super()._prepare_X(X)
        if (X != np.floor(X)).any():
            raise ValueError("类别特征必须是从 0 开始的整数编码")
        return X.astype(int)

    def _fit(self, X, y):
        self.n_categories_ = np.maximum(X.max(axis=0) + 1, self.min_categories or 1)
        self.category_count_ = []
        self.feature_log_prob_ = []
        for j, size in enumerate(self.n_categories_):
            counts = np.array([
                np.bincount(X[y == c, j], minlength=size) for c in self.classes_
            ])
            probs = (counts + self.alpha) / (
                self.class_count_[:, None] + self.alpha * size
            )
            self.category_count_.append(counts)
            self.feature_log_prob_.append(np.log(probs))

    def _log_score(self, X, i):
        if (X >= self.n_categories_).any():
            raise ValueError("预测中出现了超出已设定范围的类别编码")
        score = np.full(len(X), self.class_log_prior_[i])
        for j, log_prob in enumerate(self.feature_log_prob_):
            score += log_prob[i, X[:, j]]
        return score


class BernoulliNB(_DiscreteNB):
    def __init__(self, alpha=1.0, binarize=0.0):
        super().__init__(alpha)
        self.binarize = binarize

    def _prepare_X(self, X):
        X = _BaseNB._prepare_X(self, X)
        if self.binarize is not None:
            return (X > self.binarize).astype(float)
        if ((X != 0) & (X != 1)).any():
            raise ValueError("binarize=None 时输入必须为 0 或 1")
        return X

    def _fit(self, X, y):
        self.feature_count_ = np.array([
            X[y == c].sum(axis=0) for c in self.classes_
        ])
        probs = (self.feature_count_ + self.alpha) / (
            self.class_count_[:, None] + 2 * self.alpha
        )
        self.feature_log_prob_ = np.log(probs)
        self.feature_log_neg_prob_ = np.log1p(-probs)

    def _log_score(self, X, i):
        # 特征出现和未出现都参与计算。
        terms = X * self.feature_log_prob_[i]
        terms += (1 - X) * self.feature_log_neg_prob_[i]
        return self.class_log_prior_[i] + terms.sum(axis=1)


class ComplementNB(MultinomialNB):
    """对应 README 中 norm=False、至少两个类别的补集得分。"""

    def _fit(self, X, y):
        if len(self.classes_) < 2:
            raise ValueError("基础补集 NB 要求至少两个类别")
        super()._fit(X, y)
        counts = self.feature_count_.sum(axis=0) - self.feature_count_ + self.alpha
        self.feature_log_prob_ = -np.log(counts / counts.sum(axis=1, keepdims=True))

    def _log_score(self, X, i):
        # 补集概率的负对数得分，不添加类别先验。
        return (X * self.feature_log_prob_[i]).sum(axis=1)
