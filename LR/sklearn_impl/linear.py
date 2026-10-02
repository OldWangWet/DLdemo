"""普通最小二乘：稠密 lstsq、稀疏 LSQR、稠密非负最小二乘。"""

import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr

try:
    from .common import (
        LinearRegressor, centered_operator, positive_number, preprocess,
        sample_weights, set_intercept, validate_xy,
    )
except ImportError:
    from LR.sklearn_impl.common import (
        LinearRegressor, centered_operator, positive_number, preprocess,
        sample_weights, set_intercept, validate_xy,
    )


class LinearRegression(LinearRegressor):
    def __init__(self, *, fit_intercept=True, copy_X=True, tol=1e-6, positive=False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        positive_number(self.tol, "tol")
        X, y = validate_xy(self, X, y)
        if self.positive and sparse.issparse(X):
            raise ValueError("positive=True 仅支持稠密输入")
        weights = sample_weights(sample_weight, X.shape[0])
        X, y, offset, y_offset, roots = preprocess(
            X, y, fit_intercept=self.fit_intercept, copy_X=self.copy_X,
            weights=weights,
        )
        # 再次 fit 时不保留仅属于旧稠密解的属性。
        for attribute in ("rank_", "singular_"):
            self.__dict__.pop(attribute, None)
        targets = y[:, None] if y.ndim == 1 else y
        if self.positive:
            coefficients = np.vstack([
                optimize.nnls(X, targets[:, j])[0] for j in range(targets.shape[1])
            ])
        elif sparse.issparse(X):
            operator = centered_operator(X, offset, roots)
            coefficients = np.vstack([
                lsqr(operator, targets[:, j], atol=self.tol, btol=self.tol)[0]
                for j in range(targets.shape[1])
            ])
        else:
            result, _, self.rank_, self.singular_ = linalg.lstsq(X, targets, cond=self.tol)
            coefficients = result.T
        self.coef_ = coefficients[0] if y.ndim == 1 else coefficients
        set_intercept(self, offset, y_offset)
        return self
