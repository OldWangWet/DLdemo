"""本目录共享的校验、加权中心化和线性预测，不调用 sklearn 模型内核。"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class LinearRegressor(RegressorMixin, BaseEstimator):
    """只复用公共 estimator 工具，预测的矩阵计算在这里实现。"""

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = validate_data(
            self, X, accept_sparse=["csr", "csc", "coo"],
            dtype=np.float64, reset=False,
        )
        coefficients = self.coef_ if self.coef_.ndim == 1 else self.coef_.T
        return np.asarray(X @ coefficients + self.intercept_)


def validate_xy(estimator, X, y, *, multi_output=True, sparse_format=None, reset=True):
    return validate_data(
        estimator, X, y, accept_sparse=sparse_format or ["csr", "csc", "coo"],
        dtype=np.float64, y_numeric=True, multi_output=multi_output,
        force_writeable=True, reset=reset,
    )


def sample_weights(sample_weight, n_samples):
    """样本权重统一为非负 float64 向量，None 保持为 None。"""
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full(n_samples, weights.item(), dtype=np.float64)
    if weights.shape != (n_samples,):
        raise ValueError("sample_weight 必须是标量或与样本数一致的一维数组")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weight 必须有限、非负，且总和大于零")
    return weights


def preprocess(X, y, *, fit_intercept, copy_X, weights=None, rescale=True):
    """与 _preprocess_data 的 CPU 路径对应；稀疏 X 只记录均值。"""
    X = X.copy() if copy_X else X
    y = np.array(y, dtype=np.float64, copy=True)
    n_samples, n_features = X.shape
    if fit_intercept:
        if sparse.issparse(X):
            if weights is None:
                offset = np.asarray(X.mean(axis=0)).ravel()
            else:
                offset = np.asarray(X.T @ weights).ravel() / weights.sum()
        else:
            offset = np.average(X, axis=0, weights=weights)
            X -= offset
        y_offset = np.average(y, axis=0, weights=weights)
        y -= y_offset
    else:
        offset = np.zeros(n_features)
        y_offset = 0.0 if y.ndim == 1 else np.zeros(y.shape[1])

    root_weights = None
    if weights is not None and rescale:
        root_weights = np.sqrt(weights)
        if sparse.issparse(X):
            X = X.multiply(root_weights[:, None]).asformat(X.format)
        else:
            X *= root_weights[:, None]
        y *= root_weights if y.ndim == 1 else root_weights[:, None]
    return X, y, offset, y_offset, root_weights


def centered_operator(X, offset, root_weights=None):
    """表示 sqrt(S) * (X_original - mean)，不显式中心化稀疏矩阵。"""
    roots = np.ones(X.shape[0]) if root_weights is None else root_weights

    def matvec(vector):
        return np.asarray(X @ vector).ravel() - roots * (offset @ vector)

    def rmatvec(vector):
        return np.asarray(X.T @ vector).ravel() - offset * (roots @ vector)

    return LinearOperator(X.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)


def set_intercept(estimator, offset, y_offset):
    if estimator.fit_intercept:
        coefficients = estimator.coef_ if estimator.coef_.ndim == 1 else estimator.coef_.T
        estimator.intercept_ = y_offset - offset @ coefficients
    else:
        estimator.intercept_ = 0.0


def positive_number(value, name, *, allow_zero=True):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} 必须是有限标量")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} 的取值不在支持范围内")


def iteration_count(value, name="max_iter", *, allow_none=False):
    if allow_none and value is None:
        return
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} 必须为正整数")
