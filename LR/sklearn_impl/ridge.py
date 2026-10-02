"""参考 _ridge.py 的五种 CPU 求解路径，系数和截距计算手动实现。"""

import warnings

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg, lsqr
from sklearn.exceptions import ConvergenceWarning

try:
    from .common import (
        LinearRegressor, centered_operator, iteration_count, positive_number,
        preprocess, sample_weights, set_intercept, validate_xy,
    )
except ImportError:
    from LR.sklearn_impl.common import (
        LinearRegressor, centered_operator, iteration_count, positive_number,
        preprocess, sample_weights, set_intercept, validate_xy,
    )


def solve_svd(X, y, alpha):
    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    mask = singular_values > 1e-15
    factors = np.zeros((len(singular_values), len(alpha)))
    values = singular_values[mask, None]
    factors[mask] = values / (values * values + alpha)
    return (Vt.T @ (factors * (U.T @ y))).T


def solve_cholesky(X, y, alpha):
    """样本少于特征时转到样本空间；其余情况解特征空间方程。"""
    wide = X.shape[1] > X.shape[0]
    product = X @ X.T if wide else X.T @ X
    matrix = product.toarray() if sparse.issparse(product) else np.asarray(product)
    rhs = y if wide else np.asarray(X.T @ y)
    same_alpha = np.all(alpha == alpha[0])
    solutions = []
    penalties = [alpha[0]] if same_alpha else alpha
    for k, penalty in enumerate(penalties):
        regularized = matrix.copy()
        regularized.flat[:: len(matrix) + 1] += penalty
        target = rhs if same_alpha else rhs[:, k]
        try:
            solution = linalg.solve(regularized, target, assume_a="pos")
        except linalg.LinAlgError:
            if wide:
                solution = linalg.lstsq(regularized, target)[0]
            elif not sparse.issparse(X):
                return solve_svd(X, y, alpha), "svd"
            else:
                raise ValueError("稀疏 Cholesky 的奇异问题请改用 lsqr 或 sparse_cg")
        solutions.append(solution)
    result = solutions[0] if same_alpha else np.column_stack(solutions)
    if wide:
        result = np.asarray(X.T @ result)
    return result.T, "cholesky"


class Ridge(LinearRegressor):
    def __init__(
        self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None,
        tol=1e-4, solver="auto", positive=False,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        if self.solver not in {"auto", "cholesky", "svd", "lsqr", "sparse_cg"}:
            raise ValueError("支持的 solver 为 auto/cholesky/svd/lsqr/sparse_cg")
        if self.positive:
            raise ValueError("本实现未包含 positive Ridge 的 lbfgs 路径")
        positive_number(self.tol, "tol")
        iteration_count(self.max_iter, allow_none=True)
        X, y = validate_xy(self, X, y)
        is_sparse = sparse.issparse(X)
        if is_sparse and self.solver == "svd":
            raise ValueError("svd 不支持稀疏输入")
        if is_sparse and self.fit_intercept and self.solver == "cholesky":
            raise ValueError("稀疏输入拟合截距应使用 lsqr 或 sparse_cg")
        self.solver_ = ("sparse_cg" if is_sparse else "cholesky") if self.solver == "auto" else self.solver
        weights = sample_weights(sample_weight, X.shape[0])
        X, y, offset, y_offset, roots = preprocess(
            X, y, fit_intercept=self.fit_intercept, copy_X=self.copy_X,
            weights=weights,
        )
        targets = y[:, None] if y.ndim == 1 else y
        alpha = np.asarray(self.alpha, dtype=np.float64).reshape(-1)
        if alpha.size == 1:
            alpha = np.full(targets.shape[1], alpha.item())
        if alpha.shape != (targets.shape[1],) or not np.isfinite(alpha).all() or np.any(alpha < 0):
            raise ValueError("alpha 必须非负，且为标量或每目标一个值")
        self.n_iter_ = None
        if self.solver_ == "svd":
            coefficients = solve_svd(X, targets, alpha)
        elif self.solver_ == "cholesky":
            coefficients, self.solver_ = solve_cholesky(X, targets, alpha)
        else:
            operator = centered_operator(X, offset, roots) if is_sparse else aslinearoperator(X)
            coefficients = np.empty((targets.shape[1], X.shape[1]))
            if self.solver_ == "lsqr":
                self.n_iter_ = np.empty(targets.shape[1], dtype=np.int32)
                for k in range(targets.shape[1]):
                    result = lsqr(
                        operator, targets[:, k], damp=np.sqrt(alpha[k]),
                        atol=self.tol, btol=self.tol, iter_lim=self.max_iter,
                    )
                    coefficients[k], self.n_iter_[k] = result[0], result[2]
            else:
                wide = X.shape[1] > X.shape[0]
                for k in range(targets.shape[1]):
                    if wide:
                        def matvec(vector, penalty=alpha[k]):
                            return operator.matvec(operator.rmatvec(vector)) + penalty * vector
                        rhs = targets[:, k]
                        size = X.shape[0]
                    else:
                        def matvec(vector, penalty=alpha[k]):
                            return operator.rmatvec(operator.matvec(vector)) + penalty * vector
                        rhs = operator.rmatvec(targets[:, k])
                        size = X.shape[1]
                    matrix = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
                    # _solve_sparse_cg 的样本空间分支也不传 max_iter。
                    solution, info = cg(
                        matrix, rhs, rtol=self.tol, atol=0.0,
                        maxiter=None if wide else self.max_iter,
                    )
                    if info < 0:
                        raise ValueError("共轭梯度求解失败")
                    if info > 0:
                        warnings.warn("共轭梯度未达到指定容差", ConvergenceWarning)
                    coefficients[k] = operator.rmatvec(solution) if wide else solution
        self.coef_ = coefficients[0] if targets.shape[1] == 1 else coefficients
        set_intercept(self, offset, y_offset)
        return self
