"""参考 sklearn 的稠密 OLS、Ridge 与 Lasso 求解流程。"""

import numpy as np
from scipy import linalg


class LinearRegression:
    """中心化后直接求最小二乘，避免构造 OLS 正规方程。"""

    def fit(self, X, y):
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean
        self.coef_ = linalg.lstsq(Xc, yc, cond=1e-6)[0]
        self.intercept_ = float(y_mean - X_mean @ self.coef_)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class Ridge:
    """目标为 SSE + alpha * ||w||²；利用正定系统求解。"""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean
        A = Xc.T @ Xc
        A.flat[:: X.shape[1] + 1] += self.alpha
        self.coef_ = linalg.solve(A, Xc.T @ yc, assume_a="pos")
        self.intercept_ = float(y_mean - X_mean @ self.coef_)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class Lasso:
    """缓存列范数、增量维护残差，并用对偶间隙确认收敛。"""

    def __init__(self, alpha=1.0, max_iter=10000, tol=1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean
        m, n = X.shape
        w = np.zeros(n)
        residual = yc.copy()
        norm2 = np.einsum("ij,ij->j", Xc, Xc)
        alpha = m * self.alpha
        gap_tol = self.tol * (yc @ yc)

        for epoch in range(self.max_iter):
            max_change = 0.0
            for j in range(n):
                old = w[j]
                rho = Xc[:, j] @ residual + old * norm2[j]
                w[j] = np.sign(rho) * max(abs(rho) - alpha, 0.0) / norm2[j]
                residual -= (w[j] - old) * Xc[:, j]
                max_change = max(max_change, abs(w[j] - old))

            max_coef = np.max(np.abs(w))
            if (max_coef == 0.0 or max_change <= self.tol * max_coef
                    or epoch == self.max_iter - 1):
                dual_norm = np.max(np.abs(Xc.T @ residual))
                scale = alpha / dual_norm if dual_norm > alpha else 1.0
                primal = 0.5 * (residual @ residual) + alpha * np.abs(w).sum()
                dual = -0.5 * scale**2 * (residual @ residual) + scale * (residual @ yc)
                gap = primal - dual
                if gap <= gap_tol:
                    break

        self.coef_ = w
        self.intercept_ = float(y_mean - X_mean @ w)
        self.n_iter_ = epoch + 1
        self.dual_gap_ = float(gap / m)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
