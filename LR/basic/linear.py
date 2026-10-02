"""普通最小二乘、Ridge 和 Lasso，仅用于标准化后的 diabetes 数据。"""

import numpy as np


class LinearRegression:
    """中心化后，用正规方程求普通最小二乘解。"""

    def fit(self, X, y):
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean
        self.coef_ = np.linalg.solve(Xc.T @ Xc, Xc.T @ yc)
        self.intercept_ = float(y_mean - X_mean @ self.coef_)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class Ridge:
    """目标为 SSE + alpha * ||w||²；截距不参与正则化。"""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = float(y_mean - X_mean @ self.coef_)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class Lasso:
    """循环坐标下降：每次用软阈值更新一个系数。"""

    def __init__(self, alpha=1.0, max_iter=10000, tol=1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean
        m, n = X.shape
        w = np.zeros(n)

        for epoch in range(self.max_iter):
            old_w = w.copy()
            for j in range(n):
                x = Xc[:, j]
                # 排除当前特征，固定其它系数，求这一维的最优值。
                residual = yc - Xc @ w + x * w[j]
                rho = x @ residual / m
                z = x @ x / m
                w[j] = np.sign(rho) * max(abs(rho) - self.alpha, 0.0) / z

            self.n_iter_ = epoch + 1
            if np.max(np.abs(w - old_w)) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = float(y_mean - X_mean @ w)
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
