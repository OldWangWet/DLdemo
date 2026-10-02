"""每次使用全部样本的梯度下降，L1 使用普通次梯度。"""

import numpy as np


class BGDRegressor:
    """平方误差 + 无正则/L2/L1，固定学习率和更新次数。"""

    def __init__(self, penalty=None, alpha=0.0, eta0=0.1, max_iter=10000):
        self.penalty = penalty
        self.alpha = alpha
        self.eta0 = eta0
        self.max_iter = max_iter

    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0.0
        m = len(y)

        for epoch in range(self.max_iter):
            error = X @ w + b - y
            gradient = X.T @ error / m
            if self.penalty == "l2":
                gradient += self.alpha * w
            elif self.penalty == "l1":
                gradient += self.alpha * np.sign(w)

            w -= self.eta0 * gradient
            b -= self.eta0 * error.mean()

        self.coef_ = w
        self.intercept_ = float(b)
        self.n_iter_ = epoch + 1
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
