"""逐样本梯度下降，L1 使用普通次梯度。"""

import numpy as np


class SGDRegressor:
    """平方误差 + 无正则/L2/L1，按累计更新次数递减学习率。"""

    def __init__(
        self, penalty=None, alpha=0.0, max_iter=10000,
        eta0=0.01, power_t=0.25, random_state=42,
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state

    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0.0
        rng = np.random.default_rng(self.random_state)
        t = 1

        for epoch in range(self.max_iter):
            for i in rng.permutation(len(y)):
                eta = self.eta0 / t**self.power_t
                error = X[i] @ w + b - y[i]
                gradient = error * X[i]
                if self.penalty == "l2":
                    gradient += self.alpha * w
                elif self.penalty == "l1":
                    gradient += self.alpha * np.sign(w)

                w -= eta * gradient
                b -= eta * error
                t += 1

        self.coef_ = w
        self.intercept_ = float(b)
        self.n_iter_ = epoch + 1
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
