"""逐样本 SGD：L2 延迟缩放与 L1 累计截断。"""

import numpy as np


class SGDRegressor:
    """平方误差；固定轮数、每轮打乱、invscaling 学习率。"""

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
        # 实际系数 w = scale * v；L2 收缩只更新标量 scale。
        v = np.zeros(X.shape[1])
        scale, b = 1.0, 0.0
        q = np.zeros(X.shape[1])
        u = 0.0
        rng = np.random.default_rng(self.random_state)
        t = 1

        for epoch in range(self.max_iter):
            for i in rng.permutation(len(y)):
                eta = self.eta0 / t**self.power_t
                error = scale * (X[i] @ v) + b - y[i]
                update = -eta * error
                if self.penalty == "l2":
                    scale *= max(0.0, 1.0 - eta * self.alpha)
                    if scale < 1e-9:
                        v *= scale
                        scale = 1.0
                v += (update / scale) * X[i]
                b += update

                if self.penalty == "l1":
                    u += eta * self.alpha
                    # 按截断前的符号选择分支；q 记录各坐标累计实际收缩。
                    z = v.copy()
                    positive, negative = z > 0.0, z < 0.0
                    v[positive] = np.maximum(0.0, z[positive] - (u + q[positive]))
                    v[negative] = np.minimum(0.0, z[negative] + (u - q[negative]))
                    q += v - z
                t += 1

        self.coef_ = scale * v
        self.intercept_ = float(b)
        self.n_iter_ = epoch + 1
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
