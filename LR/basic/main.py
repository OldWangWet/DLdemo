"""运行九组手写模型，并与同数据、同目标函数的 sklearn 模型对照。

运行方式：python LR/basic/main.py
"""

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from LR.basic.bgd import BGDRegressor
from LR.basic.linear import Lasso, LinearRegression, Ridge
from LR.basic.sgd import SGDRegressor


TEST_SIZE = 0.2
RANDOM_STATE = 42
L2_ALPHA = 0.1
L1_ALPHA = 1.0
MAX_ITER = 10000
SGD_PARAMS = {
    "max_iter": MAX_ITER,
    "eta0": 0.01,
    "power_t": 0.25,
    "random_state": RANDOM_STATE,
}
BGD_PARAMS = {"eta0": 0.1, "max_iter": MAX_ITER}
REFERENCE_NAMES = {
    "BGD-none": "OLS", "BGD-L2": "Ridge-L2", "BGD-L1": "Lasso-L1",
}


def create_models(n_samples):
    return {
        "OLS": LinearRegression(),
        "Ridge-L2": Ridge(alpha=n_samples * L2_ALPHA),
        "Lasso-L1": Lasso(alpha=L1_ALPHA, max_iter=MAX_ITER, tol=1e-8),
        "SGD-none": SGDRegressor(penalty=None, alpha=0.0, **SGD_PARAMS),
        "SGD-L2": SGDRegressor(penalty="l2", alpha=L2_ALPHA, **SGD_PARAMS),
        "SGD-L1": SGDRegressor(penalty="l1", alpha=L1_ALPHA, **SGD_PARAMS),
        "BGD-none": BGDRegressor(penalty=None, alpha=0.0, **BGD_PARAMS),
        "BGD-L2": BGDRegressor(penalty="l2", alpha=L2_ALPHA, **BGD_PARAMS),
        "BGD-L1": BGDRegressor(penalty="l1", alpha=L1_ALPHA, **BGD_PARAMS),
    }


def create_references(n_samples):
    # 与上层 main 的六个 sklearn 模型保持同一组参数。
    params = {
        **SGD_PARAMS, "loss": "squared_error", "fit_intercept": True,
        "tol": None, "shuffle": True, "learning_rate": "invscaling",
    }
    return {
        "OLS": SklearnLinearRegression(fit_intercept=True),
        "Ridge-L2": SklearnRidge(alpha=n_samples * L2_ALPHA, fit_intercept=True),
        "Lasso-L1": SklearnLasso(
            alpha=L1_ALPHA, fit_intercept=True, max_iter=MAX_ITER, tol=1e-8,
        ),
        "SGD-none": SklearnSGDRegressor(penalty=None, alpha=0.0, **params),
        "SGD-L2": SklearnSGDRegressor(penalty="l2", alpha=L2_ALPHA, **params),
        "SGD-L1": SklearnSGDRegressor(penalty="l1", alpha=L1_ALPHA, **params),
    }


def training_objective(name, model, X, y):
    """把九组模型的目标统一写为 SSE/(2m) + 正则项。"""
    error = model.predict(X) - y
    value = error @ error / (2 * len(y))
    if name.endswith("L2"):
        value += L2_ALPHA * (model.coef_ @ model.coef_) / 2
    elif name.endswith("L1"):
        value += L1_ALPHA * np.abs(model.coef_).sum()
    return value


def main():
    dataset = load_diabetes(scaled=False)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"sklearn 版本：{sklearn.__version__}", flush=True)
    print(f"diabetes：训练集 {X_train.shape}，测试集 {X_test.shape}")
    print("划分：test_size=0.2，random_state=42；仅用训练集拟合 StandardScaler")
    print(f"L2：Ridge.alpha={len(y_train) * L2_ALPHA:g}，SGD/BGD.alpha={L2_ALPHA}")
    print(f"L1：alpha={L1_ALPHA}，Lasso.tol=1e-8（最大系数变化）")
    print(f"SGD：{SGD_PARAMS}，每轮打乱，不提前停止")
    print(f"BGD：{BGD_PARAMS}，固定学习率，不提前停止")
    print("BGD 对照 OLS/Ridge/Lasso；手写 SGD/BGD 的 L1 均使用普通次梯度")

    references = create_references(len(y_train))
    for reference in references.values():
        reference.fit(X_train, y_train)

    print("\nB=basic，SK=sklearn，MaxDiff=最大测试预测差，J=统一训练目标")
    print("Nonzero 按精确非零统计；Iter：Lasso 为坐标轮，SGD 为遍历轮，BGD 为整批更新次")
    print(f"零初始化的训练目标：{y_train @ y_train / (2 * len(y_train)):.6f}")
    print(
        f"{'Model':<12}{'MSE(B)':>13}{'MSE(SK)':>13}"
        f"{'R2(B)':>10}{'R2(SK)':>10}{'Nonzero':>9}{'Iter':>8}"
        f"{'MaxDiff':>12}{'J(B)':>13}{'J(SK)':>13}",
        flush=True,
    )
    for name, model in create_models(len(y_train)).items():
        model.fit(X_train, y_train)
        reference = references[REFERENCE_NAMES.get(name, name)]
        prediction = model.predict(X_test)
        expected = reference.predict(X_test)
        print(
            f"{name:<12}{mean_squared_error(y_test, prediction):>13.6f}"
            f"{mean_squared_error(y_test, expected):>13.6f}"
            f"{r2_score(y_test, prediction):>10.6f}{r2_score(y_test, expected):>10.6f}"
            f"{np.count_nonzero(model.coef_):>9}{getattr(model, 'n_iter_', '-'):>8}"
            f"{np.max(np.abs(prediction - expected)):>12.6g}"
            f"{training_objective(name, model, X_train, y_train):>13.6f}"
            f"{training_objective(name, reference, X_train, y_train):>13.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
