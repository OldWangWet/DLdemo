"""运行六组独立手写模型，与上层示例相同参数的 sklearn 模型对照。"""

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

from LR.optimized.linear import Lasso, LinearRegression, Ridge
from LR.optimized.sgd import SGDRegressor


L2_ALPHA = 0.1
L1_ALPHA = 1.0
MAX_ITER = 10000
SGD_PARAMS = {
    "max_iter": MAX_ITER, "eta0": 0.01, "power_t": 0.25, "random_state": 42,
}


def create_models(n_samples):
    return {
        "OLS": LinearRegression(),
        "Ridge-L2": Ridge(alpha=n_samples * L2_ALPHA),
        "Lasso-L1": Lasso(alpha=L1_ALPHA, max_iter=MAX_ITER, tol=1e-8),
        "SGD-none": SGDRegressor(penalty=None, alpha=0.0, **SGD_PARAMS),
        "SGD-L2": SGDRegressor(penalty="l2", alpha=L2_ALPHA, **SGD_PARAMS),
        "SGD-L1": SGDRegressor(penalty="l1", alpha=L1_ALPHA, **SGD_PARAMS),
    }


def create_references(n_samples):
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
    """统一为 SSE/(2m) + 正则项，截距不参与正则化。"""
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
        dataset.data, dataset.target, test_size=0.2, random_state=42,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"sklearn {sklearn.__version__}；diabetes：训练 {X_train.shape}，测试 {X_test.shape}")
    print("test_size=0.2，random_state=42；仅用训练集拟合 StandardScaler；均拟合截距")
    print(f"Ridge.alpha={len(y_train) * L2_ALPHA:g}；SGD-L2.alpha={L2_ALPHA}；L1.alpha={L1_ALPHA}")
    print("Lasso.tol=1e-8（相对系数变化 + 对偶间隙）；SGD 固定 10000 轮")
    print("SGD：eta0=0.01，power_t=0.25，invscaling；NumPy 打乱顺序不同于 sklearn")
    print("\nO=optimized，SK=sklearn，NZ=非零系数数，Iter=Lasso 坐标轮/SGD 遍历轮，J=训练目标")
    print(
        f"{'Model':<12}{'MSE(O)':>12}{'MSE(SK)':>12}"
        f"{'R2(O)':>10}{'R2(SK)':>10}{'NZ(O/SK)':>10}{'Iter(O/SK)':>15}"
        f"{'MaxDiff':>12}{'J(O)':>12}{'J(SK)':>12}", flush=True,
    )
    references = create_references(len(y_train))
    for name, model in create_models(len(y_train)).items():
        model.fit(X_train, y_train)
        reference = references[name].fit(X_train, y_train)
        prediction, expected = model.predict(X_test), reference.predict(X_test)
        nonzero = f"{np.count_nonzero(model.coef_)}/{np.count_nonzero(reference.coef_)}"
        iterations = f"{getattr(model, 'n_iter_', None) or '-'}/{getattr(reference, 'n_iter_', None) or '-'}"
        print(
            f"{name:<12}{mean_squared_error(y_test, prediction):>12.6f}"
            f"{mean_squared_error(y_test, expected):>12.6f}"
            f"{r2_score(y_test, prediction):>10.6f}{r2_score(y_test, expected):>10.6f}"
            f"{nonzero:>10}{iterations:>15}{np.max(np.abs(prediction - expected)):>12.6g}"
            f"{training_objective(name, model, X_train, y_train):>12.6f}"
            f"{training_objective(name, reference, X_train, y_train):>12.6f}", flush=True,
        )
        if name == "Lasso-L1":
            print(f"  Lasso dual_gap_：optimized={model.dual_gap_:.6g}，sklearn={reference.dual_gap_:.6g}", flush=True)


if __name__ == "__main__":
    main()
