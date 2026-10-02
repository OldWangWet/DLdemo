"""使用 sklearn 演示普通最小二乘、Ridge、Lasso 和三种 SGD 回归。

运行方式：python LR/main.py
后续三个目录的实现使用相同的数据划分和模型参数进行对照。
"""

import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TEST_SIZE = 0.2
RANDOM_STATE = 42
L2_ALPHA = 0.1
L1_ALPHA = 1.0
MAX_ITER = 10000
SGD_PARAMS = {
    "loss": "squared_error",
    "fit_intercept": True,
    "max_iter": MAX_ITER,
    # 固定训练轮数，不因随机损失波动提前停止。
    "tol": None,
    "shuffle": True,
    "random_state": RANDOM_STATE,
    "learning_rate": "invscaling",
    "eta0": 0.01,
    "power_t": 0.25,
}


def create_models(n_samples):
    """构造六组模型，保持对应的常规模型和 SGD 的目标函数一致。

    Ridge：SSE + alpha * ||w||²。
    SGD-L2：SSE / (2m) + alpha * ||w||² / 2。
    因此 Ridge 的 alpha 为 SGD-L2 的 alpha 乘训练样本数 m。
    Lasso 和 SGD-L1 均为 SSE / (2m) + alpha * ||w||₁。
    """
    return {
        "OLS": LinearRegression(fit_intercept=True),
        "Ridge-L2": Ridge(alpha=n_samples * L2_ALPHA, fit_intercept=True),
        "Lasso-L1": Lasso(
            alpha=L1_ALPHA, fit_intercept=True, max_iter=MAX_ITER, tol=1e-8
        ),
        "SGD-none": SGDRegressor(penalty=None, alpha=0.0, **SGD_PARAMS),
        "SGD-L2": SGDRegressor(penalty="l2", alpha=L2_ALPHA, **SGD_PARAMS),
        "SGD-L1": SGDRegressor(penalty="l1", alpha=L1_ALPHA, **SGD_PARAMS),
    }


def main():
    # 加载原始特征，划分后仅使用训练集拟合标准化器。
    dataset = load_diabetes(scaled=False)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = create_models(X_train.shape[0])
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    print(f"sklearn 版本：{sklearn.__version__}")
    print("数据集：糖尿病（diabetes），目标为一年后的疾病进展指标")
    print(f"训练集：{X_train.shape[0]} 个样本，{X_train.shape[1]} 个特征")
    print(f"测试集：{X_test.shape[0]} 个样本")
    print(f"划分参数：test_size={TEST_SIZE}，random_state={RANDOM_STATE}")
    print("预处理：所有模型使用同一套训练集标准化结果，均拟合截距")
    print("常规求解：OLS=LinearRegression，Ridge-L2=Ridge，Lasso-L1=Lasso")
    print("SGD 求解：SGDRegressor，分别设置无正则、L2、L1")
    print(f"L2 参数：SGD.alpha={L2_ALPHA:g}，Ridge.alpha={X_train.shape[0] * L2_ALPHA:g}")
    print(f"L1 参数：Lasso.alpha=SGD.alpha={L1_ALPHA}，Lasso.tol=1e-8")
    print(f"SGD 参数：{SGD_PARAMS}")
    print("正则参数仅用于演示，未调参；SGD 是迭代近似解，结果可能有差异")

    print("\n测试集评估（Nonzero 为非零系数数量，Iterations 为迭代次数，SGD 按轮计）：")
    print(f"{'Model':<12}{'MSE':>14}{'R2':>12}{'Nonzero':>10}{'Iterations':>12}")
    for name, model in models.items():
        y_pred = predictions[name]
        nonzero = (model.coef_ != 0).sum()
        epochs = getattr(model, "n_iter_", None)
        epoch_text = "-" if epochs is None else str(epochs)
        print(
            f"{name:<12}{mean_squared_error(y_test, y_pred):>14.6f}"
            f"{r2_score(y_test, y_pred):>12.6f}{nonzero:>10}{epoch_text:>12}"
        )

    print("\n模型参数（系数对应标准化后的特征，预测值 = 特征与系数的点积 + 截距）：")
    print(f"{'Feature':<12}" + "".join(f"{name:>14}" for name in models))
    # SGD 的 intercept_ 是单元素数组，其余三个模型为标量。
    print(
        f"{'intercept':<12}"
        + "".join(f"{model.intercept_.item():>14.6f}" for model in models.values())
    )
    for index, feature in enumerate(dataset.feature_names):
        print(
            f"{feature:<12}"
            + "".join(f"{model.coef_[index]:>14.6f}" for model in models.values())
        )

    print("\n前 5 个测试样本的预测：")
    print(f"{'Sample':<8}{'Actual':>10}" + "".join(f"{name:>14}" for name in models))
    for index, actual in enumerate(y_test[:5]):
        print(
            f"{index + 1:<8}{actual:>10.2f}"
            + "".join(f"{predictions[name][index]:>14.2f}" for name in models)
        )


if __name__ == "__main__":
    main()
