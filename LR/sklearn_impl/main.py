"""六组原例与有效输入能力对照；sklearn 模型只在这里作为参考。"""

import argparse
import warnings

import numpy as np
import sklearn
from scipy import sparse
from sklearn.datasets import load_diabetes
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .lasso import Lasso
    from .linear import LinearRegression
    from .ridge import Ridge
    from .sgd import SGDRegressor
except ImportError:
    from LR.sklearn_impl.lasso import Lasso
    from LR.sklearn_impl.linear import LinearRegression
    from LR.sklearn_impl.ridge import Ridge
    from LR.sklearn_impl.sgd import SGDRegressor


RANDOM_STATE = 42
L2_ALPHA, L1_ALPHA, MAX_ITER = 0.1, 1.0, 10000
SGD_PARAMS = {
    "loss": "squared_error", "fit_intercept": True, "max_iter": MAX_ITER,
    "tol": None, "shuffle": True, "random_state": RANDOM_STATE,
    "learning_rate": "invscaling", "eta0": 0.01, "power_t": 0.25,
}


def create_models(n_samples, *, reference=False):
    linear, ridge, lasso, sgd = (
        (SklearnLinearRegression, SklearnRidge, SklearnLasso, SklearnSGDRegressor)
        if reference else (LinearRegression, Ridge, Lasso, SGDRegressor)
    )
    return {
        "OLS": linear(fit_intercept=True),
        "Ridge-L2": ridge(alpha=n_samples * L2_ALPHA, fit_intercept=True),
        "Lasso-L1": lasso(alpha=L1_ALPHA, fit_intercept=True, max_iter=MAX_ITER, tol=1e-8),
        "SGD-none": sgd(penalty=None, alpha=0.0, **SGD_PARAMS),
        "SGD-L2": sgd(penalty="l2", alpha=L2_ALPHA, **SGD_PARAMS),
        "SGD-L1": sgd(penalty="l1", alpha=L1_ALPHA, **SGD_PARAMS),
    }


def compare_parameters(model, reference, X, *, atol):
    prediction, expected = model.predict(X), reference.predict(X)
    for actual, target in (
        (prediction, expected), (model.coef_, reference.coef_),
        (model.intercept_, reference.intercept_),
    ):
        np.testing.assert_allclose(actual, target, rtol=1e-8, atol=atol)
    if isinstance(model, SGDRegressor):
        assert model.n_iter_ == reference.n_iter_, "SGD 停止轮数不一致"
        assert model.t_ == reference.t_, "SGD 累计步数不一致"
    if isinstance(model, Lasso):
        np.testing.assert_allclose(model.dual_gap_, reference.dual_gap_, rtol=1e-4, atol=max(atol, 1e-10))
    return float(np.max(np.abs(prediction - expected)))


def training_objective(name, model, X, y):
    residual = model.predict(X) - y
    objective = residual @ residual / (2 * len(y))
    if name.endswith("L2"):
        objective += L2_ALPHA * (model.coef_ @ model.coef_) / 2
    elif name.endswith("L1"):
        objective += L1_ALPHA * np.abs(model.coef_).sum()
    return objective


def diabetes_demo():
    dataset = load_diabetes(scaled=False)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    models = create_models(len(y_train))
    references = create_models(len(y_train), reference=True)
    print(f"sklearn {sklearn.__version__}；diabetes：训练 {X_train.shape}，测试 {X_test.shape}", flush=True)
    print("test_size=0.2，random_state=42；StandardScaler 只拟合训练集")
    print(f"Ridge.alpha={len(y_train) * L2_ALPHA:g}；SGD-L2.alpha={L2_ALPHA}；L1.alpha={L1_ALPHA}")
    print(f"SGD：{SGD_PARAMS}")
    print("H=手写，SK=sklearn；J=SSE/(2m)+正则项；Nonzero 按精确非零统计")
    print(
        f"{'Model':<12}{'MSE(H)':>13}{'MSE(SK)':>13}{'R2(H)':>10}{'R2(SK)':>10}"
        f"{'Nonzero':>9}{'Iter(H)':>9}{'Iter(SK)':>9}{'MaxDiff':>12}{'J(H)':>13}{'J(SK)':>13}",
        flush=True,
    )
    predictions = {}
    for name, model in models.items():
        reference = references[name].fit(X_train, y_train)
        model.fit(X_train, y_train)
        tolerance = 1e-5 if name == "Lasso-L1" else 1e-6 if name.startswith("SGD") else 1e-8
        difference = compare_parameters(model, reference, X_test, atol=tolerance)
        prediction, expected = model.predict(X_test), reference.predict(X_test)
        predictions[name] = prediction
        if name.startswith("SGD"):
            assert model.n_iter_ == MAX_ITER
            assert model.t_ == 1 + MAX_ITER * len(y_train)
        print(
            f"{name:<12}{mean_squared_error(y_test, prediction):>13.6f}"
            f"{mean_squared_error(y_test, expected):>13.6f}{r2_score(y_test, prediction):>10.6f}"
            f"{r2_score(y_test, expected):>10.6f}{np.count_nonzero(model.coef_):>9}"
            f"{str(getattr(model, 'n_iter_', '-')):>9}{str(getattr(reference, 'n_iter_', '-')):>9}"
            f"{difference:>12.6g}{training_objective(name, model, X_train, y_train):>13.6f}"
            f"{training_objective(name, reference, X_train, y_train):>13.6f}", flush=True,
        )
    print("\n系数对应标准化后的特征，预测 = X @ coef + intercept")
    print(f"{'Feature':<12}" + "".join(f"{name:>14}" for name in models))
    print(f"{'intercept':<12}" + "".join(f"{np.asarray(model.intercept_).item():>14.6f}" for model in models.values()))
    for j, feature in enumerate(dataset.feature_names):
        print(f"{feature:<12}" + "".join(f"{model.coef_[j]:>14.6f}" for model in models.values()))
    print("\n前五个测试样本：")
    print(f"{'Sample':<8}{'Actual':>10}" + "".join(f"{name:>14}" for name in models))
    for i in range(5):
        print(f"{i + 1:<8}{y_test[i]:>10.2f}" + "".join(f"{predictions[name][i]:>14.2f}" for name in models))
    print("六组模型的系数、截距、预测与迭代属性核对通过。", flush=True)


def capability_checks():
    """只核对实际承诺的有效输入路径，不构造全面异常测试。"""
    rng = np.random.RandomState(7)
    X = rng.normal(size=(48, 6))
    X[np.abs(X) < 0.6] = 0
    X += np.array([0.5, -0.3, 0, 0.2, 0, -0.1])
    y = X @ np.array([0.8, -0.4, 0, 0.2, 0.5, -0.1]) + 0.7 + 0.03 * rng.normal(size=48)
    targets = np.column_stack([y, 0.4 * y + X[:, 1]])
    weights = rng.uniform(0.2, 2, size=48)
    weights[::13] = 0
    csr, csc = sparse.csr_matrix(X), sparse.csc_matrix(X)
    passed, max_difference = 0, 0.0

    def record(name, model, reference, data, *, atol=1e-8):
        nonlocal passed, max_difference
        difference = compare_parameters(model, reference, data, atol=atol)
        passed += 1
        max_difference = max(max_difference, difference)
        print(f"PASS {name:<38} MaxDiff={difference:.3g}", flush=True)

    def fitted(name, local_class, reference_class, data, target=y, *, params=None, sw=None, atol=1e-8):
        parameters = {} if params is None else params
        model, reference = local_class(**parameters), reference_class(**parameters)
        model.fit(data, target, sample_weight=sw)
        reference.fit(data, target, sample_weight=sw)
        record(name, model, reference, data, atol=atol)
        return model, reference

    fitted("OLS/weighted multi-target", LinearRegression, SklearnLinearRegression, X, targets, sw=weights)
    fitted("OLS/weighted CSR", LinearRegression, SklearnLinearRegression, csr, targets,
           params={"tol": 1e-10}, sw=weights)
    fitted("OLS/positive", LinearRegression, SklearnLinearRegression, X, targets,
           params={"positive": True}, sw=weights)
    rank_deficient = np.column_stack([X[:, :3], X[:, 0]])
    fitted("OLS/rank-deficient", LinearRegression, SklearnLinearRegression, rank_deficient)
    assert LinearRegression().fit(rank_deficient, y).rank_ == 3
    for solver in ("auto", "cholesky", "svd", "lsqr", "sparse_cg"):
        fitted(f"Ridge/{solver}/weighted multi", Ridge, SklearnRidge, X, targets,
               params={"alpha": [0.4, 0.9], "solver": solver, "tol": 1e-10}, sw=weights)
    for solver in ("auto", "lsqr", "sparse_cg"):
        fitted(f"Ridge/{solver}/weighted CSR", Ridge, SklearnRidge, csr, targets,
               params={"alpha": [0.4, 0.9], "solver": solver, "tol": 1e-10}, sw=weights)
    fitted("Ridge/sparse cholesky/no intercept", Ridge, SklearnRidge, csr, targets,
           params={"alpha": 0.4, "solver": "cholesky", "fit_intercept": False}, sw=weights)
    for solver in ("cholesky", "sparse_cg"):
        fitted(f"Ridge/{solver}/wide matrix", Ridge, SklearnRidge, X[:4], targets[:4],
               params={"alpha": [0.4, 0.9], "solver": solver, "tol": 1e-10}, sw=weights[:4])
    fitted("Ridge/svd/alpha zero", Ridge, SklearnRidge, rank_deficient,
           params={"alpha": 0, "solver": "svd"})
    fitted("Ridge/single-column target", Ridge, SklearnRidge, X, y[:, None])
    lasso_parameters = {"alpha": 0.035, "tol": 1e-11, "max_iter": 3000}
    for precompute in (False, True):
        fitted(f"Lasso/precompute={precompute}/weighted", Lasso, SklearnLasso, X, targets,
               params={**lasso_parameters, "precompute": precompute}, sw=weights, atol=1e-7)
    fitted("Lasso/provided Gram", Lasso, SklearnLasso, X, targets,
           params={**lasso_parameters, "fit_intercept": False, "precompute": X.T @ X}, atol=1e-7)
    fitted("Lasso/weighted CSC", Lasso, SklearnLasso, csc, targets,
           params=lasso_parameters, sw=weights, atol=1e-7)
    for data in (X, csc):
        fitted(f"Lasso/positive random/{'CSC' if sparse.issparse(data) else 'dense'}",
               Lasso, SklearnLasso, data, params={**lasso_parameters, "positive": True,
               "selection": "random", "random_state": 8}, sw=weights, atol=1e-7)
    model, reference = fitted("Lasso/warm-start first", Lasso, SklearnLasso, X,
                              params={**lasso_parameters, "warm_start": True}, sw=weights, atol=1e-7)
    for estimator in (model, reference):
        estimator.set_params(alpha=0.02).fit(X, y, sample_weight=weights)
    record("Lasso/warm-start second", model, reference, X, atol=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fitted("Lasso/alpha zero", Lasso, SklearnLasso, X,
               params={**lasso_parameters, "alpha": 0}, atol=1e-5)

    sgd_parameters = {
        "max_iter": 6, "tol": None, "random_state": 8, "eta0": 0.015,
        "alpha": 0.03, "epsilon": 0.2,
    }
    for loss in ("squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"):
        fitted(f"SGD/{loss}/weighted", SGDRegressor, SklearnSGDRegressor, X,
               params={**sgd_parameters, "loss": loss}, sw=weights, atol=1e-10)
    for penalty in (None, "l1", "l2", "elasticnet"):
        fitted(f"SGD/{penalty}/average dense", SGDRegressor, SklearnSGDRegressor, X,
               params={**sgd_parameters, "penalty": penalty, "average": True}, sw=weights, atol=1e-10)
        fitted(f"SGD/{penalty}/average CSR", SGDRegressor, SklearnSGDRegressor, csr,
               params={**sgd_parameters, "penalty": penalty, "average": 15}, sw=weights, atol=1e-10)
    for schedule in ("constant", "invscaling", "optimal", "adaptive"):
        fitted(f"SGD/{schedule}", SGDRegressor, SklearnSGDRegressor, X,
               params={**sgd_parameters, "learning_rate": schedule}, sw=weights, atol=1e-10)
    for data, schedule in ((X, "invscaling"), (csr, "invscaling"), (X, "adaptive")):
        fitted(f"SGD/tol/{schedule}/{'CSR' if sparse.issparse(data) else 'dense'}",
               SGDRegressor, SklearnSGDRegressor, data, params={**sgd_parameters,
               "max_iter": 100, "tol": 0.02, "learning_rate": schedule,
               "n_iter_no_change": 3, "penalty": "elasticnet"}, sw=weights, atol=1e-9)
    for average in (False, True, 100):
        params = {**sgd_parameters, "average": average, "penalty": "elasticnet"}
        model, reference = SGDRegressor(**params), SklearnSGDRegressor(**params)
        for start in (0, 16, 32):
            for estimator in (model, reference):
                estimator.partial_fit(csr[start:start + 16], y[start:start + 16],
                                      sample_weight=weights[start:start + 16])
        record(f"SGD/partial_fit/average={average}", model, reference, csr, atol=1e-10)
    for average in (False, True):
        params = {**sgd_parameters, "warm_start": True, "average": average, "penalty": "l1"}
        model, reference = SGDRegressor(**params), SklearnSGDRegressor(**params)
        for _ in range(2):
            for estimator in (model, reference):
                estimator.fit(X, y, sample_weight=weights)
        record(f"SGD/warm_start/average={average}", model, reference, X, atol=1e-10)
    model, reference = SGDRegressor(**sgd_parameters), SklearnSGDRegressor(**sgd_parameters)
    for estimator in (model, reference):
        estimator.fit(X, y, coef_init=np.full(6, 0.05), intercept_init=0.1)
    record("SGD/initial coefficients", model, reference, X, atol=1e-10)
    fitted("SGD/no shuffle/no intercept", SGDRegressor, SklearnSGDRegressor, X,
           params={**sgd_parameters, "shuffle": False, "fit_intercept": False}, atol=1e-10)
    print(f"能力核对：{passed} 组通过，最大训练预测差 {max_difference:.6g}。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-capabilities", action="store_true", help="只运行小规模有效输入能力核对")
    args = parser.parse_args()
    if args.check_capabilities:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            capability_checks()
    else:
        diabetes_demo()


if __name__ == "__main__":
    main()
