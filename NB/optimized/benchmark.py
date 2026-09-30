"""预热后分别测量 fit/predict，中位数与线程设置可复现。

不测峰值内存；仅记录输入、模型数组和主要计算数组的尺寸估算。
"""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn import naive_bayes
from threadpoolctl import threadpool_info, threadpool_limits

from NB.basic import nb as basic
from . import nb as optimized
from .main import MODEL_PARAMS, digits_split


def _array_bytes(value):
    if sparse.issparse(value):
        return value.data.nbytes + value.indices.nbytes + value.indptr.nbytes
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, list):
        return sum(_array_bytes(item) for item in value)
    return 0


def _measure(factory, X_train, y_train, X_test, repeats):
    model = factory().fit(X_train, y_train)
    prediction = model.predict(X_test)  # 预热，计时不包括数据准备和打印。
    fit_times, predict_times = [], []
    for _ in range(repeats):
        candidate = factory()
        start = perf_counter()
        candidate.fit(X_train, y_train)
        fit_times.append((perf_counter() - start) * 1000)
        start = perf_counter()
        candidate.predict(X_test)
        predict_times.append((perf_counter() - start) * 1000)
    return {
        "fit_ms": float(np.median(fit_times)),
        "predict_ms": float(np.median(predict_times)),
        "fit_samples_ms": fit_times,
        "predict_samples_ms": predict_times,
        "model_arrays_bytes": sum(_array_bytes(v) for v in vars(model).values()),
    }, model, prediction


def _cases():
    X_train, X_test, y_train, _ = digits_split()
    yield "digits", X_train.astype(float), X_test.astype(float), y_train, None
    rng = np.random.default_rng(42)
    for n, d, c in ((600, 32, 3), (2400, 128, 10)):
        X_train = rng.integers(0, 17, size=(n, d)).astype(float)
        X_test = rng.integers(0, 17, size=(n // 3, d)).astype(float)
        y = np.arange(n) % c
        rng.shuffle(y)
        yield f"synthetic-{n}-{d}-{c}", X_train, X_test, y, None
    for density in (0.01, 0.1):
        X_train = rng.integers(1, 5, size=(1800, 400)).astype(float)
        X_train[rng.random(X_train.shape) >= density] = 0
        X_test = rng.integers(1, 5, size=(600, 400)).astype(float)
        X_test[rng.random(X_test.shape) >= density] = 0
        y = np.arange(1800) % 6
        rng.shuffle(y)
        yield f"sparse-{density:g}", X_train, X_test, y, density


def run_benchmark(repeats=5, threads=1):
    records = []
    with threadpool_limits(limits=threads):
        metadata = {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "repeats": repeats, "warmups": 1, "seed": 42,
            "thread_limit": threads, "threadpools": threadpool_info(),
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )},
            "memory_note": "数组尺寸估算，不是进程峰值；合成稀疏数据在计时前转换 CSR",
        }
        for dataset, X_train, X_test, y, density in _cases():
            csr_train = sparse.csr_matrix(X_train) if density is not None else None
            csr_test = sparse.csr_matrix(X_test) if density is not None else None
            for name, params in MODEL_PARAMS:
                if density is not None and name not in ("MultinomialNB", "BernoulliNB", "ComplementNB"):
                    continue
                if density is not None and name == "BernoulliNB":
                    params = {"alpha": 1.0, "binarize": 0.0}
                n, d = X_train.shape
                m, c = X_test.shape[0], len(np.unique(y))
                implementations = (
                    [("basic", basic, X_train, X_test),
                     ("optimized", optimized, X_train, X_test),
                     ("sklearn", naive_bayes, X_train, X_test)] if density is None else
                    [("optimized_dense", optimized, X_train, X_test),
                     ("optimized_csr", optimized, csr_train, csr_test),
                     ("sklearn_csr", naive_bayes, csr_train, csr_test)]
                )
                timings, predictions, models = {}, [], []
                for label, module, train, test in implementations:
                    factory = lambda module=module, name=name, params=params: getattr(module, name)(**params)
                    timing, model, prediction = _measure(factory, train, y, test, repeats)
                    timing["input_train_bytes"] = _array_bytes(train)
                    timing["input_test_bytes"] = _array_bytes(test)
                    timings[label] = timing
                    predictions.append(prediction)
                    models.append(model)
                for prediction in predictions[1:]:
                    assert_array_equal(predictions[0], prediction)
                if density is not None:
                    assert_allclose(models[0].predict_proba(X_test), models[1].predict_proba(csr_test),
                                    rtol=1e-8, atol=1e-10)
                baseline_label, optimized_label = ("basic", "optimized") if density is None else (
                    "optimized_dense", "optimized_csr"
                )
                base_time, opt_time = timings[baseline_label], timings[optimized_label]
                record = {
                    "dataset": dataset, "model": name, "params": params,
                    "train_shape": [n, d], "test_shape": [m, d], "n_classes": c,
                    "input_dtype": str(X_train.dtype),
                    "requested_density": density,
                    "train_density": float(np.count_nonzero(X_train) / X_train.size),
                    "train_nnz": int(np.count_nonzero(X_train)),
                    "score_array_bytes": m * c * 8,
                    "label_indicator_bytes": n * c * 8 if name != "GaussianNB" else 0,
                    "gaussian_deviation_array_bytes": m * d * 8 if name == "GaussianNB" else 0,
                    "bernoulli_weight_array_bytes": c * d * 8 if name == "BernoulliNB" else 0,
                    "timings": timings, "ratio_numerator": baseline_label,
                    "ratio_denominator": optimized_label,
                    "fit_ratio": base_time["fit_ms"] / opt_time["fit_ms"],
                    "predict_ratio": base_time["predict_ms"] / opt_time["predict_ms"],
                    "predictions_equal": True,
                }
                records.append(record)
    return {"metadata": metadata, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="可选 JSON 结果路径")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats 和 threads 必须为正整数")
    report = run_benchmark(args.repeats, args.threads)
    print(json.dumps(report["metadata"], ensure_ascii=False, indent=2))
    print("数据 / 模型 | fit(ms) 对照→优化 | predict(ms) 对照→优化 | fit/predict 比值")
    for record in report["records"]:
        baseline = record["timings"][record["ratio_numerator"]]
        actual = record["timings"][record["ratio_denominator"]]
        print(f"{record['dataset']} / {record['model']} | "
              f"{baseline['fit_ms']:.3f}→{actual['fit_ms']:.3f} | "
              f"{baseline['predict_ms']:.3f}→{actual['predict_ms']:.3f} | "
              f"{record['fit_ratio']:.2f}/{record['predict_ratio']:.2f}")
    print("稠密比值为 basic/optimized；稀疏比值为 optimized_dense/optimized_csr。")
    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完整记录：{args.output}")


if __name__ == "__main__":
    main()
