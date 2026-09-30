"""预热后分别测量 fit/predict，中位数与线程设置可复现。

仅比较相同稠密数据的计算流程；线程设置用于控制测量条件。
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
from sklearn import naive_bayes
from threadpoolctl import threadpool_info, threadpool_limits

from NB.basic import nb as basic
from . import nb as optimized
from .main import MODEL_PARAMS, digits_split


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
    }, model, prediction


def _cases():
    X_train, X_test, y_train, _ = digits_split()
    yield "digits", X_train.astype(float), X_test.astype(float), y_train
    rng = np.random.default_rng(42)
    for n, d, c in ((600, 32, 3), (2400, 128, 10)):
        X_train = rng.integers(0, 17, size=(n, d)).astype(float)
        X_test = rng.integers(0, 17, size=(n // 3, d)).astype(float)
        y = np.arange(n) % c
        rng.shuffle(y)
        yield f"synthetic-{n}-{d}-{c}", X_train, X_test, y


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
            "scope": "相同稠密输入的算法计算对照；不测稀疏、增量或资源优化",
        }
        for dataset, X_train, X_test, y in _cases():
            for name, params in MODEL_PARAMS:
                n, d = X_train.shape
                m, c = X_test.shape[0], len(np.unique(y))
                timings, predictions, models = {}, [], []
                for label, module in (("basic", basic), ("optimized", optimized),
                                      ("sklearn", naive_bayes)):
                    factory = lambda module=module, name=name, params=params: getattr(module, name)(**params)
                    timing, model, prediction = _measure(factory, X_train, y, X_test, repeats)
                    timings[label] = timing
                    predictions.append(prediction)
                    models.append(model)
                for prediction in predictions[1:]:
                    assert_array_equal(predictions[0], prediction)
                assert_allclose(models[1].joint_log_likelihood(X_test),
                                models[2].predict_joint_log_proba(X_test),
                                rtol=1e-8, atol=1e-10)
                assert_allclose(models[1].predict_proba(X_test), models[2].predict_proba(X_test),
                                rtol=1e-8, atol=1e-10)
                baseline_label, optimized_label = "basic", "optimized"
                base_time, opt_time = timings[baseline_label], timings[optimized_label]
                record = {
                    "dataset": dataset, "model": name, "params": params,
                    "train_shape": [n, d], "test_shape": [m, d], "n_classes": c,
                    "input_dtype": str(X_train.dtype),
                    "timings": timings, "ratio_numerator": baseline_label,
                    "ratio_denominator": optimized_label,
                    "fit_ratio": base_time["fit_ms"] / opt_time["fit_ms"],
                    "predict_ratio": base_time["predict_ms"] / opt_time["predict_ms"],
                    "predictions_equal": True,
                    "scores_and_probabilities_close": True,
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
    print("比值为 basic/optimized；大于 1 表示优化版耗时更少。")
    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完整记录：{args.output}")


if __name__ == "__main__":
    main()
