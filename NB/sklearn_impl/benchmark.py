"""测量 NB 的 CSR 存储、复制及分批训练资源行为，计时不承诺加速。"""

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter
import tracemalloc

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
import scipy
import sklearn
from sklearn import naive_bayes as reference
from threadpoolctl import threadpool_limits

from . import nb


def storage_bytes(array):
    if sparse.issparse(array):
        return array.data.nbytes + array.indices.nbytes + array.indptr.nbytes
    return array.nbytes


def state_bytes(model):
    total = 0
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif isinstance(value, list):
            total += sum(v.nbytes for v in value if isinstance(v, np.ndarray))
    return total


def measure(call, repeats):
    call()  # 预热；不计数据准备和模型构造。
    times = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        times.append(perf_counter() - start)
    gc.collect()
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return dict(median_seconds=statistics.median(times), traced_peak_bytes=peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--features", type=int, default=8000)
    parser.add_argument("--density", type=float, default=0.002)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="NB/sklearn_impl/benchmark_results.json")
    args = parser.parse_args()
    if args.samples < 3 or args.features < 1 or args.repeats < 1 or args.threads < 1 or not 0 < args.density <= 1:
        parser.error("samples>=3, features/repeats/threads>=1，density 必须在 (0, 1] 内")
    rng = np.random.default_rng(42)
    X_csr = sparse.random(args.samples, args.features, density=args.density,
                          format="csr", random_state=rng, dtype=np.float64)
    X_dense = X_csr.toarray()  # 仅为对照准备稠密输入，不是模型中的路径。
    y = np.arange(args.samples) % 3
    test_csr = X_csr[:args.samples // 3]
    test_dense = X_dense[:args.samples // 3]
    classes = np.unique(y)
    results = []
    with threadpool_limits(limits=args.threads):
        for name in ["MultinomialNB", "BernoulliNB", "ComplementNB"]:
            paths = []
            fitted = {}
            for kind, train, test in [("dense", X_dense, test_dense), ("csr", X_csr, test_csr)]:
                models = [("manual", getattr(nb, name)()), ("sklearn", getattr(reference, name)())]
                for implementation, model in models:
                    fit = measure(lambda: model.fit(train, y), args.repeats)
                    predict = measure(lambda: model.predict(test), args.repeats)
                    paths.append(dict(input=kind, implementation=implementation,
                                      input_storage_bytes=storage_bytes(train),
                                      model_state_bytes=state_bytes(model), fit=fit, predict=predict))
                    fitted[kind, implementation] = model
                a, b = (m for _, m in models)
                assert_allclose(a.feature_count_, b.feature_count_, rtol=1e-10, atol=1e-12)
                assert_allclose(a.predict_joint_log_proba(test), b.predict_joint_log_proba(test),
                                rtol=1e-10, atol=1e-12)
                assert_array_equal(a.predict(test), b.predict(test))
            assert_allclose(fitted["dense", "manual"].feature_count_, fitted["csr", "manual"].feature_count_,
                            rtol=1e-10, atol=1e-12)
            assert_allclose(fitted["dense", "manual"].predict_joint_log_proba(test_dense),
                            fitted["csr", "manual"].predict_joint_log_proba(test_csr), rtol=1e-10, atol=1e-12)
            batches = np.array_split(np.arange(args.samples), 8)
            incremental = getattr(nb, name)()
            def partial_fit():
                # 每次资源测量从空状态开始，防止累计训练改变基准。
                for attr in list(vars(incremental)):
                    if attr.endswith("_") and not attr.startswith("__"):
                        delattr(incremental, attr)
                for indices in batches:
                    incremental.partial_fit(X_csr[indices], y[indices], classes=classes)
            batch_measure = measure(partial_fit, args.repeats)
            assert_allclose(incremental.feature_count_, fitted["csr", "manual"].feature_count_,
                            rtol=1e-10, atol=1e-12)
            incremental.predict(test_csr)
            record = dict(model=name, paths=paths,
                          partial_fit=dict(batches=len(batches), **batch_measure,
                                           model_state_bytes=state_bytes(incremental)))
            results.append(record)
            print(f"{name}：稠密/CSR、sklearn、8 批次计数与得分对照通过")
    report = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, sklearn=sklearn.__version__),
                  settings=vars(args), shape=list(X_csr.shape), nnz=X_csr.nnz, results=results,
                  memory_scope="tracemalloc 新增分配峰值；不含准备好的输入，不等于进程 RSS 或全部原生库内存")
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
