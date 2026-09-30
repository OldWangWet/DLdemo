"""运行指定版本的原始 NB 测试和公共 estimator 检查，保存真实通过/差异记录。

pytest/pandas/array-api-strict 是可选验证依赖，不参与模型实现。
原始测试只替换模型和配置入口；测试体及断言不修改。
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import scipy
import sklearn
from sklearn import config_context as reference_config_context
from sklearn.utils.estimator_checks import estimator_checks_generator

from . import nb
from ._config import config_context


@contextmanager
def verification_config_context(**kwargs):
    """原始测试的工具函数使用 sklearn 配置，手动模型使用独立本地配置。"""
    with reference_config_context(**kwargs), config_context(**kwargs):
        yield


def estimator_report():
    records = []
    for name in nb.__all__:
        model = getattr(nb, name)()
        record = dict(model=name, passed=0, skipped=0, failures=[])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for estimator, check in estimator_checks_generator(model):
                check_name = check.func.__name__ if hasattr(check, "func") else check.__name__
                try:
                    # 配置桥只存在于验证脚本，生产模型不读取 sklearn 全局配置。
                    with config_context(array_api_dispatch=True):
                        check(estimator)
                    record["passed"] += 1
                except unittest.SkipTest:
                    record["skipped"] += 1
                except Exception as exc:
                    entry = dict(check=check_name, error_type=type(exc).__name__, message=str(exc))
                    ref = getattr(sklearn.naive_bayes, name)()
                    try:
                        check(ref)
                    except Exception as ref_exc:
                        entry["reference_error_type"] = type(ref_exc).__name__
                        entry["reference_message"] = str(ref_exc)
                    record["failures"].append(entry)
        records.append(record)
    return records


def original_tests():
    import pytest

    source = Path(sklearn.__file__).parent / "tests/test_naive_bayes.py"
    text = source.read_text()
    adapted = text.replace("from sklearn.naive_bayes import (", "from NB.sklearn_impl import (")
    adapted = adapted.replace("from sklearn._config import config_context",
                              "from NB.sklearn_impl.verify_upstream import verification_config_context as config_context")

    class Results:
        def __init__(self):
            self.counts = dict(passed=0, failed=0, skipped=0, errors=0)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.counts[report.outcome] += 1
            elif report.outcome == "skipped":
                self.counts["skipped"] += 1
            elif report.outcome == "failed":
                self.counts["errors"] += 1

    results = Results()
    with tempfile.TemporaryDirectory(prefix="nb-upstream-") as tmp:
        root = Path(tmp)
        (root / "test_nb_upstream.py").write_text(adapted)
        (root / "conftest.py").write_text(
            "import numpy as np\nimport pytest\n"
            "@pytest.fixture\ndef global_random_seed():\n    return 42\n"
            "@pytest.fixture(params=[np.float32, np.float64])\n"
            "def global_dtype(request):\n    return request.param\n")
        code = pytest.main([str(root / "test_nb_upstream.py"), "-q", "--tb=short"], plugins=[results])
    return dict(source=str(source), sha256=hashlib.sha256(text.encode()).hexdigest(),
                seed=42, **results.counts, exit_code=int(code))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="NB/sklearn_impl/verification_results.json")
    args = parser.parse_args()
    if sklearn.__version__ != "1.9.1":
        raise RuntimeError("验证脚本固定对照 sklearn 1.9.1")
    tests = original_tests()
    records = estimator_report()
    report = dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__,
                  source_tests=tests, estimator_checks=records)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for record in records:
        print(f"{record['model']}：{record['passed']} 通过，{record['skipped']} 跳过，"
              f"{len(record['failures'])} 项差异/源码已有问题（见 JSON）")
    if tests["exit_code"]:
        raise SystemExit(tests["exit_code"])


if __name__ == "__main__":
    main()
