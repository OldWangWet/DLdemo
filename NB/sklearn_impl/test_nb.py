"""使用独立原始输入与 sklearn 1.9.1 对照，包含非正常数值及错误路径。"""

import ast
from dataclasses import asdict
import inspect
import pickle
import subprocess
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
import sklearn
from sklearn import naive_bayes as reference
from sklearn.base import clone, is_classifier
from sklearn import config_context as reference_config_context
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import nb
from ._support import DataConversionWarning, InvalidParameterError, NotFittedError
from ._config import config_context, get_config


NAMES = nb.__all__
DISCRETE = [n for n in NAMES if n != "GaussianNB"]
SPARSE = ["MultinomialNB", "BernoulliNB", "ComplementNB"]
STATISTICS = ["classes_", "class_count_", "class_prior_", "class_log_prior_", "theta_",
              "var_", "epsilon_", "feature_count_", "feature_all_", "category_count_",
              "n_categories_", "feature_log_prob_", "n_features_in_", "feature_names_in_"]


def outcome(call):
    """保留异常名称和警告类别；不混淆与 sklearn 类对象的身份。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
            error = None
        except Exception as exc:
            value = None
            error = (type(exc).__name__, str(exc))
    return value, error, [(w.category.__name__, str(w.message)) for w in caught]


class NaiveBayesParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sklearn.__version__ != "1.9.1":
            raise RuntimeError("本测试对应 scikit-learn 1.9.1；升级参考版本前需审核源码差异。")
        rng = np.random.default_rng(12)
        cls.X = rng.integers(0, 6, (60, 4))
        cls.y = np.tile(["z", "a", "m"], 20)
        cls.weights = rng.uniform(0.1, 2, 60)

    def setUp(self):
        # 数值边界的警告由 outcome 单独记录和断言，常规对照不重复打印。
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def pair(self, name, **params):
        return getattr(nb, name)(**params), getattr(reference, name)(**params)

    def assert_models(self, manual, ref, X=None, rtol=1e-10, atol=1e-12):
        for attr in STATISTICS:
            self.assertEqual(hasattr(manual, attr), hasattr(ref, attr), attr)
            if not hasattr(ref, attr):
                continue
            a, b = getattr(manual, attr), getattr(ref, attr)
            if isinstance(b, list):
                self.assertEqual(len(a), len(b), attr)
                for x, y in zip(a, b):
                    assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=attr)
            elif attr in ["classes_", "feature_names_in_"]:
                assert_array_equal(a, b, err_msg=attr)
                self.assertEqual(a.dtype, b.dtype, attr)
            else:
                assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=attr)
                if hasattr(b, "dtype"):
                    self.assertEqual(a.dtype, b.dtype, attr)
        if X is not None:
            for method in ["predict_joint_log_proba", "predict_log_proba", "predict_proba"]:
                a, b = getattr(manual, method)(X), getattr(ref, method)(X)
                assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=method)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.dtype, b.dtype)
            assert_array_equal(manual.predict(X), ref.predict(X))

    def assert_errors(self, call_a, call_b, exact_message=False):
        _, a, wa = outcome(call_a)
        _, b, wb = outcome(call_b)
        self.assertIsNotNone(b, "参考模型应当抛出异常")
        self.assertEqual(a[0] if a else None, b[0], (a, b))
        if exact_message:
            self.assertEqual(a[1], b[1])
        self.assertEqual([w[0] for w in wa], [w[0] for w in wb])

    def test_dense_dtypes_labels_and_weights(self):
        for name in NAMES:
            for dtype in [np.int32, np.int64, np.float32, np.float64, np.float16]:
                for labels in [self.y, np.tile([7, -3, 2], 20), np.tile([False, True], 30)]:
                    for weights in [None, self.weights, self.weights.astype(np.float32), 2.0]:
                        with self.subTest(name=name, dtype=dtype, labels=labels.dtype, weights=str(type(weights))):
                            X = self.X.astype(dtype)
                            a, b = self.pair(name)
                            a.fit(X, labels, sample_weight=weights)
                            b.fit(X, labels, sample_weight=weights)
                            self.assert_models(a, b, X, rtol=1e-6 if dtype in [np.float16, np.float32] else 1e-10)

    def test_digits(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X.astype(int), y, test_size=0.3, random_state=42, stratify=y)
        for name in NAMES:
            with self.subTest(name=name):
                params = {"binarize": 8} if name == "BernoulliNB" else {"min_categories": 17} if name == "CategoricalNB" else {}
                a, b = self.pair(name, **params)
                a.fit(X_train, y_train)
                b.fit(X_train, y_train)
                self.assert_models(a, b, X_test)
                self.assertEqual(a.score(X_test, y_test), b.score(X_test, y_test))

    def test_hand_calculations(self):
        X = [[1, 1], [1, 5], [3, 1], [3, 5]] + [[5, 5], [5, 7], [9, 5], [9, 7]] * 2
        a = nb.GaussianNB(var_smoothing=0).fit(X, ["A"] * 4 + ["B"] * 8)
        assert_allclose(a.theta_, [[2, 3], [7, 6]])
        assert_allclose(a.var_, [[1, 4], [4, 1]])
        self.assertEqual(a.predict([[4, 5]])[0], "B")
        X = [[3, 1, 0], [3, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
        a = nb.MultinomialNB().fit(X, [0, 0, 1, 1, 1, 1])
        assert_allclose(np.exp(a.feature_log_prob_), [[7/11, 2/11, 2/11], [2/11, 5/11, 4/11]])
        self.assertEqual(a.predict([[2, 0, 1]])[0], 0)

    def test_sparse_formats_arrays_and_weights(self):
        formats = [sparse.csr_matrix, sparse.csr_array, sparse.csc_matrix, sparse.csc_array,
                   sparse.coo_matrix, sparse.coo_array, sparse.lil_matrix, sparse.dok_matrix,
                   sparse.bsr_matrix, sparse.dia_matrix]
        for name in SPARSE:
            for container in formats:
                with self.subTest(name=name, container=container.__name__):
                    X = container(self.X)
                    a, b = self.pair(name)
                    a.fit(X, self.y, sample_weight=self.weights)
                    b.fit(X, self.y, sample_weight=self.weights)
                    self.assert_models(a, b, X)

    def test_sparse_duplicate_indices_and_explicit_zeros(self):
        # 相同位置 [0.7, 0.7] 逐项二值化和先求和二值化会得出不同结果。
        X = sparse.csr_matrix(([0.7, 0.7, 0., 2., -0.2, 1.2],
                               [0, 0, 1, 1, 0, 0], [0, 3, 4, 6]), shape=(3, 2))
        y = [0, 1, 0]
        original = (X.data.copy(), X.indices.copy(), X.indptr.copy())
        a, b = self.pair("BernoulliNB", binarize=1.0)
        a.fit(X, y)
        b.fit(X, y)
        self.assert_models(a, b, X)
        for actual, expected in zip((X.data, X.indices, X.indptr), original):
            assert_array_equal(actual, expected)
        assert_allclose(a.feature_count_, [[1, 0], [0, 1]])
        for name in ["MultinomialNB", "ComplementNB"]:
            a, b = self.pair(name)
            self.assert_errors(lambda: a.fit(X, y), lambda: b.fit(X, y), True)

    def test_sparse_input_never_densified(self):
        X = sparse.random(40, 20000, density=0.0005, format="csr", random_state=2)
        y = np.arange(40) % 3
        # label 指示矩阵可以是稠密；这里禁止的是高维输入的 toarray。
        for name in SPARSE:
            with self.subTest(name=name), patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("input densified")):
                a = getattr(nb, name)().fit(X, y)
                a.partial_fit(X, y)
                self.assertEqual(a.predict_proba(X).shape, (40, 3))
                self.assertTrue(sparse.issparse(a._check_X(X)))

    def test_copy_controls_and_input_immutability(self):
        for name in NAMES:
            with self.subTest(name=name):
                X = self.X.copy()
                y, weights = self.y.copy(), self.weights.copy()
                a = getattr(nb, name)().fit(X, y, sample_weight=weights)
                a.predict(X)
                assert_array_equal(X, self.X)
                assert_array_equal(y, self.y)
                assert_array_equal(weights, self.weights)
                if name not in ["BernoulliNB", "CategoricalNB"]:
                    self.assertIs(a._check_X(X), X)
        for name in ["MultinomialNB", "ComplementNB"]:
            X = sparse.csr_matrix(self.X)
            a = getattr(nb, name)().fit(X, self.y)
            self.assertIs(a._check_X(X), X)
        X = self.X.copy()
        X.flags.writeable = False
        for name in NAMES:
            a = getattr(nb, name)().fit(X, self.y)
            a.predict(X)

    def test_partial_fit_each_batch(self):
        for name in NAMES:
            for weights in [None, self.weights]:
                with self.subTest(name=name, weighted=weights is not None):
                    a, b = self.pair(name)
                    classes = ["z", "unused", "m", "a"]
                    for i, (start, stop) in enumerate([(0, 1), (1, 14), (14, 31), (31, 60)]):
                        params = {"classes": classes if i == 0 else None}
                        if weights is not None:
                            params["sample_weight"] = weights[start:stop]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            a.partial_fit(self.X[start:stop], self.y[start:stop], **params)
                            b.partial_fit(self.X[start:stop], self.y[start:stop], **params)
                            self.assert_models(a, b, self.X[:stop])

    def test_sparse_partial_fit(self):
        for name in SPARSE:
            a, b = self.pair(name)
            for start, stop in [(0, 15), (15, 50), (50, 60)]:
                X = sparse.csc_array(self.X[start:stop])
                w = self.weights[start:stop]
                a.partial_fit(X, self.y[start:stop], classes=["a", "m", "z"], sample_weight=w)
                b.partial_fit(X, self.y[start:stop], classes=["a", "m", "z"], sample_weight=w)
                self.assert_models(a, b, sparse.csr_array(self.X))

    def test_partial_fit_class_contract(self):
        for name in NAMES:
            with self.subTest(name=name):
                a, b = self.pair(name)
                self.assert_errors(lambda: a.partial_fit(self.X, self.y), lambda: b.partial_fit(self.X, self.y), True)
                a, b = self.pair(name)
                a.partial_fit(self.X, self.y, classes=["z", "m", "a"])
                b.partial_fit(self.X, self.y, classes=["z", "m", "a"])
                self.assert_errors(lambda: a.partial_fit(self.X, self.y, classes=["a", "m"]),
                                   lambda: b.partial_fit(self.X, self.y, classes=["a", "m"]), True)
                a.partial_fit(self.X, self.y, classes=["m", "a", "z"])
                b.partial_fit(self.X, self.y, classes=["m", "a", "z"])
                self.assert_models(a, b, self.X)

    def test_unknown_incremental_labels_source_behavior(self):
        # Gaussian 拒绝未知类别；离散 LabelBinarizer 会忽略，二分类补列后归入首类。
        for name in NAMES:
            for classes in [[0], [0, 1], [0, 1, 2]]:
                with self.subTest(name=name, classes=classes):
                    a, b = self.pair(name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        a.partial_fit([[1, 2], [2, 1]], [0, 0], classes=classes)
                        b.partial_fit([[1, 2], [2, 1]], [0, 0], classes=classes)
                        ca, cb = lambda: a.partial_fit([[3, 1]], [9]), lambda: b.partial_fit([[3, 1]], [9])
                        if name == "GaussianNB":
                            self.assert_errors(ca, cb, True)
                        else:
                            ca(); cb()
                            self.assert_models(a, b, [[1, 1]])

    def test_fit_resets_and_fit_then_partial_fit(self):
        for name in NAMES:
            with self.subTest(name=name):
                a, b = self.pair(name)
                a.fit(self.X, self.y); b.fit(self.X, self.y)
                a.partial_fit(self.X[:5], self.y[:5]); b.partial_fit(self.X[:5], self.y[:5])
                self.assert_models(a, b, self.X)
                X, y = self.X[:10, :2], np.arange(10) % 2
                a.fit(X, y); b.fit(X, y)
                self.assert_models(a, b, X)

    def test_custom_priors_and_uniform_priors(self):
        for name in NAMES:
            configs = [{"priors": [0.2, 0.3, 0.5]}, {"priors": [0, 0.4, 0.6]}] if name == "GaussianNB" else [
                {"fit_prior": False}, {"class_prior": [0.2, 0.3, 0.5]},
                {"fit_prior": False, "class_prior": [0.2, 0.3, 0.5]},
                {"class_prior": [1, 2, 3]}, {"class_prior": [0, 0.4, 0.6]}]
            for params in configs:
                with self.subTest(name=name, params=params), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    a, b = self.pair(name, **params)
                    a.fit(self.X, self.y, sample_weight=self.weights)
                    b.fit(self.X, self.y, sample_weight=self.weights)
                    self.assert_models(a, b, self.X)
                    a.partial_fit(self.X[:8], self.y[:8]); b.partial_fit(self.X[:8], self.y[:8])
                    self.assert_models(a, b, self.X)

    def test_alpha_scalar_vector_and_clipping(self):
        # Bernoulli 的向量 alpha 原源码依照 class_count_ 广播；维数不匹配会报错。
        for name in DISCRETE:
            for alpha in [0, 1e-30, 0.7, 3, [0.2, 0.3, 0.4, 0.5], [0, 1e-20, 1, 2]]:
                if name == "CategoricalNB" and isinstance(alpha, list):
                    continue
                for force in [True, False]:
                    with self.subTest(name=name, alpha=alpha, force=force):
                        a, b = self.pair(name, alpha=alpha, force_alpha=force)
                        va, ea, wa = outcome(lambda: a.fit(self.X, self.y))
                        vb, eb, wb = outcome(lambda: b.fit(self.X, self.y))
                        self.assertEqual(ea, eb)
                        self.assertEqual(wa, wb)
                        if eb is None:
                            self.assert_models(va, vb, self.X)
                            self.assertEqual(repr(a.alpha), repr(alpha))
        X, y = self.X, np.arange(60) % 4
        a, b = self.pair("BernoulliNB", alpha=[0.2, 0.3, 0.4, 0.5])
        a.fit(X, y); b.fit(X, y)
        self.assert_models(a, b, X)

    def test_zero_alpha_and_degenerate_scores_warnings(self):
        cases = [([[0, 0], [0, 0]], [0, 1]), ([[1, 0], [0, 1]], [0, 1]),
                 ([[1], [1]], [0, 1]), ([[0, 0], [0, 0]], [0, 0])]
        for name in NAMES:
            for X, y in cases:
                configs = [{"var_smoothing": 0}] if name == "GaussianNB" else [{"alpha": 0}]
                if name == "ComplementNB":
                    configs.append({"alpha": 0, "norm": True})
                for params in configs:
                    with self.subTest(name=name, X=X, y=y, params=params):
                        a, b = self.pair(name, **params)
                        _, ea, wa = outcome(lambda: a.fit(X, y))
                        _, eb, wb = outcome(lambda: b.fit(X, y))
                        self.assertEqual(ea, eb)
                        self.assertEqual(wa, wb)
                        for method in ["predict", "predict_joint_log_proba", "predict_log_proba", "predict_proba"]:
                            ra, ea, wa = outcome(lambda: getattr(a, method)(X))
                            rb, eb, wb = outcome(lambda: getattr(b, method)(X))
                            self.assertEqual(ea, eb)
                            self.assertEqual(wa, wb)
                            assert_allclose(ra, rb, equal_nan=True)

    def test_complement_normalization_and_one_class(self):
        for norm in [False, True]:
            for y in [self.y, np.zeros(60, dtype=int)]:
                a, b = self.pair("ComplementNB", norm=norm)
                a.fit(self.X, y); b.fit(self.X, y)
                self.assert_models(a, b, self.X)

    def test_categorical_range_growth_and_batch_attribute(self):
        for minimum in [None, 7, [2, 4]]:
            a, b = self.pair("CategoricalNB", min_categories=minimum)
            for X in [[[0, 0], [1, 1]], [[5, 2], [4, 6]], [[0, 0], [1, 1]]]:
                a.partial_fit(X, [0, 1], classes=[0, 1]); b.partial_fit(X, [0, 1], classes=[0, 1])
                self.assert_models(a, b, X)
            self.assert_models(a, b, [[5, 6]])
            # 最后一批的 n_categories_ 可以变小，历史计数表不会缩小。
            self.assertGreaterEqual(a.category_count_[0].shape[1], a.n_categories_[0])

    def test_categorical_float_truncation(self):
        X = [[-0.7, 1.9], [2.8, 0.2], [1.2, 2.6], [0.9, 1.1]]
        a, b = self.pair("CategoricalNB", min_categories=[3, 4])
        a.fit(X, [0, 0, 1, 1]); b.fit(X, [0, 0, 1, 1])
        self.assert_models(a, b, [[-0.3, 1.8], [2.9, 3.1]])
        self.assert_errors(lambda: a.predict([[3, 0]]), lambda: b.predict([[3, 0]]), True)

    def test_categorical_invalid_min_categories(self):
        for value in [0, -1, 1.5, [2., 3., 4., 5.], np.array([2, 3, 4, 5], dtype=np.uint64),
                      [2, 3], [[2, 3, 4, 5], [2, 3, 4, 5]]]:
            with self.subTest(value=value):
                a, b = self.pair("CategoricalNB", min_categories=value)
                self.assert_errors(lambda: a.fit(self.X, self.y), lambda: b.fit(self.X, self.y), True)

    def test_gaussian_current_batch_epsilon(self):
        a, b = self.pair("GaussianNB", var_smoothing=0.1)
        for X in [[[0., 0.], [2., 4.], [1., 2.], [3., 6.]],
                  [[100., 10.], [300., 20.], [200., 15.], [400., 25.]]]:
            a.partial_fit(X, [0, 0, 1, 1], classes=[0, 1])
            b.partial_fit(X, [0, 0, 1, 1], classes=[0, 1])
            self.assert_models(a, b, X)
            self.assertEqual(a.epsilon_, 0.1 * np.var(X, axis=0).max())

    def test_gaussian_update_empty_and_zero_weight_class(self):
        fn_a, fn_b = nb.GaussianNB._update_mean_variance, reference.GaussianNB._update_mean_variance
        mu, var = np.array([1., 2.]), np.array([3., 4.])
        for X, w in [(np.empty((0, 2)), None), (np.ones((3, 2)), np.zeros(3)),
                     (self.X[:3, :2], self.weights[:3])]:
            for old in [0, 7]:
                for a, b in zip(fn_a(old, mu, var, X, w), fn_b(old, mu, var, X, w)):
                    assert_allclose(a, b)
        weights = np.where(self.y == "z", 0, self.weights)
        a, b = self.pair("GaussianNB")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a.fit(self.X, self.y, sample_weight=weights); b.fit(self.X, self.y, sample_weight=weights)
            self.assert_models(a, b, self.X)

    def test_weight_validation_and_negative_weights(self):
        for name in NAMES:
            for weights in [np.zeros(60), 0, np.ones((60, 1)), np.ones(59),
                            np.full(60, np.nan), np.full(60, np.inf)]:
                with self.subTest(name=name, weights=str(weights.shape) if hasattr(weights, "shape") else weights):
                    a, b = self.pair(name)
                    self.assert_errors(lambda: a.fit(self.X, self.y, sample_weight=weights),
                                       lambda: b.fit(self.X, self.y, sample_weight=weights))
            weights = self.weights.copy(); weights[0] = -0.01
            a, b = self.pair(name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a.fit(self.X, self.y, sample_weight=weights); b.fit(self.X, self.y, sample_weight=weights)
                self.assert_models(a, b, self.X)

    def test_constructor_validation(self):
        for name in NAMES:
            params = [{"var_smoothing": -1}, {"var_smoothing": np.inf}, {"var_smoothing": np.nan},
                      {"priors": "bad"}] if name == "GaussianNB" else [
                          {"alpha": -1}, {"alpha": np.inf}, {"alpha": np.nan},
                          {"alpha": "bad"}, {"force_alpha": 1}, {"fit_prior": "yes"},
                          {"class_prior": 0.2}]
            if name == "BernoulliNB":
                params += [{"binarize": -1}, {"binarize": np.nan}]
            if name == "ComplementNB":
                params += [{"norm": 1}]
            for config in params:
                with self.subTest(name=name, config=config):
                    a, b = self.pair(name, **config)
                    self.assert_errors(lambda: a.fit(self.X, self.y), lambda: b.fit(self.X, self.y), True)

    def test_prior_validation_source_behavior(self):
        for name in NAMES:
            priors = [[0.2, 0.8], [0.2, 0.3, 0.4], [-0.2, 0.3, 0.9]]
            for prior in priors:
                with self.subTest(name=name, prior=prior):
                    param = "priors" if name == "GaussianNB" else "class_prior"
                    a, b = self.pair(name, **{param: prior})
                    va, ea, wa = outcome(lambda: a.fit(self.X, self.y))
                    vb, eb, wb = outcome(lambda: b.fit(self.X, self.y))
                    self.assertEqual(ea, eb)
                    self.assertEqual(wa, wb)
                    if eb is None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            self.assert_models(va, vb, self.X)

    def test_input_validation(self):
        cases = [(np.zeros((0, 4)), []), (np.zeros((2, 0)), [0, 1]), ([1, 2], [0, 1]),
                 (1, [0]), (np.ones((2, 3, 4)), [0, 1]), (self.X, self.y[:2]),
                 (self.X, None), (self.X, np.ones((60, 2))),
                 (np.full((2, 4), np.nan), [0, 1]), (np.full((2, 4), np.inf), [0, 1]),
                 (np.ones((2, 4), dtype=complex), [0, 1]),
                 (self.X, np.full(60, np.nan)), (self.X, np.ones(60, dtype=complex))]
        for name in NAMES:
            for X, y in cases:
                with self.subTest(name=name, shape=np.shape(X), y_shape=np.shape(y)):
                    a, b = self.pair(name)
                    self.assert_errors(lambda: a.fit(X, y), lambda: b.fit(X, y))
            a, b = self.pair(name)
            a.fit(self.X, self.y); b.fit(self.X, self.y)
            self.assert_errors(lambda: a.predict([[0, 0]]), lambda: b.predict([[0, 0]]), True)
            self.assert_errors(lambda: a.partial_fit(self.X[:, :2], self.y),
                               lambda: b.partial_fit(self.X[:, :2], self.y), True)

    def test_sparse_rejection_and_prediction_negative_domain(self):
        X = sparse.csr_matrix(self.X)
        for name in ["GaussianNB", "CategoricalNB"]:
            a, b = self.pair(name)
            self.assert_errors(lambda: a.fit(X, self.y), lambda: b.fit(X, self.y))
        for name in ["MultinomialNB", "ComplementNB"]:
            a, b = self.pair(name)
            self.assert_errors(lambda: a.fit(-self.X, self.y), lambda: b.fit(-self.X, self.y), True)
            a.fit(self.X, self.y); b.fit(self.X, self.y)
            self.assert_models(a, b, -self.X)  # sklearn 只在训练时检查非负。

    def test_binary_threshold_none_and_ties(self):
        for threshold in [None, 0, 1, 2.5]:
            a, b = self.pair("BernoulliNB", binarize=threshold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a.fit(self.X, self.y); b.fit(self.X, self.y)
                self.assert_models(a, b, self.X)
        for name in NAMES:
            a, b = self.pair(name)
            a.fit([[1, 2], [2, 1]] * 2, ["z", "z", "a", "a"])
            b.fit([[1, 2], [2, 1]] * 2, ["z", "z", "a", "a"])
            self.assert_models(a, b, [[1, 1]])
            self.assertEqual(a.predict([[1, 1]])[0], "a")

    def test_column_labels_and_unfitted(self):
        for name in NAMES:
            a, b = self.pair(name)
            _, ea, wa = outcome(lambda: a.fit(self.X, self.y[:, None]))
            _, eb, wb = outcome(lambda: b.fit(self.X, self.y[:, None]))
            self.assertEqual(ea, eb)
            self.assertEqual(wa, wb)
            self.assertEqual(wa[0][0], "DataConversionWarning")
            for method in ["predict", "predict_joint_log_proba", "predict_log_proba", "predict_proba"]:
                a, b = self.pair(name)
                self.assert_errors(lambda: getattr(a, method)(self.X), lambda: getattr(b, method)(self.X), True)
                with self.assertRaises(NotFittedError):
                    getattr(a, method)(self.X)

    def test_clone_params_pickle_and_tags(self):
        for name in NAMES:
            params = {"priors": [0.2, 0.3, 0.5]} if name == "GaussianNB" else {"alpha": 0.7, "force_alpha": False}
            a, b = self.pair(name, **params)
            self.assertEqual(a.get_params(), b.get_params())
            self.assertEqual(repr(a), repr(b))
            self.assertTrue(is_classifier(a))
            self.assertEqual(asdict(a.__sklearn_tags__().input_tags), asdict(b.__sklearn_tags__().input_tags))
            a.fit(self.X, self.y); b.fit(self.X, self.y)
            c = clone(a)
            self.assertFalse(hasattr(c, "classes_"))
            self.assertEqual(c.get_params(), a.get_params())
            self.assert_models(pickle.loads(pickle.dumps(a)), b, self.X)
            key = "var_smoothing" if name == "GaussianNB" else "alpha"
            self.assertIs(a.set_params(**{key: 0.01}), a)
            self.assertEqual(a.get_params()[key], 0.01)
            with self.assertRaises(ValueError):
                a.set_params(does_not_exist=0)

    def test_pipeline_and_grid_search(self):
        X, y = self.X, np.arange(60) % 3
        for name in NAMES:
            with self.subTest(name=name):
                params = {"min_categories": 6} if name == "CategoricalNB" else {}
                steps = [("nb", getattr(nb, name)(**params))]
                ref_steps = [("nb", getattr(reference, name)(**params))]
                if name == "GaussianNB":
                    steps.insert(0, ("scale", StandardScaler()))
                    ref_steps.insert(0, ("scale", StandardScaler()))
                key = "nb__var_smoothing" if name == "GaussianNB" else "nb__alpha"
                a = GridSearchCV(Pipeline(steps), {key: [0.1, 0.5]}, cv=3).fit(X, y)
                b = GridSearchCV(Pipeline(ref_steps), {key: [0.1, 0.5]}, cv=3).fit(X, y)
                self.assertEqual(a.best_params_, b.best_params_)
                assert_allclose(a.cv_results_["mean_test_score"], b.cv_results_["mean_test_score"])
                assert_array_equal(a.predict(X), b.predict(X))

    def test_model_has_no_sklearn_dependency(self):
        root = Path(__file__).parent
        for filename in ["nb.py", "_support.py", "_config.py", "_metadata.py", "__init__.py"]:
            tree = ast.parse((root / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(a.name.startswith("sklearn") for a in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("sklearn"))
        # 新进程屏蔽 sklearn 导入，证明训练和预测不经由隐藏的 sklearn 调用。
        code = """
import sys
class BlockSklearn:
    def find_spec(self, fullname, *args):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise ImportError('sklearn blocked')
sys.meta_path.insert(0, BlockSklearn())
from NB.sklearn_impl import nb
for name in nb.__all__:
    model = getattr(nb, name)().fit([[1, 2], [2, 1], [3, 4], [4, 3]], [0, 0, 1, 1])
    model.partial_fit([[1, 2]], [0])
    assert model.predict_proba([[1, 2]]).shape == (1, 2)
assert not any(n == 'sklearn' or n.startswith('sklearn.') for n in sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], cwd=root.parents[1], check=True,
                       capture_output=True, text=True)

    def test_local_configuration_restores_state(self):
        original = get_config()
        with config_context(print_changed_only=False):
            self.assertEqual(repr(nb.GaussianNB()), "GaussianNB(priors=None, var_smoothing=1e-09)")
            with config_context(skip_parameter_validation=True):
                self.assertTrue(get_config()["skip_parameter_validation"])
            self.assertFalse(get_config()["skip_parameter_validation"])
        self.assertEqual(get_config(), original)
        with config_context(assume_finite=True), reference_config_context(assume_finite=True):
            a, b = self.pair("GaussianNB")
            X = [[1., np.nan], [3., 1.], [5., 2.], [7., 3.]]
            a.fit(X, [0, 0, 1, 1]); b.fit(X, [0, 0, 1, 1])
            self.assert_models(a, b, X)

    def test_metadata_requests_and_cloning(self):
        for name in NAMES:
            a, b = self.pair(name)
            self.assertEqual(repr(a.get_metadata_routing()), repr(b.get_metadata_routing()))
            with self.assertRaises(RuntimeError):
                a.set_fit_request(sample_weight=True)
            for alias in [True, False, None, "weights"]:
                with config_context(enable_metadata_routing=True), reference_config_context(enable_metadata_routing=True):
                    self.assertIs(a.set_fit_request(sample_weight=alias), a)
                    b.set_fit_request(sample_weight=alias)
                    a.set_partial_fit_request(classes=False, sample_weight=alias)
                    b.set_partial_fit_request(classes=False, sample_weight=alias)
                    a.set_score_request(sample_weight=alias); b.set_score_request(sample_weight=alias)
                self.assertEqual(repr(a.get_metadata_routing()), repr(b.get_metadata_routing()))
                self.assertEqual(repr(clone(a).get_metadata_routing()), repr(b.get_metadata_routing()))
                for method in ["fit", "partial_fit", "score"]:
                    self.assertEqual(a.get_metadata_routing().consumes(method, ["classes", "sample_weight", "weights"]),
                                     b.get_metadata_routing().consumes(method, ["classes", "sample_weight", "weights"]))
            with config_context(enable_metadata_routing=True), self.assertRaises(ValueError):
                a.set_fit_request(sample_weight="bad alias")
            with config_context(enable_metadata_routing=True), reference_config_context(enable_metadata_routing=True):
                a.set_fit_request(); b.set_fit_request()
            self.assertEqual(repr(a.get_metadata_routing()), repr(b.get_metadata_routing()))

    def test_metadata_routes_through_sklearn_pipeline(self):
        with config_context(enable_metadata_routing=True), reference_config_context(enable_metadata_routing=True):
            a = Pipeline([("nb", nb.MultinomialNB().set_fit_request(sample_weight="weights"))])
            b = Pipeline([("nb", reference.MultinomialNB().set_fit_request(sample_weight="weights"))])
            a.fit(self.X, self.y, weights=self.weights)
            b.fit(self.X, self.y, weights=self.weights)
            self.assert_models(a.named_steps["nb"], b.named_steps["nb"], self.X)

    def test_pandas_names_nullable_and_sparse_inputs(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("可选 pandas 验证依赖未安装")
        for name in NAMES:
            with self.subTest(name=name):
                X = pd.DataFrame(self.X, columns=["one", "two", "three", "four"])
                a, b = self.pair(name)
                a.fit(X, self.y); b.fit(X, self.y)
                self.assert_models(a, b, X)
                self.assert_errors(lambda: a.predict(X.iloc[:, ::-1]), lambda: b.predict(X.iloc[:, ::-1]), True)
                _, ea, wa = outcome(lambda: a.predict(self.X))
                _, eb, wb = outcome(lambda: b.predict(self.X))
                self.assertEqual(wa, wb); self.assertEqual(ea, eb)
                # refit 时删除历史特征名。
                a.fit(self.X, self.y); b.fit(self.X, self.y)
                self.assert_models(a, b, self.X)
                X = X.astype("Int64")
                a.fit(X, self.y); b.fit(X, self.y)
                self.assert_models(a, b, X)
                X.iloc[0, 0] = pd.NA
                self.assert_errors(lambda: a.fit(X, self.y), lambda: b.fit(X, self.y))
        for name in SPARSE:
            X = pd.DataFrame(self.X).astype(pd.SparseDtype("float64", 0))
            a, b = self.pair(name)
            a.fit(X, self.y); b.fit(X, self.y)
            self.assert_models(a, b, X)

    def test_gaussian_array_api_devices_labels_and_weights(self):
        try:
            import array_api_strict as xp
        except ImportError:
            self.skipTest("可选 array-api-strict 验证依赖未安装")
        for dtype in [xp.float32, xp.float64]:
            for device in [xp.Device("CPU_DEVICE"), xp.Device("device1")]:
                for string_y in [False, True]:
                    with self.subTest(dtype=dtype, device=device, string_y=string_y):
                        X = xp.asarray(self.X, dtype=dtype, device=device)
                        y = self.y if string_y else xp.asarray(np.arange(60) % 3, device=device)
                        for weights in [None, self.weights]:
                            with config_context(array_api_dispatch=True), reference_config_context(array_api_dispatch=True):
                                a, b = self.pair("GaussianNB")
                                a.fit(X, y, sample_weight=weights); b.fit(X, y, sample_weight=weights)
                                from sklearn.utils._array_api import move_to
                                for attr in ["theta_", "var_", "class_count_", "class_prior_"]:
                                    ra, rb = getattr(a, attr), getattr(b, attr)
                                    self.assertEqual(ra.dtype, rb.dtype)
                                    self.assertEqual(ra.device, rb.device)
                                    assert_allclose(move_to(ra, xp=np, device="cpu"), move_to(rb, xp=np, device="cpu"),
                                                    rtol=1e-6, atol=1e-7)
                                for method in ["predict", "predict_joint_log_proba", "predict_log_proba", "predict_proba"]:
                                    ra, rb = getattr(a, method)(X), getattr(b, method)(X)
                                    if method == "predict" and string_y:
                                        assert_array_equal(ra, rb)
                                    else:
                                        assert_allclose(move_to(ra, xp=np, device="cpu"), move_to(rb, xp=np, device="cpu"),
                                                        rtol=1e-6, atol=1e-7)
                                self.assertAlmostEqual(a.score(X, y), b.score(X, y))


if __name__ == "__main__":
    unittest.main()
