"""数学等价性、参数统计、数值边界和源码扩展行为验证。"""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn import naive_bayes

from NB.basic import nb as basic
from . import nb
from .main import MODEL_PARAMS, digits_split


class TestOptimizedNB(unittest.TestCase):
    @np.errstate(divide="ignore")  # 未观察类别的先验为 0，源码会计算 log(0)。
    def assert_reference(self, actual, reference, X):
        assert_array_equal(actual.classes_, reference.classes_)
        assert_array_equal(actual.class_count_, reference.class_count_)
        prior = (np.log(reference.class_prior_) if isinstance(actual, nb.GaussianNB)
                 else reference.class_log_prior_)
        assert_allclose(actual.class_log_prior_, prior, rtol=1e-8, atol=1e-10)
        if isinstance(actual, nb.GaussianNB):
            for attr in ("theta_", "var_", "epsilon_", "class_prior_"):
                assert_allclose(getattr(actual, attr), getattr(reference, attr),
                                rtol=1e-8, atol=1e-10)
        elif isinstance(actual, nb.CategoricalNB):
            for counts, expected in zip(actual.category_count_, reference.category_count_):
                assert_array_equal(counts, expected)
            for table, expected in zip(actual.feature_log_prob_, reference.feature_log_prob_):
                assert_allclose(table, expected, rtol=1e-8, atol=1e-10)
        else:
            assert_array_equal(actual.feature_count_, reference.feature_count_)
            assert_allclose(actual.feature_log_prob_, reference.feature_log_prob_,
                            rtol=1e-8, atol=1e-10)
            if isinstance(actual, nb.ComplementNB):
                assert_array_equal(actual.feature_all_, reference.feature_all_)
        raw = actual._validate_predict_X(X)
        assert_allclose(actual.joint_log_likelihood(X),
                        reference._joint_log_likelihood(raw), rtol=1e-8, atol=1e-10)
        assert_array_equal(actual.predict(X), reference.predict(X))
        assert_allclose(actual.predict_log_proba(X), reference.predict_log_proba(X),
                        rtol=1e-8, atol=1e-10)
        proba = actual.predict_proba(X)
        assert_allclose(proba, reference.predict_proba(X), rtol=1e-8, atol=1e-10)
        assert_allclose(proba.sum(axis=1), 1, atol=1e-10)
        assert_allclose(proba, np.exp(actual.predict_log_proba(X)))

    def test_digits_three_way(self):
        X_train, X_test, y_train, _ = digits_split()
        for name, params in MODEL_PARAMS:
            with self.subTest(model=name):
                actual = getattr(nb, name)(**params).fit(X_train, y_train)
                baseline = getattr(basic, name)(**params).fit(X_train, y_train)
                reference = getattr(naive_bayes, name)(**params).fit(X_train, y_train)
                self.assert_reference(actual, reference, X_test)
                assert_array_equal(actual.predict(X_test), baseline.predict(X_test))
                assert_array_equal(actual.class_count_, baseline.class_count_)
                raw = actual._validate_predict_X(X_test)
                original = np.column_stack([
                    baseline._log_score(raw, c) for c in range(len(actual.classes_))
                ])
                assert_allclose(actual.joint_log_likelihood(X_test), original,
                                rtol=1e-8, atol=1e-10)
                if name == "CategoricalNB":
                    assert_array_equal(actual.n_categories_, reference.n_categories_)

    def test_readme_examples(self):
        X = [[1, 1], [1, 5], [3, 1], [3, 5]]
        X += [[5, 5], [5, 7], [9, 5], [9, 7]] * 2
        gaussian = nb.GaussianNB(var_smoothing=0).fit(X, ["A"] * 4 + ["B"] * 8)
        assert_allclose(gaussian.theta_, [[2, 3], [7, 6]])
        assert_allclose(gaussian.var_, [[1, 4], [4, 1]])
        assert_array_equal(gaussian.predict([[4, 5]]), ["B"])
        assert_allclose(gaussian.joint_log_likelihood([[4, 5]]), [[
            np.log(1 / 3) - np.log(4 * np.pi) - 2.5,
            np.log(2 / 3) - np.log(4 * np.pi) - 13 / 8,
        ]])

        X = [[3, 1, 0], [3, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
        y = [0, 0, 1, 1, 1, 1]
        multinomial = nb.MultinomialNB().fit(X, y)
        complement = nb.ComplementNB().fit(X, y)
        assert_allclose(np.exp(multinomial.feature_log_prob_),
                        np.array([[7, 2, 2], [2, 5, 4]]) / 11)
        assert_allclose(complement.feature_log_prob_,
                        -np.log(np.array([[2, 5, 4], [7, 2, 2]]) / 11))
        assert_allclose(np.exp(multinomial.joint_log_likelihood([[2, 0, 1]])),
                        [[98 / 3993, 32 / 3993]])
        assert_array_equal(complement.predict([[2, 0, 1]]), [0])

        categorical = nb.CategoricalNB().fit(
            [[0, 0], [0, 0], [1, 0], [2, 1], [2, 1], [2, 0]], [0, 0, 0, 1, 1, 1]
        )
        assert_allclose(np.exp(categorical.joint_log_likelihood([[0, 0]])),
                        [[1 / 5, 1 / 30]])
        assert_array_equal(categorical.predict([[0, 0]]), [0])
        X = [[int(i < 7), int(i < 5)] for i in range(8)]
        X += [[int(i < 1), int(i < 2)] for i in range(8)]
        bernoulli = nb.BernoulliNB(binarize=None).fit(X, [0] * 8 + [1] * 8)
        assert_allclose(np.exp(bernoulli.feature_log_prob_), [[0.8, 0.6], [0.2, 0.3]])
        assert_allclose(np.exp(bernoulli.joint_log_likelihood([[1, 0]])), [[0.16, 0.07]])
        assert_array_equal(bernoulli.predict([[1, 0]]), [0])

    def test_random_score_transformations_and_labels(self):
        rng = np.random.default_rng(42)
        X = rng.integers(0, 5, size=(36, 7))
        X[:, 0] = 0
        test = rng.integers(0, 5, size=(13, 7))
        for y in (np.tile([-12, 8, 300], 12), np.tile(["z", "a", "q"], 12)):
            for name, params in MODEL_PARAMS[1:]:
                with self.subTest(model=name, labels=y.dtype):
                    if name == "BernoulliNB":
                        params = {"binarize": 2}
                    actual = getattr(nb, name)(**params).fit(X, y)
                    baseline = getattr(basic, name)(**params).fit(X, y)
                    raw = actual._validate_predict_X(test)
                    original = np.column_stack([
                        baseline._log_score(raw, c) for c in range(3)
                    ])
                    assert_allclose(actual.joint_log_likelihood(test), original,
                                    rtol=1e-8, atol=1e-10)

    def test_probability_stability(self):
        cases = [
            (nb.MultinomialNB(), [[4, 1], [1, 4]], [[1e6, 2e6]]),
            (nb.ComplementNB(), [[4, 1], [1, 4]], [[1e6, 2e6]]),
            (nb.GaussianNB(), [[0, 0], [1, 2], [3, 4], [4, 6]], [[1e5, 1e5]]),
        ]
        for model, X, test in cases:
            with self.subTest(model=type(model).__name__):
                y = [0, 1] if len(X) == 2 else [0, 0, 1, 1]
                model.fit(X, y)
                self.assertTrue(np.isfinite(model.predict_log_proba(test)).all())
                self.assertTrue(np.isfinite(model.predict_proba(test)).all())
                assert_allclose(model.predict_proba(test).sum(axis=1), 1, atol=1e-6)

    def test_single_class_and_tie_order(self):
        X = [[0, 1], [1, 0], [2, 2]]
        for name in ("GaussianNB", "MultinomialNB", "CategoricalNB", "BernoulliNB"):
            with self.subTest(model=name):
                actual = getattr(nb, name)().fit(X, ["only"] * 3)
                reference = getattr(naive_bayes, name)().fit(X, ["only"] * 3)
                self.assert_reference(actual, reference, X)
        model = nb.MultinomialNB().fit([[1, 1], [1, 1]], [9, -2])
        assert_array_equal(model.predict([[0, 0], [1, 1]]), [-2, -2])
        assert_allclose(model.predict_proba([[1, 1]]), [[0.5, 0.5]])
        near_X = [[1, 1 + 1e-12], [1 + 1e-12, 1]]
        near = nb.MultinomialNB().fit(near_X, [9, -2])
        baseline = basic.MultinomialNB().fit(near_X, [9, -2])
        probe = np.array([[1, 1 + 2e-12]])
        scores = near.joint_log_likelihood(probe)
        original = np.column_stack([baseline._log_score(probe, c) for c in range(2)])
        assert_allclose(scores, original, rtol=1e-8, atol=1e-10)
        assert_array_equal(near.predict(probe), near.classes_[scores.argmax(axis=1)])

    def test_categorical_ranges_and_threshold(self):
        model = nb.CategoricalNB(min_categories=4).fit([[0], [1]], [0, 1])
        assert_allclose(model.predict_proba([[3]]), [[0.5, 0.5]])
        with self.assertRaises(ValueError):
            model.predict([[4]])
        with self.assertRaises(ValueError):
            nb.CategoricalNB().fit([[0.5], [1]], [0, 1])
        binary = nb.BernoulliNB(binarize=8).fit([[8], [9], [0], [16]], [0, 0, 1, 1])
        assert_array_equal(binary.feature_count_, [[1], [1]])
        assert_array_equal(binary._validate_predict_X([[8], [9]]), [[0], [1]])
        with self.assertRaises(ValueError):
            nb.BernoulliNB(binarize=None).fit([[0], [2]], [0, 1])

    def test_validation(self):
        for value in (0, -1, np.inf, np.nan, [1, 2]):
            with self.subTest(alpha=value), self.assertRaises(ValueError):
                nb.MultinomialNB(alpha=value)
        for value in (-1, np.inf, np.nan):
            with self.subTest(smoothing=value), self.assertRaises(ValueError):
                nb.GaussianNB(var_smoothing=value)
        for value in (0, 1.5, np.inf, [2, 3]):
            with self.subTest(categories=value), self.assertRaises(ValueError):
                nb.CategoricalNB(min_categories=value)
        for X, y in (([], []), ([[1]], []), ([[np.nan]], [0]),
                     ([[1]], [[0]]), ([[-1]], [0]), ([[1]], [np.inf])):
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                nb.MultinomialNB().fit(X, y)
        for method in ("predict", "joint_log_likelihood", "predict_proba", "predict_log_proba"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                getattr(nb.MultinomialNB(), method)([[1]])
        model = nb.MultinomialNB().fit([[0, 1]], [0])
        with self.assertRaises(ValueError):
            model.predict([[1]])
        with self.assertRaises(ValueError):
            nb.ComplementNB().fit([[0, 1]], [0])

    def test_gaussian_degenerate_and_single_sample_class(self):
        with self.assertRaises(ValueError):
            nb.GaussianNB().fit([[1, 1], [1, 1]], [0, 1])
        with self.assertRaises(ValueError):
            nb.GaussianNB(var_smoothing=0).fit([[0, 0], [1, 0]], [0, 1])
        X, y = [[0, 0], [1, 0], [4, 0]], [0, 0, 1]
        actual = nb.GaussianNB().fit(X, y)
        reference = naive_bayes.GaussianNB().fit(X, y)
        self.assert_reference(actual, reference, X)

    def test_gaussian_mean_variance_merge(self):
        rng = np.random.default_rng(1)
        X = 1e8 + rng.normal(0, 0.2, size=(101, 4))
        for chunks in (1, 2, 7, 101):
            count, mu, var = 0, np.zeros(4), np.zeros(4)
            for chunk in np.array_split(X, chunks):
                mu, var = nb.GaussianNB._update_mean_variance(count, mu, var, chunk)
                count += len(chunk)
            assert_allclose(mu, X.mean(axis=0), rtol=0, atol=2e-7)
            assert_allclose(var, X.var(axis=0), rtol=2e-6, atol=1e-10)
        unchanged = nb.GaussianNB._update_mean_variance(count, mu, var, X[:0])
        assert_array_equal(unchanged[0], mu)
        assert_array_equal(unchanged[1], var)
        actual = nb.GaussianNB().fit(X, np.tile([0, 1], 51)[:101])
        reference = naive_bayes.GaussianNB().fit(X, np.tile([0, 1], 51)[:101])
        self.assert_reference(actual, reference, X)

    def test_discrete_partial_fit_and_reset(self):
        rng = np.random.default_rng(5)
        X = rng.integers(0, 6, size=(39, 5))
        y = np.repeat(["a", "b", "c"], 13)
        for name, params in MODEL_PARAMS[1:]:
            with self.subTest(model=name):
                actual = getattr(nb, name)(**params)
                reference = getattr(naive_bayes, name)(**params)
                for indices in np.array_split(np.arange(39), 7):
                    actual.partial_fit(X[indices], y[indices], classes=["c", "a", "b"])
                    reference.partial_fit(X[indices], y[indices], classes=["c", "a", "b"])
                    self.assert_reference(actual, reference, X)
                fitted = getattr(nb, name)(**params).fit(X, y)
                self.assert_reference(actual, getattr(naive_bayes, name)(**params).fit(X, y), X)
                assert_allclose(actual.predict_proba(X), fitted.predict_proba(X))
                actual.fit(X[:26], y[:26])
                reference.fit(X[:26], y[:26])
                self.assert_reference(actual, reference, X)

    def test_categorical_incremental_table_growth(self):
        actual, reference = nb.CategoricalNB(), naive_bayes.CategoricalNB()
        for X, y in (([[0, 0], [1, 1]], [0, 1]),
                     ([[4, 2], [3, 1]], [0, 1]), ([[0, 0]], [1])):
            actual.partial_fit(X, y, classes=[0, 1])
            reference.partial_fit(X, y, classes=[0, 1])
            self.assert_reference(actual, reference, [[4, 2]] if actual.n_categories_[0] == 5 else X)
        assert_array_equal(actual.n_categories_, [5, 3])
        assert_array_equal(actual.n_categories_, [t.shape[1] for t in reference.category_count_])

    def test_partial_fit_class_contract(self):
        for name, params in MODEL_PARAMS:
            with self.subTest(model=name):
                model = getattr(nb, name)(**params)
                with self.assertRaises(ValueError):
                    model.partial_fit([[0], [1]], [0, 1])
                model.partial_fit([[0], [1]], [0, 1], classes=[1, 0])
                for X, y, classes in (([[1]], [2], None), ([[1]], [0], [0, 2]),
                                      ([[1, 2]], [0], None)):
                    with self.assertRaises(ValueError):
                        model.partial_fit(X, y, classes=classes)
                assert_array_equal(model.class_count_, [1, 1])

    def test_gaussian_partial_fit_same_batch_order(self):
        rng = np.random.default_rng(30)
        actual = nb.GaussianNB(var_smoothing=0.01)
        reference = naive_bayes.GaussianNB(var_smoothing=0.01)
        test = rng.normal(size=(10, 3))
        batches = [(rng.normal(size=(12, 3)), np.repeat([0, 1], 6)),
                   (rng.normal(2, 3, size=(18, 3)), np.tile([0, 1, 2], 6)),
                   (rng.normal(-1, 0.7, size=(9, 3)), np.tile([0, 1, 2], 3))]
        for X, y in batches:
            actual.partial_fit(X, y, classes=[0, 1, 2])
            reference.partial_fit(X, y, classes=[0, 1, 2])
            self.assert_reference(actual, reference, test)
        actual.fit(batches[-1][0], batches[-1][1])
        reference.fit(batches[-1][0], batches[-1][1])
        self.assert_reference(actual, reference, test)

    def test_csr_dense_equivalence_without_densifying(self):
        rng = np.random.default_rng(4)
        X = rng.integers(0, 5, size=(45, 30)).astype(float)
        X[rng.random(X.shape) < 0.9] = 0
        y = np.tile([2, 7, 19], 15)
        for name, params in (("MultinomialNB", {}), ("BernoulliNB", {"binarize": 1}),
                             ("ComplementNB", {}), ("ComplementNB", {"norm": True})):
            for container in (sparse.csr_matrix, sparse.csr_array):
                with self.subTest(model=name, params=params, csr=container.__name__):
                    dense = getattr(nb, name)(**params).fit(X, y)
                    csr = container(X)
                    original_data = csr.data.copy()
                    reference = getattr(naive_bayes, name)(**params).fit(csr, y)
                    with patch.object(container, "toarray", side_effect=AssertionError("禁止稠密化")):
                        actual = getattr(nb, name)(**params).fit(csr, y)
                        self.assert_reference(actual, reference, csr)
                        assert_allclose(actual.predict_proba(csr), dense.predict_proba(X))
                        incremental = getattr(nb, name)(**params)
                        for indices in np.array_split(np.arange(45), 3):
                            incremental.partial_fit(csr[indices], y[indices], classes=[2, 7, 19])
                        self.assert_reference(incremental, reference, csr)
                    assert_array_equal(csr.data, original_data)

    def test_csr_binary_validation(self):
        X = sparse.csr_matrix([[0, 1], [1, 0]])
        actual = nb.BernoulliNB(binarize=None).fit(X, [0, 1])
        reference = naive_bayes.BernoulliNB(binarize=None).fit(X, [0, 1])
        self.assert_reference(actual, reference, X)
        with self.assertRaises(ValueError):
            nb.BernoulliNB(binarize=None).fit(X * 2, [0, 1])
        with self.assertRaises(ValueError):
            nb.BernoulliNB(binarize=-1).fit(X, [0, 1])
        for model in (nb.GaussianNB(), nb.CategoricalNB()):
            with self.assertRaises(ValueError):
                model.fit(X, [0, 1])
        # 重复列索引先求和，然后二值化，输入保持不变。
        duplicate = sparse.csr_matrix(([0.6, 0.6, 1.0], [0, 0, 1], [0, 2, 3]), shape=(2, 2))
        binary = nb.BernoulliNB(binarize=1).fit(duplicate, [0, 1])
        assert_array_equal(binary.feature_count_, [[1, 0], [0, 0]])
        self.assertEqual(duplicate.nnz, 3)

    def test_complement_norm_and_no_prior(self):
        X, y = [[4, 0, 1], [2, 1, 0], [0, 3, 2], [1, 4, 0]], [0, 0, 1, 1]
        for norm in (False, True):
            model = nb.ComplementNB(norm=norm).fit(X, y)
            reference = naive_bayes.ComplementNB(norm=norm).fit(X, y)
            self.assert_reference(model, reference, X)
            assert_allclose(model.joint_log_likelihood(X), np.asarray(X) @ model.feature_log_prob_.T)
            if norm:
                assert_allclose(model.feature_log_prob_.sum(axis=1), 1)
        with self.assertRaises(ValueError):
            nb.ComplementNB(norm=True).fit([[1], [2]], [0, 1])


if __name__ == "__main__":
    unittest.main()
