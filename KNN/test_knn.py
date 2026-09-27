"""Regression tests for the hand-written KNN implementation.

Run from the project root with:
    python -m unittest test02.test_knn -v
"""

import unittest
import warnings

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.preprocessing import StandardScaler

from .kdtree import KDTree, _introsort_2way
from .knn import KNeighborsClassifier, argkmin


class KNNParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260921)
        cls.X = rng.normal(size=(80, 4))
        cls.y = rng.integers(0, 4, size=80)
        cls.query = rng.normal(size=(19, 4))

    def assert_matches_sklearn(self, **params):
        manual = KNeighborsClassifier(n_neighbors=5, **params).fit(self.X, self.y)
        sklearn = SkKNN(n_neighbors=5, **params).fit(self.X, self.y)

        manual_dist, manual_ind = manual.kneighbors(self.query)
        sklearn_dist, sklearn_ind = sklearn.kneighbors(self.query)
        np.testing.assert_allclose(manual_dist, sklearn_dist, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(manual_ind, sklearn_ind)
        np.testing.assert_array_equal(manual.predict(self.query), sklearn.predict(self.query))
        np.testing.assert_allclose(
            manual.predict_proba(self.query),
            sklearn.predict_proba(self.query),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_main_iris_path(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        manual = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
        sklearn = SkKNN(n_neighbors=3).fit(X_train, y_train)
        self.assertEqual(manual._fit_method, "kd_tree")
        self.assertEqual(sklearn._fit_method, "kd_tree")
        np.testing.assert_array_equal(manual.predict(X_test), sklearn.predict(X_test))
        np.testing.assert_allclose(
            manual.predict_proba(X_test), sklearn.predict_proba(X_test), atol=0, rtol=0
        )

    def test_named_metrics_ignore_standalone_p(self):
        for algorithm in ("auto", "kd_tree", "brute"):
            for metric in ("euclidean", "manhattan", "chebyshev"):
                for weights in ("uniform", "distance"):
                    with self.subTest(
                        algorithm=algorithm, metric=metric, weights=weights
                    ):
                        self.assert_matches_sklearn(
                            algorithm=algorithm,
                            metric=metric,
                            p=7,
                            weights=weights,
                        )

    def test_minkowski_metrics_and_fused_brute_vote(self):
        for algorithm in ("auto", "kd_tree", "brute"):
            for p in (1, 2, 3, np.inf):
                for weights in ("uniform", "distance"):
                    with self.subTest(
                        algorithm=algorithm, p=p, weights=weights
                    ):
                        self.assert_matches_sklearn(
                            algorithm=algorithm,
                            metric="minkowski",
                            p=p,
                            weights=weights,
                        )

    def test_brute_force_two_dimensional_chunking(self):
        # Tiny memory forces both the query and training axes into many blocks.
        for metric, p in (
            ("euclidean", 2),
            ("manhattan", 1),
            ("chebyshev", np.inf),
            ("minkowski", 3),
        ):
            with self.subTest(metric=metric, p=p):
                distances, indices = argkmin(
                    self.query,
                    self.X,
                    5,
                    metric,
                    p,
                    working_memory=128,
                )
                sklearn = SkKNN(
                    n_neighbors=5, algorithm="brute", metric=metric, p=p
                ).fit(self.X, self.y)
                expected_dist, expected_ind = sklearn.kneighbors(self.query)
                np.testing.assert_allclose(
                    distances, expected_dist, rtol=1e-12, atol=1e-12
                )
                np.testing.assert_array_equal(indices, expected_ind)

    def test_auto_uses_brute_for_semimetric_and_ball_tree_is_explicit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sklearn = SkKNN(metric="minkowski", p=0.5).fit(self.X, self.y)
        manual = KNeighborsClassifier(metric="minkowski", p=0.5).fit(
            self.X, self.y
        )
        self.assertEqual(manual._fit_method, sklearn._fit_method)
        self.assertEqual(manual._fit_method, "brute")

        with self.assertRaisesRegex(ValueError, "not implemented"):
            KNeighborsClassifier(algorithm="ball_tree").fit(self.X, self.y)

    def test_kdtree_named_metric_and_introsort_fallback(self):
        manual = KDTree(self.X, metric="manhattan", p=7)
        manual_dist, manual_ind = manual.query(self.query, k=5)
        sklearn = SkKNN(
            n_neighbors=5, algorithm="kd_tree", metric="manhattan", p=7
        ).fit(self.X, self.y)
        expected_dist, expected_ind = sklearn.kneighbors(self.query)
        np.testing.assert_allclose(manual_dist, expected_dist, rtol=0, atol=0)
        np.testing.assert_array_equal(manual_ind, expected_ind)

        rng = np.random.default_rng(17)
        values = rng.normal(size=96)
        indices = np.arange(values.size)
        original_values = values.copy()
        original_indices = indices.copy()
        _introsort_2way(values, indices, 7, 89, maxd=0)
        np.testing.assert_array_equal(values[:7], original_values[:7])
        np.testing.assert_array_equal(values[89:], original_values[89:])
        np.testing.assert_allclose(values[7:89], np.sort(original_values[7:89]))
        np.testing.assert_allclose(values[7:89], original_values[indices[7:89]])
        np.testing.assert_array_equal(indices[:7], original_indices[:7])
        np.testing.assert_array_equal(indices[89:], original_indices[89:])


if __name__ == "__main__":
    unittest.main()
