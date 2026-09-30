"""验证 README 算例，并以 sklearn 对照手写数字上的预测和参数。"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import naive_bayes
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .nb import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB


class TestBasicNB(unittest.TestCase):
    def test_digits_against_sklearn(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, _ = train_test_split(
            X.astype(int), y, test_size=0.3, random_state=42, stratify=y
        )
        for model_class, params in [
            (GaussianNB, {}),
            (MultinomialNB, {}),
            (CategoricalNB, {"min_categories": 17}),
            (BernoulliNB, {"binarize": 8}),
            (ComplementNB, {}),
        ]:
            with self.subTest(model=model_class.__name__):
                manual = model_class(**params).fit(X_train, y_train)
                reference = getattr(naive_bayes, model_class.__name__)(**params)
                reference.fit(X_train, y_train)
                assert_array_equal(manual.predict(X_test), reference.predict(X_test))
                assert_array_equal(manual.classes_, reference.classes_)
                assert_allclose(manual.class_log_prior_, np.log(reference.class_prior_)
                                if model_class is GaussianNB
                                else reference.class_log_prior_)
                if model_class is GaussianNB:
                    assert_allclose(manual.theta_, reference.theta_)
                    assert_allclose(manual.var_, reference.var_)
                elif model_class is CategoricalNB:
                    for actual, expected in zip(
                        manual.feature_log_prob_, reference.feature_log_prob_
                    ):
                        assert_allclose(actual, expected)
                else:
                    assert_allclose(manual.feature_log_prob_, reference.feature_log_prob_)

    def test_readme_gaussian(self):
        X = [[1, 1], [1, 5], [3, 1], [3, 5]]
        X += [[5, 5], [5, 7], [9, 5], [9, 7]] * 2
        model = GaussianNB(var_smoothing=0).fit(X, ["A"] * 4 + ["B"] * 8)
        assert_allclose(model.theta_, [[2, 3], [7, 6]])
        assert_allclose(model.var_, [[1, 4], [4, 1]])
        assert_array_equal(model.predict([[4, 5]]), ["B"])

    def test_readme_multinomial_and_complement(self):
        # 两封垃圾邮件与四封正常邮件，累计词频分别为 [6,1,1]、[1,4,3]。
        X = [[3, 1, 0], [3, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
        y = [0, 0, 1, 1, 1, 1]
        multinomial = MultinomialNB().fit(X, y)
        assert_allclose(np.exp(multinomial.feature_log_prob_),
                        np.array([[7, 2, 2], [2, 5, 4]]) / 11)
        complement = ComplementNB().fit(X, y)
        assert_allclose(complement.feature_log_prob_,
                        -np.log(np.array([[2, 5, 4], [7, 2, 2]]) / 11))
        for model in [multinomial, complement]:
            assert_array_equal(model.predict([[2, 0, 1]]), [0])

    def test_readme_categorical(self):
        # 颜色：红=0、绿=1、黄=2；形状：圆=0、长=1。
        X = [[0, 0], [0, 0], [1, 0], [2, 1], [2, 1], [2, 0]]
        model = CategoricalNB().fit(X, [0, 0, 0, 1, 1, 1])
        assert_allclose(np.exp(model.feature_log_prob_[0][:, 0]), [1 / 2, 1 / 6])
        assert_allclose(np.exp(model.feature_log_prob_[1][:, 0]), [4 / 5, 2 / 5])
        assert_array_equal(model.predict([[0, 0]]), [0])

    def test_readme_bernoulli(self):
        # alpha=1 时，8 个样本中的出现次数 [7,5]、[1,2] 对应 README 概率。
        X = [[int(i < 7), int(i < 5)] for i in range(8)]
        X += [[int(i < 1), int(i < 2)] for i in range(8)]
        model = BernoulliNB(binarize=None).fit(X, [0] * 8 + [1] * 8)
        assert_allclose(np.exp(model.feature_log_prob_), [[0.8, 0.6], [0.2, 0.3]])
        # “附件”未出现的概率也参与得分。
        scores = [np.exp(model._log_score(np.array([[1, 0]]), i))[0]
                  for i in range(2)]
        assert_allclose(scores, [0.16, 0.07])
        assert_array_equal(model.predict([[1, 0]]), [0])

    def test_reserved_categories_and_threshold(self):
        X, y = [[0], [0], [1], [1]], [0, 0, 1, 1]
        manual = CategoricalNB(min_categories=3).fit(X, y)
        reference = naive_bayes.CategoricalNB(min_categories=3).fit(X, y)
        assert_array_equal(manual.predict([[2]]), reference.predict([[2]]))
        with self.assertRaises(ValueError):
            manual.predict([[3]])
        binary = BernoulliNB(binarize=8).fit([[8], [9], [0], [16]], y)
        assert_array_equal(binary.feature_count_, [[1], [1]])


if __name__ == "__main__":
    unittest.main()
