"""digits 上 basic、optimized、sklearn 的三方对照。"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import naive_bayes
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from NB.basic import nb as basic
from NB.optimized import nb as optimized


MODEL_PARAMS = [
    ("GaussianNB", {"var_smoothing": 1e-9}),
    ("MultinomialNB", {"alpha": 1.0}),
    ("CategoricalNB", {"alpha": 1.0, "min_categories": 17}),
    ("BernoulliNB", {"alpha": 1.0, "binarize": 8}),
    ("ComplementNB", {"alpha": 1.0}),
]


def digits_split():
    X, y = load_digits(return_X_y=True)
    return train_test_split(
        X.astype(int), y, test_size=0.3, random_state=42, stratify=y
    )


def main():
    X_train, X_test, y_train, y_test = digits_split()
    print(f"训练样本：{len(y_train)}，测试样本：{len(y_test)}")
    for name, params in MODEL_PARAMS:
        models = [getattr(module, name)(**params).fit(X_train, y_train)
                  for module in (basic, optimized, naive_bayes)]
        predictions = [model.predict(X_test) for model in models]
        for prediction in predictions[1:]:
            assert_array_equal(prediction, predictions[0])
        baseline, manual, reference = models
        raw = manual._validate_predict_X(X_test)
        basic_scores = np.column_stack([
            baseline._log_score(raw, i) for i in range(len(baseline.classes_))
        ])
        scores = manual.joint_log_likelihood(X_test)
        assert_allclose(scores, basic_scores, rtol=1e-8, atol=1e-10)
        assert_allclose(scores, reference._joint_log_likelihood(raw),
                        rtol=1e-8, atol=1e-10)
        assert_allclose(manual.predict_proba(X_test), reference.predict_proba(X_test),
                        rtol=1e-8, atol=1e-10)
        accuracy = [np.mean(prediction == y_test) for prediction in predictions]
        print(f"{name}：basic {accuracy[0]:.2%}，optimized {accuracy[1]:.2%}，"
              f"sklearn {accuracy[2]:.2%}；预测、得分、概率对照通过")


if __name__ == "__main__":
    main()
