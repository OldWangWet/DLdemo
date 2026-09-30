"""手动 NB 演示与 sklearn 对比，可直接运行或通过 python -m NB.basic.main 运行。"""

import numpy as np
from sklearn import naive_bayes
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    from nb import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
else:
    from .nb import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB


def main():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(int), y, test_size=0.3, random_state=42, stratify=y
    )
    models = [
        (GaussianNB, {}),
        (MultinomialNB, {}),
        (CategoricalNB, {"min_categories": 17}),
        (BernoulliNB, {"binarize": 8}),
        (ComplementNB, {}),
    ]
    print(f"训练样本：{len(y_train)}，测试样本：{len(y_test)}")
    for model_class, params in models:
        name = model_class.__name__
        manual = model_class(**params).fit(X_train, y_train)
        reference = getattr(naive_bayes, name)(**params).fit(X_train, y_train)
        prediction = manual.predict(X_test)
        reference_prediction = reference.predict(X_test)
        print(
            f"{name}：手动 {accuracy_score(y_test, prediction):.2%}，"
            f"sklearn {accuracy_score(y_test, reference_prediction):.2%}，"
            f"预测一致：{np.array_equal(prediction, reference_prediction)}"
        )


if __name__ == "__main__":
    main()
