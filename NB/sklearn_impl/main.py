"""digits 上五种模型的一次性及分批训练，对照 sklearn 1.9.1。"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn import naive_bayes as reference
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from . import nb


def compare(manual, ref, X):
    assert_array_equal(manual.predict(X), ref.predict(X))
    for method in ["predict_joint_log_proba", "predict_log_proba", "predict_proba"]:
        assert_allclose(getattr(manual, method)(X), getattr(ref, method)(X),
                        rtol=1e-10, atol=1e-12)


def main():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(int), y, test_size=0.3, random_state=42, stratify=y)
    print(f"训练样本：{len(y_train)}，测试样本：{len(y_test)}")
    for name in nb.__all__:
        params = {"min_categories": 17} if name == "CategoricalNB" else {"binarize": 8} if name == "BernoulliNB" else {}
        use_sparse = name in ["MultinomialNB", "BernoulliNB", "ComplementNB"]
        train = sparse.csr_array(X_train) if use_sparse else X_train
        test = sparse.csr_array(X_test) if use_sparse else X_test
        manual, ref = getattr(nb, name)(**params), getattr(reference, name)(**params)
        manual.fit(train, y_train)
        ref.fit(train, y_train)
        compare(manual, ref, test)
        accuracy = manual.score(test, y_test)
        manual, ref = getattr(nb, name)(**params), getattr(reference, name)(**params)
        for indices in np.array_split(np.arange(len(y_train)), 4):
            args = dict(classes=np.unique(y_train))
            manual.partial_fit(train[indices], y_train[indices], **args)
            ref.partial_fit(train[indices], y_train[indices], **args)
        compare(manual, ref, test)
        print(f"{name}：fit {accuracy:.2%}，partial_fit {manual.score(test, y_test):.2%}；"
              f"{'CSR' if use_sparse else '稠密'}输入，预测、得分和概率对照通过")


if __name__ == "__main__":
    main()
