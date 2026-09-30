"""使用 sklearn 的五种朴素贝叶斯算法识别手写数字。"""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)


def main():
    X, y = load_digits(return_X_y=True)
    # 灰度等级为 0～16，转为整数以供类别 NB 使用。
    X = X.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = [
        ("高斯 NB", GaussianNB()),
        ("多项式 NB", MultinomialNB()),
        # 指定每个像素的 17 种等级，覆盖训练集中可能未出现的等级。
        ("类别 NB", CategoricalNB(min_categories=17)),
        # 像素值大于 8 时记为 1，否则记为 0。
        ("伯努利 NB", BernoulliNB(binarize=8)),
        ("补集 NB", ComplementNB()),
    ]

    # 多项式和补集 NB 将非负灰度值作为计数式权重，用于算法演示。
    print(f"训练样本：{len(y_train)}，测试样本：{len(y_test)}")
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name}（{type(model).__name__}）准确率："
              f"{accuracy_score(y_test, y_pred):.2%}")


if __name__ == "__main__":
    main()
