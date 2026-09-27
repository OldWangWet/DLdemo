"""Demo: manual KNN implementation, mirroring main.py.

Run either way:
    python -m knn.main          (from the project root)
    python knn/main.py          (as a plain script)

Uses the hand-written KNeighborsClassifier from knn in place of sklearn's,
then cross-checks the result against scikit-learn.
"""

import os
import sys

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    # Executed as a plain script: make the parent dir importable so that
    # ``knn`` resolves as a package.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from KNN.knn import KNeighborsClassifier
else:
    from .knn import KNeighborsClassifier


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print("=== manual KNN (knn) ===")
    print(f"fit method: {knn._fit_method}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Cross-check against scikit-learn's implementation.
    sk_knn = SkKNN(n_neighbors=3)
    sk_knn.fit(X_train, y_train)
    sk_pred = sk_knn.predict(X_test)
    print("=== sklearn cross-check ===")
    print(f"predictions identical: {bool((y_pred == sk_pred).all())}")
    print(f"accuracy identical: {accuracy_score(y_test, y_pred) == accuracy_score(y_test, sk_pred)}")


if __name__ == "__main__":
    main()
