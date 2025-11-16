
import numpy as np
from ml_lib.neighbors import KNeighborsClassifier

def test_knn_basic():
    X_train = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [6.0, 9.0]
    ])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([
        [1.2, 1.9], 
        [5.5, 8.5]
    ])

    # Case 1: k = 1
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("k=1 predictions:", y_pred)
    assert np.array_equal(y_pred, np.array([0, 1])), "KNN k=1 predictions incorrect"

    # Case 2: k = 3 (majority vote)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("k=3 predictions:", y_pred)
    assert np.array_equal(y_pred, np.array([0, 1])), "KNN k=3 predictions incorrect"
    X_train = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [6.0, 9.0]
    ])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([
        [1.2, 1.9],
        [5.5, 8.5]
    ])

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    assert np.array_equal(y_pred, np.array([0, 1])), "KNN predictions incorrect"

if __name__ == "__main__":
    test_knn_basic()
    print("KNeighborsClassifier passed the test!")
