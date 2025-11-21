import unittest
import numpy as np
import ml_lib.svm as svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class TestSvcKernel(unittest.TestCase):
    def test_when_kernel_is_linear_then_svc_predicts_well(self):
        # Arrange
        # Generate a simple binary classification dataset, use linear kernel, expect accuracy > 0.4
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X.astype(np.float64))
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        expected = True  # accuracy > 0.4
        # Act
        clf = svm.SVC(c=5.0, gamma=1, kernel="linear")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        actual = acc > 0.4
        # Assert
        self.assertTrue(actual)

    def test_when_kernel_is_polynomial_then_svc_predicts_well(self):
        # Arrange
        # Generate a simple binary classification dataset, use polynomial kernel, expect accuracy > 0.4
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X.astype(np.float64))
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        expected = True  # accuracy > 0.4
        # Act
        clf = svm.SVC(c=5.0, gamma=1, kernel="polynomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        actual = acc > 0.4
        # Assert
        self.assertTrue(actual)

    def test_when_kernel_is_rbf_then_svc_predicts_well(self):
        # Arrange
        # Generate a simple binary classification dataset, use rbf kernel, expect accuracy > 0.4
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X.astype(np.float64))
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        expected = True  # accuracy > 0.4
        # Act
        clf = svm.SVC(c=5.0, gamma=1, kernel="rbf")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        actual = acc > 0.4
        # Assert
        self.assertTrue(actual)

if __name__ == "__main__":
    unittest.main()
