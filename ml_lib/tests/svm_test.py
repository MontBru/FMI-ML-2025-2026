import numpy as np
import ml_lib.svm as svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def test_svc_kernel(kernel):
    """Generic test for SVC with a specific kernel."""
    # Generate a simple binary classification dataset
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

    # Split train/test
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # Initialize and train the model
    clf = svm.SVC(c=5.0, gamma=1, kernel=kernel)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Check output type and shape
    assert isinstance(y_pred, np.ndarray), f"{kernel} kernel: predict() must return np.ndarray"
    assert y_pred.shape == y_test.shape, f"{kernel} kernel: wrong prediction shape"

    # Check that all labels are valid
    unique_labels = np.unique(y)
    assert np.all(np.isin(y_pred, unique_labels)), f"{kernel} kernel: invalid predicted labels {np.unique(y_pred)}"

    # Check reasonable accuracy
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.4, f"{kernel} kernel: accuracy too low ({acc:.2f})"

    print(f"✅ {kernel} kernel test passed with accuracy {acc:.2f}")


def main():
    print("Running tests for custom SVC class...\n")

    for kernel in ["linear", "polynomial", "rbf"]:
        test_svc_kernel(kernel)

    print("\nAll SVC kernel tests passed successfully!")


if __name__ == "__main__":
    main()
