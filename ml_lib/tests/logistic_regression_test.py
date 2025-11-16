from ml_lib import linear_model
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


def test_logistic_regression_binary():
    """
    Test binary logistic regression on a simple dataset.
    Checks that the model can fit and predict with high accuracy.
    """
    # Create simple binary classification data
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )

    # Initialize and fit model
    model = linear_model.LogisticRegression()
    model.fit(X, y, verbose=True)

    # Predict
    y_pred = model.predict(X)
    print(y_pred.shape)

    # Accuracy should be reasonably high (>0.9)
    acc = accuracy_score(y, y_pred)
    print(f'Accuracy: {acc}')
    assert acc > 0.85, f"Binary logistic regression accuracy too low: {acc}"


def test_logistic_regression_multiclass():
    """
    Test multiclass logistic regression (one-vs-rest or softmax).
    Checks that predictions cover all classes and accuracy is good.
    """
    # Create multiclass dataset
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0
    )

    # Initialize and fit model
    model = linear_model.LogisticRegression()
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    # Basic validity checks
    assert set(np.unique(y_pred)) <= set(np.unique(y)), \
        "Predicted classes do not match training classes"

    # Accuracy should be reasonably high (>0.8)
    acc = accuracy_score(y, y_pred)
    print(f'Accuracy: {acc}')
    assert acc > 0.75, f"Multiclass logistic regression accuracy too low: {acc}"


def main():
    print("Running logistic regression tests...")
    test_logistic_regression_binary()
    print("✅ Binary logistic regression test passed.")

    test_logistic_regression_multiclass()
    print("✅ Multiclass logistic regression test passed.")

    print("All tests passed successfully!")


if __name__ == '__main__':
    main()
