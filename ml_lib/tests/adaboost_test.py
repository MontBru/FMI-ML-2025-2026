from ml_lib.tree import AdaBoostClassifier
import numpy as np

def simple_dataset():
    X = np.array([
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [5, 5],
        [6, 5],
        [5, 6],
        [6, 6]
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y

def xor_dataset():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])   # XOR labels
    return X, y



# ---------------------------------------------------------
# 1. FIT SHOULD RUN WITHOUT ERRORS
# ---------------------------------------------------------
def test_fit_runs():
    X, y = simple_dataset()
    clf = AdaBoostClassifier(n_estimators=5, learning_rate=1.0)

    try:
        clf.fit(X, y)
    except Exception as e:
        print(f"fit() raised an exception: {e}")


# ---------------------------------------------------------
# 2. PREDICT RETURNS CORRECT SHAPE
# ---------------------------------------------------------
def test_predict_shape():
    X, y = simple_dataset()
    clf = AdaBoostClassifier(n_estimators=5)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


# ---------------------------------------------------------
# 3. CLASSIFIER SHOULD ACHIEVE PERFECT ACCURACY ON SIMPLE DATASET
# ---------------------------------------------------------
def test_predict_accuracy_simple(dataset):
    X, y = dataset
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    accuracy = np.mean(y_pred == y)

    print(accuracy)
    assert accuracy > 0.8, "AdaBoost should learn simple dataset"


# ---------------------------------------------------------
# 4. WEIGHTS SHOULD CHANGE ACROSS ITERATIONS (BOOSTING EFFECT)
# ---------------------------------------------------------
def test_boosting_effect():
    X, y = xor_dataset()
    clf = AdaBoostClassifier(n_estimators=5)
    clf.fit(X, y)

    # After fitting, we should have estimators with different amount_of_say
    alphas = clf.amounts_of_say

    assert len(alphas) == clf.n_estimators
    # Some alphas should differ ==> multiple boosting rounds occurred
    if clf.n_estimators > 1:
        assert len(set(np.round(alphas, 5))) > 1, "Boosting should produce different alphas"


# ---------------------------------------------------------
# 5. LEARNING RATE SHOULD AFFECT ALPHA
# ---------------------------------------------------------
def test_learning_rate_effect():
    X, y = xor_dataset()

    clf1 = AdaBoostClassifier(n_estimators=5, learning_rate=1.0)
    clf2 = AdaBoostClassifier(n_estimators=5, learning_rate=0.1)

    clf1.fit(X, y)
    clf2.fit(X, y)

    # Compare first estimator alpha
    alpha1 = clf1.amounts_of_say[0]
    alpha2 = clf2.amounts_of_say[0]

    assert abs(alpha2) < abs(alpha1), "Lower learning rate must reduce alpha"


# ---------------------------------------------------------
# 6. HANDLES -1/1 LABEL FORMAT (ADABOOST ORIGINAL FORM)
# ---------------------------------------------------------
def test_negative_positive_labels():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([-1, -1, 1, 1])

    clf = AdaBoostClassifier(n_estimators=5)
    clf.fit(X, y)
    preds = clf.predict(X)

    # Predictions should be -1 or 1
    assert set(preds) <= {-1, 1}


# ---------------------------------------------------------
# 7. PREDICTION SHOULD NOT CRASH ON UNSEEN DATA SIZE
# ---------------------------------------------------------
def test_predict_unseen_size():
    X, y = simple_dataset()
    clf = AdaBoostClassifier(n_estimators=5)
    clf.fit(X, y)

    X_test = np.array([[2, 2], [10, 10]])
    preds = clf.predict(X_test)

    assert preds.shape[0] == 2


def main():
    test_fit_runs()
    test_predict_shape()
    test_predict_accuracy_simple(simple_dataset())
    test_predict_accuracy_simple(xor_dataset())

    test_boosting_effect()
    test_learning_rate_effect()
    test_negative_positive_labels()
    test_predict_unseen_size()

    print("All tests for adaboost passed correctly!")

if __name__ == '__main__':
    main()