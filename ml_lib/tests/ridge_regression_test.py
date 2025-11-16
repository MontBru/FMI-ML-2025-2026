import numpy as np
from ml_lib.linear_model import Ridge


def test_fit_predict_perfect():
    # y = 2x + 1 — learnable linear relationship
    X = np.array([[0], [1], [2], [3]], dtype=float)  # shape (N,1)
    y = 2*X.flatten() + 1  # shape (N,)

    model = Ridge()
    model.fit(X, y)
    y_pred = model.predict(X)

    # perfect reconstruction
    assert np.allclose(y_pred, y, atol=1e-8)


def test_score_perfect_r2_equals_1():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = 2*X.flatten() + 1

    model = Ridge()
    model.fit(X, y)
    r2 = model.score(X, y)

    assert np.isclose(r2, 1.0)


def test_score_worse_than_baseline_r2_negative():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 0, 0], dtype=float)
    y_bad_pred = np.array([10, 10, 10, 10])  # what model will predict after bad fit?

    model = Ridge()
    model.fit(X, y_bad_pred)   # model learns something irrelevant
    r2 = model.score(X, y)

    assert r2 < 0  # worse than predicting the mean




def test_fit_mismatch_shapes_should_raise():
    X = np.random.randn(10, 3)
    y = np.random.randn(9)  # wrong length

    model = Ridge()
    try:
        model.fit(X, y)
    except:
        return
    raise Exception("Expected to raise exception because of size mismatch")


def main():
    test_fit_predict_perfect()
    test_fit_mismatch_shapes_should_raise()
    test_score_perfect_r2_equals_1()
    test_score_worse_than_baseline_r2_negative()

if __name__ == '__main__':
    main()