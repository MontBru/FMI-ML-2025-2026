from ml_lib import metrics
import numpy as np
from fractions import Fraction

def test_accuracy_score():
    y = np.array([0, 1, 2, 2, 1, 0, 2])
    y_pred = np.array([0, 0, 2, 0, 1, 0, 2])
    assert metrics.accuracy_score(y_true=y, y_pred=y_pred, normalize=False) == 5/7, "Didn't work with normalize = False"
    assert metrics.accuracy_score(y_true=y, y_pred=y_pred, normalize=True) == Fraction(5,7), "Didn't work with normalize = True"
    print("All tests for accuracy passed successfully!")

def test_euclidean_distance():
    x = np.array([0,0,0])
    y = np.array([3,4,0])
    assert metrics.euclidean_distance(x,y) == 5, "Didn't calculate euclidean distance correctly"

    x = np.array([0,0,0])
    y = np.array([3,-4,0])
    assert metrics.euclidean_distance(x,y) == 5, "Didn't calculate euclidean distance correctly with negative numbers"

    print("All tests for euclidean distance passed successfully!")

def test_manhattan_distance():
    x = np.array([1, 2])
    y = np.array([4, 0])
    assert metrics.manhattan_distance(x,y) == 5, "Didn't calculate manhattan distance correctly"

    x = np.array([0,0])
    y = np.array([3,-4])
    assert metrics.manhattan_distance(x,y) == 7, "Didn't calculate manhattan distance correctly with negative numbers"

    print("All tests for manhattan distance passed successfully!")

def test_r2_score():
    # exact match → R2 = 1
    assert np.isclose(metrics.r2_score([1, 2, 3], [1, 2, 3]), 1.0)

    # constant prediction equal to mean → R2 = 0
    assert np.isclose(metrics.r2_score([1, 2, 3, 4], [2.5, 2.5, 2.5, 2.5]), 0.0)

    # worse than baseline → R2 < 0
    assert metrics.r2_score([1, 2, 3], [3, 3, 3]) < 0

    # trivial check: length mismatch should error
    try:
        metrics.r2_score(np.array([1,2]), np.array([1,2,3]))
    except Exception:
        pass
    else:
        raise AssertionError("Expected error on size mismatch")
    print("All tests for R2 metric passed successfully!")

def test_root_mean_squared_error():
    # perfect prediction → RMSE = 0
    assert np.isclose(metrics.root_mean_squared_error([1, 2, 3], [1, 2, 3]), 0.0)

    # errors = [1,1,1,1] → RMSE = 1
    assert np.isclose(metrics.root_mean_squared_error([0, 0, 0, 0], [1, -1, 1, -1]), 1.0)

    # residuals = [1,1] → RMSE = 1
    assert np.isclose(metrics.root_mean_squared_error([2, 4], [3, 5]), 1.0)

    # mismatch lengths → should raise
    try:
        metrics.root_mean_squared_error([1,2], [1])
    except Exception:
        pass
    else:
        raise AssertionError("Expected an exception for size mismatch")
    print("All tests for RMSE passed successfully!")

def test_recall_score():
    # TP=2, FN=1 → recall=2/3
    y_pred = [1,1,1,0,0]
    y_true = [1,1,0,0,0]
    assert np.isclose(metrics.recall_score(y_true,y_pred), 2/3)

    print("All tests for recall_score passed!")


def test_precision_score():
    # TP=2, FP=1 → precision=2/3
    y_pred = [1,1,0,0]
    y_true = [1,1,1,0]
    assert np.isclose(metrics.precision_score(y_true,y_pred), 2/3)

    print("All tests for precision_score passed!")


def test_f1_score():
    # TP=2, FP=1, FN=1 → P=2/3,R=2/3 → F1 = 2/3
    y_pred = [1,1,1,0]
    y_true = [1,1,0,1]
    assert np.isclose(metrics.f1_score(y_true,y_pred), 2/3)

    # trivial mismatch length → should error
    try:
        metrics.f1_score([1,0],[1,0,1])
    except Exception:
        pass
    else:
        raise AssertionError("Expected error on size mismatch")

    print("All tests for f1_score passed!")

def test_log_loss():
    y_true = [1, 2]
    y_pred = [[.1, .5, .4], 
              [.2, .35, .45]]

    assert np.isclose(metrics.log_loss(y_true, y_pred), -.5*(np.log(.5) + np.log(.45)))
    print("All tests for log_loss passed successfully")



def main():
    test_accuracy_score()
    test_euclidean_distance()
    test_manhattan_distance()
    test_r2_score()
    test_root_mean_squared_error()
    test_recall_score()
    test_precision_score()
    test_f1_score()
    test_log_loss()
    
    print("All tests passed successfully")

if __name__ == '__main__':
    main()