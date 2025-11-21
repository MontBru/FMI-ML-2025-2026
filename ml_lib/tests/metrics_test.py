import unittest
import numpy as np
from fractions import Fraction
from ml_lib import metrics

class TestAccuracyScore(unittest.TestCase):
    def test_when_normalize_false_then_fraction(self):
        # Arrange
        y = np.array([0, 1, 2, 2, 1, 0, 2])
        y_pred = np.array([0, 0, 2, 0, 1, 0, 2])
        expected = 5/7
        # Act
        actual = metrics.accuracy_score(y_true=y, y_pred=y_pred, normalize=False)
        # Assert
        self.assertEqual(actual, expected)

    def test_when_normalize_true_then_fraction_type(self):
        # Arrange
        y = np.array([0, 1, 2, 2, 1, 0, 2])
        y_pred = np.array([0, 0, 2, 0, 1, 0, 2])
        expected = Fraction(5,7)
        # Act
        actual = metrics.accuracy_score(y_true=y, y_pred=y_pred, normalize=True)
        # Assert
        self.assertEqual(actual, expected)

class TestEuclideanDistance(unittest.TestCase):
    def test_when_simple_points_then_distance(self):
        # Arrange
        x = np.array([0,0,0])
        y = np.array([3,4,0])
        expected = 5
        # Act
        actual = metrics.euclidean_distance(x, y)
        # Assert
        self.assertEqual(actual, expected)

    def test_when_negative_coords_then_distance(self):
        # Arrange
        x = np.array([0,0,0])
        y = np.array([3,-4,0])
        expected = 5
        # Act
        actual = metrics.euclidean_distance(x, y)
        # Assert
        self.assertEqual(actual, expected)

class TestManhattanDistance(unittest.TestCase):
    def test_when_simple_points_then_distance(self):
        # Arrange
        x = np.array([1, 2])
        y = np.array([4, 0])
        expected = 5
        # Act
        actual = metrics.manhattan_distance(x, y)
        # Assert
        self.assertEqual(actual, expected)

    def test_when_negative_coords_then_distance(self):
        # Arrange
        x = np.array([0,0])
        y = np.array([3,-4])
        expected = 7
        # Act
        actual = metrics.manhattan_distance(x, y)
        # Assert
        self.assertEqual(actual, expected)

class TestR2Score(unittest.TestCase):
    def test_when_perfect_match_then_r2_one(self):
        # Arrange
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        expected = 1.0
        # Act
        actual = metrics.r2_score(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_constant_prediction_then_r2_zero(self):
        # Arrange
        y_true = [1, 2, 3, 4]
        y_pred = [2.5, 2.5, 2.5, 2.5]
        expected = 0.0
        # Act
        actual = metrics.r2_score(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_worse_than_baseline_then_r2_negative(self):
        # Arrange
        y_true = [1, 2, 3]
        y_pred = [3, 3, 3]
        # Act
        actual = metrics.r2_score(y_true, y_pred)
        # Assert
        self.assertTrue(actual < 0)

    def test_when_length_mismatch_then_raises(self):
        # Arrange
        y_true = np.array([1,2])
        y_pred = np.array([1,2,3])
        # Act & Assert
        with self.assertRaises(Exception):
            metrics.r2_score(y_true, y_pred)

class TestRootMeanSquaredError(unittest.TestCase):
    def test_when_perfect_prediction_then_zero(self):
        # Arrange
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        expected = 0.0
        # Act
        actual = metrics.root_mean_squared_error(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_errors_one_then_rmse_one(self):
        # Arrange
        y_true = [0, 0, 0, 0]
        y_pred = [1, -1, 1, -1]
        expected = 1.0
        # Act
        actual = metrics.root_mean_squared_error(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_residuals_one_then_rmse_one(self):
        # Arrange
        y_true = [2, 4]
        y_pred = [3, 5]
        expected = 1.0
        # Act
        actual = metrics.root_mean_squared_error(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_length_mismatch_then_raises(self):
        # Arrange
        y_true = [1,2]
        y_pred = [1]
        # Act & Assert
        with self.assertRaises(Exception):
            metrics.root_mean_squared_error(y_true, y_pred)

class TestRecallScore(unittest.TestCase):
    def test_when_tp2_fn1_then_recall_two_thirds(self):
        # Arrange
        y_pred = [1,1,1,0,0]
        y_true = [1,1,0,0,0]
        expected = 2/3
        # Act
        actual = metrics.recall_score(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

class TestPrecisionScore(unittest.TestCase):
    def test_when_tp2_fp1_then_precision_two_thirds(self):
        # Arrange
        y_pred = [1,1,0,0]
        y_true = [1,1,1,0]
        expected = 2/3
        # Act
        actual = metrics.precision_score(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

class TestF1Score(unittest.TestCase):
    def test_when_tp2_fp1_fn1_then_f1_two_thirds(self):
        # Arrange
        y_pred = [1,1,1,0]
        y_true = [1,1,0,1]
        expected = 2/3
        # Act
        actual = metrics.f1_score(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_length_mismatch_then_raises(self):
        # Arrange
        y_true = [1,0]
        y_pred = [1,0,1]
        # Act & Assert
        with self.assertRaises(Exception):
            metrics.f1_score(y_true, y_pred)

class TestLogLoss(unittest.TestCase):
    def test_when_multiclass_then_log_loss(self):
        # Arrange
        y_true = [1, 2]
        y_pred = [[.1, .5, .4], [.2, .35, .45]]
        expected = -.5*(np.log(.5) + np.log(.45))
        # Act
        actual = metrics.log_loss(y_true, y_pred)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

if __name__ == '__main__':
    unittest.main()