from ml_lib import model_selection
import numpy as np
from collections import Counter
import pandas as pd

def test_train_test_split():

    X = pd.DataFrame(np.arange(20).reshape(10, 2))
    y = pd.Series(np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, stratify=None)

    assert len(X_train) + len(X_test) == len(X), "Split sizes don't add up"
    assert len(y_train) + len(y_test) == len(y), "Split sizes don't add up"
    assert X_train.shape[1] == X.shape[1], "Feature dimension mismatch"

    for index in range(X_train.shape[0]):
        # Get one training row (as Series)
        row = X_train.iloc[index]

        # Find rows in X that match this row exactly
        mask = (X == row).all(axis=1)
        match_idx = X.index[mask]

        # Check that there’s exactly one match
        assert len(match_idx) == 1, "Feature appears more than one time or 0 times"

        # Compare y values
        assert y.loc[match_idx[0]] == y_train.iloc[index], "The y values aren't correct"

    for index in range(X_test.shape[0]):
        # Get one training row (as Series)
        row = X_test.iloc[index]

        # Find rows in X that match this row exactly
        mask = (X == row).all(axis=1)
        match_idx = X.index[mask]

        # Check that there’s exactly one match
        assert len(match_idx) == 1, "Feature appears more than one time or 0 times"

        # Compare y values
        assert y.loc[match_idx[0]] == y_test.iloc[index], "The y values aren't correct"


    print("Basic split checks passed")

    rng = np.random.default_rng(123)


    X = pd.DataFrame(np.arange(200).reshape(100, 2))
    y = pd.Series(np.array([rng.integers(0,3) for i in range(100)]))
    # Test stratify functionality
    X_train_s, X_test_s, y_train_s, y_test_s = model_selection.train_test_split(X, y, test_size=0.3, stratify=y)

    orig_dist = np.array(list(Counter(y).values()), dtype=float)
    train_dist = np.array(list(Counter(y_train_s).values()), dtype=float)
    test_dist = np.array(list(Counter(y_test_s).values()), dtype=float)

    orig_dist /= orig_dist.sum()
    train_dist /= train_dist.sum()
    test_dist /= test_dist.sum()

    # Check that distributions are similar (within tolerance)
    assert np.allclose(train_dist, orig_dist, atol=0.1), f"Train distribution differs too much: {train_dist} vs {orig_dist}"
    assert np.allclose(test_dist, orig_dist, atol=0.1), f"Test distribution differs too much: {test_dist} vs {orig_dist}"

    print("Stratified split checks passed")
    print("All tests passed for train_test_split!")

def main():
    test_train_test_split()

if __name__ == '__main__':
    main()