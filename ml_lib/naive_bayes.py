import numpy as np
from scipy.sparse import issparse
from sklearn import base

class MultinomialNB(base.BaseEstimator):
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        X = X.astype(float)

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Count samples per class
        class_count = np.bincount(y_idx)

        # Class prior
        if self.fit_prior:
            self.class_log_prior_ = np.log(class_count / class_count.sum())
        else:
            self.class_log_prior_ = np.log(np.ones(n_classes) / n_classes)

        # Feature counts per class
        feature_count = np.zeros((n_classes, n_features))

        for c in range(n_classes):
            X_c = X[y_idx == c]
            if issparse(X_c):
                feature_count[c] = np.asarray(X_c.sum(axis=0)).ravel()
            else:
                feature_count[c] = X_c.sum(axis=0)

        # Apply Laplace smoothing
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)

        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)

        return self

    def predict(self, X):
        X = X.astype(float)

        if issparse(X):
            jll = X @ self.feature_log_prob_.T
        else:
            jll = np.dot(X, self.feature_log_prob_.T)

        jll += self.class_log_prior_

        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
