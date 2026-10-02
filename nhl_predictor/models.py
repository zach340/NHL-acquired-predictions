"""
models.py
=========
Regressors used on top of the baselines. Each has fit(X, y, sample_weight),
predict(X) and feature_importances_ (normalised to sum to 1), so training,
prediction and the importance chart treat them the same way.

The shipped models are blends: tree models capture interactions, a ridge
regression adds a smooth linear signal they fit poorly, and averaging them
beat any single model on season-based CV and the 2025-26 holdout.
"""

import numpy as np
from sklearn.linear_model import Ridge


def _normalise(imp):
    imp = np.abs(np.asarray(imp, dtype=float))
    total = imp.sum()
    return imp / total if total > 0 else imp


class RidgeRegressor:
    """Ridge on features winsorised at the training 1st/99th percentiles and standardised."""

    def __init__(self, alpha=30.0):
        self.alpha = alpha

    def _prep(self, X):
        Z = np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
        return (Z - self.mu_) / self.sd_

    def fit(self, X, y, sample_weight=None):
        A = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.percentile(A, 1, axis=0), np.percentile(A, 99, axis=0)
        Z = np.clip(A, self.lo_, self.hi_)
        self.mu_, self.sd_ = Z.mean(axis=0), Z.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        self.model_ = Ridge(alpha=self.alpha).fit(self._prep(A), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X))

    @property
    def feature_importances_(self):
        return _normalise(self.model_.coef_)


class BlendRegressor:
    """Equal-weight average of several regressors built by zero-argument factories."""

    def __init__(self, factories):
        self.factories = factories

    def fit(self, X, y, sample_weight=None):
        self.models_ = [make().fit(X, y, sample_weight=sample_weight) for make in self.factories]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)

    @property
    def feature_importances_(self):
        return np.mean([_normalise(m.feature_importances_) for m in self.models_], axis=0)


def make_catboost(loss="RMSE"):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=800, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, loss_function=loss,
                             random_seed=42, verbose=0, thread_count=-1, allow_writing_files=False)
