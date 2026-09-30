"""
training.py
===========
Shared residual-model training loop, progress reporting and the
ModelBundle container that both models return and cache to disk.
"""

import os
from typing import NamedTuple

import joblib
import numpy as np
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from .config import CV_FOLDS, ELITE_QUANTILE


class ModelBundle(NamedTuple):
    """Everything a trained model family needs at prediction time."""
    df: object
    team_ctx: object
    has_age: bool
    profiles: dict              # player_id -> (profile Series, [seasons])
    fit_models: dict            # target -> {"global": model}
    fit_metrics: dict
    fit_feature_names: list
    next_models: dict
    next_metrics: dict
    next_feature_names: list


def load_bundle(path):
    """Load a cached bundle (stored as a plain tuple for pickle stability)."""
    return ModelBundle(*joblib.load(path)) if os.path.exists(path) else None


def save_bundle(bundle, path):
    joblib.dump(tuple(bundle), path)


class StreamlitProgress:
    """Status line + progress bar shown while models train."""

    def __init__(self, total_steps):
        self.total  = total_steps
        self.step   = 0
        self._status = st.empty()
        self._bar    = st.progress(0, text="Starting up...")

    def status(self, msg):
        self._status.markdown(msg)

    def advance(self, msg):
        self.step += 1
        self._bar.progress(min(self.step / self.total, 0.99), text=msg)

    def finish(self, msg):
        self._bar.progress(1.0, text=msg)
        self._status.empty()
        self._bar.empty()


def train_residual_models(X, df, targets, target_col_map, *, labels, make_model,
                          baseline_fn, weight_fn, label_prefix, progress, track_elite=False):
    """
    For each target, fit `make_model()` on (target − baseline) with sample
    weights, report CV-fold MAE/RMSE, then refit on all rows.

    baseline_fn(df, target) -> Series; weight_fn(y, target) -> array.
    Returns (models, metrics) where models[target] = {"global": model}.
    """
    kf      = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    models  = {}
    metrics = {}

    for target in targets:
        label    = labels[target]
        y        = np.clip(df[target_col_map[target]].values, 0, None)
        baseline = baseline_fn(df, target).values
        y_resid  = y - baseline
        sample_w = weight_fn(y, target)
        elite_cut = np.quantile(y, ELITE_QUANTILE)

        fold_maes, fold_rmses, fold_elite_maes = [], [], []
        for fold, (tr, val) in enumerate(kf.split(X), 1):
            progress.status(f"🔁 **{label_prefix} — {label}** CV fold {fold}/{CV_FOLDS}")
            m = make_model()
            m.fit(X.iloc[tr], y_resid[tr], sample_weight=sample_w[tr])
            preds = np.clip(baseline[val] + m.predict(X.iloc[val]), 0, None)
            fold_maes.append(mean_absolute_error(y[val], preds))
            fold_rmses.append(np.sqrt(mean_squared_error(y[val], preds)))
            elite_mask = y[val] >= elite_cut
            if track_elite and elite_mask.any():
                fold_elite_maes.append(mean_absolute_error(y[val][elite_mask], preds[elite_mask]))
            progress.advance(f"{label_prefix} {label}: fold {fold}/{CV_FOLDS} — MAE {np.mean(fold_maes):.3f}")

        progress.status(f"✅ **{label_prefix} — {label}** fitting final residual model...")
        final = make_model()
        final.fit(X, y_resid, sample_weight=sample_w)
        models[target]  = {"global": final}
        metrics[target] = {
            "mae":  (float(np.mean(fold_maes)),  float(np.std(fold_maes))),
            "rmse": (float(np.mean(fold_rmses)), float(np.std(fold_rmses))),
        }
        if track_elite:
            metrics[target]["elite_mae"] = (
                (float(np.mean(fold_elite_maes)), float(np.std(fold_elite_maes)))
                if fold_elite_maes else (np.nan, np.nan)
            )
        progress.advance(f"{label_prefix} {label} done — MAE {np.mean(fold_maes):.3f}")

    return models, metrics


def total_training_steps(n_targets):
    """3 setup steps + (folds + final fit) per target, for both model versions."""
    return 3 + 2 * n_targets * (CV_FOLDS + 1)
