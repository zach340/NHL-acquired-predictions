"""
features.py
===========
Feature-engineering helpers shared by the forward and defenseman models.
Everything that looks at a player's history uses prior seasons only.
"""

import warnings

import numpy as np
import pandas as pd


def safe_div(a, b, fill=0.0):
    return np.where(b == 0, fill, a / b)


def _slope(y, x):
    """OLS slope of y on x (0.0 when x has no variance)."""
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    return 0.0 if denom == 0 else float(((x - x_mean) * (y - y_mean)).sum() / denom)


def prior_slope(s, window=None):
    """
    For each season, the linear slope of the *previous* seasons' values
    (the last `window` of them, or the whole career if None).
    Needs at least two prior non-null values.
    """
    vals = s.shift(1).values
    out  = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        start = 0 if window is None else max(0, i - window + 1)
        win   = vals[start:i + 1]
        mask  = ~np.isnan(win)
        if mask.sum() < 2:
            continue
        out[i] = _slope(win[mask], np.arange(len(win))[mask].astype(float))
    return pd.Series(out, index=s.index)


def per_player(grouped, col, fn):
    """Apply a per-player Series transform and realign to the original index."""
    return grouped[col].apply(fn).reset_index(level=0, drop=True)


def prior_mean(s):
    return s.shift(1).expanding().mean()


def prior_rolling_mean(s, window=3):
    return s.shift(1).rolling(window, min_periods=1).mean()


def prior_max(s):
    return s.shift(1).cummax()


def add_career_curve_features(df, curve_stats, peak_stats, pct_peak_stats, age_slope_pairs, prefix=""):
    """
    Non-linear career-arc features from per-player quadratic fits of stat vs
    age on prior seasons. No-op if age data is absent or sparse (< 50%).

    curve_stats      {short: stat_col}  → {prefix}curve_accel_{short}, {prefix}curve_local_deriv_{short}
    peak_stats       shorts that also get {prefix}seasons_from_est_peak_{short}
    pct_peak_stats   {short: pct_of_peak_col} → {prefix}pct_peak_{short}_slope
    age_slope_pairs  [(slope_col, out_col)] → out_col = age × slope_col
    """
    if "age" not in df.columns or df["age"].isna().mean() > 0.5:
        return df

    d = df.sort_values(["player_id", "season"]).copy()

    new_cols = (
        [f"{prefix}curve_accel_{k}"       for k in curve_stats] +
        [f"{prefix}curve_local_deriv_{k}" for k in curve_stats] +
        [f"{prefix}seasons_from_est_peak_{k}" for k in peak_stats] +
        [f"{prefix}pct_peak_{k}_slope" for k in pct_peak_stats] +
        [out_col for _, out_col in age_slope_pairs]
    )
    for col in new_cols:
        d[col] = np.nan

    for _, grp in d.groupby("player_id", sort=False):
        idx      = grp.index
        ages_arr = grp["age"].values

        # Quadratic fit of stat vs age over prior seasons
        for k, stat_col in curve_stats.items():
            if stat_col not in grp.columns:
                continue
            vals_arr    = grp[stat_col].values
            accel       = np.full(len(grp), np.nan)
            local_deriv = np.full(len(grp), np.nan)
            from_peak   = np.full(len(grp), np.nan)

            for i in range(len(grp)):
                prior_ages = ages_arr[:i]
                prior_vals = vals_arr[:i]
                mask = ~(np.isnan(prior_ages) | np.isnan(prior_vals))
                pa, pv = prior_ages[mask], prior_vals[mask]
                if len(pa) < 3:
                    continue
                try:
                    with warnings.catch_warnings():
                        # Few, closely spaced ages make the fit ill-conditioned; that's expected
                        warnings.simplefilter("ignore", np.exceptions.RankWarning)
                        a, b, _ = np.polyfit(pa, pv, 2)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                curr_age = ages_arr[i]
                if np.isnan(curr_age):
                    continue
                accel[i]       = a
                local_deriv[i] = 2.0 * a * curr_age + b
                if k in peak_stats and abs(a) > 1e-9:
                    from_peak[i] = curr_age - (-b / (2.0 * a))

            d.loc[idx, f"{prefix}curve_accel_{k}"]       = accel
            d.loc[idx, f"{prefix}curve_local_deriv_{k}"] = local_deriv
            if k in peak_stats:
                d.loc[idx, f"{prefix}seasons_from_est_peak_{k}"] = from_peak

        # Slope of pct-of-peak over the prior ≤3 seasons
        for k, peak_col in pct_peak_stats.items():
            if peak_col not in grp.columns:
                continue
            peak_vals = grp[peak_col].values
            pct_slope = np.full(len(grp), np.nan)
            for i in range(len(grp)):
                window = peak_vals[max(0, i - 3):i]
                mask   = ~np.isnan(window)
                if mask.sum() < 2:
                    continue
                pct_slope[i] = _slope(window[mask], np.arange(len(window))[mask].astype(float))
            d.loc[idx, f"{prefix}pct_peak_{k}_slope"] = pct_slope

        # Age × slope interactions
        for slope_col, out_col in age_slope_pairs:
            if slope_col in grp.columns:
                d.loc[idx, out_col] = ages_arr * grp[slope_col].values

    return d


def compute_baseline(df_like, candidate_cols):
    """First non-null of the candidate history columns (0 if none)."""
    cols = [c for c in candidate_cols if c in df_like.columns]
    if not cols:
        return pd.Series(np.zeros(len(df_like)), index=df_like.index, dtype=float)
    return df_like[cols].bfill(axis=1).iloc[:, 0].fillna(0.0).astype(float)


def design_matrix(frame, feature_cols, extra=None):
    """
    Select `feature_cols` (missing ones filled with 0) plus optional extra
    columns, with ±inf and NaN mapped to 0.
    """
    X = frame.reindex(columns=feature_cols).reset_index(drop=True)
    if extra is not None:
        X = pd.concat([X, extra.reset_index(drop=True)], axis=1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def latest_team_contexts(df, team_ctx, keys):
    """
    Team context rows for the latest season, falling back to each team's most
    recent available season for any (team, *keys) combination missing from it.
    """
    latest = df["season"].max()
    ctx = team_ctx[team_ctx["season"] == latest].copy()
    if ctx["player_team"].nunique() < team_ctx["player_team"].nunique():
        group_cols = ["player_team"] + keys
        fallback = (
            team_ctx.sort_values("season", ascending=False)
            .groupby(group_cols).first().reset_index()
        )
        present = set(map(tuple, ctx[group_cols].values))
        is_missing = [tuple(key) not in present for key in fallback[group_cols].values]
        ctx = pd.concat([ctx, fallback[is_missing]], ignore_index=True)
    return ctx

