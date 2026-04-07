"""
NHL Player Team Fit Predictor — Streamlit App
=============================================
Run with:  python -m streamlit run app.py

Tabs:
  1. Team Fit       — current skill profile projected across all 32 teams
  2. Next Season    — next-season forecast projected across all 32 teams
"""

import warnings
import os
import joblib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_FILE      = "offensive_performance_by_season_per60_renamed.csv"
AGES_FILE      = "player_ages.csv"
PP_FILE        = "pp_features.csv"
LINEMATE_FILE  = "linemate_features.csv"
CACHE_FILE     = "trained_models_forwards_v5.joblib"
TARGETS        = ["game_score_per_game", "points_per_game", "goals_per_game"]
MIN_GP         = 20
MIN_ICE        = 300
CV_FOLDS       = 3
N_SEASONS      = 3
SEASON_WEIGHTS = [3, 2, 1]

NHL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

ROLE_MIN_PPG = {
    "elite": 0.75,
    "top6": 0.40,
    "bottom6": 0.20,
    "depth": 0.00,
}

ROLE_LABELS = {
    "elite": "Top 2 Elite",
    "top6": "Top 6",
    "bottom6": "Bottom 6",
    "depth": "Depth",
}

FORWARD_POSITIONS = ["C", "L", "R"]

TARGET_LABELS = {
    "game_score_per_game": "Game Score / Game",
    "points_per_game":     "Points / Game",
    "goals_per_game":      "Goals / Game",
}

ELITE_QUANTILE = 0.90

BASELINE_FEATURES = {
    "game_score_per_game": [
        "recent_3yr_mean_gamescore_pg",
        "career_prev_mean_gamescore_pg",
        "prev_season_gamescore_pg",
    ],
    "points_per_game": [
        "recent_3yr_mean_points_pg",
        "career_prev_mean_points_pg",
        "prev_season_points_pg",
        "league_avg_points_pg",
    ],
    "goals_per_game": [
        "recent_3yr_mean_goals_pg",
        "career_prev_mean_goals_pg",
        "prev_season_goals_pg",
        "league_avg_goals_pg",
    ],
}

# ── Feature lists ──────────────────────────────────────────────────────────────

PLAYER_FEATURES = [
    "finishing_skill",
    "finishing_skill_adj",
    "flurry_reliance",
    "hd_shot_share",
    "hd_finishing",
    "hd_xg_outperformance",
    "xg_per_attempt",
    "on_target_rate",
    "primary_assist_share",
    "primary_vs_secondary",
    "ind_shot_attempts_per60",
    "ind_high_danger_shots_per60",
    "ind_medium_danger_shots_per60",
    "ind_low_danger_shots_per60",
    "shifts_per60",
    # Scoring environment — captures league-wide trends by season
    "league_avg_points_pg",
    "league_avg_goals_pg",
    # Career peak features — player ceiling signal
    "career_peak_points_pg",
    "career_peak_goals_pg",
    "pct_of_peak_points",
    "pct_of_peak_goals",
    # Powerplay & zone start features — key deployment signals
    "pp_icetime_pct",
    "pp_points_per60",
    "pp_goals_per60",
    "pp_xg_per60",
    "pp_points_share",
    "o_zone_start_pct",
    "zone_start_diff",
    # Career history features — keep older seasons in the signal without leakage
    "career_seasons_prior",
    "prev_season_points_pg",
    "prev_season_goals_pg",
    "prev_season_gamescore_pg",
    "career_prev_mean_points_pg",
    "career_prev_mean_goals_pg",
    "career_prev_mean_gamescore_pg",
    "career_prev_peak_points_pg",
    "career_prev_peak_goals_pg",
    "recent_3yr_mean_points_pg",
    "recent_3yr_mean_goals_pg",
    "recent_3yr_mean_gamescore_pg",
    # Explicit trend features — slope of prior seasons only
    "recent_3yr_points_slope",
    "recent_3yr_goals_slope",
    "recent_3yr_gamescore_slope",
    "career_points_slope",
    "career_goals_slope",
    "career_gamescore_slope",
]

# Age features — only added when age data is available
AGE_FEATURES = ["age", "age_sq", "age_x_shot_attempts", "age_x_finishing", "age_x_hd_share"]

TEAM_FEATURES = [
    "team_median_toi_pg",
    "team_avg_hd_share",
    "team_avg_adj_xg_per60",
    "team_adj_ratio",
    "team_avg_primary_rate",
    "team_avg_on_target",
    # Team-level line quality — swapped per team at prediction time
    "team_avg_line_adj_xg_per60",
    "team_avg_line_xg_pct",
    "team_avg_line_hd_xg_per60",
    "team_avg_line_corsi_pct",
]

# Next-season model also uses trajectory (YoY delta) features
TRAJECTORY_FEATURES = [
    "yoy_points_delta",
    "yoy_goals_delta",
    "yoy_gamescore_delta",
    "games_played_pct",
    "career_year",
]

POSITION_DUMMIES = ["pos_C", "pos_D", "pos_L", "pos_R"]

# ── Feature engineering ────────────────────────────────────────────────────────

def safe_div(a, b, fill=0.0):
    return np.where(b == 0, fill, a / b)


def engineer_player_features(df):
    d = df.copy()
    d["finishing_skill"]      = safe_div(d["ind_goals_per60"], d["ind_expected_goals_per60"])
    d["finishing_skill_adj"]  = safe_div(d["ind_goals_per60"], d["ind_flurry_score_venue_adj_expected_goals_per60"])
    d["flurry_reliance"]      = safe_div(d["ind_expected_goals_per60"], d["ind_flurry_adj_expected_goals_per60"])
    d["hd_shot_share"]        = safe_div(d["ind_high_danger_shots_per60"], d["ind_shots_on_goal_per60"])
    d["hd_finishing"]         = safe_div(d["ind_high_danger_goals_per60"], d["ind_high_danger_shots_per60"])
    d["hd_xg_outperformance"] = safe_div(d["ind_high_danger_goals_per60"], d["ind_high_danger_expected_goals_per60"])
    d["primary_assist_share"] = safe_div(d["ind_primary_assists_per60"], d["ind_points_per60"])
    d["primary_vs_secondary"] = safe_div(d["ind_primary_assists_per60"], d["ind_secondary_assists_per60"])
    d["xg_per_attempt"]       = safe_div(d["ind_expected_goals_per60"], d["ind_shot_attempts_per60"])
    d["on_target_rate"]       = safe_div(d["ind_shots_on_goal_per60"], d["ind_shot_attempts_per60"])
    d["toi_per_game"]         = d["ice_time"] / d["games_played"]
    d["game_score_per_game"]  = d["game_score"] / d["games_played"]
    d["points_per_game"]      = d["ind_points_per60"] * d["toi_per_game"] / 60
    d["goals_per_game"]       = d["ind_goals_per60"]  * d["toi_per_game"] / 60
    # League scoring environment — lets model learn how scoring rates
    # vary by season and adjust predictions accordingly
    d["league_avg_points_pg"] = d.groupby("season")["points_per_game"].transform("mean")
    d["league_avg_goals_pg"]  = d.groupby("season")["goals_per_game"].transform("mean")
    # Career peak features — anchors the model to a player's ceiling
    # rather than letting weighted averaging compress top performers
    d["career_peak_points_pg"] = d.groupby("player_id")["points_per_game"].transform("max")
    d["career_peak_goals_pg"]  = d.groupby("player_id")["goals_per_game"].transform("max")
    d["pct_of_peak_points"]    = safe_div(
        d["points_per_game"], d["career_peak_points_pg"], fill=0.0
    )
    d["pct_of_peak_goals"]     = safe_div(
        d["goals_per_game"], d["career_peak_goals_pg"], fill=0.0
    )
    # Age interaction features — lets model learn that the same skill level
    # means something different at 25 vs 35
    if "age" in d.columns:
        d["age_x_shot_attempts"] = d["age"] * d["ind_shot_attempts_per60"]
        d["age_x_finishing"]     = d["age"] * d["finishing_skill_adj"]
        d["age_x_hd_share"]      = d["age"] * d["hd_shot_share"]
    return d


def engineer_trajectory_features(df):
    """Add YoY delta and career stage features. Requires age already joined."""
    d = df.sort_values(["player_id", "season"]).copy()

    d["yoy_points_delta"]    = d.groupby("player_id")["ind_points_per60"].diff()
    d["yoy_goals_delta"]     = d.groupby("player_id")["ind_goals_per60"].diff()
    d["yoy_gamescore_delta"] = d.groupby("player_id")["game_score_per_game"].diff()
    d["games_played_pct"]    = d["games_played"] / 82.0
    d["career_year"]         = d.groupby("player_id").cumcount() + 1

    return d


def engineer_career_history_features(df):
    """Add leakage-safe career history features based on prior seasons only."""
    d = df.sort_values(["player_id", "season"]).copy()

    def prior_slope(s, window=None):
        vals = s.shift(1).values
        out = np.full(len(vals), np.nan)
        for i in range(len(vals)):
            start = 0 if window is None else max(0, i - window + 1)
            win = vals[start:i + 1]
            mask = ~np.isnan(win)
            if mask.sum() < 2:
                continue
            y = win[mask]
            x = np.arange(len(win))[mask].astype(float)
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            out[i] = 0.0 if denom == 0 else float(((x - x_mean) * (y - y_mean)).sum() / denom)
        return pd.Series(out, index=s.index)

    grouped = d.groupby("player_id", sort=False)
    d["career_seasons_prior"] = grouped.cumcount().astype(float)

    d["prev_season_points_pg"] = grouped["points_per_game"].shift(1)
    d["prev_season_goals_pg"] = grouped["goals_per_game"].shift(1)
    d["prev_season_gamescore_pg"] = grouped["game_score_per_game"].shift(1)

    d["career_prev_mean_points_pg"] = grouped["points_per_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    d["career_prev_mean_goals_pg"] = grouped["goals_per_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    d["career_prev_mean_gamescore_pg"] = grouped["game_score_per_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)

    d["career_prev_peak_points_pg"] = grouped["points_per_game"].apply(lambda s: s.shift(1).cummax()).reset_index(level=0, drop=True)
    d["career_prev_peak_goals_pg"] = grouped["goals_per_game"].apply(lambda s: s.shift(1).cummax()).reset_index(level=0, drop=True)

    d["recent_3yr_mean_points_pg"] = grouped["points_per_game"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
    d["recent_3yr_mean_goals_pg"] = grouped["goals_per_game"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
    d["recent_3yr_mean_gamescore_pg"] = grouped["game_score_per_game"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)

    d["recent_3yr_points_slope"] = grouped["points_per_game"].apply(lambda s: prior_slope(s, window=3)).reset_index(level=0, drop=True)
    d["recent_3yr_goals_slope"] = grouped["goals_per_game"].apply(lambda s: prior_slope(s, window=3)).reset_index(level=0, drop=True)
    d["recent_3yr_gamescore_slope"] = grouped["game_score_per_game"].apply(lambda s: prior_slope(s, window=3)).reset_index(level=0, drop=True)

    d["career_points_slope"] = grouped["points_per_game"].apply(lambda s: prior_slope(s, window=None)).reset_index(level=0, drop=True)
    d["career_goals_slope"] = grouped["goals_per_game"].apply(lambda s: prior_slope(s, window=None)).reset_index(level=0, drop=True)
    d["career_gamescore_slope"] = grouped["game_score_per_game"].apply(lambda s: prior_slope(s, window=None)).reset_index(level=0, drop=True)

    return d


# ── Team context ───────────────────────────────────────────────────────────────

def build_team_context(df):
    agg_dict = dict(
        team_median_toi_pg    = ("toi_per_game",                                    "median"),
        team_avg_hd_share     = ("hd_shot_share",                                   "mean"),
        team_avg_adj_xg_per60 = ("ind_flurry_score_venue_adj_expected_goals_per60", "mean"),
        _team_avg_raw_xg      = ("ind_expected_goals_per60",                        "mean"),
        team_avg_primary_rate = ("primary_assist_share",                            "mean"),
        team_avg_on_target    = ("on_target_rate",                                  "mean"),
    )
    # Add team line quality if available
    if "line_adj_xg_per60" in df.columns:
        agg_dict.update(
            team_avg_line_adj_xg_per60 = ("line_adj_xg_per60", "mean"),
            team_avg_line_xg_pct       = ("line_xg_pct",       "mean"),
            team_avg_line_hd_xg_per60  = ("line_hd_xg_per60",  "mean"),
            team_avg_line_corsi_pct    = ("line_corsi_pct",     "mean"),
        )
    team_ctx = (
        df.groupby(["player_team", "season", "position"])
        .agg(**agg_dict)
        .reset_index()
    )
    team_ctx["team_adj_ratio"] = safe_div(
        team_ctx["team_avg_adj_xg_per60"], team_ctx["_team_avg_raw_xg"], fill=1.0
    )
    team_ctx = team_ctx.drop(columns=["_team_avg_raw_xg"])
    # Fill line quality cols if missing
    for col in ["team_avg_line_adj_xg_per60","team_avg_line_xg_pct",
                "team_avg_line_hd_xg_per60","team_avg_line_corsi_pct"]:
        if col not in team_ctx.columns:
            team_ctx[col] = 0.0
    return team_ctx


def get_latest_team_contexts(df, team_ctx):
    latest_season = df["season"].max()
    ctx = team_ctx[team_ctx["season"] == latest_season].copy()
    if ctx["player_team"].nunique() < team_ctx["player_team"].nunique():
        fallback = (
            team_ctx.sort_values("season", ascending=False)
            .groupby(["player_team", "position"]).first().reset_index()
        )
        present = set(zip(ctx["player_team"], ctx["position"]))
        missing = fallback[~fallback.apply(
            lambda r: (r["player_team"], r["position"]) in present, axis=1
        )]
        ctx = pd.concat([ctx, missing], ignore_index=True)
    return ctx


# ── Player profile ─────────────────────────────────────────────────────────────

def build_weighted_player_profile(player_rows, has_age):
    latest_season = player_rows["season"].max()
    latest_rows = player_rows[player_rows["season"] == latest_season].copy()
    profile = latest_rows.sort_values("ice_time", ascending=False).iloc[0].copy()
    seasons = [latest_season]
    return profile, seasons


# ── Feature matrix builders ────────────────────────────────────────────────────

def _pos_dummies(df):
    pos_d = pd.get_dummies(df["position"], prefix="pos")
    for c in POSITION_DUMMIES:
        if c not in pos_d.columns:
            pos_d[c] = 0
    return pos_d[POSITION_DUMMIES]


def build_feature_matrix(df, has_age):
    feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + TEAM_FEATURES
    X = pd.concat(
        [df[feats].reset_index(drop=True), _pos_dummies(df).reset_index(drop=True)], axis=1
    )
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_next_season_dataset(df, has_age):
    """
    For each player-season, pair current features with NEXT season's targets.
    Drops the most recent season per player (no future labels available).
    """
    d = df.sort_values(["player_id", "season"]).copy()
    next_targets = d.groupby("player_id")[TARGETS].shift(-1)
    next_targets.columns = [f"next_{t}" for t in TARGETS]
    d = pd.concat([d, next_targets], axis=1).dropna(subset=[f"next_{t}" for t in TARGETS])
    return d


def build_next_feature_matrix(df, has_age):
    traj_present = [f for f in TRAJECTORY_FEATURES if f in df.columns]
    feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + traj_present + TEAM_FEATURES
    feats = [f for f in feats if f in df.columns]
    X = pd.concat(
        [df[feats].reset_index(drop=True), _pos_dummies(df).reset_index(drop=True)], axis=1
    )
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def _canonical_target_name(target_col):
    return target_col[5:] if target_col.startswith("next_") else target_col


def compute_target_baseline(df_like, target_col):
    """Build leakage-safe baseline for a target using prior-season history features."""
    base_target = _canonical_target_name(target_col)
    candidate_cols = [c for c in BASELINE_FEATURES.get(base_target, []) if c in df_like.columns]
    if not candidate_cols:
        return pd.Series(np.zeros(len(df_like)), index=df_like.index, dtype=float)

    baseline = df_like[candidate_cols].bfill(axis=1).iloc[:, 0]
    baseline = baseline.fillna(0.0)
    return baseline.astype(float)


def make_elite_sample_weights(y):
    """Upweight high-end outcomes so the model spends more capacity on elite players."""
    arr = np.asarray(y, dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=float)

    q75, q90, q95 = np.quantile(arr, [0.75, 0.90, 0.95])
    weights = np.ones(len(arr), dtype=float)
    weights += 0.5 * (arr >= q75)
    weights += 1.0 * (arr >= q90)
    weights += 1.5 * (arr >= q95)

    denom = q95 if q95 > 0 else 1.0
    weights += 0.5 * np.clip(arr / denom, 0.0, 2.0)
    return weights


# ── Training ───────────────────────────────────────────────────────────────────

def make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, min_child_samples=2,
        reg_alpha=0.01, reg_lambda=0.01,
        objective="regression_l2", random_state=42, verbose=-1,
    )


def train_models_with_progress(X, df, targets, target_col_map, label_prefix, status, bar, step, total_steps):
    kf      = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    models  = {}
    metrics = {}

    for target in targets:
        label     = TARGET_LABELS[target]
        target_col = target_col_map[target]
        y         = np.clip(df[target_col].values, 0, None)
        baseline  = compute_target_baseline(df, target_col).values
        y_resid   = y - baseline
        sample_w  = make_elite_sample_weights(y)
        elite_cut = np.quantile(y, ELITE_QUANTILE)

        # ── Cross-validation on full dataset for metrics ──────────────────
        fold_maes, fold_rmses, fold_elite_maes = [], [], []
        for fold, (tr, val) in enumerate(kf.split(X), 1):
            status.markdown(f"🔁 **{label_prefix} — {label}** CV fold {fold}/{CV_FOLDS}")
            gm = make_lgbm()
            gm.fit(X.iloc[tr], y_resid[tr], sample_weight=sample_w[tr])
            fold_preds = baseline[val] + gm.predict(X.iloc[val])
            fold_preds = np.clip(fold_preds, 0, None)
            fold_maes.append(mean_absolute_error(y[val], fold_preds))
            fold_rmses.append(np.sqrt(mean_squared_error(y[val], fold_preds)))
            elite_mask = y[val] >= elite_cut
            if elite_mask.any():
                fold_elite_maes.append(mean_absolute_error(y[val][elite_mask], fold_preds[elite_mask]))
            step += 1
            bar.progress(min(step / total_steps, 1.0),
                         text=f"{label_prefix} {label}: fold {fold}/{CV_FOLDS} — MAE {np.mean(fold_maes):.3f}")

        # ── Train final model on full data ───────────────────────────────
        status.markdown(f"✅ **{label_prefix} — {label}** fitting final residual model...")
        gm = make_lgbm()
        gm.fit(X, y_resid, sample_weight=sample_w)
        models[target]  = {"global": gm}
        metrics[target] = {
            "mae":  (float(np.mean(fold_maes)),  float(np.std(fold_maes))),
            "rmse": (float(np.mean(fold_rmses)), float(np.std(fold_rmses))),
            "elite_mae": (float(np.mean(fold_elite_maes)), float(np.std(fold_elite_maes))) if fold_elite_maes else (np.nan, np.nan),
        }
        step += 1
        bar.progress(min(step / total_steps, 1.0),
                     text=f"{label_prefix} {label} done — MAE {np.mean(fold_maes):.3f}")

    return models, metrics, step


def load_and_train_with_progress(path, ages_path):
    # Steps: 3 setup + (CV_FOLDS+1)*len(TARGETS) for team fit + same for next season
    total_steps = 3 + 2 * len(TARGETS) * (CV_FOLDS + 1)
    step = 0

    status = st.empty()
    bar    = st.progress(0, text="Starting up...")

    def advance(msg):
        nonlocal step
        step += 1
        bar.progress(min(step / total_steps, 1.0), text=msg)

    # ── Load ──────────────────────────────────────────────────────────────────
    status.markdown("⚙️ **Loading data...**")
    df   = pd.read_csv(path)
    raw_targets = ["game_score", "ind_points_per60", "ind_goals_per60", "ice_time", "games_played"]
    df   = df[(df["games_played"] >= MIN_GP) & (df["ice_time"] >= MIN_ICE)].dropna(subset=raw_targets).copy()
    # Train forwards-only models.
    df   = df[df["position"].isin(FORWARD_POSITIONS)].copy()
    ages = pd.read_csv(ages_path)[["player_id", "season", "age", "age_sq"]]
    df   = df.merge(ages, on=["player_id", "season"], how="left")
    # Join powerplay and zone start features
    pp_cols = ["player_id", "season", "pp_icetime_pct", "pp_points_per60",
               "pp_goals_per60", "pp_xg_per60", "pp_points_share",
               "o_zone_start_pct", "zone_start_diff"]
    pp   = pd.read_csv(PP_FILE)[pp_cols]
    df   = df.merge(pp, on=["player_id", "season"], how="left")
    df[pp_cols[2:]] = df[pp_cols[2:]].fillna(0)
    # Join linemate quality features
    lm_cols = ["player_id", "season", "line_adj_xg_per60", "line_xg_per60",
               "line_hd_xg_per60", "line_goals_per60", "line_xg_pct",
               "line_corsi_pct", "n_distinct_lines"]
    lm   = pd.read_csv(LINEMATE_FILE)[lm_cols]
    df   = df.merge(lm, on=["player_id", "season"], how="left")
    df[lm_cols[2:]] = df[lm_cols[2:]].fillna(0)
    has_age = df["age"].notna().mean() > 0.5
    advance(f"Data loaded (forwards only) — {len(df):,} rows  |  age matched: {df['age'].notna().sum():,}")

    # ── Engineer ───────────────────────────────────────────────────────────────
    status.markdown("⚙️ **Engineering features...**")
    df       = engineer_player_features(df)
    df       = engineer_trajectory_features(df)
    df       = engineer_career_history_features(df)
    team_ctx = build_team_context(df)
    df       = df.merge(team_ctx, on=["player_team", "season", "position"], how="left")
    advance("Features engineered")

    # ── Player profiles ────────────────────────────────────────────────────────
    status.markdown("⚙️ **Building latest-season player profiles...**")
    player_profiles = {}
    for pid, group in df.groupby("player_id"):
        profile, seasons = build_weighted_player_profile(group, has_age)
        player_profiles[pid] = (profile, seasons)
    advance(f"Profiles built from latest seasons — {len(player_profiles):,} players")

    # ── Team fit model ─────────────────────────────────────────────────────────
    status.markdown("⚙️ **Training Team Fit models...**")
    X_fit  = build_feature_matrix(df, has_age)
    fit_feature_names = X_fit.columns.tolist()
    fit_models, fit_metrics, step = train_models_with_progress(
        X_fit, df, TARGETS,
        {t: t for t in TARGETS},
        "Team Fit", status, bar, step, total_steps
    )

    # ── Next season model ──────────────────────────────────────────────────────
    status.markdown("⚙️ **Training Next Season models...**")
    df_next  = build_next_season_dataset(df, has_age)
    X_next   = build_next_feature_matrix(df_next, has_age)
    next_feature_names = X_next.columns.tolist()
    next_models, next_metrics, step = train_models_with_progress(
        X_next, df_next, TARGETS,
        {t: f"next_{t}" for t in TARGETS},
        "Next Season", status, bar, step, total_steps
    )

    bar.progress(1.0, text="✅ All models trained and ready!")
    status.empty()
    bar.empty()

    return (df, team_ctx, has_age, player_profiles,
            fit_models, fit_metrics, fit_feature_names,
            next_models, next_metrics, next_feature_names)


# ── Prediction (shared) ────────────────────────────────────────────────────────

def get_latest_league_env(df):
    """Return the most recent season's league-wide scoring averages."""
    latest    = df["season"].max()
    latest_df = df[df["season"] == latest]
    return {
        "league_avg_points_pg": latest_df["points_per_game"].mean(),
        "league_avg_goals_pg":  latest_df["goals_per_game"].mean(),
    }


def _build_team_predictions(profile, position, all_teams, models, has_age, use_next_features=False, df=None):
    # Use most recent season's league scoring environment for all predictions
    league_env = get_latest_league_env(df) if df is not None else {}

    records = []
    for _, team_row in all_teams.iterrows():
        row = profile.copy()
        for col in TEAM_FEATURES:
            row[col] = team_row[col]
        for k, v in league_env.items():
            row[k] = v
        records.append(row)

    pred_df = pd.DataFrame(records)
    pos_d   = pd.get_dummies(pred_df["position"], prefix="pos")
    for c in POSITION_DUMMIES:
        if c not in pos_d.columns:
            pos_d[c] = 0

    if use_next_features and df is not None:
        traj_present = [f for f in TRAJECTORY_FEATURES if f in pred_df.columns]
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + traj_present + TEAM_FEATURES
    else:
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + TEAM_FEATURES

    feats   = [f for f in feats if f in pred_df.columns]
    X_pred  = pd.concat(
        [pred_df[feats].reset_index(drop=True), pos_d[POSITION_DUMMIES].reset_index(drop=True)], axis=1
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    results = all_teams[["player_team"]].reset_index(drop=True).copy()
    for target, model_dict in models.items():
        try:
            baseline = compute_target_baseline(pred_df, target).values
            resid = model_dict["global"].predict(X_pred)
            results[f"pred_{target}"] = np.clip(baseline + resid, 0, None)
        except Exception:
            results[f"pred_{target}"] = np.nan

    return results


def predict_player(player_name, df, team_ctx, fit_models, next_models,
                   player_profiles, has_age, override_team=None):
    mask = df["player_name"].str.lower() == player_name.strip().lower()
    rows = df[mask]
    if rows.empty:
        mask = df["player_name"].str.lower().str.contains(player_name.strip().lower(), na=False)
        rows = df[mask]
        if rows.empty:
            return None

    pid              = rows["player_id"].iloc[0]
    profile, seasons = player_profiles[pid]
    position         = profile["position"]
    season           = int(profile["season"])
    matched          = profile["player_name"]

    latest_season = rows["season"].max()
    latest_rows   = rows[rows["season"] == latest_season]
    traded_teams  = sorted(latest_rows["player_team"].unique().tolist()) if len(latest_rows) > 1 else []
    actual_team   = override_team if override_team else profile["player_team"]

    all_teams = get_latest_team_contexts(df, team_ctx)
    all_teams = all_teams[all_teams["position"] == position].copy()

    # Team fit predictions (current skill)
    fit_results = _build_team_predictions(profile, position, all_teams, fit_models, has_age, df=df)
    fit_results = fit_results.sort_values("pred_points_per_game", ascending=False).reset_index(drop=True)
    fit_results.index += 1
    fit_results["is_actual"] = fit_results["player_team"] == actual_team

    # Next season predictions
    next_results = _build_team_predictions(profile, position, all_teams, next_models, has_age,
                                           use_next_features=True, df=df)
    next_results = next_results.sort_values("pred_points_per_game", ascending=False).reset_index(drop=True)
    next_results.index += 1
    next_results["is_actual"] = next_results["player_team"] == actual_team

    return {
        "matched":      matched,
        "actual_team":  actual_team,
        "season":       season,
        "position":     position,
        "seasons":      seasons,
        "traded_teams": traded_teams,
        "fit_results":  fit_results,
        "next_results": next_results,
        "age":          profile.get("age") if has_age else None,
    }


# ── Charts ─────────────────────────────────────────────────────────────────────

def make_bar_chart(results, player_name, actual_team, title):
    metric_cols   = ["pred_game_score_per_game", "pred_points_per_game", "pred_goals_per_game"]
    metric_labels = ["Game Score / Game", "Points / Game", "Goals / Game"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.patch.set_facecolor("#0e1117")
    fig.suptitle(title, color="white", fontsize=14, y=1.01)

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        ax.set_facecolor("#0e1117")
        sr = results.sort_values(col)
        bars = ax.barh(sr["player_team"], sr[col], color="#4a90d9")
        for bar, team in zip(bars, sr["player_team"]):
            if team == actual_team:
                bar.set_edgecolor("#c8102e")
                bar.set_linewidth(2.5)
        actual_val = results.loc[results["player_team"] == actual_team, col].values[0]
        ax.axvline(actual_val, color="#c8102e", linestyle="--", linewidth=1.2)
        ax.set_xlabel(label, color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    actual_patch = mpatches.Patch(edgecolor="#c8102e", facecolor="none",
                                  linewidth=2.5, label=f"Actual team ({actual_team})")
    axes[-1].legend(handles=[actual_patch],
                    facecolor="#1a1a2e", labelcolor="white", loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def make_importance_chart(models, feature_names, top_n=15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0e1117")
    for ax, target in zip(axes, TARGETS):
        ax.set_facecolor("#0e1117")
        m   = models[target].get("global") or next(iter(models[target].values()))
        imp = m.feature_importances_
        idx = np.argsort(imp)[-top_n:]
        ax.barh([feature_names[i] for i in idx], imp[idx], color="#4a90d9")
        ax.set_title(TARGET_LABELS[target], color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


def show_results_table(results, actual_team):
    display = results[[
        "player_team", "pred_game_score_per_game",
        "pred_points_per_game", "pred_goals_per_game", "is_actual"
    ]].copy()
    display.columns = ["Team", "GS/GP", "Points/GP", "Goals/GP", "Actual Team"]
    for col in ["GS/GP", "Points/GP", "Goals/GP"]:
        display[col] = display[col].round(3)
    st.dataframe(
        display.style.apply(
            lambda row: ["background-color: #3a1a1a" if row["Actual Team"] else "" for _ in row],
            axis=1,
        ),
        use_container_width=True, height=400,
    )
    return display


def show_metrics(metrics, label):
    st.markdown(f"**{label} model quality ({CV_FOLDS}-fold CV)**")
    st.caption("MAE = avg absolute error in same units as stat. RMSE penalises large errors more. Lower is better.")
    for target in TARGETS:
        mae_mean,  mae_std  = metrics[target]["mae"]
        rmse_mean, rmse_std = metrics[target]["rmse"]
        st.markdown(f"*{TARGET_LABELS[target]}*")
        c1, c2, _ = st.columns(3)
        c1.metric("MAE",  f"{mae_mean:.3f}", f"± {mae_std:.3f}")
        c2.metric("RMSE", f"{rmse_mean:.3f}", f"± {rmse_std:.3f}")
        elite_mae_mean, elite_mae_std = metrics[target].get("elite_mae", (np.nan, np.nan))
        if not pd.isna(elite_mae_mean):
            st.caption(f"Elite MAE (top 10% actual): {elite_mae_mean:.3f} ± {elite_mae_std:.3f}")


def elite_segment_stats(val_df, actual_col, pred_col, quantile=0.90):
    if val_df.empty:
        return np.nan, np.nan, 0
    cutoff = val_df[actual_col].quantile(quantile)
    seg = val_df[val_df[actual_col] >= cutoff]
    if seg.empty:
        return np.nan, np.nan, 0
    mae = mean_absolute_error(seg[actual_col], seg[pred_col])
    # Positive bias means model overpredicts; negative means underpredicts.
    bias = (seg[pred_col] - seg[actual_col]).mean()
    return float(mae), float(bias), int(len(seg))


def calibration_slope(val_df, actual_col, pred_col):
    x = val_df[pred_col].values
    y = val_df[actual_col].values
    if len(x) < 2 or np.std(x) == 0:
        return np.nan
    return np.polyfit(x, y, 1)[0]



# ── 2025-26 Validation ─────────────────────────────────────────────────────────

CURRENT_SEASON = "20252026"

@st.cache_data(show_spinner=False)
def fetch_nhl_current_season():
    """
    Pull 2025-26 skater stats from the NHL API.
    Returns a DataFrame with player_id, goals_per_game, points_per_game, games_played.
    """
    url = (
        f"https://api.nhle.com/stats/rest/en/skater/summary"
        f"?limit=-1&start=0&cayenneExp=seasonId={CURRENT_SEASON}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        df   = pd.json_normalize(data)
        if df.empty:
            return None, "No data returned from NHL API."

        df = df.rename(columns={
            "playerId":      "player_id",
            "skaterFullName":"player_name",
            "goals":         "goals",
            "points":        "points",
            "gamesPlayed":   "games_played",
        })

        # Keep only needed columns (handle missing gracefully)
        keep = ["player_id", "player_name", "goals", "points", "games_played"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df = df.dropna(subset=["player_id", "goals", "points", "games_played"])
        df = df[df["games_played"] >= 10]

        # Convert to per-game
        df["goals_per_game"]  = df["goals"]  / df["games_played"]
        df["points_per_game"] = df["points"] / df["games_played"]

        df["player_id"] = df["player_id"].astype(int)
        return df, None

    except Exception as e:
        return None, str(e)


def build_validation_results(actual_df, df, team_ctx, fit_models,
                              player_profiles, has_age):
    """
    For each player in actual_df, look up the model's prediction for their
    actual current team and compare to real 2025-26 stats.
    """
    rows = []
    for _, actual in actual_df.iterrows():
        pid = int(actual["player_id"])
        if pid not in player_profiles:
            continue

        profile, seasons = player_profiles[pid]
        position         = profile["position"]
        actual_team      = profile["player_team"]

        # Get team context for their actual team
        all_teams = get_latest_team_contexts(df, team_ctx)
        team_row  = all_teams[
            (all_teams["position"] == position) &
            (all_teams["player_team"] == actual_team)
        ]
        if team_row.empty:
            continue

        # Build single prediction row
        row = profile.copy()
        for col in TEAM_FEATURES:
            row[col] = team_row.iloc[0][col]

        pred_df = pd.DataFrame([row])
        pos_d   = pd.get_dummies(pred_df["position"], prefix="pos")
        for c in POSITION_DUMMIES:
            if c not in pos_d.columns:
                pos_d[c] = 0

        feats  = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + TEAM_FEATURES
        feats  = [f for f in feats if f in pred_df.columns]
        X_pred = pd.concat(
            [pred_df[feats].reset_index(drop=True),
             pos_d[POSITION_DUMMIES].reset_index(drop=True)], axis=1
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        base_pts   = compute_target_baseline(pred_df, "points_per_game").values[0]
        base_goals = compute_target_baseline(pred_df, "goals_per_game").values[0]
        base_gs    = compute_target_baseline(pred_df, "game_score_per_game").values[0]

        pred_pts   = np.clip(base_pts + fit_models["points_per_game"]["global"].predict(X_pred)[0], 0, None)
        pred_goals = np.clip(base_goals + fit_models["goals_per_game"]["global"].predict(X_pred)[0], 0, None)
        pred_gs    = np.clip(base_gs + fit_models["game_score_per_game"]["global"].predict(X_pred)[0], 0, None)

        rows.append({
            "player_name":       actual["player_name"],
            "team":              actual_team,
            "games_played":      actual["games_played"],
            "actual_points_gp": round(actual["points_per_game"], 3),
            "pred_points_gp":   round(pred_pts, 3),
            "points_gp_error":  round(actual["points_per_game"] - pred_pts, 3),
            "actual_goals_gp":  round(actual["goals_per_game"], 3),
            "pred_goals_gp":    round(pred_goals, 3),
            "goals_gp_error":   round(actual["goals_per_game"] - pred_goals, 3),
            "pred_gs_per_game":  round(pred_gs, 3),
            "seasons_used":      " → ".join(str(s) for s in seasons),
        })

    return pd.DataFrame(rows)


def make_scatter(val_df, actual_col, pred_col, label, ax):
    ax.set_facecolor("#0e1117")
    ax.scatter(val_df[pred_col], val_df[actual_col],
               alpha=0.5, color="#4a90d9", s=20)
    mn = min(val_df[pred_col].min(), val_df[actual_col].min()) - 0.1
    mx = max(val_df[pred_col].max(), val_df[actual_col].max()) + 0.1
    ax.plot([mn, mx], [mn, mx], color="#c8102e", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Predicted {label}", color="white", fontsize=10)
    ax.set_ylabel(f"Actual {label}", color="white", fontsize=10)
    ax.set_title(label, color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    mae  = mean_absolute_error(val_df[actual_col], val_df[pred_col])
    corr = val_df[[actual_col, pred_col]].corr().iloc[0, 1]
    ax.text(0.05, 0.92, f"MAE {mae:.3f}  |  r {corr:.2f}",
            transform=ax.transAxes, color="white", fontsize=9)


def normalize_roster_position(raw_pos):
    p = str(raw_pos).upper()
    if p in {"C", "L", "R", "LW", "RW"}:
        return "L" if p == "LW" else "R" if p == "RW" else p
    return None


def parse_roster_entries(entries, team_code):
    rows = []
    for p in entries:
        pid = p.get("id") or p.get("playerId")
        if pid is None:
            continue
        pos = normalize_roster_position(p.get("positionCode") or p.get("position"))
        if pos is None:
            continue
        first = p.get("firstName", {}).get("default") if isinstance(p.get("firstName"), dict) else p.get("firstName")
        last = p.get("lastName", {}).get("default") if isinstance(p.get("lastName"), dict) else p.get("lastName")
        full_name = p.get("fullName") or " ".join([str(first or "").strip(), str(last or "").strip()]).strip()
        if not full_name:
            full_name = str(p.get("name", "Unknown Player"))
        rows.append({
            "player_id": int(pid),
            "player_name": full_name,
            "position": pos,
            "nhl_team": team_code,
        })
    return rows


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_active_team_roster(team_code, season=CURRENT_SEASON):
    url = f"https://api-web.nhle.com/v1/roster/{team_code}/{season}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        forwards = parse_roster_entries(data.get("forwards", []), team_code)
        defense = parse_roster_entries(data.get("defensemen", []), team_code)
        roster_df = pd.DataFrame(forwards + defense)
        if roster_df.empty:
            return None, f"No skater roster data found for {team_code}."
        roster_df = roster_df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
        return roster_df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_active_rosters(season=CURRENT_SEASON):
    rosters = {}
    errors = {}
    for team in NHL_TEAMS:
        roster_df, err = fetch_active_team_roster(team, season)
        if err:
            errors[team] = err
        elif roster_df is not None:
            rosters[team] = roster_df
    return rosters, errors


def assign_deployment_bucket(rank, pred_points_gp):
    if rank <= 2:
        bucket = "elite"
    elif rank <= 6:
        bucket = "top6"
    elif rank <= 12:
        bucket = "bottom6"
    else:
        bucket = "depth"

    if bucket == "elite" and pred_points_gp < ROLE_MIN_PPG["elite"]:
        bucket = "top6"
    if bucket == "top6" and pred_points_gp < ROLE_MIN_PPG["top6"]:
        bucket = "bottom6"
    if bucket == "bottom6" and pred_points_gp < ROLE_MIN_PPG["bottom6"]:
        bucket = "depth"
    return bucket


def build_roster_deployment(team_code, roster_df, df, team_ctx, fit_models, player_profiles, has_age):
    latest_ctx = get_latest_team_contexts(df, team_ctx)
    rows = []
    skipped = 0

    for _, rp in roster_df.iterrows():
        pid = int(rp["player_id"])
        if pid not in player_profiles:
            skipped += 1
            continue
        profile, seasons = player_profiles[pid]
        position = profile.get("position")
        if position not in ["C", "L", "R", "D"]:
            skipped += 1
            continue

        all_teams = latest_ctx[latest_ctx["position"] == position].copy()
        if all_teams.empty:
            skipped += 1
            continue

        team_fit = _build_team_predictions(profile, position, all_teams, fit_models, has_age, df=df)
        team_fit = team_fit.sort_values("pred_points_per_game", ascending=False).reset_index(drop=True)
        team_fit.index += 1

        current_row = team_fit[team_fit["player_team"] == team_code]
        if current_row.empty:
            skipped += 1
            continue

        best_row = team_fit.iloc[0]
        cur = current_row.iloc[0]
        rows.append({
            "player_id": pid,
            "player_name": rp["player_name"],
            "position": position,
            "nhl_team": team_code,
            "pred_game_score_gp": float(cur["pred_game_score_per_game"]),
            "pred_points_gp": float(cur["pred_points_per_game"]),
            "pred_goals_gp": float(cur["pred_goals_per_game"]),
            "best_fit_team": best_row["player_team"],
            "best_fit_points_gp": float(best_row["pred_points_per_game"]),
            "seasons_used": " -> ".join(str(s) for s in seasons),
        })

    if not rows:
        return pd.DataFrame(), skipped

    out = pd.DataFrame(rows).sort_values("pred_points_gp", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["deployment_bucket"] = out.apply(
        lambda r: assign_deployment_bucket(int(r["rank"]), float(r["pred_points_gp"])), axis=1
    )
    out["deployment_role"] = out["deployment_bucket"].map(ROLE_LABELS)
    return out, skipped

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="NHL Player Predictor", page_icon="🏒", layout="wide")
st.title("NHL Acquisition Player Predictor")
st.caption("Forwards-only mode: defensemen are excluded from training and predictions.")

if "fit_models" not in st.session_state:
    if os.path.exists(CACHE_FILE):
        with st.spinner("Loading saved models from disk..."):
            cached = joblib.load(CACHE_FILE)
            (
                st.session_state["df"],
                st.session_state["team_ctx"],
                st.session_state["has_age"],
                st.session_state["player_profiles"],
                st.session_state["fit_models"],
                st.session_state["fit_metrics"],
                st.session_state["fit_feature_names"],
                st.session_state["next_models"],
                st.session_state["next_metrics"],
                st.session_state["next_feature_names"],
            ) = cached
    else:
        st.info("Training models for the first time — this takes 5–8 minutes. Won't happen again until you retrain.")
        results = load_and_train_with_progress(DATA_FILE, AGES_FILE)
        joblib.dump(results, CACHE_FILE)
        (
            st.session_state["df"],
            st.session_state["team_ctx"],
            st.session_state["has_age"],
            st.session_state["player_profiles"],
            st.session_state["fit_models"],
            st.session_state["fit_metrics"],
            st.session_state["fit_feature_names"],
            st.session_state["next_models"],
            st.session_state["next_metrics"],
            st.session_state["next_feature_names"],
        ) = results
        st.rerun()

df                 = st.session_state["df"]
team_ctx           = st.session_state["team_ctx"]
has_age            = st.session_state["has_age"]
player_profiles    = st.session_state["player_profiles"]
fit_models         = st.session_state["fit_models"]
fit_metrics        = st.session_state["fit_metrics"]
fit_feature_names  = st.session_state["fit_feature_names"]
next_models        = st.session_state["next_models"]
next_metrics       = st.session_state["next_metrics"]
next_feature_names = st.session_state["next_feature_names"]

if has_age:
    st.caption("Age data loaded ✅ — next-season forecasting active.")
else:
    st.caption("⚠️ Age data not found — next-season model running without age features.")

# ── Player search (shared across tabs) ────────────────────────────────────────
st.divider()
all_players  = sorted(df["player_name"].unique())
player_input = st.selectbox("Search for a player", options=[""] + all_players, index=0)

pred = None
if player_input:
    first = predict_player(player_input, df, team_ctx, fit_models, next_models,
                           player_profiles, has_age)
    if first is None:
        st.error(f"No player found matching '{player_input}'.")
    else:
        override_team = None
        if first["traded_teams"]:
            st.warning(
                f"⚠️ {first['matched']} was traded in {first['seasons'][0]}. "
                f"Select their current team:"
            )
            override_team = st.radio(
                "Current team", options=first["traded_teams"],
                horizontal=True, key="team_override"
            )
        pred = predict_player(player_input, df, team_ctx, fit_models, next_models,
                              player_profiles, has_age, override_team=override_team)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Team Fit",
    "🔮 Next Season",
    "🔬 Model Info",
    "✅ 2025-26 Validation",
    "🏒 Active Roster Roles",
])

with tab1:
    if pred:
        seasons_str = " → ".join(str(s) for s in pred["seasons"])
        age_str     = f"  |  Age {pred['age']:.0f}" if pred["age"] else ""
        st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}  |  Seasons: {seasons_str}")
        st.caption("Predicted performance based on current weighted skill profile across all 32 teams.")

        title = f"{pred['matched']}  |  Current skill profile  |  Seasons: {seasons_str}"
        st.pyplot(make_bar_chart(pred["fit_results"], pred["matched"], pred["actual_team"], title))

        st.markdown("#### Rankings Table")
        display = show_results_table(pred["fit_results"], pred["actual_team"])
        csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
        st.download_button("⬇ Download CSV", data=csv,
                           file_name=f"{pred['matched'].replace(' ','_')}_team_fit.csv",
                           mime="text/csv")
    else:
        st.info("Search for a player above to see predictions.")

with tab2:
    if pred:
        age_str = f"  |  Age {pred['age']:.0f} → {pred['age']+1:.0f}" if pred["age"] else ""
        st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}")
        st.caption(
            "Predicted next-season performance across all 32 teams, "
            "based on current skill profile + trajectory + age curve."
        )

        title = f"{pred['matched']}  |  Next season forecast"
        st.pyplot(make_bar_chart(pred["next_results"], pred["matched"], pred["actual_team"], title))

        st.markdown("#### Rankings Table")
        display = show_results_table(pred["next_results"], pred["actual_team"])
        csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
        st.download_button("⬇ Download CSV", data=csv,
                           file_name=f"{pred['matched'].replace(' ','_')}_next_season.csv",
                           mime="text/csv")
    else:
        st.info("Search for a player above to see predictions.")

with tab3:
    st.markdown("#### Model Cache")
    st.caption("Models are saved to disk and loaded instantly on reopen. Delete the cache to retrain from scratch.")
    if st.button("🔄 Retrain model (deletes cache)"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        for key in ["df","team_ctx","has_age","player_profiles","fit_models",
                    "fit_metrics","fit_feature_names","next_models","next_metrics","next_feature_names"]:
            st.session_state.pop(key, None)
        st.success("Cache cleared — refresh the page to retrain.")

    st.divider()
    with st.expander("Team Fit model quality", expanded=True):
        show_metrics(fit_metrics, "Team Fit")
    with st.expander("Next Season model quality", expanded=True):
        show_metrics(next_metrics, "Next Season")
    with st.expander("Team Fit — feature importance"):
        st.pyplot(make_importance_chart(fit_models, fit_feature_names))
    with st.expander("Next Season — feature importance"):
        st.pyplot(make_importance_chart(next_models, next_feature_names))

with tab4:
    st.subheader("2025-26 Season Validation")
    st.caption(
        "Pulls live 2025-26 stats from the NHL API and compares against "
        "the model's predictions based on each player's historical profile. "
        "Goals and Points are converted to per-game rates using games played."
    )

    if st.button("🔄 Fetch 2025-26 stats from NHL API"):
        st.cache_data.clear()

    actual_df, err = fetch_nhl_current_season()

    if err:
        st.error(f"Could not fetch NHL API data: {err}")
    elif actual_df is not None:
        st.success(f"Fetched {len(actual_df):,} skaters with 10+ games played.")

        with st.spinner("Comparing predictions to actual stats..."):
            val_df = build_validation_results(
                actual_df, df, team_ctx, fit_models, player_profiles, has_age
            )

        if val_df.empty:
            st.warning("No players matched between NHL API and model profiles.")
        else:
            st.markdown(f"**{len(val_df):,} players matched**")

            # Scatter plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.patch.set_facecolor("#0e1117")
            make_scatter(val_df, "actual_points_gp", "pred_points_gp", "Points / Game", axes[0])
            make_scatter(val_df, "actual_goals_gp",  "pred_goals_gp",  "Goals / Game",  axes[1])
            plt.tight_layout()
            st.pyplot(fig)

            # Overall MAE
            c1, c2 = st.columns(2)
            c1.metric("Points/GP MAE",
                      f"{mean_absolute_error(val_df['actual_points_gp'], val_df['pred_points_gp']):.3f}")
            c2.metric("Goals/GP MAE",
                      f"{mean_absolute_error(val_df['actual_goals_gp'], val_df['pred_goals_gp']):.3f}")

            pred_range = val_df["pred_points_gp"].max() - val_df["pred_points_gp"].min()
            actual_range = val_df["actual_points_gp"].max() - val_df["actual_points_gp"].min()
            compression_ratio = pred_range / actual_range if actual_range > 0 else np.nan
            slope = calibration_slope(val_df, "actual_points_gp", "pred_points_gp")
            c3, c4 = st.columns(2)
            c3.metric("Prediction Spread Ratio", f"{compression_ratio:.2%}" if not pd.isna(compression_ratio) else "n/a")
            c4.metric("Calibration Slope", f"{slope:.2f}" if not pd.isna(slope) else "n/a")

            elite_pts_mae, elite_pts_bias, elite_pts_n = elite_segment_stats(
                val_df, "actual_points_gp", "pred_points_gp", quantile=ELITE_QUANTILE
            )
            elite_goals_mae, elite_goals_bias, elite_goals_n = elite_segment_stats(
                val_df, "actual_goals_gp", "pred_goals_gp", quantile=ELITE_QUANTILE
            )
            e1, e2 = st.columns(2)
            e1.metric(
                f"Elite Points/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                f"{elite_pts_mae:.3f}" if not pd.isna(elite_pts_mae) else "n/a",
                f"bias {elite_pts_bias:+.3f}" if not pd.isna(elite_pts_bias) else None,
            )
            e2.metric(
                f"Elite Goals/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                f"{elite_goals_mae:.3f}" if not pd.isna(elite_goals_mae) else "n/a",
                f"bias {elite_goals_bias:+.3f}" if not pd.isna(elite_goals_bias) else None,
            )
            st.caption(f"Elite segment sample sizes: Points {elite_pts_n}, Goals {elite_goals_n}")

            st.divider()

            # Biggest misses
            st.markdown("#### Biggest Misses (model most wrong on Points/GP)")
            misses = val_df.reindex(
                val_df["points_gp_error"].abs().nlargest(15).index
            )[["player_name","team","games_played","actual_points_gp",
               "pred_points_gp","points_gp_error","actual_goals_gp",
               "pred_goals_gp","seasons_used"]]
            st.dataframe(misses, use_container_width=True)

            # Best predictions
            st.markdown("#### Best Predictions (model closest on Points/GP)")
            best = val_df.reindex(
                val_df["points_gp_error"].abs().nsmallest(15).index
            )[["player_name","team","games_played","actual_points_gp",
               "pred_points_gp","points_gp_error","actual_goals_gp",
               "pred_goals_gp","seasons_used"]]
            st.dataframe(best, use_container_width=True)

            # Full table download
            csv = val_df.to_csv(index=False)
            st.download_button("⬇ Download full validation CSV", data=csv,
                               file_name="validation_2025_26.csv", mime="text/csv")

with tab5:
    st.subheader("Active NHL Roster Deployment")
    st.caption(
        "Ranks active skaters on each NHL roster by predicted Points/Game on their current team context. "
        "Buckets use rank cutoffs (1-2 elite, 3-6 top-6, 7-12 bottom-6, 13+ depth) plus minimum Points/Game floors."
    )

    c1, c2 = st.columns([1, 1])
    selected_team = c1.selectbox("Team", options=NHL_TEAMS, index=NHL_TEAMS.index("TOR") if "TOR" in NHL_TEAMS else 0)
    view_mode = c2.radio("View", options=["Current Team Roles", "All-Team Fit Context"], horizontal=True)

    c3, c4 = st.columns([1, 1])
    if c3.button("🔄 Refresh selected team roster"):
        fetch_active_team_roster.clear()
    if c4.button("🔄 Refresh all 32 rosters"):
        fetch_all_active_rosters.clear()

    roster_df, roster_err = fetch_active_team_roster(selected_team, CURRENT_SEASON)
    if roster_err:
        st.error(f"Could not fetch {selected_team} roster: {roster_err}")
    elif roster_df is not None:
        with st.spinner("Predicting roster roles..."):
            deployment_df, skipped = build_roster_deployment(
                selected_team, roster_df, df, team_ctx, fit_models, player_profiles, has_age
            )

        if deployment_df.empty:
            st.warning("No active skaters from this roster matched the model profile set.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Skaters on API roster", len(roster_df))
            m2.metric("Matched to model", len(deployment_df))
            m3.metric("Skipped", skipped)

            if view_mode == "Current Team Roles":
                show_cols = [
                    "rank", "player_name", "position", "deployment_role",
                    "pred_points_gp", "pred_goals_gp", "pred_game_score_gp"
                ]
            else:
                show_cols = [
                    "rank", "player_name", "position", "deployment_role", "pred_points_gp",
                    "best_fit_team", "best_fit_points_gp", "seasons_used"
                ]

            show_df = deployment_df[show_cols].copy()
            show_df.columns = [
                "Rank", "Player", "Pos", "Current Role", "Points/GP",
                "Goals/GP", "GS/GP"
            ] if view_mode == "Current Team Roles" else [
                "Rank", "Player", "Pos", "Current Role", "Current Team Points/GP",
                "Best Fit Team", "Best Fit Points/GP", "Seasons Used"
            ]

            st.dataframe(show_df, use_container_width=True, height=500)

            bucket_counts = deployment_df["deployment_role"].value_counts()
            st.markdown("#### Role Counts")
            rc = st.columns(4)
            rc[0].metric("Top 2 Elite", int(bucket_counts.get("Top 2 Elite", 0)))
            rc[1].metric("Top 6", int(bucket_counts.get("Top 6", 0)))
            rc[2].metric("Bottom 6", int(bucket_counts.get("Bottom 6", 0)))
            rc[3].metric("Depth", int(bucket_counts.get("Depth", 0)))

            csv = deployment_df.to_csv(index=False)
            st.download_button(
                "⬇ Download roster deployment CSV",
                data=csv,
                file_name=f"{selected_team}_active_roster_roles.csv",
                mime="text/csv",
            )