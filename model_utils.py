"""
model_utils.py
==============
Shared constants, feature engineering, training, and prediction functions
for both the offensive (forwards) and defensive (defensemen) models.

Imported by app.py — do not run directly.
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

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_FILE      = "season_dataset.csv"
AGES_FILE      = "player_ages.csv"
PP_FILE        = "pp_features.csv"
LINEMATE_FILE  = "linemate_features.csv"
CACHE_FILE     = "trained_models_forwards_v5.joblib"
DEF_FILE       = "defensive_dataset.csv"

# Danger zone weights (empirically derived from historical conversion rates)
HD_WEIGHT = 11.2
MD_WEIGHT = 4.1
LD_WEIGHT = 1.0
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

# Position display mapping: MoneyPuck/model codes → NHL display codes
MODEL_TO_NHL_POS = {"C": "C", "L": "LW", "R": "RW", "D": "D"}
NHL_TO_MODEL_POS = {"C": "C", "LW": "L", "RW": "R", "L": "L", "R": "R", "D": "D"}

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
    d["toi_per_game"]         = (d["ice_time"] / 60) / d["games_played"]
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


def _make_X_from_profile(profile, has_age, use_traj=False):
    """Build a single-row feature matrix from a player profile dict/Series."""
    pred_df = pd.DataFrame([profile])
    pos_d   = _pos_dummies(pred_df)
    if use_traj:
        traj  = [f for f in TRAJECTORY_FEATURES if f in pred_df.columns]
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + traj + TEAM_FEATURES
    else:
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + TEAM_FEATURES
    feats = [f for f in feats if f in pred_df.columns]
    X = pd.concat(
        [pred_df[feats].reset_index(drop=True),
         pos_d[POSITION_DUMMIES].reset_index(drop=True)], axis=1
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

def _load_ages(ages_path):
    """
    Load player_ages.csv and compute any missing ages from birthDate.
    Returns a DataFrame with player_id, season, age, age_sq.
    """
    from datetime import datetime as _dt
    df_ages = pd.read_csv(ages_path)

    # Compute missing ages from birthDate if column exists
    if "birthDate" in df_ages.columns and df_ages["age"].isna().any():
        def _calc(row):
            if pd.notna(row["age"]):
                return row["age"]
            if pd.isna(row.get("birthDate")):
                return None
            try:
                birth = _dt.strptime(str(row["birthDate"]), "%Y-%m-%d")
                ref   = _dt(int(row["season"]) - 1, 10, 1)
                return round((ref - birth).days / 365.25, 1)
            except Exception:
                return None
        df_ages["age"]    = df_ages.apply(_calc, axis=1)
        df_ages["age_sq"] = df_ages["age"] ** 2

    keep = ["player_id", "season", "age", "age_sq"]
    return df_ages[[c for c in keep if c in df_ages.columns]]


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
    raw_targets = ["game_score_per_game", "points_per_game", "goals_per_game", "ice_time", "games_played"]
    df   = df[(df["games_played"] >= MIN_GP) & (df["ice_time"] >= MIN_ICE)].dropna(subset=raw_targets).copy()
    # Train forwards-only models.
    df   = df[df["position"].isin(FORWARD_POSITIONS)].copy()
    ages = _load_ages(ages_path)
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
        "pid":          pid,
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

        base_pts    = compute_target_baseline(pred_df, "points_per_game").values[0]
        base_goals  = compute_target_baseline(pred_df, "goals_per_game").values[0]
        base_gs     = compute_target_baseline(pred_df, "game_score_per_game").values[0]

        pred_pts   = np.clip(base_pts   + fit_models["points_per_game"]["global"].predict(X_pred)[0], 0, None)
        pred_goals = np.clip(base_goals + fit_models["goals_per_game"]["global"].predict(X_pred)[0], 0, None)
        pred_gs    = np.clip(base_gs    + fit_models["game_score_per_game"]["global"].predict(X_pred)[0], 0, None)

        rows.append({
            "player_name":       actual["player_name"],
            "team":              actual_team,
            "games_played":      actual["games_played"],
            "actual_points_gp": round(actual["points_per_game"], 3),
            "pred_points_gp":   round(pred_pts, 3),
            "points_gp_error":  round(actual["points_per_game"] - pred_pts, 3),
            "actual_goals_gp":  round(float(actual.get("goals_per_game", 0)), 3),
            "pred_goals_gp":    round(pred_goals, 3),
            "goals_gp_error":   round(float(actual.get("goals_per_game", 0)) - pred_goals, 3),
            "pred_gs_per_game":  round(pred_gs, 3),
            "seasons_used":      " → ".join(str(s) for s in seasons),
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nhl_defensive_stats():
    """
    Pull 2025-26 defenseman stats from the NHL API.
    Merges realtime endpoint (hits, takeaways, penalties) with
    timeonice endpoint (PK time) to validate all 5 defensive targets.
    """
    try:
        # ── Realtime stats (hits, takeaways, giveaways, penalties) ────────────
        rt_url = (
            "https://api.nhle.com/stats/rest/en/skater/realtime"
            "?limit=-1&start=0"
            "&cayenneExp=seasonId=20252026"
        )
        rt_resp = requests.get(rt_url, timeout=15)
        rt_resp.raise_for_status()
        rt_data = rt_resp.json().get("data", [])
        rt_df   = pd.json_normalize(rt_data)
        if rt_df.empty:
            return None, "No realtime data returned."

        rt_df = rt_df.rename(columns={
            "playerId":         "player_id",
            "skaterFullName":   "player_name",
            "positionCode":     "position",
            "hits":             "hits",
            "blockedShots":     "blocked_shots",
            "takeaways":        "takeaways",
            "giveaways":        "giveaways",
            "penaltyMinutes":   "penalty_minutes",
            "gamesPlayed":      "games_played",
        })
        rt_df = rt_df[[c for c in ["player_id","player_name","position","hits",
                                    "blocked_shots","takeaways","giveaways",
                                    "penalty_minutes","games_played"]
                       if c in rt_df.columns]].copy()
        # Filter to defensemen only
        if "position" in rt_df.columns:
            rt_df = rt_df[rt_df["position"] == "D"]
        rt_df = rt_df.dropna(subset=["player_id","games_played"])
        rt_df = rt_df[rt_df["games_played"] >= 10]
        rt_df["player_id"] = rt_df["player_id"].astype(int)

        # ── Time on ice stats (PK time) ────────────────────────────────────────
        toi_url = (
            "https://api.nhle.com/stats/rest/en/skater/timeonice"
            "?limit=-1&start=0"
            "&cayenneExp=seasonId=20252026"
        )
        toi_resp = requests.get(toi_url, timeout=15)
        toi_resp.raise_for_status()
        toi_data = toi_resp.json().get("data", [])
        toi_df   = pd.json_normalize(toi_data)
        if not toi_df.empty:
            toi_df = toi_df.rename(columns={
                "playerId":         "player_id",
                "positionCode":     "position",
                "shTimeOnIce":      "pk_time_on_ice",   # shorthanded = penalty kill
                "timeOnIce":        "total_time_on_ice",
            })
            toi_cols = [c for c in ["player_id","position","pk_time_on_ice","total_time_on_ice"]
                        if c in toi_df.columns]
            toi_df = toi_df[toi_cols].copy()
            if "position" in toi_df.columns:
                toi_df = toi_df[toi_df["position"] == "D"]
            toi_df["player_id"] = toi_df["player_id"].astype(int)
            df = rt_df.merge(toi_df[["player_id","pk_time_on_ice","total_time_on_ice"]],
                             on="player_id", how="left")
        else:
            df = rt_df.copy()
            df["pk_time_on_ice"]    = np.nan
            df["total_time_on_ice"] = np.nan

        # ── Compute per-game rates ─────────────────────────────────────────────
        gp = df["games_played"]
        df["hits_pg"]      = df["hits"]          / gp
        df["blocks_pg"]    = df["blocked_shots"] / gp
        df["takeaways_pg"] = df["takeaways"]     / gp
        df["giveaways_pg"] = df["giveaways"]     / gp

        # Estimate penalty count from penalty minutes (avg penalty = 2 min)
        if "penalty_minutes" in df.columns:
            df["pim_pg"]        = df["penalty_minutes"] / gp
        else:
            df["pim_pg"] = np.nan

        return df.fillna(0), None

    except Exception as e:
        return None, str(e)


def build_defensive_validation(actual_df, def_df, def_team_ctx,
                                def_fit_models, def_player_profiles, def_has_age,
                                feature_names=None):
    """Compare defensive model predictions against 2025-26 actual NHL stats."""
    latest_ctx = def_get_latest_team_contexts(def_df, def_team_ctx)
    rows = []
    for _, actual in actual_df.iterrows():
        pid = int(actual["player_id"])
        if pid not in def_player_profiles:
            continue
        profile, seasons = def_player_profiles[pid]
        team     = profile["player_team"]
        team_row = latest_ctx[latest_ctx["player_team"] == team]
        if team_row.empty:
            continue
        preds = def_predict_for_team(
            profile, team_row.iloc[0], def_fit_models, def_has_age,
            feature_names=feature_names
        )
        rows.append({
            "player_name":    actual["player_name"],
            "team":           team,
            "games_played":   actual["games_played"],
            "actual_hits_pg": round(float(actual.get("hits_pg",      0)), 3),
            "pred_hits_pg":   round(preds.get("ind_hits_pg",          0), 3),
            "hits_error":     round(float(actual.get("hits_pg",       0)) - preds.get("ind_hits_pg", 0), 3),
            "actual_tk_pg":   round(float(actual.get("takeaways_pg",  0)), 3),
            "pred_tk_pg":     round(preds.get("ind_takeaways_pg",     0), 3),
            "tk_error":       round(float(actual.get("takeaways_pg",  0)) - preds.get("ind_takeaways_pg", 0), 3),
            "actual_pk_pct":  round(float(actual.get("pk_ice_pct",   0)), 4),
            "pred_pk_pct":    round(preds.get("pk_ice_pct",           0), 4),
            "pk_error":       round(float(actual.get("pk_ice_pct",   0)) - preds.get("pk_ice_pct", 0), 4),
            "actual_pim_pg":  round(float(actual.get("pim_pg",       0)), 3),
            "pred_pim_pg":    round(preds.get("pim_pg",               0), 3),
            "pim_error":      round(float(actual.get("pim_pg",       0)) - preds.get("pim_pg", 0), 3),
            "seasons_used":   " → ".join(str(s) for s in seasons),
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
    """Return the NHL display position (LW/RW/C/D). Use NHL_TO_MODEL_POS to get model code."""
    p = str(raw_pos).upper().strip()
    if p in {"LW", "L"}:
        return "LW"
    if p in {"RW", "R"}:
        return "RW"
    if p == "C":
        return "C"
    if p == "D":
        return "D"
    return None


def parse_roster_entries(entries, team_code):
    rows = []
    for p in entries:
        pid = p.get("id") or p.get("playerId")
        if pid is None:
            continue
        nhl_pos = normalize_roster_position(p.get("positionCode") or p.get("position"))
        if nhl_pos is None:
            continue
        model_pos = NHL_TO_MODEL_POS.get(nhl_pos)  # LW→L, RW→R, C→C, D→D
        first = p.get("firstName", {}).get("default") if isinstance(p.get("firstName"), dict) else p.get("firstName")
        last = p.get("lastName", {}).get("default") if isinstance(p.get("lastName"), dict) else p.get("lastName")
        full_name = p.get("fullName") or " ".join([str(first or "").strip(), str(last or "").strip()]).strip()
        if not full_name:
            full_name = str(p.get("name", "Unknown Player"))
        rows.append({
            "player_id":    int(pid),
            "player_name":  full_name,
            "nhl_position": nhl_pos,    # display: LW / RW / C / D
            "position":     model_pos,  # model:   L  / R  / C / D
            "nhl_team":     team_code,
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
            "pred_goals_gp":  float(cur.get("pred_goals_per_game", 0)),
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

# ── Player roster insertion ────────────────────────────────────────────────────

SLOT_COLORS = {
    "1st Line": "#FFD700", "1st Pair": "#FFD700",
    "2nd Line": "#4a90d9", "2nd Pair": "#4a90d9",
    "3rd Line": "#57a85a", "3rd Pair": "#57a85a",
    "4th Line": "#888888", "3rd Pair (extra)": "#888888",
}

FWD_SLOT_MAP = {
    1: "1st Line", 2: "1st Line",  3: "1st Line",
    4: "2nd Line", 5: "2nd Line",  6: "2nd Line",
    7: "3rd Line", 8: "3rd Line",  9: "3rd Line",
    10: "4th Line", 11: "4th Line", 12: "4th Line",
}

DEF_SLOT_MAP = {
    1: "1st Pair", 2: "1st Pair",
    3: "2nd Pair", 4: "2nd Pair",
    5: "3rd Pair", 6: "3rd Pair",
}


def build_player_insertion(player_id, team_code, df, team_ctx,
                           fit_models, player_profiles, has_age):
    """
    Insert the searched player into the selected team's roster.
    Ranks all rostered players + the searched player by predicted Points/GP
    on that team context. Existing players shift down if the new player
    ranks above them.

    Returns a DataFrame with columns:
        rank, player_name, player_id, position, pred_points_gp,
        pred_goals_gp, lineup_slot, slot_color, is_searched_player
    """
    roster_df, err = fetch_active_team_roster(team_code)
    if err or roster_df is None:
        return None, err or "Could not fetch roster."

    latest_ctx = get_latest_team_contexts(df, team_ctx)

    if player_id not in player_profiles:
        return None, "Player profile not found in model data."

    searched_profile, searched_seasons = player_profiles[player_id]
    model_pos = searched_profile.get("position", "C")   # L / R / C / D (MoneyPuck)
    is_fwd    = model_pos in ("C", "L", "R")
    # Accept both model codes and NHL display codes in the roster filter
    model_fwd_group = {"C", "L", "R"}
    nhl_fwd_group   = {"C", "LW", "RW"}

    team_row = latest_ctx[
        (latest_ctx["player_team"] == team_code) &
        (latest_ctx["position"] == model_pos)
    ]
    if team_row.empty:
        return None, f"No team context found for {team_code}."
    team_row    = team_row.iloc[0]
    league_env  = get_latest_league_env(df)

    def predict_pts(profile):
        row = profile.copy()
        for col in TEAM_FEATURES:
            if col in team_row.index:
                row[col] = team_row[col]
        for k, v in league_env.items():
            row[k] = v
        X = _make_X_from_profile(row, has_age)
        baseline = compute_target_baseline(pd.DataFrame([row]), "points_per_game").values[0]
        raw      = fit_models["points_per_game"]["global"].predict(X)[0]
        return float(np.clip(baseline + raw, 0, None))

    def predict_goals(profile):
        row = profile.copy()
        for col in TEAM_FEATURES:
            if col in team_row.index:
                row[col] = team_row[col]
        for k, v in league_env.items():
            row[k] = v
        X = _make_X_from_profile(row, has_age)
        baseline = compute_target_baseline(pd.DataFrame([row]), "goals_per_game").values[0]
        raw      = fit_models["goals_per_game"]["global"].predict(X)[0]
        return float(np.clip(baseline + raw, 0, None))

    # NHL display position for the searched player (LW / RW / C)
    searched_nhl_pos = MODEL_TO_NHL_POS.get(model_pos, model_pos)

    rows = []

    # Add searched player
    rows.append({
        "player_id":           player_id,
        "player_name":         searched_profile.get("player_name", "Selected Player"),
        "position":            model_pos,
        "nhl_position":        searched_nhl_pos,
        "pred_points_gp":      predict_pts(searched_profile),
        "pred_goals_gp":       predict_goals(searched_profile),
        "is_searched_player":  True,
    })

    # Add rostered players in same position group
    for _, rp in roster_df.iterrows():
        pid = int(rp["player_id"])
        rp_model_pos = rp.get("position")       # L / R / C from parse_roster_entries
        rp_nhl_pos   = rp.get("nhl_position")   # LW / RW / C from NHL API
        # Accept both coding conventions in case nhl_position is absent
        in_group = (rp_model_pos in model_fwd_group) if is_fwd else (rp_model_pos == "D")
        if not in_group:
            in_group = (rp_nhl_pos in nhl_fwd_group) if is_fwd else (rp_nhl_pos == "D")
        if not in_group:
            continue
        if pid == player_id:
            continue
        if pid not in player_profiles:
            continue
        profile, _ = player_profiles[pid]
        rows.append({
            "player_id":          pid,
            "player_name":        rp["player_name"],
            "position":           rp_model_pos or model_pos,
            "nhl_position":       rp_nhl_pos or MODEL_TO_NHL_POS.get(rp_model_pos, rp_model_pos),
            "pred_points_gp":     predict_pts(profile),
            "pred_goals_gp":      predict_goals(profile),
            "is_searched_player": False,
        })

    if not rows:
        return None, "No players could be matched."

    if not is_fwd:
        # Defensemen: simple rank by predicted score
        result = (
            pd.DataFrame(rows)
            .sort_values("pred_points_gp", ascending=False)
            .reset_index(drop=True)
        )
        result["rank"] = result.index + 1
        result["lineup_slot"] = result["rank"].apply(
            lambda r: DEF_SLOT_MAP.get(r, "3rd Pair (extra)")
        )
    else:
        # ── Position-aware line building ──────────────────────────────────────
        # Split forwards into LW / C / RW groups, rank within each,
        # then build lines with one player per position per line.
        # Use line_adj_xg_per60 from MoneyPuck (already in player profiles)
        # as a secondary signal alongside predicted points/gp.
        def _line_score(row):
            pts    = row.get("pred_points_gp", 0)
            lm_xg  = float(row.get("line_adj_xg_per60", 0) or 0)
            # Blend: 85% predicted output, 15% historical line quality
            return pts * 0.85 + lm_xg * 0.15

        by_pos = {"LW": [], "C": [], "RW": []}
        for r in rows:
            p = r.get("nhl_position", "")
            if p in by_pos:
                by_pos[p].append(r)
            else:
                # Fall back: map model position if nhl_position is missing
                mp = r.get("position", "")
                mapped = MODEL_TO_NHL_POS.get(mp, "")
                if mapped in by_pos:
                    by_pos[mapped].append(r)

        # Sort each position group by composite line score
        for pos in by_pos:
            # Attach line metrics from player profile if available
            for r in by_pos[pos]:
                pid = r.get("player_id")
                if pid and pid in player_profiles:
                    prof, _ = player_profiles[pid]
                    r["line_adj_xg_per60"] = float(prof.get("line_adj_xg_per60", 0) or 0)
                    r["line_corsi_pct"]    = float(prof.get("line_corsi_pct",    0) or 0)
            by_pos[pos].sort(key=_line_score, reverse=True)

        # Build up to 4 lines: each line gets the Nth-ranked LW + C + RW
        LINE_LABELS = ["1st Line", "2nd Line", "3rd Line", "4th Line"]
        assigned_rows = []
        max_depth = max(len(v) for v in by_pos.values()) if any(by_pos.values()) else 0

        for line_idx in range(min(max_depth, 4)):
            line_label = LINE_LABELS[line_idx]
            for pos in ("LW", "C", "RW"):
                group = by_pos[pos]
                if line_idx < len(group):
                    group[line_idx]["lineup_slot"] = line_label
                    assigned_rows.append(group[line_idx])

        # Any overflow (5th+ LW etc.) labelled 4th Line
        for pos in by_pos:
            for r in by_pos[pos][4:]:
                r["lineup_slot"] = "4th Line"
                assigned_rows.append(r)

        if not assigned_rows:
            # Fallback to flat sort if position data is missing
            assigned_rows = sorted(rows, key=lambda r: r.get("pred_points_gp", 0), reverse=True)
            for i, r in enumerate(assigned_rows):
                r["lineup_slot"] = FWD_SLOT_MAP.get(i + 1, "4th Line")

        result = pd.DataFrame(assigned_rows).reset_index(drop=True)
        result["rank"] = result.index + 1

    result["slot_color"]     = result["lineup_slot"].map(SLOT_COLORS).fillna("#888888")
    result["pred_points_gp"] = result["pred_points_gp"].round(3)
    result["pred_goals_gp"]  = result["pred_goals_gp"].round(3)

    if "nhl_position" not in result.columns:
        result["nhl_position"] = result["position"].map(MODEL_TO_NHL_POS).fillna(result["position"])

    return result, None


# ── Defensive data ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_defensive_data():
    try:
        df = pd.read_csv(DEF_FILE)
        df["season"] = df["season"].astype(int)
        return df, None
    except Exception as e:
        return None, str(e)


def show_defensive_profile(player_id, player_name, def_df):
    """Show full defensive profile for a defenseman."""
    rows = def_df[def_df["player_id"] == player_id].sort_values("season")
    if rows.empty:
        st.warning(f"No defensive data found for {player_name}. They may not be in the training data as a defenseman.")
        return

    latest = rows.iloc[-1]
    season = int(latest["season"])
    team   = latest["player_team"]
    gp     = int(latest["games_played"])

    # ── Season selector ────────────────────────────────────────────────────────
    available_seasons = sorted(rows["season"].unique(), reverse=True)
    sel_season = st.selectbox(
        "Season", options=available_seasons,
        format_func=lambda s: f"{s}-{str(s+1)[-2:]}",
        key="def_season_select"
    )
    row = rows[rows["season"] == sel_season].iloc[0]
    gp  = int(row["games_played"])

    # ── Compute league percentiles for that season ─────────────────────────────
    season_df = def_df[def_df["season"] == sel_season]

    def pct_rank(col, row=row, df=season_df, higher_better=True):
        """Return percentile rank 0-100 among all D that season."""
        if col not in df.columns or col not in row.index:
            return None
        vals = df[col].dropna()
        if len(vals) == 0:
            return None
        val = row[col]
        if higher_better:
            return float((vals < val).mean() * 100)
        else:
            return float((vals > val).mean() * 100)

    def fmt(val, decimals=2, pct=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return f"{val*100:.1f}%" if pct else f"{val:.{decimals}f}"

    def pct_color(p):
        if p is None:
            return "white"
        if p >= 80: return "#FFD700"
        if p >= 60: return "#4a90d9"
        if p >= 40: return "#57a85a"
        return "#888888"

    # ── Header ────────────────────────────────────────────────────────────────
    st.subheader(f"{player_name} — {row['player_team']} | {sel_season}-{str(sel_season+1)[-2:]} | {gp} GP")
    st.caption("Percentile ranks shown in brackets — compared to all NHL defensemen that season.")

    # ── Key metrics row ───────────────────────────────────────────────────────
    st.markdown("#### Physical & Defensive Impact")
    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        (c1, "Hits / Game",     "ind_hits_pg",                True,  2),
        (c2, "Blocks / Game",   "shots_blocked_by_player_pg", True,  2),
        (c3, "Takeaways / Game","ind_takeaways_pg",            True,  3),
        (c4, "Giveaways / Game","ind_giveaways_pg",            False, 3),
        (c5, "Take/Give Ratio", "take_give_ratio",             True,  2),
    ]
    for col_ui, label, stat, hb, dec in metrics:
        val = row.get(stat, None)
        p   = pct_rank(stat, higher_better=hb)
        col_ui.metric(
            label,
            fmt(val, dec),
            f"p{p:.0f}" if p is not None else None,
        )

    c6, c7, c8, c9, c10 = st.columns(5)
    metrics2 = [
        (c6,  "Penalty Min/GP",  "ind_penalty_minutes_pg", False, 2),
        (c7,  "Penalties Drawn/GP","penalties_drawn_pg",    True,  2),
        (c8,  "D-Zone Start %",  "d_zone_start_pct",       False, 3),
        (c9,  "Faceoff Win %",   "faceoff_win_pct",        True,  3),
        (c10, "PK Time / Game",  "pk_toi_per_game",        True,  2),
    ]
    for col_ui, label, stat, hb, dec in metrics2:
        val = row.get(stat, None)
        p   = pct_rank(stat, higher_better=hb)
        col_ui.metric(
            label,
            fmt(val, dec),
            f"p{p:.0f}" if p is not None else None,
        )

    # ── Penalty Kill section ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Penalty Kill")
    pk1, pk2, pk3 = st.columns(3)
    pk_pct = row.get("pk_ice_pct", 0)
    pk_toi = row.get("pk_toi_per_game", 0)
    pk_p   = pct_rank("pk_ice_pct")
    pk1.metric("PK Ice Time %",    f"{pk_pct*100:.1f}%", f"p{pk_p:.0f}" if pk_p else None)
    pk2.metric("PK TOI / Game",    f"{pk_toi:.2f} min")
    pk3.metric("PK Seasons Avg",   f"{rows['pk_ice_pct'].mean()*100:.1f}%" if len(rows) > 1 else "—")

    # PK usage bar
    fig_pk, ax_pk = plt.subplots(figsize=(10, 1.2))
    fig_pk.patch.set_facecolor("#0e1117")
    ax_pk.set_facecolor("#0e1117")
    ax_pk.barh(["PK%"], [pk_pct * 100], color="#4a90d9", height=0.5)
    ax_pk.barh(["PK%"], [season_df["pk_ice_pct"].mean() * 100],
               color="#888888", height=0.3, alpha=0.6, label="League avg")
    ax_pk.set_xlim(0, max(season_df["pk_ice_pct"].max() * 100 + 1, 5))
    ax_pk.set_xlabel("% of ice time on penalty kill", color="white", fontsize=9)
    ax_pk.tick_params(colors="white", labelsize=8)
    ax_pk.axvline(season_df["pk_ice_pct"].mean() * 100,
                  color="#888888", linestyle="--", linewidth=1)
    for spine in ax_pk.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    st.pyplot(fig_pk)
    plt.close()

    # ── On-ice defensive impact ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### On-Ice Defensive Impact (5v5)")
    st.caption("Lower xG against and HD shots against = better defensive suppression.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("xG Against / 60",    fmt(row.get("xg_against_per60_5v5"),    2),
              f"p{pct_rank('xg_against_per60_5v5', higher_better=False):.0f}"
              if pct_rank("xg_against_per60_5v5") is not None else None)
    d2.metric("HD Shots Against/60",fmt(row.get("hd_shots_against_per60_5v5"), 2),
              f"p{pct_rank('hd_shots_against_per60_5v5', higher_better=False):.0f}"
              if pct_rank("hd_shots_against_per60_5v5") is not None else None)
    d3.metric("xGA/60 (Zone Adj)", fmt(row.get("xga_per60_zone_adj") or row.get("goals_against_per60"), 2),
              f"p{pct_rank('xga_per60_zone_adj', higher_better=False):.0f}"
              if pct_rank("xga_per60_zone_adj") is not None else None)
    d4.metric("5v5 Corsi %",        fmt(row.get("on_ice_corsi_pct"), 1),
              f"p{pct_rank('on_ice_corsi_pct'):.0f}"
              if pct_rank("on_ice_corsi_pct") is not None else None)

    # ── Career trend chart ─────────────────────────────────────────────────────
    if len(rows) > 1:
        st.divider()
        st.markdown("#### Career Trends")
        trend_cols = {
            "Hits/Game":    "ind_hits_pg",
            "Blocks/Game":  "shots_blocked_by_player_pg",
            "Takeaways/Game": "ind_takeaways_pg",
            "PK Ice %":     "pk_ice_pct",
        }
        fig_t, axes_t = plt.subplots(1, 4, figsize=(18, 3.5))
        fig_t.patch.set_facecolor("#0e1117")
        for ax_t, (label, col) in zip(axes_t, trend_cols.items()):
            ax_t.set_facecolor("#0e1117")
            if col in rows.columns:
                ax_t.plot(rows["season"], rows[col], color="#4a90d9",
                          marker="o", linewidth=2, markersize=6)
                ax_t.set_title(label, color="white", fontsize=10)
                ax_t.tick_params(colors="white", labelsize=7)
                ax_t.set_xlabel("Season", color="white", fontsize=8)
                # Shade selected season
                ax_t.axvline(sel_season, color="#FFD700",
                             linestyle="--", linewidth=1, alpha=0.7)
            for spine in ax_t.spines.values():
                spine.set_edgecolor("#333")
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close()

    # ── League leaderboard context ─────────────────────────────────────────────
    st.divider()
    st.markdown(f"#### League Rankings — {sel_season}-{str(sel_season+1)[-2:]}")
    rank_cols = {
        "Hits/GP":    ("ind_hits_pg",                True),
        "Blocks/GP":  ("shots_blocked_by_player_pg", True),
        "TK/GP":      ("ind_takeaways_pg",            True),
        "PK%":        ("pk_ice_pct",                  True),
        "xGA/60":     ("xg_against_per60_5v5",        False),
    }
    rank_rows = []
    for label, (col, hb) in rank_cols.items():
        if col not in season_df.columns:
            continue
        ranked = season_df[["player_name", col]].dropna().sort_values(col, ascending=not hb).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        player_rank = ranked[ranked["player_name"] == player_name]
        if not player_rank.empty:
            r = int(player_rank["rank"].iloc[0])
            v = player_rank[col].iloc[0]
            rank_rows.append({
                "Metric": label,
                "Value":  round(v * 100, 1) if "pct" in col or col == "pk_ice_pct" else round(v, 3),
                "Rank":   f"{r} / {len(ranked)}",
                "Percentile": f"p{100*(1-r/len(ranked)):.0f}",
            })
    if rank_rows:
        st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# DEFENSIVE MODEL — Defensemen prediction engine
# ══════════════════════════════════════════════════════════════════════════════

# ── Defensive model config ────────────────────────────────────────────────────

# DEF_FILE already defined above
# AGES_FILE already defined above
DEF_CACHE_FILE = "defensive_models.joblib"
# CURRENT_SEASON already defined above

# MIN_GP already defined above
# CV_FOLDS already defined above
# N_SEASONS already defined above
# SEASON_WEIGHTS already defined above

# NHL_TEAMS already defined above

# Defensive targets — what we're predicting
DEF_TARGETS = [
    "ind_hits_pg",
    "ind_takeaways_pg",
    "xga_per60_zone_adj",          # deployment-adjusted xGA/60
    "pk_ice_pct",
    "pim_pg",
    "shots_blocked_by_player_pg",  # article: blocking shots is explicit D responsibility
    "ind_giveaways_pg",            # article: puck decisions / disrupting offensive flow
]

DEF_TARGET_LABELS = {
    "ind_hits_pg":              "Hits / Game",
    "ind_takeaways_pg":         "Takeaways / Game",
    "xga_per60_zone_adj":         "xGA / 60 — Zone-Start Adjusted",
    "goals_against_per60":        "GA / 60 — 5v5 (ice-time adjusted)",
    "shots_blocked_by_player_pg": "Shots Blocked / Game",
    "ind_giveaways_pg":           "Giveaways / Game",
    "take_give_ratio":            "Takeaway / Giveaway Ratio",
    "pk_ice_pct":               "PK Ice Time %",
    "pim_pg":               "PIM / Game",
}

# Lower is better for these targets
DEF_LOWER_IS_BETTER = {"goals_against_per60", "xga_per60_zone_adj", "pim_pg", "ind_giveaways_pg"}

# Defensive score weights (used for pairing)
# Weights derived from the article's framework:
#   Analytics (xGA, Corsi) → highest weight
#   Active puck recovery (takeaways) + coaching trust (PK) → medium
#   Physical play (blocks, hits) → secondary
#   Puck decisions (giveaways, PIM) → discipline penalty
DEF_SCORE_WEIGHTS = {
    "xga_per60_zone_adj":          0.25,  # zone-adj xGA/60 — quality of chances against
    "ind_takeaways_pg":            0.18,  # active puck recovery
    "pk_ice_pct":                  0.15,  # coaching trust in defensive situations
    "shots_blocked_by_player_pg":  0.13,  # sacrificing body — explicit D responsibility
    "take_give_ratio":             0.12,  # puck decision balance (higher = more takes than gives)
    "ind_hits_pg":                 0.09,  # physicality (article: important but not everything)
    "pim_pg":                      0.08,  # discipline — lower is better
}

# Pairing slot definitions
DEF_PAIR_SLOTS = {1: "1st Pair", 2: "1st Pair",
              3: "2nd Pair", 4: "2nd Pair",
              5: "3rd Pair", 6: "3rd Pair"}
DEF_PAIR_COLORS = {
    "1st Pair": "#FFD700",
    "2nd Pair": "#4a90d9",
    "3rd Pair": "#57a85a",
    "3rd Pair (extra)": "#888888",
}

# ── Baseline features (leakage-safe prior season history) ──────────────────────

DEF_BASELINE_FEATURES = {
    "ind_hits_pg":             ["prev_season_hits_pg",     "recent_3yr_mean_hits_pg",     "career_prev_mean_hits_pg"],
    "ind_takeaways_pg":        ["prev_season_takeaways_pg","recent_3yr_mean_takeaways_pg","career_prev_mean_takeaways_pg"],
    "xga_per60_zone_adj":  ["prev_season_xga_pg",      "recent_3yr_mean_xga_pg",      "career_prev_mean_xga_pg"],
    "goals_against_per60": ["prev_season_xga_pg",      "recent_3yr_mean_xga_pg",      "career_prev_mean_xga_pg"],
    "pk_ice_pct":                  ["prev_season_pk_pct",          "recent_3yr_mean_pk_pct",          "career_prev_mean_pk_pct"],
    "pim_pg":                      ["prev_season_pim_pg",           "recent_3yr_mean_pim_pg",           "career_prev_mean_pim_pg"],
    "shots_blocked_by_player_pg":  ["prev_season_blocks_pg",        "recent_3yr_mean_blocks_pg",        "career_prev_mean_blocks_pg"],
    "ind_giveaways_pg":            ["prev_season_giveaways_pg",     "recent_3yr_mean_giveaways_pg",     "career_prev_mean_giveaways_pg"],
}

# ── Player features ────────────────────────────────────────────────────────────

DEF_PLAYER_FEATURES = [
    # Physical skill signals
    "ind_hits_per60",
    "ind_takeaways_per60",
    "ind_giveaways_per60",
    "shots_blocked_by_player_per60",
    "ind_penalty_minutes_per60",
    "take_give_ratio",
    "d_zone_start_pct",
    "faceoff_win_pct",
    # On-ice defensive impact
    "xga_per60_zone_adj",      # zone-adjusted xGA/60 — used in defensive score
    "goals_against_per60",     # raw GA/60 fallback
    "xg_against_per60_5v5",
    "hd_shots_against_per60_5v5",
    "on_ice_corsi_pct",
    "on_ice_fenwick_pct",
    # Career peak signals
    "career_peak_hits_pg",
    "career_peak_takeaways_pg",
    "career_peak_pk_pct",
    "pct_of_peak_hits",
    "pct_of_peak_takeaways",
    # Career history (leakage-safe — prior seasons only)
    "prev_season_hits_pg",
    "prev_season_takeaways_pg",
    "prev_season_xga_pg",
    "prev_season_pk_pct",
    "prev_season_pim_pg",
    "recent_3yr_mean_hits_pg",
    "recent_3yr_mean_takeaways_pg",
    "recent_3yr_mean_xga_pg",
    "recent_3yr_mean_pk_pct",
    "recent_3yr_mean_pim_pg",
    "career_prev_mean_hits_pg",
    "career_prev_mean_takeaways_pg",
    "career_prev_mean_xga_pg",
    "career_prev_mean_pk_pct",
    "career_prev_mean_pim_pg",
    "career_seasons_prior",
    # Slopes
    "recent_3yr_hits_slope",
    "recent_3yr_takeaways_slope",
    "recent_3yr_xga_slope",
    "recent_3yr_pk_slope",
    "recent_3yr_pim_slope",
    # League environment
    "league_avg_hits_pg",
    "league_avg_pk_pct",
]

DEF_AGE_FEATURES = ["age", "age_sq", "age_x_hits", "age_x_takeaways", "age_x_pk"]

DEF_TEAM_FEATURES = [
    "team_avg_hits_pg",
    "team_avg_takeaways_pg",
    "team_avg_xga_per60",
    "team_avg_pk_pct",
    "team_avg_pim_pg",
    "team_avg_toi_pg",
    "team_avg_d_zone_start_pct",
]

DEF_TRAJECTORY_FEATURES = [
    "yoy_hits_delta",
    "yoy_takeaways_delta",
    "yoy_xga_delta",
    "games_played_pct",
    "career_year",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def def_safe_div(a, b, fill=0.0):
    return np.where(b == 0, fill, a / b)


def def_prior_slope(s, window=None):
    vals = s.shift(1).values
    out  = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        start = 0 if window is None else max(0, i - window + 1)
        win   = vals[start:i + 1]
        mask  = ~np.isnan(win)
        if mask.sum() < 2:
            continue
        y       = win[mask]
        x       = np.arange(len(win))[mask].astype(float)
        x_mean  = x.mean()
        y_mean  = y.mean()
        denom   = ((x - x_mean) ** 2).sum()
        out[i]  = 0.0 if denom == 0 else float(((x - x_mean) * (y - y_mean)).sum() / denom)
    return pd.Series(out, index=s.index)


# ── Feature engineering ────────────────────────────────────────────────────────

def def_engineer_features(df):
    """Add derived features, league environment, career peaks."""
    d = df.copy()

    # League scoring environment
    d["league_avg_hits_pg"] = d.groupby("season")["ind_hits_pg"].transform("mean")
    d["league_avg_pk_pct"]  = d.groupby("season")["pk_ice_pct"].transform("mean")

    # Career peak features
    d["career_peak_hits_pg"]      = d.groupby("player_id")["ind_hits_pg"].transform("max")
    d["career_peak_takeaways_pg"] = d.groupby("player_id")["ind_takeaways_pg"].transform("max")
    d["career_peak_pk_pct"]       = d.groupby("player_id")["pk_ice_pct"].transform("max")

    d["pct_of_peak_hits"]      = def_safe_div(d["ind_hits_pg"],      d["career_peak_hits_pg"])
    d["pct_of_peak_takeaways"] = def_safe_div(d["ind_takeaways_pg"], d["career_peak_takeaways_pg"])

    # Goals Against / 60 (5v5) — controls for ice time so high-minute D-men
    # aren't penalised just for being on the ice more.
    # Priority: use MoneyPuck's pre-computed per-60 column; fall back to
    # computing it from raw on-ice goals divided by 5v5 TOI in hours.
    if "goals_against_per60_5v5" in d.columns:
        d["goals_against_per60"] = d["goals_against_per60_5v5"]
    elif "on_ice_against_goals" in d.columns:
        if "fv5_ice_time" in d.columns:
            fv5_hours = (d["fv5_ice_time"] / 3600).replace(0, np.nan)
        elif "ice_time" in d.columns:
            # Approximate: assume ~60% of total TOI is 5v5
            fv5_hours = (d["ice_time"] * 0.60 / 3600).replace(0, np.nan)
        else:
            fv5_hours = None
        if fv5_hours is not None:
            d["goals_against_per60"] = d["on_ice_against_goals"] / fv5_hours
        else:
            # Last resort: per-game (not ideal but labeled correctly in display)
            gp_safe = d["games_played"].replace(0, np.nan)
            d["goals_against_per60"] = d["on_ice_against_goals"] / gp_safe

    # ── Zone-start adjusted xGA/60 (training label + feature) ─────────────────
    # This is the primary quality-against target the model trains to predict.
    # Adjusting BEFORE training means the model learns the true relationship
    # between deployment, shot suppression, and defensive quality — rather than
    # us bolting on an adjustment after the fact.
    #
    # Formula:  adj_xga = xg_against_per60_5v5 * (league_avg_dzone / player_dzone)^0.5
    # Square-root dampens the correction; clamped to [0.70, 1.40].
    if "xg_against_per60_5v5" in d.columns and "d_zone_start_pct" in d.columns:
        league_avg_dzone = d.groupby("season")["d_zone_start_pct"].transform("mean")
        dzone_ratio = def_safe_div(league_avg_dzone, d["d_zone_start_pct"].replace(0, np.nan), fill=1.0)
        dzone_ratio = dzone_ratio.clip(0.70, 1.40)
        d["xga_per60_zone_adj"] = d["xg_against_per60_5v5"] * (dzone_ratio ** 0.5)
    elif "goals_against_per60" in d.columns and "d_zone_start_pct" in d.columns:
        league_avg_dzone = d.groupby("season")["d_zone_start_pct"].transform("mean")
        dzone_ratio = def_safe_div(league_avg_dzone, d["d_zone_start_pct"].replace(0, np.nan), fill=1.0)
        dzone_ratio = dzone_ratio.clip(0.70, 1.40)
        d["xga_per60_zone_adj"] = d["goals_against_per60"] * (dzone_ratio ** 0.5)
    elif "xg_against_per60_5v5" in d.columns:
        d["xga_per60_zone_adj"] = d["xg_against_per60_5v5"]
    elif "goals_against_per60" in d.columns:
        d["xga_per60_zone_adj"] = d["goals_against_per60"]

    # Age interactions
    if "age" in d.columns:
        d["age_x_hits"]      = d["age"] * d["ind_hits_per60"]
        d["age_x_takeaways"] = d["age"] * d["ind_takeaways_per60"]
        d["age_x_pk"]        = d["age"] * d["pk_ice_pct"]

    return d


def def_engineer_career_history(df):
    """Leakage-safe prior-season history features."""
    d = df.sort_values(["player_id", "season"]).copy()
    g = d.groupby("player_id", sort=False)

    d["career_seasons_prior"] = g.cumcount().astype(float)
    d["career_year"]          = d["career_seasons_prior"] + 1
    d["games_played_pct"]     = d["games_played"] / 82.0

    # Previous season values
    for col, name in [
        ("ind_hits_pg",             "hits_pg"),
        ("ind_takeaways_pg",        "takeaways_pg"),
        # Use zone-adjusted xGA as the historical quality-against signal.
        ("xga_per60_zone_adj",        "xga_pg"),
        ("pk_ice_pct",                "pk_pct"),
        ("ind_penalty_minutes_pg",    "pim_pg"),
        ("shots_blocked_by_player_pg","blocks_pg"),
        ("ind_giveaways_pg",          "giveaways_pg"),
    ]:
        d[f"prev_season_{name}"] = g[col].shift(1)
        d[f"recent_3yr_mean_{name}"] = (
            g[col].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        d[f"career_prev_mean_{name}"] = (
            g[col].apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
        d[f"recent_3yr_{name}_slope"] = (
            g[col].apply(lambda s: def_prior_slope(s, window=3))
            .reset_index(level=0, drop=True)
        )

    # YoY deltas
    d["yoy_hits_delta"]      = g["ind_hits_pg"].diff()
    d["yoy_takeaways_delta"] = g["ind_takeaways_pg"].diff()
    d["yoy_xga_delta"]       = g["xga_per60_zone_adj"].diff()

    return d


def def_build_team_context(df):
    """Aggregate team-level defensive context per season."""
    team_ctx = (
        df.groupby(["player_team", "season"])
        .agg(
            team_avg_hits_pg         = ("ind_hits_pg",             "mean"),
            team_avg_takeaways_pg    = ("ind_takeaways_pg",        "mean"),
            team_avg_xga_per60       = ("xga_per60_zone_adj", "mean"),
            team_avg_pk_pct          = ("pk_ice_pct",              "mean"),
            team_avg_pim_pg          = ("ind_penalty_minutes_pg",  "mean"),
            team_avg_toi_pg          = ("pk_toi_per_game",         "mean"),
            team_avg_d_zone_start_pct = ("d_zone_start_pct",       "mean"),
        )
        .reset_index()
    )
    return team_ctx


def def_get_latest_team_contexts(df, team_ctx):
    latest = df["season"].max()
    ctx    = team_ctx[team_ctx["season"] == latest].copy()
    if ctx["player_team"].nunique() < team_ctx["player_team"].nunique():
        fallback = (
            team_ctx.sort_values("season", ascending=False)
            .groupby("player_team").first().reset_index()
        )
        present = set(ctx["player_team"])
        missing = fallback[~fallback["player_team"].isin(present)]
        ctx     = pd.concat([ctx, missing], ignore_index=True)
    return ctx


def def_build_player_profile(player_rows):
    latest_season = player_rows["season"].max()
    profile       = player_rows[player_rows["season"] == latest_season].iloc[0].copy()
    seasons       = [latest_season]
    return profile, seasons


def def_build_feature_matrix(df, has_age):
    feats = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + DEF_TEAM_FEATURES
    feats = [f for f in feats if f in df.columns]
    X     = df[feats].copy()
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def def_build_next_feature_matrix(df, has_age):
    traj  = [f for f in DEF_TRAJECTORY_FEATURES if f in df.columns]
    feats = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + traj + DEF_TEAM_FEATURES
    feats = [f for f in feats if f in df.columns]
    X     = df[feats].copy()
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def def_compute_target_baseline(df_like, target):
    candidates = [c for c in DEF_BASELINE_FEATURES.get(target, []) if c in df_like.columns]
    if not candidates:
        return pd.Series(np.zeros(len(df_like)), index=df_like.index)
    baseline = df_like[candidates].bfill(axis=1).iloc[:, 0]
    return baseline.fillna(0.0).astype(float)


def def_make_elite_sample_weights(y, lower_is_better=False):
    arr = np.asarray(y, dtype=float)
    if lower_is_better:
        arr = -arr  # flip so elite = extreme low = high weight
    q75, q90, q95 = np.quantile(arr, [0.75, 0.90, 0.95])
    weights = np.ones(len(arr), dtype=float)
    weights += 0.5 * (arr >= q75)
    weights += 1.0 * (arr >= q90)
    weights += 1.5 * (arr >= q95)
    return weights


# ── Training ───────────────────────────────────────────────────────────────────

def def_make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=0.1,
        objective="huber", random_state=42, verbose=-1,
    )


def def_train_models_with_progress(X, df, targets, target_col_map,
                                label_prefix, status, bar, step, total_steps):
    kf      = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    models  = {}
    metrics = {}

    for target in targets:
        label      = DEF_TARGET_LABELS[target]
        target_col = target_col_map[target]
        y          = np.clip(df[target_col].values, 0, None)
        baseline   = def_compute_target_baseline(df, target).values
        y_resid    = y - baseline
        sample_w   = def_make_elite_sample_weights(y, lower_is_better=(target in DEF_LOWER_IS_BETTER))

        fold_maes, fold_rmses = [], []
        for fold, (tr, val) in enumerate(kf.split(X), 1):
            status.markdown(f"🔁 **{label_prefix} — {label}** fold {fold}/{CV_FOLDS}")
            m = clone(def_make_lgbm())
            m.fit(X.iloc[tr], y_resid[tr], sample_weight=sample_w[tr])
            preds = np.clip(baseline[val] + m.predict(X.iloc[val]), 0, None)
            fold_maes.append(mean_absolute_error(y[val], preds))
            fold_rmses.append(np.sqrt(mean_squared_error(y[val], preds)))
            step += 1
            bar.progress(min(step / total_steps, 0.99),
                         text=f"{label_prefix} {label}: fold {fold}/{CV_FOLDS} — MAE {np.mean(fold_maes):.3f}")

        status.markdown(f"✅ **{label_prefix} — {label}** fitting final model...")
        gm = def_make_lgbm()
        gm.fit(X, y_resid, sample_weight=sample_w)
        models[target]  = {"global": gm}
        metrics[target] = {
            "mae":  (float(np.mean(fold_maes)),  float(np.std(fold_maes))),
            "rmse": (float(np.mean(fold_rmses)), float(np.std(fold_rmses))),
        }
        step += 1
        bar.progress(min(step / total_steps, 0.99),
                     text=f"{label_prefix} {label} done — MAE {np.mean(fold_maes):.3f}")

    return models, metrics, step


def def_load_and_train(def_path, ages_path):
    total_steps = 3 + 2 * len(DEF_TARGETS) * (CV_FOLDS + 1)
    step = 0

    status = st.empty()
    bar    = st.progress(0, text="Starting up...")

    def advance(msg):
        nonlocal step
        step += 1
        bar.progress(min(step / total_steps, 0.99), text=msg)

    # Load
    status.markdown("⚙️ **Loading defensive data...**")
    df = pd.read_csv(def_path)
    df = df[df["games_played"] >= MIN_GP].copy()

    ages = _load_ages(ages_path)
    df   = df.merge(ages, on=["player_id", "season"], how="left")
    has_age = df["age"].notna().mean() > 0.5
    advance(f"Data loaded — {len(df):,} defenseman-seasons | age matched: {df['age'].notna().sum():,}")

    # Engineer
    status.markdown("⚙️ **Engineering features...**")
    df       = def_engineer_features(df)
    df       = def_engineer_career_history(df)
    # pim_pg is the model target name — alias from the source column
    df["pim_pg"] = df["ind_penalty_minutes_pg"]
    team_ctx = def_build_team_context(df)
    df       = df.merge(team_ctx, on=["player_team", "season"], how="left")
    advance("Features engineered")

    # Profiles
    status.markdown("⚙️ **Building player profiles...**")
    player_profiles = {}
    for pid, group in df.groupby("player_id"):
        profile, seasons = def_build_player_profile(group)
        player_profiles[pid] = (profile, seasons)
    advance(f"Profiles built — {len(player_profiles):,} defensemen")

    # Current fit model
    status.markdown("⚙️ **Training Current Fit models...**")
    X_fit = def_build_feature_matrix(df, has_age)
    fit_feature_names = X_fit.columns.tolist()
    fit_models, fit_metrics, step = def_train_models_with_progress(
        X_fit, df, DEF_TARGETS, {t: t for t in DEF_TARGETS},
        "Current Fit", status, bar, step, total_steps
    )

    # Next season model
    status.markdown("⚙️ **Training Next Season models...**")
    df_next = df.sort_values(["player_id", "season"]).copy()
    next_targets_df = df_next.groupby("player_id")[DEF_TARGETS].shift(-1)
    next_targets_df.columns = [f"next_{t}" for t in DEF_TARGETS]
    df_next = pd.concat([df_next, next_targets_df], axis=1).dropna(
        subset=[f"next_{t}" for t in DEF_TARGETS]
    )
    X_next = def_build_next_feature_matrix(df_next, has_age)
    next_feature_names = X_next.columns.tolist()
    next_models, next_metrics, step = def_train_models_with_progress(
        X_next, df_next, DEF_TARGETS, {t: f"next_{t}" for t in DEF_TARGETS},
        "Next Season", status, bar, step, total_steps
    )

    bar.progress(1.0, text="✅ All defensive models trained!")
    status.empty()
    bar.empty()

    return (df, team_ctx, has_age, player_profiles,
            fit_models, fit_metrics, fit_feature_names,
            next_models, next_metrics, next_feature_names)


# ── Prediction ─────────────────────────────────────────────────────────────────

def def_predict_for_team(profile, team_row, models, has_age, use_traj=False,
                         feature_names=None):
    """Predict all 5 targets for a player on a specific team context."""
    row = profile.copy()
    for col in DEF_TEAM_FEATURES:
        if col in team_row.index:
            row[col] = team_row[col]

    pred_df = pd.DataFrame([row])

    if feature_names is not None:
        # Use exact feature names the model was trained on — avoids count mismatches
        feats = [f for f in feature_names if f in pred_df.columns]
        # Fill any missing expected features with 0
        for f in feature_names:
            if f not in pred_df.columns:
                pred_df[f] = 0.0
        feats = feature_names
    else:
        if use_traj:
            traj  = [f for f in DEF_TRAJECTORY_FEATURES if f in pred_df.columns]
            feats = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + traj + DEF_TEAM_FEATURES
        else:
            feats = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + DEF_TEAM_FEATURES
        feats = [f for f in feats if f in pred_df.columns]

    X = pred_df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)

    preds = {}
    for target, model_dict in models.items():
        baseline = def_compute_target_baseline(pred_df, target).values[0]
        raw      = model_dict["global"].predict(X)[0]
        preds[target] = float(np.clip(baseline + raw, 0, None))
    return preds


def def_build_all_team_predictions(profile, all_teams, models, has_age,
                                     use_traj=False, feature_names=None):
    """Predict all 5 targets for a player across all 32 teams."""
    rows = []
    for _, team_row in all_teams.iterrows():
        preds = def_predict_for_team(profile, team_row, models, has_age,
                                     use_traj, feature_names=feature_names)
        preds["player_team"] = team_row["player_team"]
        rows.append(preds)
    return pd.DataFrame(rows)


def classify_defenseman_role(profile):
    """
    Classify a defenseman as Offensive D / Two-Way D / Defensive D.

    Based on the article framework:
      Offensive D  — puck-movers, high Corsi, fewer D-zone starts, contribute offensively
      Defensive D  — physical, shot-blockers, shutdown, high PK time, heavy D-zone starts
      Two-Way D    — balanced across all dimensions (most versatile)

    Returns (role_label, role_color, role_description).
    """
    corsi     = float(profile.get("on_ice_corsi_pct", 0.50) or 0.50)
    d_zone    = float(profile.get("d_zone_start_pct", 0.40) or 0.40)
    pk_pct    = float(profile.get("pk_ice_pct", 0.0) or 0.0)
    hits      = float(profile.get("ind_hits_pg", 0) or 0)
    blocks    = float(profile.get("shots_blocked_by_player_pg", 0) or 0)
    takeaways = float(profile.get("ind_takeaways_pg", 0) or 0)
    giveaways = float(profile.get("ind_giveaways_pg", 0.5) or 0.5)

    # Offensive signal: above-average Corsi, fewer D-zone starts
    off_score = (corsi - 0.50) * 200 + max(0, 0.40 - d_zone) * 150

    # Defensive signal: physicality, shot blocking, PK usage, heavy D-zone deployment
    def_score_val = (hits * 10) + (blocks * 10) + (pk_pct * 100) + max(0, d_zone - 0.40) * 150

    # Puck decision quality: high takeaway/giveaway ratio tilts toward D
    tg_ratio = takeaways / max(giveaways, 0.1)
    def_score_val += max(0, tg_ratio - 1.5) * 5

    OFF_THRESHOLD = 8
    DEF_THRESHOLD = 8

    if off_score >= OFF_THRESHOLD and def_score_val < DEF_THRESHOLD:
        return (
            "Offensive D", "#4a90d9",
            "Puck-mover with high Corsi and offensive zone deployment. "
            "Contributes by moving pucks up ice and generating offense."
        )
    elif def_score_val >= DEF_THRESHOLD and off_score < OFF_THRESHOLD:
        return (
            "Defensive D", "#57a85a",
            "Shutdown D with physical play, shot blocking, and heavy defensive-zone deployment. "
            "Trusted on the penalty kill."
        )
    else:
        return (
            "Two-Way D", "#FFD700",
            "Balanced across offense and defense. "
            "Versatile — can contribute in all situations."
        )


def def_compute_defensive_score(df_preds, season_df=None):
    """
    Compute a composite defensive score 0-100 for each row.

    Normalises each metric against fixed league-wide empirical ranges
    rather than within the prediction set — avoids the collapse-to-45
    issue when all 32-team predictions are similar.

    The quality-against component uses zone-start adjusted xGA/60:
    the model predicts goals_against_per60, which is then adjusted by
    d_zone_start_pct so that D-men eating harder minutes aren't penalised
    for their deployment.
    """
    LEAGUE_RANGES = {
        "ind_hits_pg":        (0.0,  3.5),
        "ind_takeaways_pg":   (0.0,  0.65),
        "xga_per60_zone_adj": (1.5,  3.5),  # zone-adjusted xGA/60
        "goals_against_per60":(1.5,  3.5),  # fallback if zone-adj unavailable
        "pk_ice_pct":                  (0.0,  0.18),
        "pim_pg":                      (0.0,  1.2),
        "shots_blocked_by_player_pg":  (0.0,  3.5),
        "ind_giveaways_pg":            (0.0,  1.5),
        "take_give_ratio":             (0.5,  4.0),
    }

    result = df_preds.copy()

    # xga_per60_zone_adj is now a direct model prediction — no post-hoc adjustment needed.
    # Fall back to goals_against_per60 only if the model hasn't been retrained yet.
    if "xga_per60_zone_adj" not in result.columns and "goals_against_per60" in result.columns:
        result["xga_per60_zone_adj"] = result["goals_against_per60"]

    score = np.zeros(len(result))

    for target, weight in DEF_SCORE_WEIGHTS.items():
        if target not in result.columns:
            continue
        vals = result[target].values.astype(float)

        # Use season_df for live normalization if provided, else use fixed ranges
        if season_df is not None and target in season_df.columns:
            lo = np.nanpercentile(season_df[target], 5)
            hi = np.nanpercentile(season_df[target], 95)
        else:
            lo, hi = LEAGUE_RANGES.get(target, (vals.min(), vals.max()))

        rng = hi - lo
        if rng == 0:
            norm = np.full(len(vals), 0.5)
        else:
            norm = np.clip((vals - lo) / rng, 0, 1)

        # Invert for lower-is-better metrics
        if target in DEF_LOWER_IS_BETTER:
            norm = 1 - norm

        score += norm * weight

    result["defensive_score"] = np.round(score * 100, 1)
    return result


def def_predict_defenseman(player_name, df, team_ctx, fit_models, next_models,
                        player_profiles, has_age,
                        fit_feature_names=None, next_feature_names=None):
    """Main prediction entry point."""
    mask = df["player_name"].str.lower() == player_name.strip().lower()
    rows = df[mask]
    if rows.empty:
        mask = df["player_name"].str.lower().str.contains(player_name.strip().lower(), na=False)
        rows = df[mask]
        if rows.empty:
            return None

    pid              = rows["player_id"].iloc[0]
    profile, seasons = player_profiles[pid]
    actual_team      = profile["player_team"]
    matched          = profile["player_name"]

    all_teams = def_get_latest_team_contexts(df, team_ctx)

    # Current fit predictions
    fit_results  = def_build_all_team_predictions(profile, all_teams, fit_models, has_age,
                                                    feature_names=fit_feature_names)
    fit_results  = def_compute_defensive_score(fit_results)
    fit_results  = fit_results.sort_values("defensive_score", ascending=False).reset_index(drop=True)
    fit_results.index += 1
    fit_results["is_actual"] = fit_results["player_team"] == actual_team

    # Next season predictions
    next_results = def_build_all_team_predictions(profile, all_teams, next_models, has_age,
                                                     use_traj=True, feature_names=next_feature_names)
    next_results = def_compute_defensive_score(next_results)
    next_results = next_results.sort_values("defensive_score", ascending=False).reset_index(drop=True)
    next_results.index += 1
    next_results["is_actual"] = next_results["player_team"] == actual_team

    return {
        "pid":          pid,
        "matched":      matched,
        "actual_team":  actual_team,
        "seasons":      seasons,
        "fit_results":  fit_results,
        "next_results": next_results,
        "profile":      profile,
    }


# ── Roster & pairing ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def def_fetch_team_roster_d(team_code):
    """Fetch current defensemen from NHL API."""
    try:
        url  = f"https://api-web.nhle.com/v1/roster/{team_code}/{CURRENT_SEASON}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        players = []
        for p in data.get("defensemen", []):
            players.append({
                "player_id":   int(p["id"]),
                "player_name": f"{p['firstName']['default']} {p['lastName']['default']}",
                "position":    "D",
            })
        return players
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_actual_pairs(team_code):
    """
    Fetch actual D-pair combinations using shared TOI.

    Strategy (tries in order, returns on first success):
      1. NHL stats REST API  — pre-computed on-ice-with data for the season.
      2. NHL shift chart API — parses raw per-game shift data.

    Returns ([(pid1, pid2, shared_seconds), ...], error_string_or_None).
    """
    import datetime
    from collections import defaultdict

    FINISHED_STATES = {"OFF", "FINAL", "CRIT", "OVER"}  # all states meaning game ended

    def _to_int_pid(v):
        try:
            return int(v)
        except Exception:
            return None

    def _to_seconds(v):
        if v is None:
            return None
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(v)
        s = str(v).strip()
        if not s or s == "None":
            return None
        if ":" in s:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except Exception:
                return None
        try:
            return int(float(s))
        except Exception:
            return None

    # ── Fetch current D-men roster (used by both methods) ─────────────────────
    try:
        roster_url  = f"https://api-web.nhle.com/v1/roster/{team_code}/{CURRENT_SEASON}"
        roster_resp = requests.get(roster_url, timeout=10)
        roster_resp.raise_for_status()
        d_pids = {int(p["id"]) for p in roster_resp.json().get("defensemen", [])}
    except Exception:
        d_pids = set()

    # ── Method 1: NHL stats REST API (on-ice-with, pre-computed) ──────────────
    # Note: api.nhle.com is a separate domain from api-web.nhle.com
    # and may be unavailable in some network environments.
    try:
        import urllib.parse
        import json as _json
        sort_param  = _json.dumps([{"property": "timeOnIceWith", "direction": "DESC"}])
        cayenne_exp = f'seasonId={CURRENT_SEASON} and gameTypeId=2 and teamAbbrevs="{team_code}"'
        fact_exp    = "gamesPlayedWith>=5"
        stats_url   = (
            "https://api.nhle.com/stats/rest/en/skater/oniceWith"
            f"?isAggregate=true&isGame=false"
            f"&sort={urllib.parse.quote(sort_param)}"
            f"&start=0&limit=500"
            f"&factCayenneExp={urllib.parse.quote(fact_exp)}"
            f"&cayenneExp={urllib.parse.quote(cayenne_exp)}"
        )
        stats_resp = requests.get(stats_url, timeout=15)
        if stats_resp.ok:
            stats_data = stats_resp.json().get("data", [])
            pair_toi   = defaultdict(int)
            for row in stats_data:
                pid1 = _to_int_pid(row.get("playerId"))
                pid2 = _to_int_pid(row.get("withPlayerId"))
                toi  = row.get("timeOnIceWith", 0)
                if pid1 is None or pid2 is None or toi <= 0:
                    continue
                if d_pids and (pid1 not in d_pids or pid2 not in d_pids):
                    continue
                key = tuple(sorted([pid1, pid2]))
                pair_toi[key] = max(pair_toi[key], int(toi))
            if pair_toi:
                pairs = sorted(pair_toi.items(), key=lambda x: x[1], reverse=True)
                return [(p[0], p[1], toi) for (p, toi) in pairs], None
    except Exception:
        pass  # fall through to shift chart method

    # ── Method 2: Raw shift chart parsing ─────────────────────────────────────
    errors = []
    try:
        # Scan back week by week until we have at least 10 finished games
        TARGET_GAMES  = 10
        MAX_WEEKS_BACK = 12   # extended to cover playoffs / schedule gaps
        all_games  = []
        seen_ids   = set()

        for weeks_back in range(MAX_WEEKS_BACK):
            date_str = (datetime.date.today() - datetime.timedelta(weeks=weeks_back)).strftime("%Y-%m-%d")
            url = f"https://api-web.nhle.com/v1/club-schedule/{team_code}/week/{date_str}"
            try:
                resp = requests.get(url, timeout=10)
                if not resp.ok:
                    continue
                for g in resp.json().get("games", []):
                    gid = g.get("id")
                    if gid and gid not in seen_ids:
                        seen_ids.add(gid)
                        all_games.append(g)
            except Exception as e:
                errors.append(f"schedule week {date_str}: {e}")
                continue

            # Accept any game state that isn't clearly "not yet played"
            n_finished = sum(
                1 for g in all_games
                if str(g.get("gameState", "")).upper() not in ("FUT", "LIVE", "PRE", "CRIT", "")
                or str(g.get("gameState", "")).upper() in FINISHED_STATES
            )
            if n_finished >= TARGET_GAMES:
                break

        # Filter to finished games
        finished = sorted(
            [g for g in all_games
             if str(g.get("gameState", "")).upper() in FINISHED_STATES
             or str(g.get("gameState", "")).upper() not in ("FUT", "LIVE", "PRE", "")],
            key=lambda g: g.get("startTimeUTC", ""),
            reverse=True,
        )[:TARGET_GAMES]

        if not finished:
            return [], f"No finished games found in last {MAX_WEEKS_BACK} weeks."

        pair_toi = defaultdict(int)
        games_processed = 0

        for game in finished:
            game_id = game.get("id")
            if not game_id:
                continue
            try:
                shift_url  = f"https://api-web.nhle.com/v1/shiftcharts/{game_id}"
                shift_resp = requests.get(shift_url, timeout=10)
                shift_resp.raise_for_status()
                shifts_raw = shift_resp.json().get("data", [])

                # Identify which field holds the team abbreviation
                # (NHL API has used both "teamAbbrev" and "teamTriCode")
                sample = next((s for s in shifts_raw if s), {})
                team_field = None
                for candidate in ("teamAbbrev", "teamTriCode", "team", "teamCode"):
                    if candidate in sample:
                        team_field = candidate
                        break

                # Filter to this team's D-men. Try teamAbbrev first; if that
                # yields nothing, fall back to d_pids-only filter.
                if team_field:
                    team_d_shifts = [
                        s for s in shifts_raw
                        if s.get(team_field) == team_code
                        and (not d_pids or _to_int_pid(s.get("playerId")) in d_pids)
                    ]
                else:
                    team_d_shifts = []

                # Fallback: if team field filter got nothing, just use d_pids
                if not team_d_shifts and d_pids:
                    team_d_shifts = [
                        s for s in shifts_raw
                        if _to_int_pid(s.get("playerId")) in d_pids
                    ]

                if len(team_d_shifts) < 2:
                    continue  # not enough D-men to form a pair

                by_period = defaultdict(list)
                for s in team_d_shifts:
                    period = s.get("period") or s.get("periodNumber")
                    if period is not None:
                        by_period[period].append(s)

                for period_shifts in by_period.values():
                    period_shifts.sort(key=lambda x: _to_seconds(x.get("startTime") or x.get("shiftStart")) or 0)
                    n = len(period_shifts)
                    for i in range(n):
                        si      = period_shifts[i]
                        pid_i   = _to_int_pid(si.get("playerId"))
                        start_i = _to_seconds(si.get("startTime") or si.get("shiftStart"))
                        end_i   = _to_seconds(si.get("endTime")   or si.get("shiftEnd"))
                        if pid_i is None or start_i is None or end_i is None:
                            continue
                        for j in range(i + 1, n):
                            sj       = period_shifts[j]
                            sj_start = _to_seconds(sj.get("startTime") or sj.get("shiftStart"))
                            if sj_start is None:
                                continue
                            if sj_start >= end_i:
                                break
                            pid_j  = _to_int_pid(sj.get("playerId"))
                            sj_end = _to_seconds(sj.get("endTime") or sj.get("shiftEnd"))
                            if pid_j is None or sj_end is None or pid_i == pid_j:
                                continue
                            overlap = min(end_i, sj_end) - max(start_i, sj_start)
                            if overlap > 0:
                                pair_toi[tuple(sorted([pid_i, pid_j]))] += overlap

                games_processed += 1
            except Exception as e:
                errors.append(f"game {game_id}: {e}")
                continue

        if not pair_toi:
            err_detail = "; ".join(errors[:3]) if errors else f"{games_processed} games processed, no overlapping shifts found"
            return [], f"Shift data unavailable ({err_detail})."

        pairs = sorted(pair_toi.items(), key=lambda x: x[1], reverse=True)
        return [(p[0], p[1], toi) for (p, toi) in pairs], None

    except Exception as e:
        return [], str(e)


def build_actual_pairing_insertion(player_id, team_code, df, team_ctx,
                                    fit_models, player_profiles, has_age,
                                    feature_names=None):
    """
    Build pairing view using ACTUAL NHL pair combinations from shift data.
    Shows the real current pairs with model defensive scores, then inserts
    the searched player into their best fit pair slot.
    """
    # 1. Fetch current roster and actual pair combinations
    roster = def_fetch_team_roster_d(team_code)
    if not roster:
        return None, "Could not fetch roster.", None

    roster_pids = {p["player_id"]: p["player_name"] for p in roster}

    # Get all team predictions from model
    all_teams = def_get_latest_team_contexts(df, team_ctx)
    team_row  = all_teams[all_teams["player_team"] == team_code]
    if team_row.empty:
        return None, f"No team context found for {team_code}.", None
    team_row = team_row.iloc[0]

    if player_id not in player_profiles:
        return None, "Player not found in model data.", None

    # 2. Get model predictions for every rostered D-man
    player_scores = {}  # pid -> {metric: val, defensive_score: val, player_name: str}
    missing_model_players = []
    for pid, pname in roster_pids.items():
        if pid not in player_profiles:
            missing_model_players.append({"player_id": pid, "player_name": pname})
            continue
        prof, _ = player_profiles[pid]
        preds   = def_predict_for_team(prof, team_row, fit_models, has_age,
                                        feature_names=feature_names)
        score_df = def_compute_defensive_score(pd.DataFrame([preds]))
        player_scores[pid] = {
            **preds,
            "defensive_score": round(score_df["defensive_score"].iloc[0], 1),
            "player_name":     pname,
        }

    # Also score the searched player
    search_profile, _ = player_profiles[player_id]
    search_preds = def_predict_for_team(search_profile, team_row, fit_models, has_age,
                                         feature_names=feature_names)
    search_score_df = def_compute_defensive_score(pd.DataFrame([search_preds]))
    player_scores[player_id] = {
        **search_preds,
        "defensive_score": round(search_score_df["defensive_score"].iloc[0], 1),
        "player_name":     search_profile.get("player_name", "Selected Player"),
        "is_searched_player": True,
    }

    # 3. Fetch actual pairs from shift data
    actual_pairs, pair_err = fetch_actual_pairs(team_code)

    # 4. Build current pair view (without searched player)
    current_pairs = []
    assigned      = set()

    if actual_pairs:
        for (pid1, pid2, shared_toi) in actual_pairs:
            if pid1 not in roster_pids or pid2 not in roster_pids:
                continue
            if pid1 in assigned or pid2 in assigned:
                continue
            if pid1 == player_id or pid2 == player_id:
                continue  # exclude searched player from current pairs
            if pid1 not in player_scores or pid2 not in player_scores:
                continue
            current_pairs.append({
                "pid1":      pid1,
                "pid2":      pid2,
                "name1":     player_scores[pid1]["player_name"],
                "name2":     player_scores[pid2]["player_name"],
                "score1":    player_scores[pid1]["defensive_score"],
                "score2":    player_scores[pid2]["defensive_score"],
                "pair_score": round((player_scores[pid1]["defensive_score"] +
                                     player_scores[pid2]["defensive_score"]) / 2, 1),
                "shared_toi": shared_toi,
                "from_shifts": True,
            })
            assigned.add(pid1)
            assigned.add(pid2)

    # Any D-man not found in shift data — add as unassigned
    unassigned = [pid for pid in player_scores
                  if pid not in assigned and pid != player_id
                  and pid in roster_pids]

    # If real pair data isn't available, fall back to MoneyPuck model-ranked pairs.
    # These are clearly labelled so the user knows they're estimated, not actual.
    if not current_pairs:
        scored_roster = sorted(
            [(pid, info) for pid, info in player_scores.items()
             if pid != player_id and pid in roster_pids],
            key=lambda x: x[1]["defensive_score"],
            reverse=True,
        )
        for k in range(0, len(scored_roster) - 1, 2):
            pid1, info1 = scored_roster[k]
            pid2, info2 = scored_roster[k + 1]
            current_pairs.append({
                "pid1":       pid1,  "pid2":       pid2,
                "name1":      info1["player_name"], "name2": info2["player_name"],
                "score1":     info1["defensive_score"],
                "score2":     info2["defensive_score"],
                "pair_score": round((info1["defensive_score"] + info2["defensive_score"]) / 2, 1),
                "shared_toi": None,
                "from_shifts": False,  # clearly marks as estimated
            })
        unassigned = []

    # 5b. Sort current pairs by pair_score (best pair = 1st pair)
    current_pairs.sort(key=lambda x: x["pair_score"], reverse=True)
    for i, pair in enumerate(current_pairs):
        slot_num = i + 1
        pair["slot"] = f"{slot_num}{'st' if slot_num==1 else 'nd' if slot_num==2 else 'rd' if slot_num==3 else 'th'} Pair"
        pair["slot_color"] = DEF_PAIR_COLORS.get(pair["slot"].split()[0] + " Pair",
                             list(DEF_PAIR_COLORS.values())[-1])

    # 6. Find best partner for searched player using defensive score + Corsi style
    searched_score  = player_scores[player_id]["defensive_score"]
    searched_corsi  = float(player_scores[player_id].get("on_ice_corsi_pct", 0.50) or 0.50)
    best_partner_pid  = None
    best_partner_name = "—"
    best_partner_slot = "—"
    pushed_out_name   = "—"

    def _pair_compatibility(pair, searched_score, searched_corsi):
        """
        Score how well the searched player fits into this pair.
        Uses two signals from MoneyPuck data:
          1. Defensive score proximity — searched player should slot near the pair average
          2. Corsi style complementarity — a high-Corsi D pairs well with a lower-Corsi
             defensive D, and vice versa (complementary styles create balanced pairings)
        Returns a combined score (higher = better fit).
        """
        score_delta = abs(searched_score - pair["pair_score"])
        score_fit   = 1.0 / (1.0 + score_delta / 10.0)   # normalised 0-1

        # Get Corsi for both players in the pair
        c1 = float(player_scores[pair["pid1"]].get("on_ice_corsi_pct", 0.50) or 0.50)
        c2 = float(player_scores[pair["pid2"]].get("on_ice_corsi_pct", 0.50) or 0.50)
        partner_corsi = (c1 + c2) / 2.0
        # Ideal complement: searched Corsi + partner average ≈ 0.50 each, or complementary spread
        corsi_complement = 1.0 - abs(searched_corsi - (1.0 - partner_corsi))
        corsi_complement = max(corsi_complement, 0.0)

        # 65% defensive score fit, 35% Corsi style compatibility
        return score_fit * 0.65 + corsi_complement * 0.35

    if current_pairs:
        best_compat = -1.0
        for pair in current_pairs:
            s1, s2 = pair["score1"], pair["score2"]
            weaker_pid    = pair["pid1"] if s1 <= s2 else pair["pid2"]
            weaker_name   = pair["name1"] if s1 <= s2 else pair["name2"]
            stronger_pid  = pair["pid2"] if s1 <= s2 else pair["pid1"]
            stronger_name = pair["name2"] if s1 <= s2 else pair["name1"]
            compat = _pair_compatibility(pair, searched_score, searched_corsi)
            if compat > best_compat:
                best_compat       = compat
                best_partner_pid  = stronger_pid
                best_partner_name = stronger_name
                best_partner_slot = pair["slot"]
                pushed_out_name   = weaker_name
    elif unassigned:
        # No shift data — use defensive score + Corsi complement on unassigned D-men
        def _unassigned_compat(pid):
            other_score  = player_scores[pid]["defensive_score"]
            other_corsi  = float(player_scores[pid].get("on_ice_corsi_pct", 0.50) or 0.50)
            score_fit    = 1.0 / (1.0 + abs(searched_score - other_score) / 10.0)
            corsi_comp   = 1.0 - abs(searched_corsi - (1.0 - other_corsi))
            return score_fit * 0.65 + max(corsi_comp, 0.0) * 0.35

        best_partner_pid  = max(unassigned, key=_unassigned_compat)
        best_partner_name = player_scores[best_partner_pid]["player_name"]
        best_partner_slot = "Projected"

    return current_pairs, unassigned, player_scores, {
        "partner_name":       best_partner_name,
        "partner_slot":       best_partner_slot,
        "pushed_out_name":    pushed_out_name,
        "searched_score":     searched_score,
        "pair_err":           pair_err,
        "missing_model_players": missing_model_players,
        "best_partner_pid":   best_partner_pid,
    }


def def_build_pairing_insertion(player_id, team_code, df, team_ctx,
                             fit_models, player_profiles, has_age,
                             feature_names=None):
    """Wrapper that calls the actual-pair-based insertion."""
    return build_actual_pairing_insertion(
        player_id, team_code, df, team_ctx,
        fit_models, player_profiles, has_age, feature_names
    )



# ── Charts ─────────────────────────────────────────────────────────────────────

def def_make_bar_chart(results, player_name, actual_team, title):
    metric_cols   = ["ind_hits_pg", "ind_takeaways_pg",
                     "xga_per60_zone_adj", "pk_ice_pct", "defensive_score"]
    metric_labels = ["Hits / Game", "Takeaways / Game",
                     "xGA/60 (Zone Adj)", "PK Ice %", "Defensive Score"]

    fig, axes = plt.subplots(1, 5, figsize=(28, 8))
    fig.patch.set_facecolor("#0e1117")
    fig.suptitle(title, color="white", fontsize=13, y=1.01)

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        ax.set_facecolor("#0e1117")
        sr         = results.sort_values(col, ascending=(col in DEF_LOWER_IS_BETTER))
        bar_colors = ["#FFD700" if t == actual_team else "#4a90d9"
                      for t in sr["player_team"]]
        ax.barh(sr["player_team"], sr[col], color=bar_colors)
        actual_val = results.loc[results["player_team"] == actual_team, col].values[0]
        ax.axvline(actual_val, color="#c8102e", linestyle="--", linewidth=1.2)
        ax.set_xlabel(label, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        if col in DEF_LOWER_IS_BETTER:
            ax.invert_xaxis()

    actual_patch = mpatches.Patch(color="#FFD700", label=f"Actual team ({actual_team})")
    other_patch  = mpatches.Patch(color="#4a90d9", label="Other teams")
    axes[-1].legend(handles=[actual_patch, other_patch],
                    facecolor="#1a1a2e", labelcolor="white", loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def def_show_results_table(results, actual_team):
    display = results[[
        "player_team", "ind_hits_pg", "ind_takeaways_pg",
        "xga_per60_zone_adj", "pk_ice_pct", "pim_pg",
        "defensive_score", "is_actual"
    ]].copy()
    display.columns = [
        "Team", "Hits/GP", "TK/GP", "xGA/60 (adj)",
        "PK%", "PEN/GP", "Def Score", "Actual Team"
    ]
    for col in ["Hits/GP", "TK/GP", "xGA/60 (adj)", "PEN/GP", "Def Score"]:
        display[col] = display[col].round(3)
    display["PK%"] = (display["PK%"] * 100).round(1)
    st.dataframe(
        display.style.apply(
            lambda row: ["background-color: #3a1a1a" if row["Actual Team"] else "" for _ in row],
            axis=1,
        ),
        use_container_width=True, height=400,
    )
    return display


def def_show_metrics(metrics, label):
    st.markdown(f"**{label} model quality ({CV_FOLDS}-fold CV)**")
    for target in DEF_TARGETS:
        mae_mean, mae_std   = metrics[target]["mae"]
        rmse_mean, rmse_std = metrics[target]["rmse"]
        direction = "↓ lower is better" if target in DEF_LOWER_IS_BETTER else "↑ higher is better"
        st.markdown(f"*{DEF_TARGET_LABELS[target]}* — {direction}")
        c1, c2, _ = st.columns(3)
        c1.metric("MAE",  f"{mae_mean:.3f}", f"± {mae_std:.3f}")
        c2.metric("RMSE", f"{rmse_mean:.3f}", f"± {rmse_std:.3f}")

# ── Contract evaluator ────────────────────────────────────────────────────────

# NHL age curves — empirical annual decline rates by age bracket
# Derived from typical production curves in hockey analytics literature
OFF_AGE_CURVE = {
    # age: annual multiplier on production (1.0 = no change)
    (18, 22): 1.05,   # rapid development
    (23, 25): 1.02,   # continued growth
    (26, 28): 1.00,   # peak years
    (29, 30): 0.97,   # early decline
    (31, 32): 0.94,   # moderate decline
    (33, 34): 0.90,   # significant decline
    (35, 99): 0.85,   # steep decline
}

DEF_AGE_CURVE = {
    (18, 22): 1.04,
    (23, 25): 1.02,
    (26, 29): 1.00,   # defensemen peak slightly later
    (30, 31): 0.97,
    (32, 33): 0.94,
    (34, 35): 0.90,
    (36, 99): 0.85,
}

# Confidence decay per year beyond year 1
CONFIDENCE_DECAY = {1: 1.0, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.40}


def get_age_multiplier(age, is_defenseman=False):
    """Return the annual production multiplier for a given age."""
    curve = DEF_AGE_CURVE if is_defenseman else OFF_AGE_CURVE
    for (lo, hi), mult in curve.items():
        if lo <= age <= hi:
            return mult
    return 0.85


def age_profile(profile, years_ahead, is_defenseman=False):
    """
    Return a copy of the player profile aged forward by years_ahead.
    Applies age curve to skill features and updates history features.
    """
    p = profile.copy()
    current_age = float(p.get("age", 28))

    # Compound the age multiplier over each year
    cumulative_mult = 1.0
    for y in range(years_ahead):
        age_y = current_age + y
        cumulative_mult *= get_age_multiplier(age_y, is_defenseman)

    # Update age fields
    new_age = current_age + years_ahead
    p["age"]    = new_age
    p["age_sq"] = new_age ** 2

    # Apply multiplier to skill rate features
    skill_cols = (
        ["ind_hits_per60", "ind_takeaways_per60", "ind_giveaways_per60",
         "shots_blocked_by_player_per60", "xg_against_per60_5v5",
         "hd_shots_against_per60_5v5", "on_ice_corsi_pct"]
        if is_defenseman else
        ["finishing_skill", "finishing_skill_adj", "ind_shot_attempts_per60",
         "ind_high_danger_shots_per60", "ind_medium_danger_shots_per60",
         "ind_points_per60", "ind_goals_per60"]
    )
    for col in skill_cols:
        if col in p.index:
            p[col] = float(p[col]) * cumulative_mult

    # Update career history features to reflect the aged season
    if is_defenseman:
        for col in ["prev_season_hits_pg", "recent_3yr_mean_hits_pg",
                    "prev_season_takeaways_pg", "recent_3yr_mean_takeaways_pg",
                    "prev_season_xga_pg", "recent_3yr_mean_xga_pg",
                    "prev_season_pk_pct", "recent_3yr_mean_pk_pct"]:
            if col in p.index:
                p[col] = float(p[col]) * cumulative_mult
    else:
        for col in ["prev_season_points_pg", "recent_3yr_mean_points_pg",
                    "prev_season_goals_pg", "recent_3yr_mean_goals_pg",
                    "career_prev_mean_points_pg", "career_prev_mean_goals_pg"]:
            if col in p.index:
                p[col] = float(p[col]) * cumulative_mult

    # Advance career year
    if "career_seasons_prior" in p.index:
        p["career_seasons_prior"] = float(p.get("career_seasons_prior", 5)) + years_ahead
    if "career_year" in p.index:
        p["career_year"] = float(p.get("career_year", 5)) + years_ahead

    # Update age interactions
    if "age_x_shot_attempts" in p.index:
        p["age_x_shot_attempts"] = new_age * float(p.get("ind_shot_attempts_per60", 0))
    if "age_x_finishing" in p.index:
        p["age_x_finishing"] = new_age * float(p.get("finishing_skill_adj", 0))
    if "age_x_hd_share" in p.index:
        p["age_x_hd_share"] = new_age * float(p.get("hd_shot_share", 0))
    if "age_x_hits" in p.index:
        p["age_x_hits"] = new_age * float(p.get("ind_hits_per60", 0))
    if "age_x_takeaways" in p.index:
        p["age_x_takeaways"] = new_age * float(p.get("ind_takeaways_per60", 0))
    if "age_x_pk" in p.index:
        p["age_x_pk"] = new_age * float(p.get("pk_ice_pct", 0))

    # Ensure all expected non-age feature columns are present (fill missing with 0)
    # Note: age features are excluded here — they are only included at prediction time
    # if has_age=True, matching exactly how the model was trained.
    all_expected = (PLAYER_FEATURES + TEAM_FEATURES
                    if not is_defenseman else
                    DEF_PLAYER_FEATURES + DEF_TEAM_FEATURES)
    for col in all_expected:
        if col not in p.index:
            p[col] = 0.0

    return p


def build_contract_projection(player_name, pred, dpred, df, team_ctx,
                               fit_models, next_models, player_profiles, has_age,
                               def_df, def_team_ctx, def_fit_models,
                               def_player_profiles, def_has_age,
                               team, n_years, def_fit_feature_names=None):
    """
    Build a multi-year contract projection for a player on a specific team.
    Works for both forwards (pred) and defensemen (dpred).

    Returns a list of dicts, one per year, with predicted stats and confidence.
    """
    is_d = pred is not None and pred.get("position") == "D"

    if is_d:
        if dpred is None:
            return None, "Defensive model not loaded."
        pid     = dpred["pid"]
        profile = dpred["profile"]
        matched = dpred["matched"]
        age     = float(profile.get("age", 28))

        # Get team context for defensemen
        latest_ctx = def_get_latest_team_contexts(def_df, def_team_ctx)
        team_row   = latest_ctx[latest_ctx["player_team"] == team]
        if team_row.empty:
            return None, f"No defensive team context for {team}."
        team_row = team_row.iloc[0]

    else:
        if pred is None or pred.get("fit_results") is None:
            return None, "Offensive model not loaded or player not found."
        pid      = pred["pid"]
        profile  = player_profiles[pid][0]
        matched  = pred["matched"]
        age      = float(pred.get("age") or profile.get("age", 28) or 28)

        # Get team context for forwards
        all_teams = get_latest_team_contexts(df, team_ctx)
        pos       = profile["position"]
        team_row_df = all_teams[
            (all_teams["player_team"] == team) &
            (all_teams["position"] == pos)
        ]
        if team_row_df.empty:
            return None, f"No offensive team context for {team}."
        team_row = team_row_df.iloc[0]

    league_env = get_latest_league_env(df)
    rows = []

    for year in range(1, n_years + 1):
        aged = age_profile(profile, year - 1, is_defenseman=is_d)
        conf  = CONFIDENCE_DECAY.get(year, 0.35)
        age_y = age + year - 1
        mult  = 1.0
        for y in range(year - 1):
            mult *= get_age_multiplier(age + y, is_d)

        if is_d:
            # Set team context
            for col in DEF_TEAM_FEATURES:
                if col in team_row.index:
                    aged[col] = team_row[col]
            preds = def_predict_for_team(aged, team_row, def_fit_models, def_has_age,
                                            feature_names=def_fit_feature_names)
            row = {
                "year":           year,
                "age":            round(age_y, 0),
                "confidence":     conf,
                "hits_pg":        round(max(preds.get("ind_hits_pg", 0), 0), 2),
                "takeaways_pg":   round(max(preds.get("ind_takeaways_pg", 0), 0), 3),
                "goals_against_pg": round(max(preds.get("xga_per60_zone_adj") or preds.get("goals_against_per60", 0), 0), 3),
                "pk_pct":         round(max(preds.get("pk_ice_pct", 0), 0), 3),
                "pim_pg":         round(max(preds.get("pim_pg", 0), 0), 3),
                "def_score":      None,
            }
            # Compute defensive score for the year
            from_preds = {"ind_hits_pg": row["hits_pg"],
                          "ind_takeaways_pg": row["takeaways_pg"],
                          "xga_per60_zone_adj": row["goals_against_pg"],
                          "pk_ice_pct": row["pk_pct"],
                          "pim_pg": row["pim_pg"]}
            tmp = pd.DataFrame([from_preds])
            row["def_score"] = round(def_compute_defensive_score(tmp)["defensive_score"].iloc[0], 1)

        else:
            # Forward — set team and league context
            aged_row = aged.copy()
            for col in TEAM_FEATURES:
                if col in team_row.index:
                    aged_row[col] = team_row[col]
                elif col not in aged_row.index:
                    aged_row[col] = 0.0   # ensure all TEAM_FEATURES present
            for k, v in league_env.items():
                aged_row[k] = v
            # Ensure position dummies can be built
            if "position" not in aged_row.index:
                aged_row["position"] = profile.get("position", "C")

            pred_df    = pd.DataFrame([aged_row])
            X          = _make_X_from_profile(aged_row, has_age)

            base_pts   = compute_target_baseline(pred_df, "points_per_game").values[0]
            base_goals = compute_target_baseline(pred_df, "goals_per_game").values[0]
            base_gs    = compute_target_baseline(pred_df, "game_score_per_game").values[0]

            pred_pts   = max(base_pts   + fit_models["points_per_game"]["global"].predict(X)[0], 0)
            pred_goals = max(base_goals + fit_models["goals_per_game"]["global"].predict(X)[0], 0)
            pred_gs    = max(base_gs    + fit_models["game_score_per_game"]["global"].predict(X)[0], 0)

            row = {
                "year":       year,
                "age":        round(age_y, 0),
                "confidence": conf,
                "points_pg":  round(pred_pts,   3),
                "goals_pg":   round(pred_goals,  3),
                "gs_pg":      round(pred_gs,     3),
            }

        rows.append(row)

    return rows, None


def get_cba_limits(current_age, actual_team, signing_team):
    """
    Return CBA hard limits for contract length.
      - Same team (re-signing):   max 7 years
      - New team (UFA/trade):     max 6 years
      - Age 35+ at signing:       flags cap recapture risk (35+ rule)
    Note: 'recommended' here is an age-based floor used before projections are
    available. Call recommend_contract_length() for the performance-aware value.
    """
    is_same_team  = actual_team == signing_team
    max_years     = 7 if is_same_team else 6
    is_35_signing = current_age >= 35
    age_at_expiry = current_age + max_years
    hits_35_rule  = (current_age + 1) >= 35

    # Hard age cap — absolute ceiling regardless of projections
    if current_age >= 35:
        age_cap = 1
    elif current_age >= 33:
        age_cap = 2
    elif current_age >= 31:
        age_cap = 3
    else:
        age_cap = max_years

    return {
        "max_years":       max_years,
        "age_cap":         min(age_cap, max_years),
        "recommended":     min(age_cap, max_years),  # overridden by recommend_contract_length()
        "is_same_team":    is_same_team,
        "is_35_signing":   is_35_signing,
        "hits_35_rule":    hits_35_rule,
        "age_at_expiry":   age_at_expiry,
    }


def get_roster_cutline(team_code, is_d, df, team_ctx, fit_models, player_profiles, has_age,
                       def_df=None, def_team_ctx=None, def_fit_models=None,
                       def_player_profiles=None, def_has_age=False, def_feature_names=None):
    """
    Return the model score of the last player who would dress for team_code:
      - Forwards:   points_pg of the 12th-ranked forward  (4th-line threshold)
      - Defensemen: defensive_score of the 6th-ranked D-man (3rd-pair threshold)

    A contract year where the signed player projects below this score means they
    would no longer hold a roster spot on this team.

    Returns (cutline_score, cutline_label) or (None, None) on failure.
    """
    ROSTER_SPOTS = 6 if is_d else 12  # last spot that dresses

    if is_d:
        if not def_fit_models or not def_player_profiles:
            return None, None
        try:
            roster = def_fetch_team_roster_d(team_code)
            if not roster:
                return None, None

            all_teams = def_get_latest_team_contexts(def_df, def_team_ctx)
            team_row  = all_teams[all_teams["player_team"] == team_code]
            if team_row.empty:
                return None, None
            team_row = team_row.iloc[0]

            scores = []
            for p in roster:
                pid = p["player_id"]
                if pid not in def_player_profiles:
                    continue
                prof, _ = def_player_profiles[pid]
                preds   = def_predict_for_team(prof, team_row, def_fit_models,
                                               def_has_age, feature_names=def_feature_names)
                score_df = def_compute_defensive_score(pd.DataFrame([preds]))
                scores.append(round(score_df["defensive_score"].iloc[0], 1))

            if len(scores) < ROSTER_SPOTS:
                return None, None

            scores.sort(reverse=True)
            cutline = scores[ROSTER_SPOTS - 1]  # score of the last D-man who dresses
            return cutline, "3rd-pair threshold (6th D-man)"
        except Exception:
            return None, None

    else:  # forwards
        try:
            roster_df, err = fetch_active_team_roster(team_code)
            if err or roster_df is None:
                return None, None

            latest_ctx  = get_latest_team_contexts(df, team_ctx)
            league_env  = get_latest_league_env(df)

            # Use center context as the forward baseline (most common position)
            team_row = latest_ctx[
                (latest_ctx["player_team"] == team_code) &
                (latest_ctx["position"] == "C")
            ]
            if team_row.empty:
                return None, None
            team_row = team_row.iloc[0]

            fwd_positions = {"C", "L", "R"}
            scores = []
            for _, rp in roster_df.iterrows():
                if rp.get("position") not in fwd_positions:
                    continue
                pid = int(rp["player_id"])
                if pid not in player_profiles:
                    continue
                prof, _ = player_profiles[pid]
                row = prof.copy()
                for col in TEAM_FEATURES:
                    if col in team_row.index:
                        row[col] = team_row[col]
                for k, v in league_env.items():
                    row[k] = v
                X        = _make_X_from_profile(row, has_age)
                baseline = compute_target_baseline(pd.DataFrame([row]), "points_per_game").values[0]
                raw      = fit_models["points_per_game"]["global"].predict(X)[0]
                scores.append(float(np.clip(baseline + raw, 0, None)))

            if len(scores) < ROSTER_SPOTS:
                return None, None

            scores.sort(reverse=True)
            cutline = scores[ROSTER_SPOTS - 1]  # score of the last forward who dresses
            return cutline, "4th-line threshold (12th forward)"
        except Exception:
            return None, None


def recommend_contract_length(rows, is_d, cba_max, current_age, roster_cutline=None, cutline_label=None):
    """
    Recommend contract length from actual projected performance values.

    Logic:
      1. Find the last year the player projects at >= 80% of their year-1 value
         (the 'value retention' cutoff).
      2. Detect cliff years — any year where the single-year drop exceeds
         10% (D) or 12% (F). Cap the recommendation before the cliff.
      3. If the player is ascending (year-N > year-1), allow up to CBA max.
      4. Hard-cap by age: 35+ → 1 yr, 33-34 → 2 yrs, 31-32 → 3 yrs.
         This overrides the performance signal for extreme ages.

    Returns (recommended_years, explanation_string).
    """
    if not rows:
        return min(2, cba_max), "No projection data — defaulting to conservative estimate."

    score_key = "def_score" if is_d else "points_pg"
    y1_val    = rows[0].get(score_key, 0)

    if y1_val <= 0:
        return 1, "Year 1 projected value is zero — cannot assess contract length."

    # ── Step 1: value-retention cutoff (last year >= 80% of year-1) ──────────
    RETENTION_FLOOR = 0.80
    retention_cutoff = 1
    for row in rows:
        val = row.get(score_key, 0)
        if (val / y1_val) >= RETENTION_FLOOR:
            retention_cutoff = row["year"]
        # Don't break early — a plateau after a dip still counts

    # ── Step 2: cliff detection ───────────────────────────────────────────────
    MAX_YOY_DROP = 0.10 if is_d else 0.12
    cliff_year = cba_max  # assume no cliff unless found
    for i in range(1, len(rows)):
        prev_val = rows[i - 1].get(score_key, 0)
        curr_val = rows[i].get(score_key, 0)
        if prev_val > 0 and (prev_val - curr_val) / prev_val > MAX_YOY_DROP:
            cliff_year = rows[i - 1]["year"]  # recommend stopping before the cliff
            break

    # ── Step 3: ascending player gets full runway ─────────────────────────────
    last_val = rows[-1].get(score_key, 0)
    if last_val >= y1_val:
        perf_rec = cba_max
        reason   = "ascending trajectory — production projected to improve or hold steady"
    else:
        perf_rec = min(retention_cutoff, cliff_year)
        decline_pct = (y1_val - last_val) / y1_val * 100
        reason = (
            f"production projected to decline ~{decline_pct:.0f}% over {cba_max} years; "
            f"value retention holds through year {retention_cutoff}"
            + (f", cliff detected at year {cliff_year}" if cliff_year < cba_max else "")
        )

    # ── Step 4: hard age cap ──────────────────────────────────────────────────
    if current_age >= 35:
        age_hard_cap, age_note = 1, "age 35+ hard cap"
    elif current_age >= 33:
        age_hard_cap, age_note = 2, "age 33-34 hard cap"
    elif current_age >= 31:
        age_hard_cap, age_note = 3, "age 31-32 hard cap"
    else:
        age_hard_cap, age_note = cba_max, None

    recommended = min(perf_rec, age_hard_cap, cba_max)
    recommended = max(recommended, 1)  # always at least 1

    # ── Step 5: roster cutline ───────────────────────────────────────────────
    # If the player would fall below the team's 4th-line/3rd-pair threshold,
    # the contract should expire before that year — they won't hold a spot.
    cutline_hit_year = None
    if roster_cutline is not None:
        for row in rows:
            val = row.get(score_key, 0)
            if val < roster_cutline:
                cutline_hit_year = row["year"]
                break

    if cutline_hit_year is not None:
        cutline_rec = max(cutline_hit_year - 1, 1)
        lbl = cutline_label or ("3rd-pair" if is_d else "4th-line")
        if cutline_rec < recommended:
            recommended = cutline_rec
            if age_hard_cap < cutline_rec:
                explanation = (
                    f"Recommended {recommended} yr(s) — {age_note} overrides; "
                    f"player also projects below {lbl} threshold in year {cutline_hit_year}."
                )
            else:
                explanation = (
                    f"Recommended {recommended} yr(s) — player projected below {lbl} threshold "
                    f"starting year {cutline_hit_year}, so contract should not extend beyond year {cutline_rec}."
                )
        else:
            # Cutline not a binding constraint — mention it passes cleanly
            extra = f" Player stays above {lbl} threshold for the full term."
            if age_hard_cap < perf_rec:
                explanation = f"Recommended {recommended} yr(s) — {age_note} overrides projections ({reason}).{extra}"
            else:
                explanation = f"Recommended {recommended} yr(s) — {reason}.{extra}"
    elif age_hard_cap < perf_rec:
        explanation = f"Recommended {recommended} yr(s) — {age_note} overrides projections ({reason})."
    else:
        explanation = f"Recommended {recommended} yr(s) — {reason}."

    return recommended, explanation


def contract_risk_rating(rows, is_d):
    """
    Assess contract risk based on predicted performance trajectory.
    Returns (rating_label, rating_color, explanation).
    """
    if not rows:
        return "Unknown", "#888888", ""

    year1 = rows[0]
    last  = rows[-1]
    n     = len(rows)

    score_key   = "def_score" if is_d else "points_pg"
    y1_val      = year1.get(score_key, 50 if is_d else 0)
    yn_val      = last.get(score_key,  50 if is_d else 0)
    age_yr1     = year1["age"]
    decline_pct = (y1_val - yn_val) / max(y1_val, 0.01)

    # Check for a single large cliff year in the middle
    max_yoy_drop = 0.0
    for i in range(1, len(rows)):
        prev = rows[i - 1].get(score_key, 0)
        curr = rows[i].get(score_key, 0)
        if prev > 0:
            max_yoy_drop = max(max_yoy_drop, (prev - curr) / prev)

    if age_yr1 >= 35:
        return "Very High Risk", "#c8102e", f"Age {age_yr1:.0f} at signing — steep decline modeled. 35+ rule applies."
    elif decline_pct > 0.25 or max_yoy_drop > 0.15:
        return "High Risk", "#e8622a", (
            f"Age {age_yr1:.0f} — model projects {decline_pct*100:.0f}% total decline over {n} years"
            + (f" with a single-year drop of {max_yoy_drop*100:.0f}%" if max_yoy_drop > 0.15 else "") + "."
        )
    elif decline_pct > 0.12 or (age_yr1 >= 30 and decline_pct > 0.06):
        return "Moderate Risk", "#FFD700", f"Age {age_yr1:.0f} — model projects {decline_pct*100:.0f}% decline; manageable with right length."
    elif decline_pct <= 0:
        return "Low Risk", "#57a85a", f"Age {age_yr1:.0f} — ascending player, production projected to improve."
    else:
        return "Low Risk", "#57a85a", f"Age {age_yr1:.0f} — stable production projected through contract."