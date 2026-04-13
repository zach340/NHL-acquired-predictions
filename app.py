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
    raw_targets = ["game_score_per_game", "points_per_game", "goals_per_game", "ice_time", "games_played"]
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
            df["penalties_pg"] = (df["penalty_minutes"] / 2) / gp
        else:
            df["penalties_pg"] = np.nan

        return df.fillna(0), None

    except Exception as e:
        return None, str(e)


def build_defensive_validation(actual_df, def_df, def_team_ctx,
                                def_fit_models, def_player_profiles, def_has_age):
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
            profile, team_row.iloc[0], def_fit_models, def_has_age
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
    1: "1st Line", 2: "1st Line",
    3: "2nd Line", 4: "2nd Line",
    5: "3rd Line", 6: "3rd Line",
    7: "4th Line", 8: "4th Line",
    9: "4th Line", 10: "4th Line",
    11: "4th Line", 12: "4th Line",
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
    position = searched_profile.get("position", "C")
    is_fwd   = position in ("C", "L", "R")
    pos_group = {"C", "L", "R"} if is_fwd else {"D"}

    team_row = latest_ctx[
        (latest_ctx["player_team"] == team_code) &
        (latest_ctx["position"] == position)
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

    rows = []

    # Add searched player
    rows.append({
        "player_id":           player_id,
        "player_name":         searched_profile.get("player_name", "Selected Player"),
        "position":            position,
        "pred_points_gp":      predict_pts(searched_profile),
        "pred_goals_gp":       predict_goals(searched_profile),
        "is_searched_player":  True,
    })

    # Add rostered players in same position group
    for _, rp in roster_df.iterrows():
        pid = int(rp["player_id"])
        if rp.get("position") not in pos_group:
            continue
        if pid == player_id:
            continue  # already added above
        if pid not in player_profiles:
            continue
        profile, _ = player_profiles[pid]
        rows.append({
            "player_id":          pid,
            "player_name":        rp["player_name"],
            "position":           rp.get("position", position),
            "pred_points_gp":     predict_pts(profile),
            "pred_goals_gp":      predict_goals(profile),
            "is_searched_player": False,
        })

    if not rows:
        return None, "No players could be matched."

    result = (
        pd.DataFrame(rows)
        .sort_values("pred_points_gp", ascending=False)
        .reset_index(drop=True)
    )
    result["rank"] = result.index + 1

    slot_map = FWD_SLOT_MAP if is_fwd else DEF_SLOT_MAP
    result["lineup_slot"] = result["rank"].apply(
        lambda r: slot_map.get(r, "4th Line" if is_fwd else "3rd Pair (extra)")
    )
    result["slot_color"] = result["lineup_slot"].map(SLOT_COLORS).fillna("#888888")
    result["pred_points_gp"] = result["pred_points_gp"].round(3)
    result["pred_goals_gp"]  = result["pred_goals_gp"].round(3)

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
    d3.metric("Goals Against / 60", fmt(row.get("goals_against_per_game"), 2),
              f"p{pct_rank('goals_against_per_game', higher_better=False):.0f}"
              if pct_rank("goals_against_per_game") is not None else None)
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
    "goals_against_per_game",
    "pk_ice_pct",
    "penalties_pg",
]

DEF_TARGET_LABELS = {
    "ind_hits_pg":              "Hits / Game",
    "ind_takeaways_pg":         "Takeaways / Game",
    "goals_against_per_game":   "Goals Against / Game",
    "pk_ice_pct":               "PK Ice Time %",
    "penalties_pg":             "Penalties / Game",
}

# Lower is better for these targets
DEF_LOWER_IS_BETTER = {"goals_against_per_game", "penalties_pg"}

# Defensive score weights (used for pairing)
DEF_SCORE_WEIGHTS = {
    "ind_hits_pg":              0.15,
    "ind_takeaways_pg":         0.20,
    "goals_against_per_game":   0.30,  # most important
    "pk_ice_pct":               0.20,
    "penalties_pg":             0.15,
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
    "goals_against_per_game": ["prev_season_xga_pg",      "recent_3yr_mean_xga_pg",      "career_prev_mean_xga_pg"],
    "pk_ice_pct":              ["prev_season_pk_pct",      "recent_3yr_mean_pk_pct",      "career_prev_mean_pk_pct"],
    "penalties_pg":            ["prev_season_penalties_pg","recent_3yr_mean_penalties_pg","career_prev_mean_penalties_pg"],
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
    "goals_against_per_game",
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
    "prev_season_penalties_pg",
    "recent_3yr_mean_hits_pg",
    "recent_3yr_mean_takeaways_pg",
    "recent_3yr_mean_xga_pg",
    "recent_3yr_mean_pk_pct",
    "recent_3yr_mean_penalties_pg",
    "career_prev_mean_hits_pg",
    "career_prev_mean_takeaways_pg",
    "career_prev_mean_xga_pg",
    "career_prev_mean_pk_pct",
    "career_prev_mean_penalties_pg",
    "career_seasons_prior",
    # Slopes
    "recent_3yr_hits_slope",
    "recent_3yr_takeaways_slope",
    "recent_3yr_xga_slope",
    "recent_3yr_pk_slope",
    "recent_3yr_penalties_slope",
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
    "team_avg_penalties_pg",
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

    # Goals against per game (cleaner than per-60 — same scale as EDGE/API)
    if "on_ice_against_goals" in d.columns and "games_played" in d.columns:
        gp_safe = d["games_played"].replace(0, np.nan)
        d["goals_against_per_game"] = d["on_ice_against_goals"] / gp_safe

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
        ("goals_against_per_game", "xga_pg"),
        ("pk_ice_pct",              "pk_pct"),
        ("penalties_pg",            "penalties_pg"),
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
    d["yoy_xga_delta"]       = g["goals_against_per_game"].diff()

    return d


def def_build_team_context(df):
    """Aggregate team-level defensive context per season."""
    team_ctx = (
        df.groupby(["player_team", "season"])
        .agg(
            team_avg_hits_pg         = ("ind_hits_pg",             "mean"),
            team_avg_takeaways_pg    = ("ind_takeaways_pg",        "mean"),
            team_avg_xga_per60       = ("goals_against_per_game", "mean"),
            team_avg_pk_pct          = ("pk_ice_pct",              "mean"),
            team_avg_penalties_pg    = ("penalties_pg",            "mean"),
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

    ages = pd.read_csv(ages_path)[["player_id", "season", "age", "age_sq"]]
    df   = df.merge(ages, on=["player_id", "season"], how="left")
    has_age = df["age"].notna().mean() > 0.5
    advance(f"Data loaded — {len(df):,} defenseman-seasons | age matched: {df['age'].notna().sum():,}")

    # Engineer
    status.markdown("⚙️ **Engineering features...**")
    df       = def_engineer_features(df)
    df       = def_engineer_career_history(df)
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

def def_predict_for_team(profile, team_row, models, has_age, use_traj=False, df_ref=None):
    """Predict all 5 targets for a player on a specific team context."""
    row = profile.copy()
    for col in DEF_TEAM_FEATURES:
        if col in team_row.index:
            row[col] = team_row[col]

    pred_df = pd.DataFrame([row])
    if use_traj:
        traj    = [f for f in DEF_TRAJECTORY_FEATURES if f in pred_df.columns]
        feats   = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + traj + DEF_TEAM_FEATURES
    else:
        feats   = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + DEF_TEAM_FEATURES
    feats   = [f for f in feats if f in pred_df.columns]
    X       = pred_df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)

    preds = {}
    for target, model_dict in models.items():
        baseline = def_compute_target_baseline(pred_df, target).values[0]
        raw      = model_dict["global"].predict(X)[0]
        preds[target] = float(np.clip(baseline + raw, 0, None))
    return preds


def defdef_build_all_team_predictions(profile, all_teams, models, has_age, use_traj=False):
    """Predict all 5 targets for a player across all 32 teams."""
    rows = []
    for _, team_row in all_teams.iterrows():
        preds = def_predict_for_team(profile, team_row, models, has_age, use_traj)
        preds["player_team"] = team_row["player_team"]
        rows.append(preds)
    return pd.DataFrame(rows)


def def_compute_defensive_score(df_preds, season_df=None):
    """
    Compute a composite defensive score 0-100 for each team prediction.
    Normalises each metric to 0-1 then applies weights.
    """
    result = df_preds.copy()
    score  = np.zeros(len(result))

    for target, weight in DEF_SCORE_WEIGHTS.items():
        if target not in result.columns:
            continue
        vals = result[target].values.astype(float)
        rng  = vals.max() - vals.min()
        if rng == 0:
            norm = np.zeros(len(vals))
        else:
            norm = (vals - vals.min()) / rng
        # Invert for lower-is-better metrics
        if target in DEF_LOWER_IS_BETTER:
            norm = 1 - norm
        score += norm * weight

    result["defensive_score"] = score * 100
    return result


def def_predict_defenseman(player_name, df, team_ctx, fit_models, next_models,
                        player_profiles, has_age):
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
    fit_results  = def_build_all_team_predictions(profile, all_teams, fit_models, has_age)
    fit_results  = def_compute_defensive_score(fit_results)
    fit_results  = fit_results.sort_values("defensive_score", ascending=False).reset_index(drop=True)
    fit_results.index += 1
    fit_results["is_actual"] = fit_results["player_team"] == actual_team

    # Next season predictions
    next_results = def_build_all_team_predictions(profile, all_teams, next_models, has_age, use_traj=True)
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


def def_build_pairing_insertion(player_id, team_code, df, team_ctx,
                             fit_models, player_profiles, has_age):
    """
    Insert the searched D-man into a team's roster.
    Returns ranked DataFrame with pair slots and pairing partner.
    """
    roster = def_fetch_team_roster_d(team_code)
    if not roster:
        return None, "Could not fetch roster."

    all_teams = def_get_latest_team_contexts(df, team_ctx)
    team_row  = all_teams[all_teams["player_team"] == team_code]
    if team_row.empty:
        return None, f"No team context found for {team_code}."
    team_row = team_row.iloc[0]

    if player_id not in player_profiles:
        return None, "Player not found in model data."

    rows = []

    # Searched player
    profile, _ = player_profiles[player_id]
    preds      = def_predict_for_team(profile, team_row, fit_models, has_age)
    preds["player_id"]          = player_id
    preds["player_name"]        = profile.get("player_name", "Selected Player")
    preds["is_searched_player"] = True
    rows.append(preds)

    # Rostered D-men
    for p in roster:
        pid = p["player_id"]
        if pid == player_id:
            continue
        if pid not in player_profiles:
            continue
        prof, _ = player_profiles[pid]
        preds_r = def_predict_for_team(prof, team_row, fit_models, has_age)
        preds_r["player_id"]          = pid
        preds_r["player_name"]        = p["player_name"]
        preds_r["is_searched_player"] = False
        rows.append(preds_r)

    if len(rows) < 2:
        return None, "Not enough players matched for pairing."

    result = pd.DataFrame(rows)
    result = def_compute_defensive_score(result)
    result = result.sort_values("defensive_score", ascending=False).reset_index(drop=True)
    result["rank"]        = result.index + 1
    result["pair_slot"]   = result["rank"].apply(
        lambda r: DEF_PAIR_SLOTS.get(r, "3rd Pair (extra)")
    )
    result["slot_color"]  = result["pair_slot"].map(DEF_PAIR_COLORS).fillna("#888888")

    # Pairing partner — player ranked adjacent in the same pair slot
    searched_rank = result[result["is_searched_player"]]["rank"].iloc[0]
    if searched_rank % 2 == 1:
        partner_rank = searched_rank + 1
    else:
        partner_rank = searched_rank - 1
    partner_rows = result[result["rank"] == partner_rank]
    partner_name = partner_rows["player_name"].iloc[0] if not partner_rows.empty else "—"

    return result, None, partner_name


# ── Charts ─────────────────────────────────────────────────────────────────────

def def_make_bar_chart(results, player_name, actual_team, title):
    metric_cols   = ["ind_hits_pg", "ind_takeaways_pg",
                     "goals_against_per_game", "pk_ice_pct", "defensive_score"]
    metric_labels = ["Hits / Game", "Takeaways / Game",
                     "Goals Against / 60", "PK Ice %", "Defensive Score"]

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
        "goals_against_per_game", "pk_ice_pct", "penalties_pg",
        "defensive_score", "is_actual"
    ]].copy()
    display.columns = [
        "Team", "Hits/GP", "TK/GP", "GA/GP",
        "PK%", "PEN/GP", "Def Score", "Actual Team"
    ]
    for col in ["Hits/GP", "TK/GP", "GA/60 (5v5)", "PEN/GP", "Def Score"]:
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

    return p


def build_contract_projection(player_name, pred, dpred, df, team_ctx,
                               fit_models, next_models, player_profiles, has_age,
                               def_df, def_team_ctx, def_fit_models,
                               def_player_profiles, def_has_age,
                               team, n_years):
    """
    Build a multi-year contract projection for a player on a specific team.
    Works for both forwards (pred) and defensemen (dpred).

    Returns a list of dicts, one per year, with predicted stats and confidence.
    """
    is_d = pred is not None and pred.get("position") == "D"

    if is_d:
        if dpred is None or not def_models_loaded:
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
            preds = def_predict_for_team(aged, team_row, def_fit_models, def_has_age)
            row = {
                "year":           year,
                "age":            round(age_y, 0),
                "confidence":     conf,
                "hits_pg":        round(max(preds.get("ind_hits_pg", 0), 0), 2),
                "takeaways_pg":   round(max(preds.get("ind_takeaways_pg", 0), 0), 3),
                "goals_against_pg": round(max(preds.get("goals_against_per_game", 0), 0), 3),
                "pk_pct":         round(max(preds.get("pk_ice_pct", 0), 0), 3),
                "penalties_pg":   round(max(preds.get("penalties_pg", 0), 0), 3),
                "def_score":      None,
            }
            # Compute defensive score for the year
            from_preds = {"ind_hits_pg": row["hits_pg"],
                          "ind_takeaways_pg": row["takeaways_pg"],
                          "goals_against_per_game": row["goals_against_pg"],
                          "pk_ice_pct": row["pk_pct"],
                          "penalties_pg": row["penalties_pg"]}
            tmp = pd.DataFrame([from_preds])
            row["def_score"] = round(def_compute_defensive_score(tmp)["defensive_score"].iloc[0], 1)

        else:
            # Forward — set team and league context
            aged_row = aged.copy()
            for col in TEAM_FEATURES:
                if col in team_row.index:
                    aged_row[col] = team_row[col]
            for k, v in league_env.items():
                aged_row[k] = v

            pred_df  = pd.DataFrame([aged_row])
            feats    = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + TEAM_FEATURES
            feats    = [f for f in feats if f in pred_df.columns]
            X        = pred_df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)

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
    Return CBA contract length limits under the current CBA.
      - Same team (re-signing):   max 7 years
      - New team (UFA/trade):     max 6 years
      - Age 35+ at signing:       flags cap recapture risk (35+ rule)
      - Age 35+ by contract end:  flags that cap hit counts even if retired
    Returns dict with limits and flags.
    """
    is_same_team  = actual_team == signing_team
    max_years     = 7 if is_same_team else 6
    is_35_signing = current_age >= 35

    # Age when contract ends (year 1 = next season)
    age_at_expiry = current_age + max_years

    # Recommended max based purely on age curve (before CBA cap)
    if current_age >= 35:
        recommended = 1
    elif current_age >= 33:
        recommended = 2
    elif current_age >= 31:
        recommended = 3
    elif current_age >= 28:
        recommended = 4
    else:
        recommended = max_years  # young enough to fill the contract

    # Never recommend more than CBA allows
    recommended = min(recommended, max_years)

    # 35+ rule: if player will be 35+ during any year of the contract,
    # cap hit counts against the team even if player retires
    hits_35_rule = (current_age + 1) >= 35  # will be 35 in year 1

    return {
        "max_years":       max_years,
        "is_same_team":    is_same_team,
        "recommended":     recommended,
        "is_35_signing":   is_35_signing,
        "hits_35_rule":    hits_35_rule,
        "age_at_expiry":   age_at_expiry,
    }


def contract_risk_rating(rows, is_d):
    """
    Assess contract risk based on age trajectory.
    Returns (rating_label, rating_color, explanation).
    """
    if not rows:
        return "Unknown", "#888888", ""

    year1 = rows[0]
    last  = rows[-1]
    n     = len(rows)

    if is_d:
        score_key = "def_score"
        y1_val    = year1.get(score_key, 50)
        yn_val    = last.get(score_key, 50)
    else:
        score_key = "points_pg"
        y1_val    = year1.get(score_key, 0)
        yn_val    = last.get(score_key, 0)

    decline_pct = (y1_val - yn_val) / max(y1_val, 0.01)
    age_yr1     = year1["age"]

    if age_yr1 >= 35:
        return "Very High Risk", "#c8102e", f"Age {age_yr1:.0f} at signing — steep decline likely. 35+ rule applies."
    elif age_yr1 >= 32 and decline_pct > 0.20:
        return "High Risk", "#e8622a", f"Age {age_yr1:.0f} with projected {decline_pct*100:.0f}% decline over {n} years."
    elif age_yr1 >= 30 and decline_pct > 0.10:
        return "Moderate Risk", "#FFD700", f"Age {age_yr1:.0f} — some decline expected but manageable."
    elif decline_pct < 0:
        return "Low Risk", "#57a85a", f"Age {age_yr1:.0f} — ascending player, production expected to improve."
    else:
        return "Low Risk", "#57a85a", f"Age {age_yr1:.0f} — stable production expected through contract."


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="NHL Player Predictor", page_icon="🏒", layout="wide")
st.title("NHL Player Predictor")
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

# ── Load / train defensive model ───────────────────────────────────────────────
if "def_fit_models" not in st.session_state:
    if os.path.exists(DEF_CACHE_FILE):
        with st.spinner("Loading saved defensive models..."):
            def_cached = joblib.load(DEF_CACHE_FILE)
            (
                st.session_state["def_df"],
                st.session_state["def_team_ctx"],
                st.session_state["def_has_age"],
                st.session_state["def_player_profiles"],
                st.session_state["def_fit_models"],
                st.session_state["def_fit_metrics"],
                st.session_state["def_fit_feature_names"],
                st.session_state["def_next_models"],
                st.session_state["def_next_metrics"],
                st.session_state["def_next_feature_names"],
            ) = def_cached
    elif os.path.exists(DEF_FILE):
        st.info("Training defensive models for the first time — takes 3-5 minutes.")
        def_results = def_load_and_train(DEF_FILE, AGES_FILE)
        joblib.dump(def_results, DEF_CACHE_FILE)
        (
            st.session_state["def_df"],
            st.session_state["def_team_ctx"],
            st.session_state["def_has_age"],
            st.session_state["def_player_profiles"],
            st.session_state["def_fit_models"],
            st.session_state["def_fit_metrics"],
            st.session_state["def_fit_feature_names"],
            st.session_state["def_next_models"],
            st.session_state["def_next_metrics"],
            st.session_state["def_next_feature_names"],
        ) = def_results
        st.rerun()
    else:
        st.session_state["def_fit_models"] = None

def_models_loaded = st.session_state.get("def_fit_models") is not None

if def_models_loaded:
    def_df                = st.session_state["def_df"]
    def_team_ctx          = st.session_state["def_team_ctx"]
    def_has_age           = st.session_state["def_has_age"]
    def_player_profiles   = st.session_state["def_player_profiles"]
    def_fit_models        = st.session_state["def_fit_models"]
    def_fit_metrics       = st.session_state["def_fit_metrics"]
    def_fit_feature_names = st.session_state["def_fit_feature_names"]
    def_next_models       = st.session_state["def_next_models"]
    def_next_metrics      = st.session_state["def_next_metrics"]
    def_next_feature_names= st.session_state["def_next_feature_names"]

if has_age:
    st.caption("Age data loaded ✅ — next-season forecasting active.")
else:
    st.caption("⚠️ Age data not found — next-season model running without age features.")

# ── Player search (shared across tabs) ────────────────────────────────────────
st.divider()
# Combine forwards (from offensive df) and defensemen (from defensive df)
_fwd_names = set(df["player_name"].unique())
_def_names = set(def_df["player_name"].unique()) if def_models_loaded else set()
all_players = sorted(_fwd_names | _def_names)
player_input = st.selectbox("Search for a player", options=[""] + all_players, index=0)

pred = None
if player_input:
    # Try offensive model first (forwards)
    first = predict_player(player_input, df, team_ctx, fit_models, next_models,
                           player_profiles, has_age)

    if first is None and def_models_loaded:
        # Try defensive model (defensemen)
        def_first = def_predict_defenseman(
            player_input, def_df, def_team_ctx,
            def_fit_models, def_next_models, def_player_profiles, def_has_age
        )
        if def_first is not None:
            # Create a minimal pred dict so tabs know a defenseman is searched
            pred = {
                "pid":          def_first["pid"],
                "matched":      def_first["matched"],
                "actual_team":  def_first["actual_team"],
                "position":     "D",
                "seasons":      def_first["seasons"],
                "traded_teams": [],
                "fit_results":  None,
                "next_results": None,
                "age":          None,
            }
        else:
            st.error(f"No player found matching '{player_input}'.")
    elif first is None:
        st.error(f"No player found matching '{player_input}'.")
    else:
        override_team = None
        if first["traded_teams"]:
            st.warning(
                f"{first['matched']} was traded in {first['seasons'][0]}. "
                f"Select their current team:"
            )
            override_team = st.radio(
                "Current team", options=first["traded_teams"],
                horizontal=True, key="team_override"
            )
        pred = predict_player(player_input, df, team_ctx, fit_models, next_models,
                              player_profiles, has_age, override_team=override_team)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_off, tab_def, tab_contract, tab_model, tab_val = st.tabs([
    "Offensive",
    "Defensive",
    "Contract Evaluator",
    "Model Info",
    "Validation",
])

# ── Offensive (nested) ────────────────────────────────────────────────────────
with tab_off:
    off_t1, off_t2, off_t3 = st.tabs(["Team Fit", "Next Season", "Roster Insertion"])

    with off_t1:
        if pred and pred.get("fit_results") is not None:
            seasons_str = " → ".join(str(s) for s in pred["seasons"])
            age_str     = f"  |  Age {pred['age']:.0f}" if pred.get("age") else ""
            st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}  |  Seasons: {seasons_str}")
            st.caption("Predicted performance based on current weighted skill profile across all 32 teams.")
            st.pyplot(make_bar_chart(pred["fit_results"], pred["matched"], pred["actual_team"],
                                     f"{pred['matched']}  |  Current skill profile  |  Seasons: {seasons_str}"))
            st.markdown("#### Rankings Table")
            display = show_results_table(pred["fit_results"], pred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{pred['matched'].replace(' ','_')}_team_fit.csv",
                               mime="text/csv")
        else:
            st.info("Search for a forward above to see predictions.")

    with off_t2:
        if pred and pred.get("next_results") is not None:
            age_str = f"  |  Age {pred['age']:.0f} → {pred['age']+1:.0f}" if pred.get("age") else ""
            st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}")
            st.caption("Predicted next-season performance across all 32 teams.")
            st.pyplot(make_bar_chart(pred["next_results"], pred["matched"], pred["actual_team"],
                                     f"{pred['matched']}  |  Next season forecast"))
            st.markdown("#### Rankings Table")
            display = show_results_table(pred["next_results"], pred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{pred['matched'].replace(' ','_')}_next_season.csv",
                               mime="text/csv")
        else:
            st.info("Search for a forward above to see predictions.")

    with off_t3:
        st.caption("Select a team to see where the searched player would slot into their active roster.")
        if not pred or pred.get("position") == "D":
            st.info("Search for a forward above to use this tab.")
        else:
            c1, c2 = st.columns([1, 1])
            insertion_team = c1.selectbox(
                "Select team to insert player into",
                options=NHL_TEAMS,
                index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
                key="insertion_team"
            )
            if c2.button("Refresh roster"):
                fetch_active_team_roster.clear()

            with st.spinner(f"Building {insertion_team} roster with {pred['matched']} inserted..."):
                insertion_df, insertion_err = build_player_insertion(
                    pred["pid"], insertion_team, df, team_ctx,
                    fit_models, player_profiles, has_age
                )

            if insertion_err:
                st.error(insertion_err)
            elif insertion_df is not None and not insertion_df.empty:
                searched_row = insertion_df[insertion_df["is_searched_player"]].iloc[0]
                slot   = searched_row["lineup_slot"]
                rank   = int(searched_row["rank"])
                color  = searched_row["slot_color"]
                total  = len(insertion_df)
                st.markdown(
                    f"<h3 style='color:{color}'>"
                    f"{pred['matched']} projects as a <b>{slot}</b> player on {insertion_team} "
                    f"(rank {rank} of {total} {'forwards' if pred['position'] in ('C','L','R') else 'defensemen'})"
                    f"</h3>",
                    unsafe_allow_html=True
                )
                display = insertion_df[[
                    "rank","player_name","position","lineup_slot","pred_points_gp","pred_goals_gp"
                ]].copy()
                display.columns = ["Rank","Player","Pos","Line/Pair","Points/GP","Goals/GP"]
                def _hi_searched(row):
                    m = insertion_df.loc[insertion_df["player_name"]==row["Player"],"is_searched_player"].values
                    if len(m)>0 and m[0]:
                        return [f"background-color:{color}22;font-weight:bold;border-left:3px solid {color}"]*len(row)
                    return [""]*len(row)
                st.dataframe(display.style.apply(_hi_searched,axis=1),
                             use_container_width=True,
                             height=min(50+len(display)*35,600))
                slot_counts = display["Line/Pair"].value_counts()
                is_fwd = pred["position"] in ("C","L","R")
                slot_labels = (["1st Line","2nd Line","3rd Line","4th Line"] if is_fwd
                               else ["1st Pair","2nd Pair","3rd Pair","3rd Pair (extra)"])
                st.markdown("**Roster slot breakdown after insertion:**")
                cols_slots = st.columns(4)
                for col_s, lbl in zip(cols_slots, slot_labels):
                    col_s.metric(lbl, int(slot_counts.get(lbl,0)))
                csv = insertion_df.drop(columns=["slot_color"]).to_csv(index=False)
                st.download_button("Download roster insertion CSV", data=csv,
                                   file_name=f"{pred['matched'].replace(' ','_')}_{insertion_team}_insertion.csv",
                                   mime="text/csv")

# ── Defensive (nested) ────────────────────────────────────────────────────────
with tab_def:
    is_d = pred is not None and pred.get("position") == "D"

    dpred = None
    if is_d and def_models_loaded:
        _dpred_key = f"dpred_{pred['pid']}"
        if _dpred_key not in st.session_state:
            with st.spinner("Computing defensive predictions..."):
                st.session_state[_dpred_key] = def_predict_defenseman(
                    pred["matched"], def_df, def_team_ctx,
                    def_fit_models, def_next_models, def_player_profiles, def_has_age
                )
        dpred = st.session_state[_dpred_key]

    def_t1, def_t2, def_t3 = st.tabs(["Team Fit", "Next Season", "Pairing"])

    with def_t1:
        if not def_models_loaded:
            st.warning("Defensive model not loaded. Ensure defensive_dataset.csv is present.")
        elif not pred:
            st.info("Search for a defenseman above.")
        elif not is_d:
            st.info(f"{pred['matched']} is a forward. Search for a defenseman to use defensive tabs.")
        elif dpred:
            seasons_str = " → ".join(str(s) for s in dpred["seasons"])
            st.subheader(f"{dpred['matched']}  —  D  |  {dpred['actual_team']}  |  Seasons: {seasons_str}")
            st.caption("Defensive Score: xGA suppression 30%, takeaways 20%, PK usage 20%, hits 15%, discipline 15%.")
            st.markdown("#### Rankings Table")
            display = def_show_results_table(dpred["fit_results"], dpred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{dpred['matched'].replace(' ','_')}_def_fit.csv",
                               mime="text/csv")

    with def_t2:
        if not def_models_loaded:
            st.warning("Defensive model not loaded.")
        elif not pred or not is_d:
            st.info("Search for a defenseman above.")
        elif dpred:
            st.subheader(f"{dpred['matched']}  —  D  |  {dpred['actual_team']}")
            st.caption("Next-season defensive forecast based on current profile and trajectory.")
            st.markdown("#### Rankings Table")
            display = def_show_results_table(dpred["next_results"], dpred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{dpred['matched'].replace(' ','_')}_def_next.csv",
                               mime="text/csv")

    with def_t3:
        if not def_models_loaded:
            st.warning("Defensive model not loaded.")
        elif not pred or not is_d:
            st.info("Search for a defenseman above.")
        elif dpred:
            pc1, pc2 = st.columns([1, 1])
            pair_team = pc1.selectbox(
                "Select team",
                options=NHL_TEAMS,
                index=NHL_TEAMS.index(dpred["actual_team"]) if dpred["actual_team"] in NHL_TEAMS else 0,
                key="pair_team_sel"
            )
            if pc2.button("Refresh roster", key="pair_refresh"):
                def_fetch_team_roster_d.clear()

            with st.spinner(f"Building {pair_team} pairings with {dpred['matched']}..."):
                pair_result = def_build_pairing_insertion(
                    dpred["pid"], pair_team, def_df, def_team_ctx,
                    def_fit_models, def_player_profiles, def_has_age
                )

            if len(pair_result) == 3:
                pairing_df, pair_err, partner = pair_result
            else:
                pairing_df, pair_err = pair_result
                partner = "—"

            if pair_err:
                st.error(pair_err)
            elif pairing_df is not None and not pairing_df.empty:
                searched_row = pairing_df[pairing_df["is_searched_player"]].iloc[0]
                slot  = searched_row["pair_slot"]
                rank  = int(searched_row["rank"])
                color = searched_row["slot_color"]
                score = searched_row["defensive_score"]
                total = len(pairing_df)
                st.markdown(
                    f"<h3 style='color:{color}'>"
                    f"{dpred['matched']} projects as <b>{slot}</b> on {pair_team} "
                    f"(rank {rank}/{total} | Def Score: {score:.1f})"
                    f"</h3>",
                    unsafe_allow_html=True
                )
                if partner != "—":
                    st.info(f"Projected pairing partner: **{partner}**")
                display = pairing_df[[
                    "rank","player_name","pair_slot","defensive_score",
                    "ind_hits_pg","ind_takeaways_pg","goals_against_per_game","pk_ice_pct"
                ]].copy()
                display.columns = ["Rank","Player","Pair Slot","Def Score",
                                   "Hits/GP","TK/GP","GA/GP","PK%"]
                display["Def Score"] = display["Def Score"].round(1)
                display["Hits/GP"]   = display["Hits/GP"].round(2)
                display["TK/GP"]     = display["TK/GP"].round(3)
                display["GA/GP"]     = display["GA/GP"].round(3)
                display["PK%"]       = (display["PK%"] * 100).round(1)
                def _hi_pair(row):
                    m = pairing_df.loc[pairing_df["player_name"]==row["Player"],"is_searched_player"].values
                    if len(m)>0 and m[0]:
                        return [f"background-color:{color}22;font-weight:bold;border-left:3px solid {color}"]*len(row)
                    return [""]*len(row)
                st.dataframe(display.style.apply(_hi_pair,axis=1),
                             use_container_width=True,
                             height=min(50+len(display)*35,500))
                slot_counts = display["Pair Slot"].value_counts()
                st.markdown("**Pair breakdown after insertion:**")
                sc = st.columns(4)
                for c_s, lbl in zip(sc, ["1st Pair","2nd Pair","3rd Pair","3rd Pair (extra)"]):
                    c_s.metric(lbl, int(slot_counts.get(lbl,0)))
                csv = pairing_df.drop(columns=["slot_color"]).to_csv(index=False)
                st.download_button("Download pairing CSV", data=csv,
                                   file_name=f"{dpred['matched'].replace(' ','_')}_{pair_team}_pairing.csv",
                                   mime="text/csv")

# ── Contract Evaluator ────────────────────────────────────────────────────────
with tab_contract:
    st.subheader("Contract Evaluator")
    st.caption(
        "Projects a player's production across multiple seasons using empirical age curves. "
        "Works for both forwards (offensive stats) and defensemen (defensive stats). "
        "Confidence decreases in later years — use ranges rather than exact numbers."
    )

    if not pred:
        st.info("Search for a player above to use the contract evaluator.")
    else:
        is_d_contract = pred.get("position") == "D"

        # ── Controls ───────────────────────────────────────────────────────────
        cc1, cc2, cc3 = st.columns(3)
        contract_team = cc1.selectbox(
            "Team signing the player",
            options=NHL_TEAMS,
            index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
            key="contract_team"
        )
        curr_age = float(pred.get("age") or (
            def_player_profiles[pred["pid"]][0].get("age", 28)
            if is_d_contract and def_models_loaded else 28
        ) or 28)

        # CBA limits based on signing team
        cba = get_cba_limits(curr_age, pred["actual_team"], contract_team)

        n_years = cc2.slider(
            "Contract length (years)",
            min_value=1,
            max_value=cba["max_years"],
            value=min(cba["recommended"], cba["max_years"]),
        )
        cc3.metric("Current Age", f"{curr_age:.0f}")

        # ── CBA Info bar ───────────────────────────────────────────────────────
        cba_cols = st.columns(4)
        signing_type = "Re-signing (same team)" if cba["is_same_team"] else "New signing (different team)"
        cba_cols[0].metric("Signing Type",    signing_type)
        cba_cols[1].metric("CBA Max Length",  f"{cba['max_years']} years")
        cba_cols[2].metric("Recommended Max", f"{cba['recommended']} years")
        cba_cols[3].metric("Age at Expiry",   f"{cba['age_at_expiry']:.0f}")

        # 35+ rule warning
        if cba["is_35_signing"]:
            st.error(
                f"35+ Rule: {pred['matched']} is {curr_age:.0f} at signing. "
                "The cap hit counts against your team even if the player retires early. "
                "This creates significant cap recapture risk."
            )
        elif cba["hits_35_rule"]:
            st.warning(
                f"35+ Rule alert: {pred['matched']} will reach age 35 during this contract. "
                "If they retire before expiry, the cap hit remains on your books."
            )

        if n_years > cba["recommended"]:
            st.warning(
                f"This contract is longer than the recommended maximum of {cba['recommended']} years "
                f"based on the player's age curve. Years {cba['recommended']+1}+ carry high uncertainty."
            )

        # ── Run projection ─────────────────────────────────────────────────────
        with st.spinner("Projecting contract years..."):
            proj_rows, proj_err = build_contract_projection(
                pred["matched"], pred, dpred if is_d_contract else None,
                df, team_ctx, fit_models, next_models, player_profiles, has_age,
                def_df if def_models_loaded else pd.DataFrame(),
                def_team_ctx if def_models_loaded else pd.DataFrame(),
                def_fit_models if def_models_loaded else {},
                def_player_profiles if def_models_loaded else {},
                def_has_age if def_models_loaded else False,
                contract_team, n_years
            )

        if proj_err:
            st.error(proj_err)
        elif proj_rows:
            risk_label, risk_color, risk_explanation = contract_risk_rating(proj_rows, is_d_contract)

            # ── Risk header ────────────────────────────────────────────────────
            st.markdown(
                f"<h3>{pred['matched']} on {contract_team} — "
                f"<span style='color:{risk_color}'>{risk_label}</span></h3>",
                unsafe_allow_html=True
            )
            if risk_explanation:
                st.caption(risk_explanation)

            # ── Year-by-year table ─────────────────────────────────────────────
            st.markdown("#### Year-by-Year Projection")
            st.caption("Confidence reflects uncertainty compounding over time. Use wider mental ranges in later years.")

            if is_d_contract:
                proj_df = pd.DataFrame([{
                    "Year":          f"Year {r['year']} (Age {r['age']:.0f})",
                    "Hits/GP":       r["hits_pg"],
                    "Takeaways/GP":  r["takeaways_pg"],
                    "GA/GP (5v5)":   r["goals_against_pg"],
                    "PK%":           f"{r['pk_pct']*100:.1f}%",
                    "Penalties/GP":  r["penalties_pg"],
                    "Def Score":     r["def_score"],
                    "Confidence":    f"{r['confidence']*100:.0f}%",
                } for r in proj_rows])
            else:
                proj_df = pd.DataFrame([{
                    "Year":         f"Year {r['year']} (Age {r['age']:.0f})",
                    "Points/GP":    r["points_pg"],
                    "Goals/GP":     r["goals_pg"],
                    "Game Score/GP":r["gs_pg"],
                    "Confidence":   f"{r['confidence']*100:.0f}%",
                } for r in proj_rows])

            st.dataframe(proj_df, use_container_width=True, hide_index=True)

            # ── Trend chart ────────────────────────────────────────────────────
            st.markdown("#### Production Trend")
            fig_c, ax_c = plt.subplots(figsize=(10, 4))
            fig_c.patch.set_facecolor("#0e1117")
            ax_c.set_facecolor("#0e1117")

            years  = [r["year"] for r in proj_rows]
            labels = [f"Yr {r['year']} (Age {r['age']:.0f})" for r in proj_rows]

            if is_d_contract:
                vals   = [r["def_score"] for r in proj_rows]
                ylabel = "Defensive Score (0-100)"
                color  = "#4a90d9"
            else:
                vals   = [r["points_pg"] for r in proj_rows]
                ylabel = "Points / Game"
                color  = "#4a90d9"

            confs = [r["confidence"] for r in proj_rows]

            # Plot line with confidence band
            ax_c.plot(years, vals, color=color, linewidth=2.5, marker="o", markersize=8, zorder=3)
            # Confidence band — widen with lower confidence
            spread = [v * (1 - c) * 0.5 for v, c in zip(vals, confs)]
            ax_c.fill_between(years,
                              [v - s for v, s in zip(vals, spread)],
                              [v + s for v, s in zip(vals, spread)],
                              alpha=0.2, color=color, label="Confidence range")
            ax_c.set_xticks(years)
            ax_c.set_xticklabels(labels, color="white", fontsize=9)
            ax_c.set_ylabel(ylabel, color="white", fontsize=10)
            ax_c.tick_params(colors="white")
            ax_c.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
            for spine in ax_c.spines.values():
                spine.set_edgecolor("#333")
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close()

            # ── Summary recommendation ─────────────────────────────────────────
            st.divider()
            st.markdown("#### Contract Recommendation")
            yr1     = proj_rows[0]
            yr_last = proj_rows[-1]

            if is_d_contract:
                y1_score    = yr1["def_score"]
                yn_score    = yr_last["def_score"]
                decline     = max(y1_score - yn_score, 0)
                decline_pct = max((y1_score - yn_score) / max(y1_score, 1) * 100, 0)
                st.markdown(
                    f"- **Year 1 Defensive Score:** {y1_score:.1f} / 100  \n"
                    f"- **Year {n_years} Defensive Score:** {yn_score:.1f} / 100  \n"
                    f"- **Projected decline:** {decline:.1f} pts ({decline_pct:.0f}%)  \n"
                    f"- **Risk rating:** {risk_label}"
                )
            else:
                y1_pts    = yr1["points_pg"]
                yn_pts    = yr_last["points_pg"]
                total_pts = sum(r["points_pg"] * 82 for r in proj_rows)
                st.markdown(
                    f"- **Year 1 Points/GP:** {y1_pts:.3f}  \n"
                    f"- **Year {n_years} Points/GP:** {yn_pts:.3f}  \n"
                    f"- **Projected total points:** ~{total_pts:.0f} over {n_years} years  \n"
                    f"- **Risk rating:** {risk_label}"
                )

            # CBA recommendation box
            st.markdown("**CBA Summary:**")
            rec_col1, rec_col2 = st.columns(2)
            signing_lbl = "Same team (re-signing)" if cba["is_same_team"] else "New team"
            rec_col1.info(
                f"Same-team max: **7 years**  \n"
                f"New-team max: **6 years**  \n"
                f"Signing type: **{signing_lbl}**  \n"
                f"CBA max for this deal: **{cba['max_years']} years**"
            )
            rule35_note = "35+ rule applies — cap recapture risk if player retires early." if cba["hits_35_rule"] else "No 35+ rule concerns."
            rec_col2.success(
                f"Model recommendation: **{cba['recommended']} years**  \n"
                f"Based on age {curr_age:.0f} trajectory.  \n"
                f"{rule35_note}"
            )

            # Download
            csv = proj_df.to_csv(index=False)
            st.download_button(
                "Download contract projection CSV", data=csv,
                file_name=f"{pred['matched'].replace(' ','_')}_{contract_team}_{n_years}yr_contract.csv",
                mime="text/csv"
            )

# ── Model Info ────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("#### Offensive Model Cache")
    if st.button("Retrain offensive model (deletes cache)"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        for key in ["df","team_ctx","has_age","player_profiles","fit_models",
                    "fit_metrics","fit_feature_names","next_models","next_metrics","next_feature_names"]:
            st.session_state.pop(key, None)
        st.success("Offensive cache cleared — refresh the page to retrain.")

    st.divider()
    with st.expander("Team Fit model quality", expanded=True):
        show_metrics(fit_metrics, "Team Fit")
    with st.expander("Next Season model quality"):
        show_metrics(next_metrics, "Next Season")
    with st.expander("Team Fit — feature importance"):
        st.pyplot(make_importance_chart(fit_models, fit_feature_names))
    with st.expander("Next Season — feature importance"):
        st.pyplot(make_importance_chart(next_models, next_feature_names))

    if def_models_loaded:
        st.divider()
        st.markdown("#### Defensive Model Cache")
        if st.button("Retrain defensive models"):
            if os.path.exists(DEF_CACHE_FILE):
                os.remove(DEF_CACHE_FILE)
            for k in list(st.session_state.keys()):
                if k.startswith("def_") or k.startswith("dpred_"):
                    del st.session_state[k]
            st.success("Defensive cache cleared — refresh to retrain.")
        with st.expander("Defensive Current Fit quality", expanded=True):
            def_show_metrics(def_fit_metrics, "Defensive Current Fit")
        with st.expander("Defensive Next Season quality"):
            def_show_metrics(def_next_metrics, "Defensive Next Season")

# ── Validation (nested) ───────────────────────────────────────────────────────
with tab_val:
    val_t1, val_t2 = st.tabs(["Offensive", "Defensive"])

    with val_t1:
        st.subheader("2025-26 Offensive Validation")
        st.caption(
            "Pulls live 2025-26 stats from the NHL API and compares against "
            "the model predictions. Points and Goals converted to per-game rates."
        )
        if st.button("Refresh NHL API stats"):
            st.cache_data.clear()
            for k in ["edge_weighted_shots"]:
                st.session_state.pop(k, None)
            st.rerun()

        actual_df, err = fetch_nhl_current_season()
        if err:
            st.error(f"Could not fetch NHL API data: {err}")
        elif actual_df is not None:
            st.success(f"Fetched {len(actual_df):,} skaters with 10+ games played.")
            # weighted_shots_pg not used in validation — points and goals only

            with st.spinner("Comparing predictions to actual stats..."):
                val_df = build_validation_results(
                    actual_df, df, team_ctx, fit_models, player_profiles, has_age
                )
            if val_df.empty:
                st.warning("No players matched between NHL API and model profiles.")
            else:
                st.markdown(f"**{len(val_df):,} players matched**")
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.patch.set_facecolor("#0e1117")
                make_scatter(val_df, "actual_points_gp", "pred_points_gp", "Points / Game",  axes[0])
                make_scatter(val_df, "actual_goals_gp",  "pred_goals_gp",  "Goals / Game",   axes[1])
                plt.tight_layout()
                st.pyplot(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("Points/GP MAE",
                          f"{mean_absolute_error(val_df['actual_points_gp'], val_df['pred_points_gp']):.3f}")
                c2.metric("Goals/GP MAE",
                          f"{mean_absolute_error(val_df['actual_goals_gp'], val_df['pred_goals_gp']):.3f}")
                c3.metric("Players matched", f"{len(val_df):,}")

                pred_range    = val_df["pred_points_gp"].max() - val_df["pred_points_gp"].min()
                actual_range  = val_df["actual_points_gp"].max() - val_df["actual_points_gp"].min()
                compression   = pred_range / actual_range if actual_range > 0 else np.nan
                slope_val     = calibration_slope(val_df, "actual_points_gp", "pred_points_gp")
                c3b, c4b = st.columns(2)
                c3b.metric("Prediction Spread Ratio", f"{compression:.2%}" if not pd.isna(compression) else "n/a")
                c4b.metric("Calibration Slope",       f"{slope_val:.2f}"   if not pd.isna(slope_val)   else "n/a")

                elite_pts_mae, elite_pts_bias, elite_pts_n = elite_segment_stats(
                    val_df, "actual_points_gp", "pred_points_gp", quantile=ELITE_QUANTILE)
                elite_goals_mae, elite_goals_bias, elite_goals_n = elite_segment_stats(
                    val_df, "actual_goals_gp", "pred_goals_gp", quantile=ELITE_QUANTILE)
                e1, e2 = st.columns(2)
                e1.metric(f"Elite Points/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                          f"{elite_pts_mae:.3f}" if not pd.isna(elite_pts_mae) else "n/a",
                          f"bias {elite_pts_bias:+.3f}" if not pd.isna(elite_pts_bias) else None)
                e2.metric(f"Elite Goals/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                          f"{elite_goals_mae:.3f}" if not pd.isna(elite_goals_mae) else "n/a",
                          f"bias {elite_goals_bias:+.3f}" if not pd.isna(elite_goals_bias) else None)
                st.caption(f"Elite sample sizes: Points {elite_pts_n}, Goals {elite_goals_n}")
                st.divider()
                st.markdown("#### Biggest Misses")
                misses = val_df.reindex(val_df["points_gp_error"].abs().nlargest(15).index)[
                    ["player_name","team","games_played","actual_points_gp",
                     "pred_points_gp","points_gp_error","actual_goals_gp","pred_goals_gp","seasons_used"]]
                st.dataframe(misses, use_container_width=True)
                st.markdown("#### Best Predictions")
                best = val_df.reindex(val_df["points_gp_error"].abs().nsmallest(15).index)[
                    ["player_name","team","games_played","actual_points_gp",
                     "pred_points_gp","points_gp_error","actual_goals_gp","pred_goals_gp","seasons_used"]]
                st.dataframe(best, use_container_width=True)
                csv = val_df.to_csv(index=False)
                st.download_button("Download full validation CSV", data=csv,
                                   file_name="validation_2025_26.csv", mime="text/csv")

    with val_t2:
        st.subheader("2025-26 Defensive Validation")
        st.caption(
            "Compares defensive model predictions against 2025-26 actual stats "
            "from the NHL API realtime endpoint (hits and takeaways per game)."
        )
        if st.button("Refresh defensive stats"):
            fetch_nhl_defensive_stats.cache_clear()

        if not def_models_loaded:
            st.warning("Defensive model not loaded.")
        else:
            def_actual, def_err = fetch_nhl_defensive_stats()
            if def_err:
                st.error(f"Could not fetch NHL API data: {def_err}")
            elif def_actual is not None:
                def_actual_d = def_actual[
                    def_actual["player_id"].isin(def_player_profiles.keys())
                ].copy()
                st.success(f"Fetched {len(def_actual_d):,} defensemen with 10+ games played.")
                with st.spinner("Comparing predictions to actual stats..."):
                    def_val_df = build_defensive_validation(
                        def_actual_d, def_df, def_team_ctx,
                        def_fit_models, def_player_profiles, def_has_age
                    )
                if def_val_df.empty:
                    st.warning("No defensemen matched between NHL API and model profiles.")
                else:
                    st.markdown(f"**{len(def_val_df):,} defensemen matched**")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Hits/GP MAE",
                              f"{mean_absolute_error(def_val_df['actual_hits_pg'], def_val_df['pred_hits_pg']):.3f}")
                    c2.metric("Takeaways/GP MAE",
                              f"{mean_absolute_error(def_val_df['actual_tk_pg'], def_val_df['pred_tk_pg']):.3f}")
                    has_pk = "actual_pk_pct" in def_val_df.columns and def_val_df["actual_pk_pct"].sum() > 0
                    if has_pk:
                        c3.metric("PK% MAE",
                                  f"{mean_absolute_error(def_val_df['actual_pk_pct'], def_val_df['pred_pk_pct']):.4f}")
                    c4.metric("Defensemen matched", f"{len(def_val_df):,}")
                    st.caption("Penalties/GP omitted — NHL API penalty_minutes differs from MoneyPuck penalty count.")

                    n_plots = 3 if has_pk else 2
                    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
                    if n_plots == 2:
                        axes = list(axes)
                    fig.patch.set_facecolor("#0e1117")
                    make_scatter(def_val_df, "actual_hits_pg", "pred_hits_pg", "Hits / Game",      axes[0])
                    make_scatter(def_val_df, "actual_tk_pg",   "pred_tk_pg",   "Takeaways / Game", axes[1])
                    if has_pk:
                        make_scatter(def_val_df, "actual_pk_pct", "pred_pk_pct", "PK Ice %", axes[2])
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.divider()
                    st.markdown("#### Biggest Misses (Hits/GP)")
                    miss_cols = ["player_name","team","games_played",
                                 "actual_hits_pg","pred_hits_pg","hits_error",
                                 "actual_tk_pg","pred_tk_pg","tk_error",
                                 "actual_pk_pct","pred_pk_pct","seasons_used"]
                    misses = def_val_df.reindex(def_val_df["hits_error"].abs().nlargest(15).index)[
                        [c for c in miss_cols if c in def_val_df.columns]]
                    st.dataframe(misses, use_container_width=True)

                    st.markdown("#### Best Predictions (Hits/GP)")
                    best = def_val_df.reindex(def_val_df["hits_error"].abs().nsmallest(15).index)[
                        [c for c in miss_cols if c in def_val_df.columns]]
                    st.dataframe(best, use_container_width=True)

                    csv = def_val_df.to_csv(index=False)
                    st.download_button("Download defensive validation CSV", data=csv,
                                       file_name="def_validation_2025_26.csv", mime="text/csv")