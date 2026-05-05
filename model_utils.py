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
import base64
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "plotly is required. Run: pip install plotly"
    )
import streamlit as st
import streamlit.components.v1 as st_components
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── CSV helper ────────────────────────────────────────────────────────────────

def _safe_read_csv(path, **kwargs):
    """
    Read a CSV trying cp1252 first (Windows default), then UTF-8 variants.
    cp1252 is tried first because pandas with utf-8 may silently substitute
    bad bytes rather than raising, so the fallback never triggers.
    Handles special characters like ä, é, ö, å that appear in player names.
    """
    for enc in ("cp1252", "utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            # Verify no replacement characters snuck in (sign of wrong encoding)
            return df
        except (UnicodeDecodeError, ValueError):
            continue
    # Last resort — replace bad bytes rather than crash
    return pd.read_csv(path, encoding="latin-1", errors="replace", **kwargs)


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_FILE      = "season_dataset.csv"
AGES_FILE      = "player_ages.csv"
PP_FILE        = "pp_features.csv"
LINEMATE_FILE  = "linemate_features.csv"
CACHE_FILE     = "trained_models_forwards_v5.joblib"
NAMES_FILE     = "player_names.csv"     # persistent NHL API name cache
DEF_FILE       = "defensive_dataset.csv"
OFF_FILE       = "season_dataset.csv"   # shared with offensive model — contains D-men too

# Shift pair data is cached on disk so it survives app restarts.
# Each file: shifts_cache/{TEAM}_{N_GAMES}.json — refreshed when > TTL hours old.
SHIFTS_CACHE_DIR   = "shifts_cache"
SHIFTS_CACHE_TTL_H = 6   # hours before a cached file is considered stale

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

# Primary brand colours for each NHL team — used to highlight the actual team in charts.
# Secondary colour used as the bar fill; primary as the vline/outline.
TEAM_COLORS = {
    "ANA": {"primary": "#F47A38", "secondary": "#B9975B"},
    "BOS": {"primary": "#FCB514", "secondary": "#000000"},
    "BUF": {"primary": "#003087", "secondary": "#FFB81C"},
    "CAR": {"primary": "#CC0000", "secondary": "#000000"},
    "CBJ": {"primary": "#CE1126", "secondary": "#002654"},
    "CGY": {"primary": "#C8102E", "secondary": "#F1BE48"},
    "CHI": {"primary": "#CF0A2C", "secondary": "#FF671B"},
    "COL": {"primary": "#6F263D", "secondary": "#236192"},
    "DAL": {"primary": "#006847", "secondary": "#8F8F8C"},
    "DET": {"primary": "#CE1126", "secondary": "#FFFFFF"},
    "EDM": {"primary": "#FC4C02", "secondary": "#041E42"},
    "FLA": {"primary": "#C8102E", "secondary": "#041E42"},
    "LAK": {"primary": "#A2AAAD", "secondary": "#111111"},
    "MIN": {"primary": "#154734", "secondary": "#A6192E"},
    "MTL": {"primary": "#AF1E2D", "secondary": "#192168"},
    "NJD": {"primary": "#CE1126", "secondary": "#003087"},
    "NSH": {"primary": "#FFB81C", "secondary": "#041E42"},
    "NYI": {"primary": "#003087", "secondary": "#FC4C02"},
    "NYR": {"primary": "#0038A8", "secondary": "#CE1126"},
    "OTT": {"primary": "#C52032", "secondary": "#C69214"},
    "PHI": {"primary": "#F74902", "secondary": "#000000"},
    "PIT": {"primary": "#FCB514", "secondary": "#000000"},
    "SEA": {"primary": "#99D9D9", "secondary": "#001628"},
    "SJS": {"primary": "#006D75", "secondary": "#EA7200"},
    "STL": {"primary": "#002F87", "secondary": "#FCB514"},
    "TBL": {"primary": "#002868", "secondary": "#FFFFFF"},
    "TOR": {"primary": "#003E7E", "secondary": "#FFFFFF"},
    "UTA": {"primary": "#6CAEDF", "secondary": "#010101"},
    "VAN": {"primary": "#00843D", "secondary": "#00205B"},
    "VGK": {"primary": "#B4975A", "secondary": "#333F42"},
    "WPG": {"primary": "#004C97", "secondary": "#041E42"},
    "WSH": {"primary": "#C8102E", "secondary": "#041E42"},
}

def get_team_color(team: str, key: str = "primary") -> str:
    """Return a team's brand colour, defaulting to a neutral if not found."""
    return TEAM_COLORS.get(team, {}).get(key, "#c8102e")

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

# Non-linear career curve features — only included when age data is available.
NONLINEAR_FEATURES = [
    "curve_accel_points",
    "curve_accel_goals",
    "curve_accel_gs",
    "curve_local_deriv_points",
    "curve_local_deriv_goals",
    "curve_local_deriv_gs",
    "seasons_from_est_peak_points",
    "seasons_from_est_peak_goals",
    "seasons_from_est_peak_gs",
    "pct_peak_points_slope",
    "pct_peak_goals_slope",
    "age_x_3yr_pts_slope",
    "age_x_3yr_goals_slope",
    "age_x_career_pts_slope",
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


def engineer_nonlinear_trajectory_features(df):
    """
    Add non-linear career curve features using per-player quadratic fits.
    No-op if age data is absent or sparse (< 50% coverage).

    Adds:
      curve_accel_{stat}            Quadratic coeff a — negative = normal inverted-U arc.
      curve_local_deriv_{stat}      2a·age + b at current age. Positive = ascending.
      seasons_from_est_peak_{stat}  current_age − peak_age. Negative = pre-peak.
      pct_peak_{stat}_slope         Slope of pct_of_peak over prior ≤3 seasons.
      age_x_3yr_{stat}_slope        age × 3-year linear slope.
      age_x_career_pts_slope        age × full-career slope.
    """
    if "age" not in df.columns or df["age"].isna().mean() > 0.5:
        return df

    d = df.sort_values(["player_id", "season"]).copy()

    stats = {
        "points": "points_per_game",
        "goals":  "goals_per_game",
        "gs":     "game_score_per_game",
    }

    new_cols = (
        [f"curve_accel_{k}"           for k in stats] +
        [f"curve_local_deriv_{k}"     for k in stats] +
        [f"seasons_from_est_peak_{k}" for k in stats] +
        ["pct_peak_points_slope", "pct_peak_goals_slope",
         "age_x_3yr_pts_slope", "age_x_3yr_goals_slope", "age_x_career_pts_slope"]
    )
    for col in new_cols:
        d[col] = np.nan

    for pid, grp in d.groupby("player_id", sort=False):
        idx      = grp.index
        ages_arr = grp["age"].values

        for k, stat_col in stats.items():
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
                    a, b, _ = np.polyfit(pa, pv, 2)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                curr_age = ages_arr[i]
                if np.isnan(curr_age):
                    continue
                accel[i]       = a
                local_deriv[i] = 2.0 * a * curr_age + b
                if abs(a) > 1e-9:
                    from_peak[i] = curr_age - (-b / (2.0 * a))

            d.loc[idx, f"curve_accel_{k}"]           = accel
            d.loc[idx, f"curve_local_deriv_{k}"]     = local_deriv
            d.loc[idx, f"seasons_from_est_peak_{k}"] = from_peak

        for stat_short, peak_col in [
            ("points", "pct_of_peak_points"),
            ("goals",  "pct_of_peak_goals"),
        ]:
            if peak_col not in grp.columns:
                continue
            peak_vals = grp[peak_col].values
            pct_slope = np.full(len(grp), np.nan)
            for i in range(len(grp)):
                window = peak_vals[max(0, i - 3):i]
                mask   = ~np.isnan(window)
                if mask.sum() < 2:
                    continue
                y = window[mask]
                x = np.arange(len(window))[mask].astype(float)
                xm, ym = x.mean(), y.mean()
                denom  = ((x - xm) ** 2).sum()
                pct_slope[i] = 0.0 if denom == 0 else float(
                    ((x - xm) * (y - ym)).sum() / denom
                )
            d.loc[idx, f"pct_peak_{stat_short}_slope"] = pct_slope

        for slope_col, out_col in [
            ("recent_3yr_points_slope", "age_x_3yr_pts_slope"),
            ("recent_3yr_goals_slope",  "age_x_3yr_goals_slope"),
            ("career_points_slope",     "age_x_career_pts_slope"),
        ]:
            if slope_col in grp.columns:
                d.loc[idx, out_col] = ages_arr * grp[slope_col].values

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
    feats = (PLAYER_FEATURES
             + (AGE_FEATURES       if has_age else [])
             + (NONLINEAR_FEATURES if has_age else [])
             + TEAM_FEATURES)
    X = pd.concat(
        [df[feats].reset_index(drop=True), _pos_dummies(df).reset_index(drop=True)], axis=1
    )
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def _make_X_from_profile(profile, has_age, use_traj=False):
    """Build a single-row feature matrix from a player profile dict/Series."""
    pred_df = pd.DataFrame([profile])
    pos_d   = _pos_dummies(pred_df)
    nl = NONLINEAR_FEATURES if has_age else []
    if use_traj:
        traj  = [f for f in TRAJECTORY_FEATURES if f in pred_df.columns]
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + nl + traj + TEAM_FEATURES
    else:
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + nl + TEAM_FEATURES
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
    feats = (PLAYER_FEATURES
             + (AGE_FEATURES       if has_age else [])
             + (NONLINEAR_FEATURES if has_age else [])
             + traj_present + TEAM_FEATURES)
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
    weights = np.ones(len(arr), dtype=float)
    if len(arr) == 0:
        return weights
    elite_cut = np.quantile(arr, ELITE_QUANTILE)
    weights[arr >= elite_cut] = 3.0
    return weights

def _load_ages(ages_path):
    """
    Load player_ages.csv and compute any missing ages from birthDate.
    Returns a DataFrame with player_id, season, age, age_sq.
    """
    from datetime import datetime as _dt
    df_ages = _safe_read_csv(ages_path)

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
    df   = _safe_read_csv(path)
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
    pp   = _safe_read_csv(PP_FILE)[pp_cols]
    df   = df.merge(pp, on=["player_id", "season"], how="left")
    df[pp_cols[2:]] = df[pp_cols[2:]].fillna(0)
    # Join linemate quality features
    lm_cols = ["player_id", "season", "line_adj_xg_per60", "line_xg_per60",
               "line_hd_xg_per60", "line_goals_per60", "line_xg_pct",
               "line_corsi_pct", "n_distinct_lines"]
    lm   = _safe_read_csv(LINEMATE_FILE)[lm_cols]
    df   = df.merge(lm, on=["player_id", "season"], how="left")
    df[lm_cols[2:]] = df[lm_cols[2:]].fillna(0)
    has_age = df["age"].notna().mean() > 0.5
    advance(f"Data loaded (forwards only) — {len(df):,} rows  |  age matched: {df['age'].notna().sum():,}")

    # ── Engineer ───────────────────────────────────────────────────────────────
    status.markdown("⚙️ **Engineering features...**")
    df       = engineer_player_features(df)
    df       = engineer_trajectory_features(df)
    df       = engineer_career_history_features(df)
    df       = engineer_nonlinear_trajectory_features(df)   # non-linear curve signals
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

    nl = NONLINEAR_FEATURES if has_age else []
    if use_next_features and df is not None:
        traj_present = [f for f in TRAJECTORY_FEATURES if f in pred_df.columns]
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + nl + traj_present + TEAM_FEATURES
    else:
        feats = PLAYER_FEATURES + (AGE_FEATURES if has_age else []) + nl + TEAM_FEATURES

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
    # Fetch the correctly-accented name from NHL API; fall back to CSV name
    _api_name = fetch_player_display_name(int(pid))
    matched   = _api_name if _api_name else profile["player_name"]

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

    # Resolve age — profile age may be NaN if player_id didn't match ages CSV.
    # Fall back to reading the ages file directly by player_id.
    _age = profile.get("age") if has_age else None
    if _age is None or (isinstance(_age, float) and np.isnan(_age)):
        try:
            _ages_df = _safe_read_csv(AGES_FILE)
            _age_row = _ages_df[_ages_df["player_id"] == pid].sort_values("season", ascending=False)
            if not _age_row.empty and pd.notna(_age_row.iloc[0].get("age")):
                _base_age   = float(_age_row.iloc[0]["age"])
                _base_season = int(_age_row.iloc[0]["season"])
                _current_season = int(df["season"].max())
                _age = _base_age + max(0, _current_season - _base_season)
        except Exception:
            _age = None

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
        "age":          _age if _age is not None and not (isinstance(_age, float) and np.isnan(_age)) else None,
    }


# ── Charts ─────────────────────────────────────────────────────────────────────

def make_bar_chart(results, player_name, actual_team, title):
    """
    Interactive Plotly bar chart. X-axis zoomed to data range so team
    differences are visible. Click the fullscreen icon (⛶) to expand.
    """
    metric_cols   = ["pred_game_score_per_game", "pred_points_per_game", "pred_goals_per_game"]
    metric_labels = ["Game Score / Game", "Points / Game", "Goals / Game"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=metric_labels,
                        horizontal_spacing=0.08)

    for col_idx, (col, label) in enumerate(zip(metric_cols, metric_labels), start=1):
        sr     = results.sort_values(col)
        team_primary   = get_team_color(actual_team, "primary")
        team_secondary = get_team_color(actual_team, "secondary")
        colors = [team_primary if t == actual_team else "#4a90d9" for t in sr["player_team"]]
        vals   = sr[col].values

        fig.add_trace(
            go.Bar(
                x=vals,
                y=sr["player_team"],
                orientation="h",
                marker_color=colors,
                marker_line_color=[team_secondary if t == actual_team else "#4a90d9"
                                   for t in sr["player_team"]],
                marker_line_width=[2 if t == actual_team else 0 for t in sr["player_team"]],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )

        # Vertical line for actual team value
        actual_val = float(results.loc[results["player_team"] == actual_team, col].values[0])
        fig.add_vline(
            x=actual_val, line_color=team_primary, line_dash="dash", line_width=2,
            row=1, col=col_idx,
        )

        # Zoom x-axis to data range so differences are visible
        spread = vals.max() - vals.min()
        pad    = max(spread * 0.1, vals.max() * 0.005)
        fig.update_xaxes(
            range=[vals.min() - pad, vals.max() + pad],
            row=1, col=col_idx,
            gridcolor="#2d3748", zerolinecolor="#2d3748",
            tickfont=dict(color="#aaa", size=9),
        )
        fig.update_yaxes(
            row=1, col=col_idx,
            tickfont=dict(color="#aaa", size=9),
            gridcolor="#2d3748",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=13)),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=700,
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(color="white"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "white"
        ann.font.size  = 12

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


# ── Theme / background ────────────────────────────────────────────────────────

def _brighten_for_gradient(hex_color: str, min_luminance: float = 0.22) -> str:
    """
    If a hex color is too dark to show against the app background, mix it
    toward white until it clears min_luminance. Returns the (possibly brightened)
    hex string so team gradients are always visible regardless of how dark the
    team's brand color is (e.g. DAL green, COL burgundy, VGK gold-black).
    Threshold 0.22 brightens only truly dark colours (e.g. near-black VGK gold)
    while leaving medium brand colours like NYR navy/red closer to their
    original hue so the gradient looks authentic rather than washed-out.
    """
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    if luminance >= min_luminance:
        return hex_color
    # Mix toward white until luminance target is met (max 5 passes)
    for _ in range(5):
        r = min(255, int(r + (255 - r) * 0.45))
        g = min(255, int(g + (255 - g) * 0.45))
        b = min(255, int(b + (255 - b) * 0.45))
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        if luminance >= min_luminance:
            break
    return f"#{r:02x}{g:02x}{b:02x}"


def _team_gradient(team_name: str) -> str:
    """Return the CSS gradient string for a team, or 'none' if not found."""
    if team_name and team_name in TEAM_COLORS:
        p = _brighten_for_gradient(TEAM_COLORS[team_name]["primary"])
        s = _brighten_for_gradient(TEAM_COLORS[team_name]["secondary"])
        # 99 ≈ 60% opacity primary, 70 ≈ 44% secondary — noticeably brighter
        # than the original 70/50 (44%/31%) without washing colours out
        return f"linear-gradient(135deg, {p}99 0%, #141414 50%, {s}70 100%)"
    return "none"


def _build_team_grad_map() -> dict:
    """Precompute {team_code: gradient_string} for all known teams."""
    return {team: _team_gradient(team) for team in TEAM_COLORS}


def _make_roster_tab_js_html() -> str:
    """
    Generate the iframe HTML with all team gradients baked in as a JS constant.
    This allows the background to switch INSTANTLY when the user changes the
    team dropdown — no Streamlit rerun needed.
    """
    import json
    grad_json = json.dumps(_build_team_grad_map())

    return f"""<!DOCTYPE html>
<html>
<head><style>html,body{{margin:0;padding:0;background:transparent;}}</style></head>
<body>
<script>
(function(){{
  var D = window.parent.document;

  /* All team gradients baked in — no round-trip to Python required. */
  var GRADS = {grad_json};

  var OVERRIDE_TABS = ['Roster Insertion', 'Pairing', 'Contract Evaluator'];

  function isRosterActive() {{
    return Array.from(D.querySelectorAll('button[role="tab"]')).some(function(t) {{
      return OVERRIDE_TABS.indexOf(t.textContent.trim()) !== -1 &&
             t.getAttribute('aria-selected') === 'true';
    }});
  }}

  function applyGrad(grad) {{
    var app = D.querySelector('.stApp');
    if (app) app.style.setProperty('background-image', grad, 'important');
  }}

  function clearGrad() {{
    var app = D.querySelector('.stApp');
    if (app) app.style.removeProperty('background-image');
  }}

  /* Read whatever team code is currently shown in a visible selectbox.
     Scans each word of the element's visible text so it works regardless
     of whether BaseWeb renders the value in a span, div, or any other tag.
     Team codes are 2-3 uppercase letters (e.g. "PIT") — player name words
     are mixed-case so they never collide. */
  function activeSelectboxTeam() {{
    var selects = D.querySelectorAll('[data-baseweb="select"]');
    for (var i = 0; i < selects.length; i++) {{
      var rect = selects[i].getBoundingClientRect();
      if (rect.width === 0) continue;   // hidden / not in active tab
      var words = (selects[i].innerText || selects[i].textContent || '')
                    .trim().split(/\s+/);
      for (var j = 0; j < words.length; j++) {{
        var w = words[j].replace(/[^A-Z]/g, '');  // strip parens etc.
        if (GRADS[w]) return w;
      }}
    }}
    return null;
  }}

  /* Full update: tab state + current selectbox value. */
  function update() {{
    if (isRosterActive()) {{
      var code = activeSelectboxTeam();
      if (code) {{
        applyGrad(GRADS[code]);
      }} else {{
        /* Fall back to the CSS-var override set by Python. */
        var cs = getComputedStyle(D.documentElement);
        var grad = cs.getPropertyValue('--team-bg-override').trim();
        if (grad && grad !== 'none') applyGrad(grad);
      }}
    }} else {{
      clearGrad();
    }}
  }}

  /* --- one-time observer setup --- */
  if (D.__teamBgReady) {{
    /* Rerun: refresh the update fn reference so the parent poll stays current. */
    window.parent.__teamBgUpdate = update;
    update();
    return;
  }}
  D.__teamBgReady = true;

  /* Expose on parent so the interval below can call the latest closure
     even after this iframe is replaced on a subsequent rerun. */
  window.parent.__teamBgUpdate = update;

  /* Instant tab-click response. */
  new MutationObserver(function(muts) {{
    for (var i = 0; i < muts.length; i++) {{
      if (muts[i].attributeName === 'aria-selected') {{ update(); return; }}
    }}
  }}).observe(D.body, {{
    attributes: true, subtree: true, attributeFilter: ['aria-selected']
  }});

  /* Poll runs in the PARENT window — survives iframe replacement on reruns.
     Each new iframe updates window.parent.__teamBgUpdate so this always
     calls the freshest closure with the latest GRADS / D references. */
  window.parent.setInterval(function() {{
    window.parent.__teamBgUpdate && window.parent.__teamBgUpdate();
  }}, 50);

  update();
}})();
</script>
</body>
</html>"""


def _team_bg_css_vars(base_grad: str, override_grad: str) -> str:
    """<style> block that sets the two live CSS custom properties.
    Streamlit updates <style> content reliably on every rerun, and because
    .stApp uses var(--team-bg-base) directly, the background updates the
    instant this block is parsed — no JS round-trip required.
    """
    return (
        f"<style>:root{{"
        f"--team-bg-base:{base_grad};"
        f"--team-bg-override:{override_grad};"
        f"}}</style>"
    )


def update_team_colors(player_team: str = None, override_team: str = None) -> None:
    """Refresh the CSS custom properties so the background updates instantly.
    Because .stApp references var(--team-bg-base) directly, changing the var
    is enough — no !important battles, no JS execution needed.
    """
    base_grad     = _team_gradient(player_team)
    override_grad = _team_gradient(override_team) if override_team else base_grad
    st.markdown(
        "\n" + _team_bg_css_vars(base_grad, override_grad),
        unsafe_allow_html=True,
    )


def apply_team_theme(player_team: str = None, override_team: str = None) -> None:
    """Inject the full dark-mode CSS + one-time tab-aware switcher JS.

    Background strategy
    ───────────────────
    • :root CSS vars (--team-bg-base / --team-bg-override) are written on
      every Streamlit rerun by this function and update_team_colors().
    • .stApp uses  background-image: var(--team-bg-base)  so it tracks the
      var live — any rerun that changes the var immediately changes the bg.
    • .stApp.team-bg-roster (higher specificity) uses var(--team-bg-override).
    • The JS observer adds/removes that class based on which sub-tab is active.

    This avoids all !important source-order cascade problems: the first player
    shows immediately because update_team_colors() sets --team-bg-base and
    .stApp picks it up in the same render pass.
    """
    base_grad     = _team_gradient(player_team)
    override_grad = _team_gradient(override_team) if override_team else base_grad


def apply_team_theme(player_team: str = None, override_team: str = None) -> None:
    """
    Inject the full dark-mode CSS + the tab-aware background switcher.
    Called on every Streamlit rerun (top of app.py).

    How the background switching works:
      • Two CSS custom properties on :root are updated every rerun:
            --team-bg-base     (player's real team gradient)
            --team-bg-override (insertion / target team gradient)
      • A one-time JS MutationObserver + 300 ms interval reads whichever
        property is appropriate for the active tab and writes it into a
        dedicated <style id="__team_bg_sw"> tag with !important.
      • Because CSS vars update reliably without script re-execution,
        every player/team change is reflected within one polling tick.
    """
    base_grad     = _team_gradient(player_team)
    override_grad = _team_gradient(override_team) if override_team else base_grad

    css = f"""
    <style>
    /* ── Live CSS custom properties (updated every rerun) ──────────── */
    :root {{
        --team-bg-base:     {base_grad};
        --team-bg-override: {override_grad};
    }}

    /* ── Base dark theme ────────────────────────────────────────────── */
    .stApp {{
        background-color: #141414 !important;
        background-image: var(--team-bg-base) !important;
        color: #f0f0f0 !important;
    }}

    /* Main content area */
    .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"] {{
        background: transparent !important;
        color: #f0f0f0 !important;
    }}

    /* Headings and specific Streamlit text containers */
    h1, h2, h3, h4, h5, h6 {{
        color: #ffffff !important;
    }}
    label, .stCaption,
    [data-testid="stMarkdownContainer"] > p,
    [data-testid="stMarkdownContainer"] > div {{
        color: #f0f0f0 !important;
    }}
    .stText, small {{
        color: #cccccc !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {{
        color: #f0f0f0 !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: #cccccc !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: #ffffff !important;
    }}

    /* Input widgets */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    [data-baseweb="select"],
    [data-baseweb="input"] {{
        background-color: #1e2a45 !important;
        color: #f0f0f0 !important;
        border-color: #4a5568 !important;
    }}
    [data-baseweb="select"] * {{
        color: #f0f0f0 !important;
        background-color: #1e2a45 !important;
    }}

    /* Dropdown popup */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    [data-baseweb="list"],
    ul[role="listbox"] {{
        background-color: #1e2a45 !important;
        border: 1px solid #4a5568 !important;
        color: #f0f0f0 !important;
    }}
    [role="option"],
    [data-baseweb="menu-item"],
    li[role="option"] {{
        background-color: #1e2a45 !important;
        color: #f0f0f0 !important;
    }}
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    li[role="option"]:hover {{
        background-color: #2d4a7a !important;
        color: #ffffff !important;
    }}

    /* Slider */
    [data-testid="stSlider"] label {{
        color: #f0f0f0 !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: #1e2a45 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a5568 !important;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        background-color: #1a2236 !important;
        border: 1px solid #4a5568 !important;
    }}
    [data-testid="stExpander"] summary {{
        color: #f0f0f0 !important;
    }}

    /* Info / warning / success / error boxes */
    [data-testid="stAlert"] {{
        background-color: #1e2a45 !important;
        color: #f0f0f0 !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        background-color: #1a2236 !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #0e1420 !important;
    }}

    /* Top toolbar / header */
    header[data-testid="stHeader"] {{
        background-color: #141414 !important;
    }}
    </style>
    """
    # css already contains :root { --team-bg-base / --team-bg-override }.
    # The JS observer is injected via st_components.html (real iframe) so its
    # script is guaranteed to execute — unlike st.markdown innerHTML scripts.
    st.markdown(css, unsafe_allow_html=True)
    st_components.html(_make_roster_tab_js_html(), height=0)


# Team logo base URL
_NHL_LOGO_URL = "https://assets.nhle.com/logos/nhl/svg/{team}_light.svg"


def get_player_headshot_html(player_id: int, size: int = 80) -> str:
    """
    Return an <img> HTML string for a player headshot.
    Uses base64-encoded image fetched server-side (avoids CDN cross-origin blocking).
    Falls back to NHL silhouette SVG inline if no cached image is available.
    """
    global _NAMES_CACHE, _NAMES_CACHE_LOADED
    if not _NAMES_CACHE_LOADED:
        _NAMES_CACHE = _load_names_cache()
        _NAMES_CACHE_LOADED = True

    pid   = int(player_id)
    entry = _NAMES_CACHE.get(pid, {})
    b64   = entry.get("headshot_b64", "") if entry else ""

    # If no cached headshot yet, try to fetch it now
    if not b64:
        _ensure_player_cached(pid)
        entry = _NAMES_CACHE.get(pid, {})
        b64   = entry.get("headshot_b64", "") if entry else ""

    style = (
        "width:" + str(size) + "px;"
        "height:" + str(size) + "px;"
        "border-radius:50%;"
        "object-fit:cover;"
        "margin-top:4px;"
        "background:#1a1a2e"
    )

    if b64:
        src = "data:image/png;base64," + b64
        return '<img src="' + src + '" style="' + style + '">'

    # Silhouette SVG fallback — no external request needed
    silhouette = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="' + str(size) + '" height="' + str(size) + '" viewBox="0 0 100 100">'
        '<circle cx="50" cy="50" r="50" fill="#2d3748"/>'
        '<circle cx="50" cy="38" r="18" fill="#718096"/>'
        '<ellipse cx="50" cy="80" rx="28" ry="20" fill="#718096"/>'
        '</svg>'
    )
    return silhouette


def _render_scrollable_table(render_df, is_actual_series, actual_team, rank_val, total, context_window=5, team_color=None):
    """
    Render a scrollable HTML table that auto-scrolls to centre the highlighted
    actual-team row on load. Uses a small inline JS scrollIntoView call.
    Team logos are embedded as inline <img> tags (fetched by the browser).

    render_df        — DataFrame without the _is_actual column (display columns only)
    is_actual_series — boolean Series aligned to render_df index
    """
    import uuid
    table_id = "tbl_" + uuid.uuid4().hex[:8]

    # ── Build column headers ───────────────────────────────────────────────────
    th_style = (
        "padding:6px 12px; text-align:left; background:#1a1a2e; "
        "color:#aaa; font-size:13px; border-bottom:1px solid #333; "
        "position:sticky; top:0; z-index:1;"
    )
    headers = "".join(f'<th style="{th_style}">{c}</th>' for c in render_df.columns)

    # ── Build rows ─────────────────────────────────────────────────────────────
    rows_html = ""
    for i, (idx, row) in enumerate(render_df.iterrows()):
        is_actual = bool(is_actual_series.iloc[i])
        if is_actual:
            _tc = team_color or "#c8102e"
            # Fill the row with the team color at ~35% opacity so it reads
            # clearly on the dark background while staying legible.
            row_style = (
                f"background:{_tc}59; "
                "font-weight:bold; color:#fff; font-size:13.5px;"
            )
            row_id = f'id="actual_row_{table_id}"'
        else:
            row_style = "background:#0e1117; color:#ccc;" if i % 2 == 0 else "background:#111827; color:#ccc;"
            row_id = ""

        td_style = ("padding:6px 14px; font-size:13.5px; border-bottom:1px solid #1f2937;"
                    if is_actual else
                    "padding:5px 12px; font-size:13px; border-bottom:1px solid #1f2937;")
        cells = ""
        for col, v in zip(render_df.columns, row):
            if col == "Team":
                logo_url = _NHL_LOGO_URL.format(team=v)
                img_err = "this.style.display='none'"
                cell_content = (
                    f'<img src="{logo_url}" height="20" '
                    f'style="vertical-align:middle;margin-right:6px" '
                    f'onerror="{img_err}"> {v}'
                )
                cells += f'<td style="{td_style}">{cell_content}</td>'
            else:
                cells += f'<td style="{td_style}">{v}</td>'
        rows_html += f'<tr style="{row_style}" {row_id}>{cells}</tr>\n'

    # ── Visible height: context_window above + actual + context_window below ──
    row_h   = 38
    header  = 38
    visible = context_window * 2 + 1
    height  = header + visible * row_h

    html = f"""
<div id="wrap_{table_id}" style="overflow-y:auto; height:{height}px; border:1px solid #2d3748; border-radius:4px;">
  <table id="{table_id}" style="width:100%; border-collapse:collapse; table-layout:auto;">
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
<script>
  (function() {{
    var wrap = document.getElementById("wrap_{table_id}");
    var row  = document.getElementById("actual_row_{table_id}");
    if (wrap && row) {{
      // Scroll the container div internally — never touches page scroll position
      wrap.scrollTop = row.offsetTop - (wrap.clientHeight / 2) + (row.offsetHeight / 2);
    }}
  }})();
</script>
"""
    st.caption(
        f"**{actual_team}** ranks **{rank_val} of {total}** — "
        f"highlighted row is centred, scroll to see all teams."
    )
    st_components.html(html, height=height + 4, scrolling=False)


def show_results_table(results, actual_team, context_window=5):
    display = results[[
        "player_team", "pred_game_score_per_game",
        "pred_points_per_game", "pred_goals_per_game", "is_actual"
    ]].copy()
    display.columns = ["Team", "GS/GP", "Points/GP", "Goals/GP", "_is_actual"]
    for col in ["GS/GP", "Points/GP", "Goals/GP"]:
        display[col] = display[col].round(3)
    display.insert(0, "Rank", range(1, len(display) + 1))

    actual_idx = display.index[display["_is_actual"]].tolist()
    rank_val   = int(display.loc[actual_idx[0], "Rank"]) if actual_idx else "?"

    render = display.drop(columns=["_is_actual"]).reset_index(drop=True)
    _render_scrollable_table(
        render, display["_is_actual"].reset_index(drop=True),
        actual_team, rank_val, len(display), context_window,
        team_color=get_team_color(actual_team)
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

def _load_names_cache() -> dict:
    """Load player_names.csv into {player_id: {name, headshot_b64}} dict."""
    if os.path.exists(NAMES_FILE):
        try:
            ndf = pd.read_csv(NAMES_FILE, dtype={"player_id": int})
            result = {}
            for _, row in ndf.iterrows():
                result[int(row["player_id"])] = {
                    "name":         str(row.get("name", "")),
                    "headshot_b64": str(row.get("headshot_b64", "")) if pd.notna(row.get("headshot_b64")) else "",
                }
            return result
        except Exception:
            pass
    return {}


def _save_names_cache(cache: dict) -> None:
    """Persist {player_id: {name, headshot_b64}} to player_names.csv."""
    try:
        rows = pd.DataFrame([
            {"player_id": pid, "name": v.get("name",""), "headshot_b64": v.get("headshot_b64","")}
            for pid, v in cache.items()
        ])
        rows.to_csv(NAMES_FILE, index=False, encoding="utf-8")
    except Exception:
        pass


# Module-level in-memory dict — loaded once per process, written back on new entries
_NAMES_CACHE: dict = {}
_NAMES_CACHE_LOADED: bool = False


def _ensure_player_cached(player_id: int) -> None:
    """Fetch name + headshot from NHL API and store in cache if not already present."""
    global _NAMES_CACHE, _NAMES_CACHE_LOADED
    if not _NAMES_CACHE_LOADED:
        _NAMES_CACHE = _load_names_cache()
        _NAMES_CACHE_LOADED = True

    pid = int(player_id)
    entry = _NAMES_CACHE.get(pid, {})

    # Already have both name and headshot
    if entry.get("name") and entry.get("headshot_b64"):
        return

    try:
        url  = f"https://api-web.nhle.com/v1/player/{pid}/landing"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data  = resp.json()

        first = data.get("firstName", {}).get("default", "")
        last  = data.get("lastName",  {}).get("default", "")
        name  = f"{first} {last}".strip()

        # Fetch headshot from the URL in the API response
        headshot_b64 = entry.get("headshot_b64", "")
        hs_url = data.get("headshot", "")
        if hs_url and not headshot_b64:
            try:
                hs_resp = requests.get(hs_url, timeout=8)
                if hs_resp.status_code == 200:
                    headshot_b64 = base64.b64encode(hs_resp.content).decode("utf-8")
            except Exception:
                headshot_b64 = ""

        if name or headshot_b64:
            _NAMES_CACHE[pid] = {
                "name":         name or entry.get("name", ""),
                "headshot_b64": headshot_b64,
            }
            _save_names_cache(_NAMES_CACHE)
    except Exception:
        pass


def fetch_player_display_name(player_id: int) -> str:
    """Return the correctly-spelled player name. Fetches from NHL API if needed."""
    global _NAMES_CACHE, _NAMES_CACHE_LOADED
    if not _NAMES_CACHE_LOADED:
        _NAMES_CACHE = _load_names_cache()
        _NAMES_CACHE_LOADED = True
    pid = int(player_id)
    if pid not in _NAMES_CACHE or not _NAMES_CACHE[pid].get("name"):
        _ensure_player_cached(pid)
    entry = _NAMES_CACHE.get(pid, {})
    return entry.get("name") or None



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

        feats  = (PLAYER_FEATURES
                  + (AGE_FEATURES       if has_age else [])
                  + (NONLINEAR_FEATURES if has_age else [])
                  + TEAM_FEATURES)
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
    Merges realtime endpoint (hits, takeaways) with
    summary endpoint (penalty minutes) and
    timeonice endpoint (PK time) to validate all 5 defensive targets.
    """
    try:
        # ── Realtime stats (hits, takeaways, giveaways) ───────────────────────
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
            "gamesPlayed":      "games_played",
        })
        rt_df = rt_df[[c for c in ["player_id","player_name","position","hits",
                                    "blocked_shots","takeaways","giveaways",
                                    "games_played"]
                       if c in rt_df.columns]].copy()
        # Filter to defensemen only
        if "position" in rt_df.columns:
            rt_df = rt_df[rt_df["position"] == "D"]
        rt_df = rt_df.dropna(subset=["player_id","games_played"])
        rt_df = rt_df[rt_df["games_played"] >= 10]
        rt_df["player_id"] = rt_df["player_id"].astype(int)

        # ── Summary stats (PIM) ────────────────────────────────────────────────
        sum_url = (
            "https://api.nhle.com/stats/rest/en/skater/summary"
            "?limit=-1&start=0"
            "&cayenneExp=seasonId=20252026"
        )
        sum_resp = requests.get(sum_url, timeout=15)
        sum_resp.raise_for_status()
        sum_data = sum_resp.json().get("data", [])
        sum_df   = pd.json_normalize(sum_data)
        if not sum_df.empty:
            sum_df = sum_df.rename(columns={
                "playerId":       "player_id",
                "positionCode":   "position",
                "penaltyMinutes": "penalty_minutes",
            })
            sum_cols = [c for c in ["player_id", "position", "penalty_minutes"] if c in sum_df.columns]
            sum_df = sum_df[sum_cols].copy()
            if "position" in sum_df.columns:
                sum_df = sum_df[sum_df["position"] == "D"]
            sum_df["player_id"] = sum_df["player_id"].astype(int)
            rt_df = rt_df.merge(sum_df[["player_id", "penalty_minutes"]], on="player_id", how="left")
        else:
            rt_df["penalty_minutes"] = np.nan

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
    # 3 forwards per line (C, LW, RW) across 4 lines = 12 forwards
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
        df = _safe_read_csv(DEF_FILE)
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
    d3.metric("xGA Against / 60", fmt(row.get("xg_against_per60_5v5"), 2),
              f"p{pct_rank('xg_against_per60_5v5', higher_better=False):.0f}"
              if pct_rank("xg_against_per60_5v5") is not None else None)
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

# ── Defensive offensive stats loader ──────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_defensive_offensive_stats():
    """
    Load offensive stats for defensemen from season_dataset.csv.
    Returns:
      - dict: player_id -> {points_pg, goals_pg, game_score_pg, pp_pct, season}
      - DataFrame: all D-men rows (most recent season per player) for live percentile ranking
    """
    try:
        df = _safe_read_csv(OFF_FILE)
        df = df[df["position"] == "D"].copy()
        df = df.sort_values("season", ascending=False)
        df = df.groupby("player_id").first().reset_index()
        result = {}
        for _, row in df.iterrows():
            result[int(row["player_id"])] = {
                "points_pg":      float(row.get("points_per_game",   0) or 0),
                "goals_pg":       float(row.get("goals_per_game",    0) or 0),
                "game_score_pg":  float(row.get("game_score_per_game", 0) or 0),
                "pp_icetime_pct": float(row.get("pp_icetime_pct",   0) or 0),
                "season":         int(row.get("season", 2024)),
            }
        return result, df, None
    except Exception as e:
        return {}, None, str(e)


def grade_offensive_defenseman(off_stats, season_off_df=None):
    """
    Grade a defenseman's offensive production on A-F scale using
    live percentile rank among all D-men in the dataset.
    Weighted 70% points/GP + 30% goals/GP.

    Grade cutoffs (percentile among all D-men):
      A   = top 10%  (p90+)
      B+  = top 25%  (p75–90)
      B   = top 50%  (p50–75)
      C+  = top 65%  (p35–50)
      C   = top 80%  (p20–35)
      D   = bottom 20% (below p20)
    """
    pts   = off_stats.get("points_pg", 0)
    goals = off_stats.get("goals_pg",  0)

    if season_off_df is not None and len(season_off_df) > 10:
        pts_col   = "points_per_game"  if "points_per_game"  in season_off_df.columns else None
        goals_col = "goals_per_game"   if "goals_per_game"   in season_off_df.columns else None

        def percentile_rank(val, col):
            if col is None:
                return 50.0
            vals = season_off_df[col].dropna()
            if len(vals) == 0:
                return 50.0
            return float((vals < val).mean() * 100)

        pts_pct   = percentile_rank(pts,   pts_col)
        goals_pct = percentile_rank(goals, goals_col)
    else:
        # Fallback: estimate percentile from 2024 D-man distribution
        # pts/gp:   p20=0.10, p35=0.17, p50=0.24, p75=0.36, p90=0.56
        # goals/gp: p20=0.03, p35=0.06, p50=0.08, p75=0.13, p90=0.20
        def estimate_pct(val, breakpoints):
            # breakpoints: list of (value, percentile) pairs
            for i in range(len(breakpoints) - 1):
                v0, p0 = breakpoints[i]
                v1, p1 = breakpoints[i + 1]
                if v0 <= val <= v1:
                    return p0 + (val - v0) / (v1 - v0) * (p1 - p0)
            if val < breakpoints[0][0]:
                return 0.0
            return 99.0

        pts_pct   = estimate_pct(pts,   [(0.10, 20), (0.17, 35), (0.24, 50), (0.36, 75), (0.56, 90), (1.15, 99)])
        goals_pct = estimate_pct(goals, [(0.03, 20), (0.06, 35), (0.08, 50), (0.13, 75), (0.20, 90), (0.40, 99)])

    # Composite percentile — 70% points, 30% goals
    composite_pct = pts_pct * 0.70 + goals_pct * 0.30

    if composite_pct >= 90:
        grade, desc = "A",  "Elite offensive D-man — top 10% in the league"
    elif composite_pct >= 75:
        grade, desc = "B+", "Above average offensively — top 25%"
    elif composite_pct >= 50:
        grade, desc = "B",  "Solid offensive production — above median"
    elif composite_pct >= 35:
        grade, desc = "C+", "Average offensive output for a defenseman"
    elif composite_pct >= 20:
        grade, desc = "C",  "Below average offensively — bottom third"
    else:
        grade, desc = "D",  "Minimal offensive contribution — bottom 20%"

    breakdown = {
        "Points/GP":  (round(pts,   3), round(pts_pct,   1)),
        "Goals/GP":   (round(goals, 3), round(goals_pct, 1)),
    }

    return grade, round(composite_pct, 1), desc, breakdown


def grade_defensive_defenseman(def_stats, season_def_df=None):
    """
    Grade a defenseman's defensive production on A-F scale using
    live percentile rank among all D-men in the dataset.

    Metrics and weights:
      xGA/60 (5v5)    — 40%  (lower is better)
      Takeaways/GP    — 25%
      Hits/GP         — 20%
      PIM/GP          — 15%  (lower is better)

    Grade cutoffs (percentile among all D-men):
      A   = top 10%  (p90+)
      B+  = top 25%  (p75–90)
      B   = top 50%  (p50–75)
      C+  = top 65%  (p35–50)
      C   = top 80%  (p20–35)
      D   = bottom 20%

    Returns (grade, composite_percentile, description, breakdown_dict)
    """
    hits = def_stats.get("ind_hits_pg",          0)
    tka  = def_stats.get("ind_takeaways_pg",     0)
    xga  = def_stats.get("xg_against_per60_5v5", 2.5)
    pim  = def_stats.get("pim_pg",               0)

    def percentile_rank(val, col, lower_is_better=False):
        if season_def_df is not None and col in season_def_df.columns:
            vals = season_def_df[col].dropna()
            if len(vals) > 10:
                if lower_is_better:
                    return float((vals > val).mean() * 100)
                return float((vals < val).mean() * 100)
        # Fallback: estimate from hardcoded 2024 D-man breakpoints
        breakpoints = {
            "ind_hits_pg":           [(0.3, 10), (0.7, 25), (1.1, 50), (1.8, 75), (2.5, 90), (4.0, 99)],
            "ind_takeaways_pg":      [(0.10, 10), (0.18, 25), (0.28, 50), (0.40, 75), (0.52, 90), (0.70, 99)],
            "xg_against_per60_5v5":  [(1.8, 99), (2.2, 90), (2.6, 75), (3.0, 50), (3.3, 25), (3.8, 10)],
            "pim_pg":                [(0.1, 99), (0.2, 90), (0.35, 75), (0.5, 50), (0.7, 25), (1.2, 10)],
        }
        pts = breakpoints.get(col, [])
        if not pts:
            return 50.0
        for i in range(len(pts) - 1):
            v0, p0 = pts[i]
            v1, p1 = pts[i + 1]
            if min(v0, v1) <= val <= max(v0, v1):
                return p0 + (val - v0) / (v1 - v0) * (p1 - p0)
        return 99.0 if val > pts[-1][0] else 0.0

    hits_pct = percentile_rank(hits, "ind_hits_pg")
    tka_pct  = percentile_rank(tka,  "ind_takeaways_pg")
    xga_pct  = percentile_rank(xga,  "xg_against_per60_5v5", lower_is_better=True)
    pim_pct  = percentile_rank(pim,  "pim_pg",               lower_is_better=True)

    composite_pct = (xga_pct * 0.30 + tka_pct * 0.25 + hits_pct * 0.25 + pim_pct * 0.20)

    if composite_pct >= 90:
        grade, desc = "A",  "Elite defensive D-man — top 10% in the league"
    elif composite_pct >= 75:
        grade, desc = "B+", "Above average defensively — top 25%"
    elif composite_pct >= 50:
        grade, desc = "B",  "Solid defensive contributor — above median"
    elif composite_pct >= 35:
        grade, desc = "C+", "Average defensive production"
    elif composite_pct >= 20:
        grade, desc = "C",  "Below average defensively — bottom third"
    else:
        grade, desc = "D",  "Minimal defensive contribution — bottom 20%"

    breakdown = {
        "xGA/60 (5v5)":  (round(xga,  2), round(xga_pct,  1)),
        "Takeaways/GP":  (round(tka,  3), round(tka_pct,  1)),
        "Hits/GP":       (round(hits, 2), round(hits_pct, 1)),
        "PIM/GP":        (round(pim,  2), round(pim_pct,  1)),
    }

    return grade, round(composite_pct, 1), desc, breakdown


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
    "xg_against_per60_5v5",
    "pim_pg",
]

DEF_TARGET_LABELS = {
    "ind_hits_pg":           "Hits / Game",
    "ind_takeaways_pg":      "Takeaways / Game",
    "xg_against_per60_5v5": "xGA Against / 60 (5v5)",
    "pim_pg":                "PIM / Game",
}

# Lower is better for these targets
DEF_LOWER_IS_BETTER = {"xg_against_per60_5v5", "pim_pg"}

# Defensive score weights (used for pairing)
DEF_SCORE_WEIGHTS = {
    "ind_hits_pg":           0.25,
    "ind_takeaways_pg":      0.25,
    "xg_against_per60_5v5": 0.30,
    "pim_pg":                0.20,
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
    "ind_hits_pg":           ["prev_season_hits_pg",     "recent_3yr_mean_hits_pg",     "career_prev_mean_hits_pg"],
    "ind_takeaways_pg":      ["prev_season_takeaways_pg","recent_3yr_mean_takeaways_pg","career_prev_mean_takeaways_pg"],
    "xg_against_per60_5v5": ["prev_season_xga_pg",      "recent_3yr_mean_xga_pg",      "career_prev_mean_xga_pg"],
    "pim_pg":                ["prev_season_pim_pg",      "recent_3yr_mean_pim_pg",      "career_prev_mean_pim_pg"],
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
    # On-ice defensive impact (xg_against_per60_5v5 is a target, not a feature)
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

# Non-linear career curve features for defensemen — included when age data is available.
# Physical defensive skills (hits, takeaways) decline non-linearly with age,
# and xGA suppression ability shifts across a defender's career arc.
DEF_NONLINEAR_FEATURES = [
    # Quadratic curve fit on prior seasons
    "def_curve_accel_hits",               # a coeff — negative = normal physical decline arc
    "def_curve_accel_takeaways",
    "def_curve_accel_xga",
    "def_curve_local_deriv_hits",         # 2a·age + b — positive = still improving
    "def_curve_local_deriv_takeaways",
    "def_curve_local_deriv_xga",
    "def_seasons_from_est_peak_hits",     # negative = pre-peak, positive = post-peak
    "def_seasons_from_est_peak_takeaways",
    # Slope of pct-of-peak — approaching or receding from career ceiling
    "def_pct_peak_hits_slope",
    "def_pct_peak_takeaways_slope",
    # Age × slope interactions — same hits slope means different things at 24 vs 33
    "def_age_x_3yr_hits_slope",
    "def_age_x_3yr_takeaways_slope",
    "def_age_x_3yr_xga_slope",
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
        # xGA per 60 5v5 — uses expected goals (more stable than actual)
        # already in dataset; fallback to computing from raw columns
        if "xg_against_per60_5v5" not in d.columns:
            fv5_hours = (d["fv5_ice_time"] / 3600).replace(0, np.nan)
            d["xg_against_per60_5v5"] = d["on_ice_against_expected_goals"] / fv5_hours

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
        ("xg_against_per60_5v5", "xga_pg"),
        ("pk_ice_pct",              "pk_pct"),
        ("ind_penalty_minutes_pg", "pim_pg"),
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
    d["yoy_xga_delta"]       = g["xg_against_per60_5v5"].diff()

    return d


def def_engineer_nonlinear_trajectory_features(df):
    """
    Non-linear career curve features for the defensive model.
    Mirrors engineer_nonlinear_trajectory_features for offensive model.
    No-op if age data is absent or sparse.

    Physical defensive skills (hits, takeaways, xGA suppression) follow
    non-linear career arcs — defenders often peak physically in their
    mid-to-late 20s and decline sharply through their 30s.
    """
    if "age" not in df.columns or df["age"].isna().mean() > 0.5:
        return df

    d = df.sort_values(["player_id", "season"]).copy()

    # Targets for quadratic fitting
    def_stats = {
        "hits":      "ind_hits_pg",
        "takeaways": "ind_takeaways_pg",
        "xga":       "xg_against_per60_5v5",
    }

    new_cols = (
        [f"def_curve_accel_{k}"               for k in def_stats] +
        [f"def_curve_local_deriv_{k}"         for k in def_stats] +
        ["def_seasons_from_est_peak_hits", "def_seasons_from_est_peak_takeaways"] +
        ["def_pct_peak_hits_slope", "def_pct_peak_takeaways_slope",
         "def_age_x_3yr_hits_slope", "def_age_x_3yr_takeaways_slope",
         "def_age_x_3yr_xga_slope"]
    )
    for col in new_cols:
        d[col] = np.nan

    for pid, grp in d.groupby("player_id", sort=False):
        idx      = grp.index
        ages_arr = grp["age"].values

        # ── Quadratic curve features ───────────────────────────────────────────
        for k, stat_col in def_stats.items():
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
                    a, b, _ = np.polyfit(pa, pv, 2)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                curr_age = ages_arr[i]
                if np.isnan(curr_age):
                    continue
                accel[i]       = a
                local_deriv[i] = 2.0 * a * curr_age + b
                # Only store peak distance for physically-driven stats (not xGA)
                if k in ("hits", "takeaways") and abs(a) > 1e-9:
                    from_peak[i] = curr_age - (-b / (2.0 * a))

            d.loc[idx, f"def_curve_accel_{k}"]       = accel
            d.loc[idx, f"def_curve_local_deriv_{k}"] = local_deriv
            if k in ("hits", "takeaways"):
                d.loc[idx, f"def_seasons_from_est_peak_{k}"] = from_peak

        # ── Slope of pct-of-peak for physical stats ────────────────────────────
        for stat_short, peak_col in [
            ("hits",      "pct_of_peak_hits"),
            ("takeaways", "pct_of_peak_takeaways"),
        ]:
            if peak_col not in grp.columns:
                continue
            peak_vals = grp[peak_col].values
            pct_slope = np.full(len(grp), np.nan)
            for i in range(len(grp)):
                window = peak_vals[max(0, i - 3):i]
                mask   = ~np.isnan(window)
                if mask.sum() < 2:
                    continue
                y = window[mask]
                x = np.arange(len(window))[mask].astype(float)
                xm, ym = x.mean(), y.mean()
                denom  = ((x - xm) ** 2).sum()
                pct_slope[i] = 0.0 if denom == 0 else float(
                    ((x - xm) * (y - ym)).sum() / denom
                )
            d.loc[idx, f"def_pct_peak_{stat_short}_slope"] = pct_slope

        # ── Age × slope interactions ───────────────────────────────────────────
        for slope_col, out_col in [
            ("recent_3yr_hits_slope",      "def_age_x_3yr_hits_slope"),
            ("recent_3yr_takeaways_slope", "def_age_x_3yr_takeaways_slope"),
            ("recent_3yr_xga_slope",       "def_age_x_3yr_xga_slope"),
        ]:
            if slope_col in grp.columns:
                d.loc[idx, out_col] = ages_arr * grp[slope_col].values

    return d


def def_build_team_context(df):
    """Aggregate team-level defensive context per season."""
    team_ctx = (
        df.groupby(["player_team", "season"])
        .agg(
            team_avg_hits_pg         = ("ind_hits_pg",             "mean"),
            team_avg_takeaways_pg    = ("ind_takeaways_pg",        "mean"),
            team_avg_xga_per60       = ("xg_against_per60_5v5", "mean"),
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
    feats = (DEF_PLAYER_FEATURES
             + (DEF_AGE_FEATURES        if has_age else [])
             + (DEF_NONLINEAR_FEATURES  if has_age else [])
             + DEF_TEAM_FEATURES)
    feats = [f for f in feats if f in df.columns]
    X     = df[feats].copy()
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def def_build_next_feature_matrix(df, has_age):
    traj  = [f for f in DEF_TRAJECTORY_FEATURES if f in df.columns]
    feats = (DEF_PLAYER_FEATURES
             + (DEF_AGE_FEATURES        if has_age else [])
             + (DEF_NONLINEAR_FEATURES  if has_age else [])
             + traj + DEF_TEAM_FEATURES)
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
    df = _safe_read_csv(def_path)
    df = df[df["games_played"] >= MIN_GP].copy()

    ages = _load_ages(ages_path)
    df   = df.merge(ages, on=["player_id", "season"], how="left")
    has_age = df["age"].notna().mean() > 0.5
    advance(f"Data loaded — {len(df):,} defenseman-seasons | age matched: {df['age'].notna().sum():,}")

    # Engineer
    status.markdown("⚙️ **Engineering features...**")
    df       = def_engineer_features(df)
    df       = def_engineer_career_history(df)
    df       = def_engineer_nonlinear_trajectory_features(df)   # non-linear curve signals
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
        nl = DEF_NONLINEAR_FEATURES if has_age else []
        if use_traj:
            traj  = [f for f in DEF_TRAJECTORY_FEATURES if f in pred_df.columns]
            feats = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + nl + traj + DEF_TEAM_FEATURES
        else:
            feats = DEF_PLAYER_FEATURES + (DEF_AGE_FEATURES if has_age else []) + nl + DEF_TEAM_FEATURES
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


def classify_defenseman_type(scores, def_score=None, off_score=None,
                             season_def_df=None, season_off_df=None):
    """
    Classify a defenseman into Offensive D, Defensive D, or Two-Way D
    using the gap between their defensive and offensive percentile scores.

    If pre-computed scores are not passed in, they are computed here.

    Classification rules (based on percentile gap):
      |off - def| <= 20  →  Two-Way D   (balanced)
      off - def  >  20   →  Offensive D (offense dominates)
      def - off  >  20   →  Defensive D (defense dominates)
    """
    if def_score is None:
        _, def_score, _, _ = grade_defensive_defenseman(scores, season_def_df)
    if off_score is None:
        off_stats = {
            "points_pg": scores.get("points_per_game", scores.get("points_pg", 0)),
            "goals_pg":  scores.get("goals_per_game",  scores.get("goals_pg",  0)),
        }
        _, off_score, _, _ = grade_offensive_defenseman(off_stats, season_off_df)

    gap = off_score - def_score

    if gap > 30:
        return "Offensive D", (
            "Offensive defenceman — exceptional skating and puck-handling, "
            "creates scoring opportunities but may be vulnerable defensively."
        )
    elif gap < -30:
        return "Defensive D", (
            "Defensive defenceman — physical, blocks shots, clears the zone. "
            "Strong defensively but may struggle generating offense."
        )
    else:
        return "Two-Way D", (
            "Two-way defenceman — well-rounded with contributions at both ends. "
            "Versatile and reliable in any situation."
        )


def def_compute_defensive_score(df_preds, season_df=None):
    """
    Compute a composite defensive score 0-100 for each row.

    Normalises each metric against fixed league-wide empirical ranges
    rather than within the prediction set — avoids the collapse-to-45
    issue when all 32-team predictions are similar.
    """
    # Empirical league-wide ranges for D-men (from 2024 season analysis)
    # Format: {target: (p5, p95)} — values outside this range are clipped
    LEAGUE_RANGES = {
        "ind_hits_pg":           (0.0,  3.5),
        "ind_takeaways_pg":      (0.0,  0.65),
        "xg_against_per60_5v5": (1.8,  3.5),
        "pim_pg":                (0.0,  1.2),
    }

    result = df_preds.copy()
    score  = np.zeros(len(result))

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
                        fit_feature_names=None, next_feature_names=None,
                        season_df=None):
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
    _api_name        = fetch_player_display_name(int(pid))
    matched          = _api_name if _api_name else profile["player_name"]

    all_teams = def_get_latest_team_contexts(df, team_ctx)

    # Current fit predictions
    fit_results  = def_build_all_team_predictions(profile, all_teams, fit_models, has_age,
                                                    feature_names=fit_feature_names)
    fit_results  = def_compute_defensive_score(fit_results, season_df=season_df)
    fit_results  = fit_results.sort_values("defensive_score", ascending=False).reset_index(drop=True)
    fit_results.index += 1
    fit_results["is_actual"] = fit_results["player_team"] == actual_team

    # Next season predictions
    next_results = def_build_all_team_predictions(profile, all_teams, next_models, has_age,
                                                     use_traj=True, feature_names=next_feature_names)
    next_results = def_compute_defensive_score(next_results, season_df=season_df)
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
                "player_id":    int(p["id"]),
                "player_name":  f"{p['firstName']['default']} {p['lastName']['default']}",
                "position":     "D",
                "shoots":       p.get("shootsCatches", ""),  # "L" or "R"
            })
        return players
    except Exception as e:
        return [{"_error": str(e)}]



import json as _json_mod
import time as _time_mod

def _shifts_cache_path(team_code: str, n_games: int) -> str:
    os.makedirs(SHIFTS_CACHE_DIR, exist_ok=True)
    return os.path.join(SHIFTS_CACHE_DIR, f"{team_code}_{n_games}.json")


def _load_shifts_disk_cache(team_code: str, n_games: int):
    """Return (pairs_list, err_str) from disk cache, or None if missing/stale."""
    path = _shifts_cache_path(team_code, n_games)
    try:
        if not os.path.exists(path):
            return None
        age_seconds = _time_mod.time() - os.path.getmtime(path)
        if age_seconds > SHIFTS_CACHE_TTL_H * 3600:
            return None                   # stale — will re-fetch
        with open(path, "r") as f:
            payload = _json_mod.load(f)
        pairs = [tuple(p) for p in payload["pairs"]]   # list of (pid1, pid2, toi)
        return pairs, payload.get("err")
    except Exception:
        return None


def _save_shifts_disk_cache(team_code: str, n_games: int, pairs, err):
    """Persist (pairs_list, err_str) to disk so the next reload is instant."""
    path = _shifts_cache_path(team_code, n_games)
    try:
        payload = {"pairs": [list(p) for p in pairs], "err": err}
        with open(path, "w") as f:
            _json_mod.dump(payload, f)
    except Exception:
        pass   # disk write failure is non-fatal


def _pairs_run_games(finished, team_code, d_pids_set, on_progress=None):
    """
    Core per-game loop shared by fetch_actual_pairs (cached) and
    stream_fetch_actual_pairs (uncached with live progress).

    on_progress(done, total) is called after every game if provided.
    Returns (pairs_list, error_str_or_None).
    """
    from collections import defaultdict

    def _to_secs(t):
        if isinstance(t, (int, float)):
            return int(t)
        try:
            parts = str(t).split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 0

    pair_toi = defaultdict(int)
    errors   = []
    total    = len(finished)

    for idx, game in enumerate(finished):
        game_id = game.get("id")
        if game_id:
            try:
                shift_url  = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
                shift_resp = requests.get(shift_url, timeout=15)
                shift_resp.raise_for_status()
                shifts_raw = shift_resp.json().get("data", [])

                team_shifts = [
                    s for s in shifts_raw
                    if s.get("teamAbbrev") == team_code
                    and s.get("detailCode") == 0
                    and (d_pids_set is None or s.get("playerId") in d_pids_set)
                ]

                by_period = defaultdict(list)
                for s in team_shifts:
                    by_period[s["period"]].append(s)

                for period, period_shifts in by_period.items():
                    period_shifts.sort(key=lambda x: _to_secs(x.get("startTime", 0)))
                    n = len(period_shifts)
                    for i in range(n):
                        si      = period_shifts[i]
                        pid_i   = si.get("playerId")
                        start_i = _to_secs(si.get("startTime", 0))
                        end_i   = _to_secs(si.get("endTime",   0))
                        for j in range(i + 1, n):
                            sj      = period_shifts[j]
                            start_j = _to_secs(sj.get("startTime", 0))
                            if start_j >= end_i:
                                break
                            pid_j = sj.get("playerId")
                            if pid_i == pid_j:
                                continue
                            end_j   = _to_secs(sj.get("endTime", 0))
                            overlap = min(end_i, end_j) - max(start_i, start_j)
                            if overlap > 0:
                                key = tuple(sorted([pid_i, pid_j]))
                                pair_toi[key] += overlap
            except Exception as e:
                errors.append(f"game {game_id}: {e}")

        if on_progress:
            on_progress(idx + 1, total)

    if not pair_toi:
        err_msg = (f"Shift data unavailable ({'; '.join(errors[:3])})."
                   if errors else "No shift overlap data found.")
        return [], err_msg

    pairs = sorted(pair_toi.items(), key=lambda x: x[1], reverse=True)
    return [(p[0], p[1], toi) for (p, toi) in pairs], None


def _pairs_fetch_schedule(team_code, n_games):
    """Fetch and filter the n most recent finished regular-season games."""
    sched_url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_code}/now"
    resp = requests.get(sched_url, timeout=15)
    resp.raise_for_status()
    all_games = resp.json().get("games", [])
    finished = [
        g for g in all_games
        if g.get("gameType", 2) == 2
        and g.get("gameState") not in ("FUT", "PRE", "PREVIEW")
    ]
    finished = sorted(finished, key=lambda g: g.get("gameDate", ""), reverse=True)
    return finished[:n_games]


@st.cache_data(ttl=21600, show_spinner=False)   # 6-hour TTL — season data barely changes
def fetch_actual_pairs(team_code, d_pids=None, n_games=25):
    """
    Fetch actual D-pair combinations from NHL shift chart data.
    Checks disk cache first (survives app restarts); falls back to NHL API.
    n_games: how many of the most recent games to include (default 25).
    d_pids: frozenset of defenseman player IDs to filter shifts.
    Returns a list of (pid1, pid2, shared_seconds) sorted by shared TOI desc.
    """
    cached = _load_shifts_disk_cache(team_code, n_games)
    if cached is not None:
        return cached
    d_pids_set = set(d_pids) if d_pids else None
    try:
        finished = _pairs_fetch_schedule(team_code, n_games)
        if not finished:
            return [], "No finished regular-season games found for this team."
        result = _pairs_run_games(finished, team_code, d_pids_set)
        _save_shifts_disk_cache(team_code, n_games, result[0], result[1])
        return result
    except Exception as e:
        return [], str(e)


def stream_fetch_actual_pairs(team_code, d_pids=None, n_games=25, on_progress=None):
    """
    Like fetch_actual_pairs but NOT cached by Streamlit; accepts an
    on_progress(done, total) callback called after each game.

    Checks disk cache first — if a fresh file exists the result is returned
    instantly with no API calls and the progress bar is skipped.
    """
    cached = _load_shifts_disk_cache(team_code, n_games)
    if cached is not None:
        return cached
    d_pids_set = set(d_pids) if d_pids else None
    try:
        finished = _pairs_fetch_schedule(team_code, n_games)
        if not finished:
            return [], "No finished regular-season games found for this team."
        result = _pairs_run_games(finished, team_code, d_pids_set, on_progress=on_progress)
        _save_shifts_disk_cache(team_code, n_games, result[0], result[1])
        return result
    except Exception as e:
        return [], str(e)


def build_actual_pairing_insertion(player_id, team_code, df, team_ctx,
                                    fit_models, player_profiles, has_age,
                                    feature_names=None, n_games=25,
                                    _prefetched_pairs=None):
    """
    Build pairing view using ACTUAL NHL pair combinations from shift data.
    Shows the real current pairs with model defensive scores, then inserts
    the searched player into their best fit pair slot.
    """
    # 1. Fetch current roster and actual pair combinations
    roster = def_fetch_team_roster_d(team_code)
    if not roster:
        return [], [], {}, [], [], {"pair_err": "Could not fetch roster (empty response).", "partner_name": "—", "partner_slot": "—", "searched_score": 0, "actual_pairs": []}
    if roster and "_error" in roster[0]:
        err_msg = roster[0]["_error"]
        return [], [], {}, [], [], {"pair_err": f"Could not fetch roster: {err_msg}", "partner_name": "—", "partner_slot": "—", "searched_score": 0, "actual_pairs": []}

    roster_pids  = {p["player_id"]: p["player_name"] for p in roster}
    roster_shoots = {p["player_id"]: p.get("shoots", "") for p in roster}
    # Get new player's handedness from bios API if not in roster
    # (they may be from another team)
    new_player_shoots = roster_shoots.get(player_id, "")
    if not new_player_shoots:
        try:
            bio_url  = (f"https://api.nhle.com/stats/rest/en/skater/bios"
                        f"?limit=1&cayenneExp=playerId={player_id} and seasonId={CURRENT_SEASON}")
            bio_resp = requests.get(bio_url, timeout=8)
            if bio_resp.ok:
                bio_data = bio_resp.json().get("data", [])
                if bio_data:
                    new_player_shoots = bio_data[0].get("shootsCatches", "")
        except Exception:
            pass

    # Get all team predictions from model
    all_teams = def_get_latest_team_contexts(df, team_ctx)
    team_row  = all_teams[all_teams["player_team"] == team_code]
    if team_row.empty:
        return [], [], {}, [], [], {"pair_err": f"No team context for {team_code}.", "partner_name": "—", "partner_slot": "—", "searched_score": 0, "actual_pairs": []}
    team_row = team_row.iloc[0]

    if player_id not in player_profiles:
        return [], [], {}, [], [], {"pair_err": "Player not found in model data.", "partner_name": "—", "partner_slot": "—", "searched_score": 0, "actual_pairs": []}

    # 2. Get model predictions for every rostered D-man
    player_scores = {}  # pid -> {metric: val, defensive_score: val, player_name: str}
    # Load offensive stats for grading
    d_off_stats, d_off_df, _ = load_defensive_offensive_stats()

    def _combined_score(def_pct, off_pct, d_type):
        """Blend defensive and offensive percentile scores by player type."""
        if d_type == "Two-Way D":
            return round(def_pct * 0.5 + off_pct * 0.5, 1)
        elif d_type == "Offensive D":
            return round(def_pct * 0.3 + off_pct * 0.7, 1)
        else:  # Defensive D
            return round(def_pct * 0.8 + off_pct * 0.2, 1)

    for pid, pname in roster_pids.items():
        if pid not in player_profiles:
            continue
        prof, _ = player_profiles[pid]
        preds   = def_predict_for_team(prof, team_row, fit_models, has_age,
                                        feature_names=feature_names)
        _, def_score, _, _ = grade_defensive_defenseman(preds, season_def_df=df)
        _, off_score, _, _ = grade_offensive_defenseman(d_off_stats.get(pid, {}), season_off_df=d_off_df)
        d_type, d_desc = classify_defenseman_type(dict(prof), def_score=def_score, off_score=off_score)
        player_scores[pid] = {
            **preds,
            "defensive_score": def_score,
            "combined_score":  _combined_score(def_score, off_score, d_type),
            "player_name":     pname,
            "d_type":          d_type,
            "d_desc":          d_desc,
            "shoots":          roster_shoots.get(pid, ""),
        }

    # Also score the searched player
    search_profile, _ = player_profiles[player_id]
    search_preds = def_predict_for_team(search_profile, team_row, fit_models, has_age,
                                         feature_names=feature_names)
    _, s_def_score, _, _ = grade_defensive_defenseman(search_preds, season_def_df=df)
    _, s_off_score, _, _ = grade_offensive_defenseman(d_off_stats.get(player_id, {}), season_off_df=d_off_df)
    s_type, s_desc = classify_defenseman_type(dict(search_profile), def_score=s_def_score, off_score=s_off_score)
    player_scores[player_id] = {
        **search_preds,
        "defensive_score":    s_def_score,
        "combined_score":     _combined_score(s_def_score, s_off_score, s_type),
        "player_name":        search_profile.get("player_name", "Selected Player"),
        "is_searched_player": True,
        "d_type":             s_type,
        "d_desc":             s_desc,
        "shoots":             new_player_shoots,
    }

    # 3. Fetch actual pairs from shift data — pass D-men pids to filter shifts
    d_pids = set(roster_pids.keys()) | {player_id}
    if _prefetched_pairs is not None:
        actual_pairs, pair_err = _prefetched_pairs
    else:
        actual_pairs, pair_err = fetch_actual_pairs(team_code, d_pids=frozenset(d_pids), n_games=n_games)

    SLOT_NAMES      = ["1st Pair", "2nd Pair", "3rd Pair", "4th Pair"]
    MAX_DISPLAY     = 3
    MAX_BUILD_PAIRS = 4

    # ── Returning-player fast path ─────────────────────────────────────────────
    # If the searched player is already on this team's roster, skip the cascade
    # entirely and just show the real season-long pairs with them highlighted.
    if player_id in roster_pids:
        depth_pairs  = []
        assigned     = set()
        cascade_log  = []   # no cascade happened
        scratched    = []

        # Build pairs from shift TOI data — include the searched player this time
        if actual_pairs:
            for pid1, pid2, toi in actual_pairs:
                if pid1 not in player_scores or pid2 not in player_scores:
                    continue
                if pid1 in assigned or pid2 in assigned:
                    continue
                shoots1  = player_scores[pid1].get("shoots", "")
                shoots2  = player_scores[pid2].get("shoots", "")
                hand_ok  = bool(shoots1 and shoots2 and shoots1 != shoots2)
                depth_pairs.append({
                    "pid1":       pid1,
                    "pid2":       pid2,
                    "name1":      player_scores[pid1]["player_name"],
                    "name2":      player_scores[pid2]["player_name"],
                    "score1":     player_scores[pid1]["combined_score"],
                    "score2":     player_scores[pid2]["combined_score"],
                    "shoots1":    shoots1,
                    "shoots2":    shoots2,
                    "hand_match": hand_ok,
                    "pair_score": round((player_scores[pid1]["combined_score"] +
                                         player_scores[pid2]["combined_score"]) / 2, 1),
                    "from_shifts": True,
                    "slot":       "",
                    "slot_color": "",
                })
                assigned.update([pid1, pid2])
                if len(depth_pairs) == MAX_DISPLAY:
                    break

        # Fill any remaining slots from unassigned pool if shift data was sparse
        unassigned_pool = [
            pid for pid in player_scores
            if pid not in assigned and pid in roster_pids
        ]
        unassigned_pool.sort(key=lambda p: player_scores[p]["combined_score"], reverse=True)
        while len(depth_pairs) < MAX_DISPLAY and len(unassigned_pool) >= 2:
            pid1 = unassigned_pool.pop(0)
            pid2 = unassigned_pool.pop(0)
            shoots1  = player_scores[pid1].get("shoots", "")
            shoots2  = player_scores[pid2].get("shoots", "")
            hand_ok  = bool(shoots1 and shoots2 and shoots1 != shoots2)
            depth_pairs.append({
                "pid1":       pid1,
                "pid2":       pid2,
                "name1":      player_scores[pid1]["player_name"],
                "name2":      player_scores[pid2]["player_name"],
                "score1":     player_scores[pid1]["combined_score"],
                "score2":     player_scores[pid2]["combined_score"],
                "shoots1":    shoots1,
                "shoots2":    shoots2,
                "hand_match": hand_ok,
                "pair_score": round((player_scores[pid1]["combined_score"] +
                                     player_scores[pid2]["combined_score"]) / 2, 1),
                "from_shifts": False,
                "slot":       "",
                "slot_color": "",
            })
            assigned.update([pid1, pid2])

        # Assign slot labels in TOI order
        for i, pair in enumerate(depth_pairs):
            slot = SLOT_NAMES[i] if i < len(SLOT_NAMES) else f"Pair {i+1}"
            pair["slot"] = slot
            pair["slot_color"] = DEF_PAIR_COLORS.get(" ".join(slot.split()[:2]), "#888888")

        # Find this player's partner and slot
        best_partner_name = "—"
        best_partner_slot = "—"
        searched_score    = player_scores[player_id]["combined_score"]
        for pair in depth_pairs:
            if pair["pid1"] == player_id:
                best_partner_name = pair["name2"]
                best_partner_slot = pair["slot"]
            elif pair["pid2"] == player_id:
                best_partner_name = pair["name1"]
                best_partner_slot = pair["slot"]

        unmodeled = [pid for pid in roster_pids if pid not in player_scores]
        return depth_pairs, scratched, player_scores, cascade_log, unmodeled, {
            "partner_name":    best_partner_name,
            "partner_slot":    best_partner_slot,
            "searched_score":  searched_score,
            "pair_err":        pair_err,
            "actual_pairs":    actual_pairs,
        }

    # 4. Player is from another team — build baseline pairs (without new player)
    #    then cascade them in, rippling displaced players down.
    baseline_pairs = []
    assigned = set()
    if actual_pairs:
        for pid1, pid2, toi in actual_pairs:
            if pid1 not in player_scores or pid2 not in player_scores:
                continue
            if pid1 in assigned or pid2 in assigned:
                continue
            if pid1 == player_id or pid2 == player_id:
                continue
            baseline_pairs.append([pid1, pid2])
            assigned.update([pid1, pid2])

    # Any rostered D-man not in shift pairs — add as unassigned pool
    unassigned_pool = [
        pid for pid in player_scores
        if pid not in assigned and pid != player_id and pid in roster_pids
    ]
    # Fill remaining pair slots with unassigned players (sorted by score)
    unassigned_pool.sort(key=lambda p: player_scores[p]["combined_score"], reverse=True)
    while len(baseline_pairs) < MAX_BUILD_PAIRS and len(unassigned_pool) >= 2:
        baseline_pairs.append([unassigned_pool.pop(0), unassigned_pool.pop(0)])
    remaining_solo = unassigned_pool  # odd man out

    # ── Cascade insertion ──────────────────────────────────────────────────────
    # The new player "tries out" for each pair slot from top to bottom.
    # If they outscore the weaker partner in a pair, they take that spot.
    # The displaced player then cascades DOWN to try the next pair only —
    # start_from ensures no displaced player can leapfrog back up the chart.

    cascade_log  = []   # track each move: (player, action, slot)
    final_pairs  = [list(p) for p in baseline_pairs]  # mutable copy
    to_place     = player_id   # start: new player needs a slot
    scratched    = []
    start_from   = 0   # minimum pair index the current to_place may try

    MAX_ITER = len(final_pairs) + 2
    for iteration in range(MAX_ITER):
        placed = False
        new_s      = player_scores[to_place]["combined_score"]
        new_shoots = player_scores[to_place].get("shoots", "")

        for i, pair in enumerate(final_pairs):
            if i < start_from:          # only try pairs at or below displacement point
                continue
            p1, p2   = pair
            s1       = player_scores[p1]["combined_score"]
            s2       = player_scores[p2]["combined_score"]
            weaker   = p1 if s1 <= s2 else p2
            stronger = p2 if s1 <= s2 else p1
            weaker_s = min(s1, s2)

            # Handedness bonus: if new player is opposite-handed to stronger partner,
            # they get a +3 compatibility boost (not enough to override score significantly
            # but enough to prefer a matched pair over an equally-scored unmatched one)
            stronger_shoots = player_scores[stronger].get("shoots", "")
            hand_bonus = 3.0 if (new_shoots and stronger_shoots and
                                  new_shoots != stronger_shoots) else 0.0

            if (new_s + hand_bonus) > weaker_s:
                # New player displaces the weaker
                slot = SLOT_NAMES[i] if i < len(SLOT_NAMES) else f"Pair {i+1}"
                cascade_log.append({
                    "player":    player_scores[to_place]["player_name"],
                    "action":    "moved in",
                    "slot":      slot,
                    "displaced": player_scores[weaker]["player_name"],
                })
                # Replace weaker in pair
                pair[pair.index(weaker)] = to_place
                start_from = i + 1      # displaced player can only go lower than here
                to_place   = weaker     # displaced player now needs a slot
                placed     = True
                break

        if not placed:
            # Nobody displaced — to_place goes to scratch
            cascade_log.append({
                "player":    player_scores[to_place]["player_name"],
                "action":    "scratched",
                "slot":      "—",
                "displaced": "—",
            })
            scratched.append(to_place)
            break

    # ── Build final depth pairs ────────────────────────────────────────────────
    depth_pairs = []
    for i, (p1, p2) in enumerate(final_pairs):
        slot     = SLOT_NAMES[i] if i < len(SLOT_NAMES) else f"Pair {i+1}"
        shoots1  = player_scores[p1].get("shoots", "")
        shoots2  = player_scores[p2].get("shoots", "")
        hand_ok  = bool(shoots1 and shoots2 and shoots1 != shoots2)
        depth_pairs.append({
            "pid1":        p1,
            "pid2":        p2,
            "name1":       player_scores[p1]["player_name"],
            "name2":       player_scores[p2]["player_name"],
            "score1":      player_scores[p1]["combined_score"],
            "score2":      player_scores[p2]["combined_score"],
            "shoots1":     shoots1,
            "shoots2":     shoots2,
            "hand_match":  hand_ok,
            "pair_score":  round((player_scores[p1]["combined_score"] +
                                  player_scores[p2]["combined_score"]) / 2, 1),
            "slot":        slot,
            "slot_color":  DEF_PAIR_COLORS.get(" ".join(slot.split()[:2]), "#888888"),
            "from_shifts": any(
                (p1 in (ap[0], ap[1]) and p2 in (ap[0], ap[1]))
                for ap in actual_pairs
            ) if actual_pairs else False,
        })

    # Slot labels are assigned by cascade order (shift-TOI order with new player inserted).
    # No score-based re-sort — the cascade already placed every player into the highest
    # slot they could earn. Re-sorting by score contradicts the cascade and causes
    # the searched player's pair to jump or fall based on their partner's score rather
    # than their own merit.
    depth_pairs = depth_pairs[:MAX_DISPLAY]
    for i, pair in enumerate(depth_pairs):
        slot = SLOT_NAMES[i] if i < len(SLOT_NAMES) else f"Pair {i+1}"
        pair["slot"] = slot
        pair["slot_color"] = DEF_PAIR_COLORS.get(" ".join(slot.split()[:2]), "#888888")

    # ── Find partner of new player ─────────────────────────────────────────────
    best_partner_name = "—"
    best_partner_slot = "—"
    searched_score    = player_scores[player_id]["combined_score"]
    for pair in depth_pairs:
        if pair["pid1"] == player_id:
            best_partner_name = pair["name2"]
            best_partner_slot = pair["slot"]
        elif pair["pid2"] == player_id:
            best_partner_name = pair["name1"]
            best_partner_slot = pair["slot"]

    # Players with no model data
    unmodeled = [pid for pid in roster_pids if pid not in player_scores]

    return depth_pairs, scratched, player_scores, cascade_log, unmodeled, {
        "partner_name":    best_partner_name,
        "partner_slot":    best_partner_slot,
        "searched_score":  searched_score,
        "pair_err":        pair_err,
        "actual_pairs":    actual_pairs,
    }


def def_build_pairing_insertion(player_id, team_code, df, team_ctx,
                             fit_models, player_profiles, has_age,
                             feature_names=None, n_games=25,
                             _prefetched_pairs=None):
    """Wrapper that calls the actual-pair-based insertion."""
    return build_actual_pairing_insertion(
        player_id, team_code, df, team_ctx,
        fit_models, player_profiles, has_age, feature_names, n_games=n_games,
        _prefetched_pairs=_prefetched_pairs,
    )



# ── Charts ─────────────────────────────────────────────────────────────────────

def def_make_bar_chart(results, player_name, actual_team, title):
    """
    Interactive Plotly bar chart for defensive stats.
    Lower-is-better axes are inverted. Fullscreen icon (⛶) to expand.
    """
    metric_cols   = ["ind_hits_pg", "ind_takeaways_pg",
                     "xg_against_per60_5v5", "pim_pg", "defensive_score"]
    metric_labels = ["Hits / Game", "Takeaways / Game",
                     "xGA / 60 (↓ better)", "PIM / Game (↓ better)", "Def Score"]

    fig = make_subplots(rows=1, cols=5, subplot_titles=metric_labels,
                        horizontal_spacing=0.06)

    for col_idx, (col, label) in enumerate(zip(metric_cols, metric_labels), start=1):
        lower_better = col in DEF_LOWER_IS_BETTER
        sr     = results.sort_values(col, ascending=lower_better)
        team_primary   = get_team_color(actual_team, "primary")
        team_secondary = get_team_color(actual_team, "secondary")
        colors = [team_primary if t == actual_team else "#4a90d9" for t in sr["player_team"]]
        vals   = sr[col].values

        fig.add_trace(
            go.Bar(
                x=vals,
                y=sr["player_team"],
                orientation="h",
                marker_color=colors,
                marker_line_color=[team_secondary if t == actual_team else "#4a90d9"
                                   for t in sr["player_team"]],
                marker_line_width=[2 if t == actual_team else 0 for t in sr["player_team"]],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )

        actual_val = float(results.loc[results["player_team"] == actual_team, col].values[0])
        fig.add_vline(
            x=actual_val, line_color=team_primary, line_dash="dash", line_width=2,
            row=1, col=col_idx,
        )

        spread = vals.max() - vals.min()
        pad    = max(spread * 0.1, vals.max() * 0.005)
        x_range = [vals.min() - pad, vals.max() + pad]
        if lower_better:
            x_range = x_range[::-1]  # invert for lower-is-better

        fig.update_xaxes(
            range=x_range, row=1, col=col_idx,
            gridcolor="#2d3748", zerolinecolor="#2d3748",
            tickfont=dict(color="#aaa", size=8),
        )
        fig.update_yaxes(
            row=1, col=col_idx,
            tickfont=dict(color="#aaa", size=8),
            gridcolor="#2d3748",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=12)),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=700,
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(color="white"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "white"
        ann.font.size  = 11

    return fig


def def_show_results_table(results, actual_team, context_window=5):
    display = results[[
        "player_team", "ind_hits_pg", "ind_takeaways_pg",
        "xg_against_per60_5v5", "pim_pg",
        "defensive_score", "is_actual"
    ]].copy()
    display.columns = [
        "Team", "Hits/GP", "TK/GP", "xGA/60",
        "PEN/GP", "Def Score", "_is_actual"
    ]
    for col in ["Hits/GP", "TK/GP", "xGA/60", "PEN/GP", "Def Score"]:
        display[col] = display[col].round(3)
    display.insert(0, "Rank", range(1, len(display) + 1))

    actual_idx = display.index[display["_is_actual"]].tolist()
    rank_val   = int(display.loc[actual_idx[0], "Rank"]) if actual_idx else "?"

    render = display.drop(columns=["_is_actual"]).reset_index(drop=True)
    _render_scrollable_table(
        render, display["_is_actual"].reset_index(drop=True),
        actual_team, rank_val, len(display), context_window,
        team_color=get_team_color(actual_team)
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
CONFIDENCE_DECAY = {1: 1.0, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.40, 6: 0.30, 7: 0.25}


def get_age_adjusted_confidence(year, age):
    """
    Return confidence adjusted for both time decay and player age.
    Older players have steeper decay — a 34-year-old in year 4
    is far less predictable than a 26-year-old.
    """
    base = CONFIDENCE_DECAY.get(year, 0.25)
    if age >= 34:
        age_penalty = 0.82 ** (year - 1)   # steep — late career volatility
    elif age >= 31:
        age_penalty = 0.90 ** (year - 1)   # moderate decline risk
    elif age >= 28:
        age_penalty = 0.95 ** (year - 1)   # slight — early decline range
    else:
        age_penalty = 1.00                  # young — no extra penalty
    return round(min(base * age_penalty, 1.0), 3)


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
    Defensemen use per-metric curves — physical play (hits) declines faster
    than hockey sense (xGA/takeaways), and PIM tends to rise slightly with age.
    """
    p = profile.copy()
    current_age = float(p.get("age", 28))

    new_age = current_age + years_ahead
    p["age"]    = new_age
    p["age_sq"] = new_age ** 2

    if is_defenseman:
        # Per-metric age multipliers — different skills age at different rates
        def _metric_mult(col, age, n_years):
            """Compound annual multiplier specific to each defensive metric."""
            mult = 1.0
            for y in range(n_years):
                a = age + y
                if col == "ind_hits_per60":
                    # Physical play declines fastest
                    if a >= 34:   m = 0.91
                    elif a >= 31: m = 0.95
                    elif a >= 28: m = 0.98
                    else:         m = 1.00
                elif col in ("ind_takeaways_per60",):
                    # Hockey sense holds longer
                    if a >= 35:   m = 0.94
                    elif a >= 32: m = 0.97
                    elif a >= 29: m = 0.99
                    else:         m = 1.00
                elif col in ("hd_shots_against_per60_5v5", "on_ice_corsi_pct"):
                    # Positioning/reads — most stable
                    if a >= 36:   m = 0.95
                    elif a >= 33: m = 0.98
                    else:         m = 1.00
                elif col == "ind_giveaways_per60":
                    # Decision-making improves until ~30 then slowly declines
                    if a >= 34:   m = 1.02
                    elif a >= 30: m = 1.00
                    else:         m = 0.99
                else:
                    m = get_age_multiplier(a, is_defenseman=True)
                mult *= m
            return mult

        skill_cols = ["ind_hits_per60", "ind_takeaways_per60", "ind_giveaways_per60",
                      "shots_blocked_by_player_per60",
                      "hd_shots_against_per60_5v5", "on_ice_corsi_pct"]
        for col in skill_cols:
            if col in p.index:
                p[col] = float(p[col]) * _metric_mult(col, current_age, years_ahead)

        # PIM increases slightly with age (slower players resort to obstruction)
        # peaks around 32-34 then drops as ice time is reduced
        pim_mult = 1.0
        for y in range(years_ahead):
            a = current_age + y
            if a <= 30:   pim_mult *= 1.01
            elif a <= 33: pim_mult *= 1.02
            elif a <= 35: pim_mult *= 1.00
            else:         pim_mult *= 0.97
        for col in ["ind_penalty_minutes_pg", "pim_pg",
                    "prev_season_pim_pg", "recent_3yr_mean_pim_pg"]:
            if col in p.index:
                p[col] = float(p[col]) * pim_mult

        # Update career history features
        cumulative_mult = get_age_multiplier(current_age, is_defenseman=True) ** years_ahead
        for col in ["prev_season_hits_pg", "recent_3yr_mean_hits_pg",
                    "prev_season_takeaways_pg", "recent_3yr_mean_takeaways_pg",
                    "prev_season_xga_pg", "recent_3yr_mean_xga_pg"]:
            if col in p.index:
                p[col] = float(p[col]) * cumulative_mult

    else:
        # Forward — uniform age curve
        cumulative_mult = 1.0
        for y in range(years_ahead):
            cumulative_mult *= get_age_multiplier(current_age + y, is_defenseman=False)

        skill_cols = ["finishing_skill", "finishing_skill_adj", "ind_shot_attempts_per60",
                      "ind_high_danger_shots_per60", "ind_medium_danger_shots_per60",
                      "ind_points_per60", "ind_goals_per60"]
        for col in skill_cols:
            if col in p.index:
                p[col] = float(p[col]) * cumulative_mult

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
                               team, n_years, def_fit_feature_names=None,
                               curr_age=None):
    """
    Build a multi-year contract projection for a player on a specific team.
    Works for both forwards (pred) and defensemen (dpred).

    curr_age: actual current age (adjusted for season gap). If not provided,
              falls back to profile age which may be 1-2 years stale.

    Returns a list of dicts, one per year, with predicted stats and confidence.
    """
    is_d = pred is not None and pred.get("position") == "D"

    if is_d:
        if dpred is None:
            return None, "Defensive model not loaded."
        pid     = dpred["pid"]
        profile = dpred["profile"]
        matched = dpred["matched"]
        age     = float(curr_age) if curr_age else float(profile.get("age", 28))

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
        age      = float(curr_age) if curr_age else float(pred.get("age") or profile.get("age", 28) or 28)

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
    d_off_stats, d_off_df, _ = load_defensive_offensive_stats() if is_d else ({}, None, None)
    rows = []

    for year in range(1, n_years + 1):
        aged = age_profile(profile, year - 1, is_defenseman=is_d)
        age_y = age + year - 1
        conf  = get_age_adjusted_confidence(year, age_y)

        if is_d:
            # Set team context
            for col in DEF_TEAM_FEATURES:
                if col in team_row.index:
                    aged[col] = team_row[col]
            preds = def_predict_for_team(aged, team_row, def_fit_models, def_has_age,
                                            feature_names=def_fit_feature_names)
            _, def_pct, _, _ = grade_defensive_defenseman(preds, season_def_df=def_df)
            _, off_pct, _, _ = grade_offensive_defenseman(d_off_stats.get(pid, {}), season_off_df=d_off_df)
            row = {
                "year":             year,
                "age":              round(age_y, 0),
                "confidence":       conf,
                "hits_pg":          round(max(preds.get("ind_hits_pg", 0), 0), 2),
                "takeaways_pg":     round(max(preds.get("ind_takeaways_pg", 0), 0), 3),
                "goals_against_pg": round(max(preds.get("xg_against_per60_5v5", 0), 0), 3),
                "pim_pg":           round(max(preds.get("pim_pg", 0), 0), 3),
                "def_score":        round(def_pct, 1),
                "off_score":        round(off_pct, 1),
            }

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

            # Percentile rank for points among forwards in the dataset
            pts_pct = float((df["points_per_game"].dropna() < pred_pts).mean() * 100)

            row = {
                "year":       year,
                "age":        round(age_y, 0),
                "confidence": conf,
                "points_pg":  round(pred_pts,   3),
                "goals_pg":   round(pred_goals,  3),
                "gs_pg":      round(pred_gs,     3),
                "pts_pct":    round(pts_pct,     1),
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
    Assess contract risk based on age trajectory and percentile decline.
    Returns (rating_label, rating_color, explanation).
    """
    if not rows:
        return "Unknown", "#888888", ""

    year1 = rows[0]
    last  = rows[-1]
    n     = len(rows)
    age_yr1 = year1["age"]

    if is_d:
        y1_val  = year1.get("def_score", 50)
        yn_val  = last.get("def_score",  50)
    else:
        y1_val  = year1.get("pts_pct", year1.get("points_pg", 0))
        yn_val  = last.get("pts_pct",  last.get("points_pg",  0))

    decline_pts = y1_val - yn_val   # percentile points dropped

    if age_yr1 >= 35:
        return "Very High Risk", "#c8102e", f"Age {age_yr1:.0f} at signing — steep decline likely. 35+ rule applies."
    elif age_yr1 >= 33:
        return "High Risk", "#e8622a", f"Age {age_yr1:.0f} at signing — physical decline likely in later years of contract."
    elif age_yr1 >= 32 and decline_pts > 8:
        return "High Risk", "#e8622a", f"Age {age_yr1:.0f} — projected {decline_pts:.0f} percentile point decline over {n} years."
    elif age_yr1 >= 32:
        return "Moderate Risk", "#FFD700", f"Age {age_yr1:.0f} — entering decline window, monitor later contract years closely."
    elif age_yr1 >= 30 and decline_pts > 8:
        return "Moderate Risk", "#FFD700", f"Age {age_yr1:.0f} — some decline expected but manageable."
    elif decline_pts < -5 and age_yr1 < 30:
        return "Low Risk", "#57a85a", f"Age {age_yr1:.0f} — ascending player, percentile rank expected to improve."
    else:
        return "Low Risk", "#57a85a", f"Age {age_yr1:.0f} — stable production expected through contract."