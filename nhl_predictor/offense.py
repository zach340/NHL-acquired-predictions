"""
offense.py
==========
Forward (offensive) model: feature engineering, training and prediction of
Game Score / Points / Goals per game on any of the 32 team contexts.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import nhl_api
from .config import (
    AGE_FEATURES, AGES_FILE, BASELINE_FEATURES, ELITE_QUANTILE, FORWARD_POSITIONS,
    FWD_SLOT_MAP, LINEMATE_FILE, MIN_GP, MIN_ICE, NEXT_BASELINE_FEATURES, NEXT_BASELINE_RATES,
    NONLINEAR_FEATURES, OUTCOME_FEATURES,
    PLAYER_FEATURES, POSITION_DUMMIES, PP_FILE, SLOT_COLORS, TARGET_LABELS, TARGETS,
    TEAM_FEATURES, TRAJECTORY_FEATURES, season_label,
)
from .data_io import latest_known_age, load_ages, safe_read_csv
from .features import (
    add_career_curve_features, compute_baseline, design_matrix,
    latest_team_contexts, per_player, prior_max, prior_mean, prior_rolling_mean,
    prior_slope, safe_div, weighted_recent_mean,
)
from .training import (
    ModelBundle, StreamlitProgress, total_training_steps, train_residual_models,
)

PP_COLS = ["pp_icetime_pct", "pp_points_per60", "pp_goals_per60", "pp_xg_per60",
           "pp_points_share", "o_zone_start_pct", "zone_start_diff"]
LINEMATE_COLS = ["line_adj_xg_per60", "line_xg_per60", "line_hd_xg_per60",
                 "line_goals_per60", "line_xg_pct", "line_corsi_pct", "n_distinct_lines"]
LINE_TEAM_FEATURES = ["team_avg_line_adj_xg_per60", "team_avg_line_xg_pct",
                      "team_avg_line_hd_xg_per60", "team_avg_line_corsi_pct"]

# Per-player history stats: short name → per-game column
_HISTORY_STATS = {"points": "points_per_game", "goals": "goals_per_game", "gamescore": "game_score_per_game"}


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_player_features(df):
    d = df.sort_values(["player_id", "season"]).copy()
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
    # League scoring environment by season
    d["league_avg_points_pg"] = d.groupby("season")["points_per_game"].transform("mean")
    d["league_avg_goals_pg"]  = d.groupby("season")["goals_per_game"].transform("mean")
    # Career peak to date (never future seasons) — anchors to a player's ceiling
    d["career_peak_points_pg"] = d.groupby("player_id")["points_per_game"].cummax()
    d["career_peak_goals_pg"]  = d.groupby("player_id")["goals_per_game"].cummax()
    d["pct_of_peak_points"]    = safe_div(d["points_per_game"], d["career_peak_points_pg"])
    d["pct_of_peak_goals"]     = safe_div(d["goals_per_game"], d["career_peak_goals_pg"])
    # Age interactions — the same skill level means something different at 25 vs 35
    if "age" in d.columns:
        d["age_x_shot_attempts"] = d["age"] * d["ind_shot_attempts_per60"]
        d["age_x_finishing"]     = d["age"] * d["finishing_skill_adj"]
        d["age_x_hd_share"]      = d["age"] * d["hd_shot_share"]
    return d


def engineer_trajectory_features(df):
    """YoY deltas and career stage."""
    d = df.sort_values(["player_id", "season"]).copy()
    g = d.groupby("player_id")
    d["yoy_points_delta"]    = g["ind_points_per60"].diff()
    d["yoy_goals_delta"]     = g["ind_goals_per60"].diff()
    d["yoy_gamescore_delta"] = g["game_score_per_game"].diff()
    d["games_played_pct"]    = d["games_played"] / 82.0
    d["career_year"]         = g.cumcount() + 1
    return d


def engineer_career_history_features(df):
    """Leakage-safe career history features based on prior seasons only."""
    d = df.sort_values(["player_id", "season"]).copy()
    g = d.groupby("player_id", sort=False)
    d["career_seasons_prior"] = g.cumcount().astype(float)

    for short, col in _HISTORY_STATS.items():
        d[f"prev_season_{short}_pg"]      = g[col].shift(1)
        d[f"career_prev_mean_{short}_pg"] = per_player(g, col, prior_mean)
    for short in ("points", "goals"):
        d[f"career_prev_peak_{short}_pg"] = per_player(g, _HISTORY_STATS[short], prior_max)
    for short, col in _HISTORY_STATS.items():
        d[f"recent_3yr_mean_{short}_pg"] = per_player(g, col, prior_rolling_mean)
    for short, col in _HISTORY_STATS.items():
        d[f"recent_3yr_{short}_slope"] = per_player(g, col, lambda s: prior_slope(s, window=3))
    for short, col in _HISTORY_STATS.items():
        d[f"career_{short}_slope"] = per_player(g, col, prior_slope)

    # 3-2-1 weighted recent form, including this season (Next Season model only)
    for short, col in _HISTORY_STATS.items():
        d[f"wavg_{short}_pg"] = weighted_recent_mean(d, col)
    for short, col in (("toi", "toi_per_game"), ("p60", "ind_points_per60"), ("g60", "ind_goals_per60")):
        d[f"wavg_{short}"] = weighted_recent_mean(d, col, weight_col="ice_time")
    return d


def engineer_nonlinear_trajectory_features(df):
    return add_career_curve_features(
        df,
        curve_stats={"points": "points_per_game", "goals": "goals_per_game", "gs": "game_score_per_game"},
        peak_stats={"points", "goals", "gs"},
        pct_peak_stats={"points": "pct_of_peak_points", "goals": "pct_of_peak_goals"},
        age_slope_pairs=[
            ("recent_3yr_points_slope", "age_x_3yr_pts_slope"),
            ("recent_3yr_goals_slope",  "age_x_3yr_goals_slope"),
            ("career_points_slope",     "age_x_career_pts_slope"),
        ],
    )


def build_team_context(df):
    agg = dict(
        team_median_toi_pg    = ("toi_per_game",                                    "median"),
        team_avg_hd_share     = ("hd_shot_share",                                   "mean"),
        team_avg_adj_xg_per60 = ("ind_flurry_score_venue_adj_expected_goals_per60", "mean"),
        _team_avg_raw_xg      = ("ind_expected_goals_per60",                        "mean"),
        team_avg_primary_rate = ("primary_assist_share",                            "mean"),
        team_avg_on_target    = ("on_target_rate",                                  "mean"),
    )
    if "line_adj_xg_per60" in df.columns:
        agg.update(
            team_avg_line_adj_xg_per60 = ("line_adj_xg_per60", "mean"),
            team_avg_line_xg_pct       = ("line_xg_pct",       "mean"),
            team_avg_line_hd_xg_per60  = ("line_hd_xg_per60",  "mean"),
            team_avg_line_corsi_pct    = ("line_corsi_pct",    "mean"),
        )
    ctx = df.groupby(["player_team", "season", "position"]).agg(**agg).reset_index()
    ctx["team_adj_ratio"] = safe_div(ctx["team_avg_adj_xg_per60"], ctx["_team_avg_raw_xg"], fill=1.0)
    ctx = ctx.drop(columns=["_team_avg_raw_xg"])
    for col in LINE_TEAM_FEATURES:
        if col not in ctx.columns:
            ctx[col] = 0.0
    return ctx


def get_latest_team_contexts(df, team_ctx):
    return latest_team_contexts(df, team_ctx, keys=["position"])


def get_latest_league_env(df):
    """Most recent season's league-wide scoring averages."""
    latest_df = df[df["season"] == df["season"].max()]
    return {
        "league_avg_points_pg": latest_df["points_per_game"].mean(),
        "league_avg_goals_pg":  latest_df["goals_per_game"].mean(),
    }


def build_player_profile(player_rows):
    """Profile = the player's highest-TOI row in their latest season."""
    latest_season = player_rows["season"].max()
    latest_rows   = player_rows[player_rows["season"] == latest_season]
    profile = latest_rows.sort_values("ice_time", ascending=False).iloc[0].copy()
    return profile, [latest_season]


# ── Feature matrices ──────────────────────────────────────────────────────────

def feature_columns(has_age, next_season=False):
    """Model inputs. Team Fit (same-season target) excludes same-season outcome features."""
    age = (AGE_FEATURES + NONLINEAR_FEATURES) if has_age else []
    traj = TRAJECTORY_FEATURES if next_season else []
    cols = PLAYER_FEATURES + age + traj + TEAM_FEATURES
    return cols if next_season else [c for c in cols if c not in OUTCOME_FEATURES]


def position_dummies(df):
    pos = pd.get_dummies(df["position"], prefix="pos")
    for c in POSITION_DUMMIES:
        if c not in pos.columns:
            pos[c] = 0
    return pos[POSITION_DUMMIES]


def build_X(frame, has_age, next_season=False):
    return design_matrix(frame, feature_columns(has_age, next_season), extra=position_dummies(frame))


def compute_target_baseline(df_like, target):
    """Team Fit baseline: prior seasons only."""
    return compute_baseline(df_like, BASELINE_FEATURES.get(target, []))


def compute_next_baseline(df_like, target):
    """
    Next Season baseline: 3-2-1 weighted recent form including this season.
    Points/Goals are built as TOI/GP × rate/60 so ice-time changes and rate
    changes are separated.
    """
    if target in NEXT_BASELINE_RATES:
        cols = ["wavg_toi", NEXT_BASELINE_RATES[target]]
        if not all(c in df_like.columns for c in cols):
            return pd.Series(np.zeros(len(df_like)), index=df_like.index, dtype=float)
        return (df_like[cols[0]] * df_like[cols[1]] / 60).fillna(0.0).astype(float)
    return compute_baseline(df_like, [NEXT_BASELINE_FEATURES[target]])


def make_elite_sample_weights(y, _target=None):
    """Upweight the top 10% of outcomes 3× so the model focuses on elite players."""
    arr = np.asarray(y, dtype=float)
    weights = np.ones(len(arr), dtype=float)
    if len(arr):
        weights[arr >= np.quantile(arr, ELITE_QUANTILE)] = 3.0
    return weights


def make_lgbm():
    # Shallow, subsampled trees with min_child_samples=40, picked on season-based
    # CV + the 2025-26 holdout (deeper, min_child_samples=2 trees overfit).
    return lgb.LGBMRegressor(
        n_estimators=600, max_depth=5, num_leaves=24, learning_rate=0.03,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=40,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="regression_l2", random_state=42, verbose=-1,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def load_training_frame(path, ages_path):
    """Forwards with enough games/ice time, joined to ages, PP and linemate features."""
    df = safe_read_csv(path)
    required = ["game_score_per_game", "points_per_game", "goals_per_game", "ice_time", "games_played"]
    df = df[(df["games_played"] >= MIN_GP) & (df["ice_time"] >= MIN_ICE)].dropna(subset=required)
    df = df[df["position"].isin(FORWARD_POSITIONS)].copy()
    df = df.merge(load_ages(ages_path), on=["player_id", "season"], how="left")

    for file, cols in ((PP_FILE, PP_COLS), (LINEMATE_FILE, LINEMATE_COLS)):
        extra = safe_read_csv(file)[["player_id", "season"] + cols]
        df = df.merge(extra, on=["player_id", "season"], how="left")
        df[cols] = df[cols].fillna(0)
    return df


def load_and_train(path, ages_path):
    progress = StreamlitProgress(total_training_steps(len(TARGETS)))

    progress.status("⚙️ **Loading data...**")
    df = load_training_frame(path, ages_path)
    has_age = df["age"].notna().mean() > 0.5
    progress.advance(f"Data loaded (forwards only) — {len(df):,} rows  |  age matched: {df['age'].notna().sum():,}")

    progress.status("⚙️ **Engineering features...**")
    df = engineer_player_features(df)
    df = engineer_trajectory_features(df)
    df = engineer_career_history_features(df)
    df = engineer_nonlinear_trajectory_features(df)
    team_ctx = build_team_context(df)
    df = df.merge(team_ctx, on=["player_team", "season", "position"], how="left")
    progress.advance("Features engineered")

    progress.status("⚙️ **Building latest-season player profiles...**")
    profiles = {pid: build_player_profile(group) for pid, group in df.groupby("player_id")}
    progress.advance(f"Profiles built from latest seasons — {len(profiles):,} players")

    common = dict(labels=TARGET_LABELS, make_model=make_lgbm, baseline_fn=compute_target_baseline,
                  weight_fn=make_elite_sample_weights, progress=progress, track_elite=True)

    progress.status("⚙️ **Training Team Fit models...**")
    X_fit = build_X(df, has_age)
    fit_models, fit_metrics = train_residual_models(
        X_fit, df, TARGETS, {t: t for t in TARGETS}, label_prefix="Team Fit", **common)

    progress.status("⚙️ **Training Next Season models...**")
    df_next = build_next_season_dataset(df)
    X_next  = build_X(df_next, has_age, next_season=True)
    next_models, next_metrics = train_residual_models(
        X_next, df_next, TARGETS, {t: f"next_{t}" for t in TARGETS}, label_prefix="Next Season",
        **(common | {"baseline_fn": compute_next_baseline}))

    progress.finish("✅ All models trained and ready!")
    return ModelBundle(df, team_ctx, has_age, profiles,
                       fit_models, fit_metrics, X_fit.columns.tolist(),
                       next_models, next_metrics, X_next.columns.tolist())


def build_next_season_dataset(df):
    """Pair each player-season's features with the NEXT season's targets."""
    d = df.sort_values(["player_id", "season"]).copy()
    next_targets = d.groupby("player_id")[TARGETS].shift(-1)
    next_targets.columns = [f"next_{t}" for t in TARGETS]
    return pd.concat([d, next_targets], axis=1).dropna(subset=list(next_targets.columns))


# ── Prediction ─────────────────────────────────────────────────────────────────

def with_context(profile, team_row, league_env=None):
    """Copy of a profile with a team's context (and optionally league env) swapped in."""
    row = profile.copy()
    for col in TEAM_FEATURES:
        if col in team_row.index:
            row[col] = team_row[col]
    for k, v in (league_env or {}).items():
        row[k] = v
    return row


def predict_frame(frame, models, has_age, next_season=False):
    """{target: array of predictions} for each row of `frame`."""
    X = build_X(frame, has_age, next_season)
    baseline = compute_next_baseline if next_season else compute_target_baseline
    return {
        target: np.clip(baseline(frame, target).values + m["global"].predict(X), 0, None)
        for target, m in models.items()
    }


def predict_row(row, models, has_age, next_season=False):
    """{target: float} for a single profile row."""
    return {t: float(v[0]) for t, v in predict_frame(pd.DataFrame([row]), models, has_age, next_season).items()}


def predict_all_teams(profile, all_teams, models, has_age, league_env, next_season=False):
    """Predictions for one player on every team in `all_teams`."""
    frame = pd.DataFrame([with_context(profile, team_row, league_env) for _, team_row in all_teams.iterrows()])
    results = all_teams[["player_team"]].reset_index(drop=True).copy()
    for target, vals in predict_frame(frame, models, has_age, next_season).items():
        results[f"pred_{target}"] = vals
    return results


def _ranked(results, actual_team):
    results = results.sort_values("pred_points_per_game", ascending=False).reset_index(drop=True)
    results.index += 1
    results["is_actual"] = results["player_team"] == actual_team
    return results


def _profile_age(profile, bundle, pid):
    """Profile age, falling back to the ages file rolled forward to the latest season."""
    age = profile.get("age") if bundle.has_age else None
    if age is None or (isinstance(age, float) and np.isnan(age)):
        try:
            age = latest_known_age(AGES_FILE, pid, int(bundle.df["season"].max()))
        except Exception:
            age = None
    if age is not None and isinstance(age, float) and np.isnan(age):
        return None
    return age


def predict_player(player_id, bundle, override_team=None):
    """Team Fit and Next Season rankings across all 32 teams for one forward (None if unknown)."""
    if player_id not in bundle.profiles:
        return None
    pid  = player_id
    rows = bundle.df[bundle.df["player_id"] == pid]
    profile, seasons = bundle.profiles[pid]
    position = profile["position"]

    latest_rows  = rows[rows["season"] == rows["season"].max()]
    traded_teams = sorted(latest_rows["player_team"].unique().tolist()) if len(latest_rows) > 1 else []
    actual_team  = override_team or profile["player_team"]

    all_teams  = get_latest_team_contexts(bundle.df, bundle.team_ctx)
    all_teams  = all_teams[all_teams["position"] == position].copy()
    league_env = get_latest_league_env(bundle.df)

    fit  = predict_all_teams(profile, all_teams, bundle.fit_models,  bundle.has_age, league_env)
    nxt  = predict_all_teams(profile, all_teams, bundle.next_models, bundle.has_age, league_env, next_season=True)

    return {
        "pid":          pid,
        "matched":      nhl_api.fetch_player_display_name(int(pid)) or profile["player_name"],
        "actual_team":  actual_team,
        "season":       int(profile["season"]),
        "position":     position,
        "seasons":      seasons,
        "traded_teams": traded_teams,
        "fit_results":  _ranked(fit, actual_team),
        "next_results": _ranked(nxt, actual_team),
        "age":          _profile_age(profile, bundle, pid),
    }


def build_validation_results(actual_df, bundle):
    """
    Next Season prediction (from each player's latest profile, on their latest
    team) vs their real stats the following season from the NHL API.
    """
    all_teams  = get_latest_team_contexts(bundle.df, bundle.team_ctx)
    league_env = get_latest_league_env(bundle.df)
    rows = []
    for _, actual in actual_df.iterrows():
        pid = int(actual["player_id"])
        if pid not in bundle.profiles:
            continue
        profile, seasons = bundle.profiles[pid]
        team = profile["player_team"]
        team_row = all_teams[(all_teams["position"] == profile["position"]) & (all_teams["player_team"] == team)]
        if team_row.empty:
            continue

        preds = predict_row(with_context(profile, team_row.iloc[0], league_env), bundle.next_models,
                            bundle.has_age, next_season=True)
        pts, goals, gs = preds["points_per_game"], preds["goals_per_game"], preds["game_score_per_game"]
        actual_goals = float(actual.get("goals_per_game", 0))
        rows.append({
            "player_name":      actual["player_name"],
            "team":             team,
            "games_played":     actual["games_played"],
            "actual_points_gp": round(actual["points_per_game"], 3),
            "pred_points_gp":   round(pts, 3),
            "points_gp_error":  round(actual["points_per_game"] - pts, 3),
            "actual_goals_gp":  round(actual_goals, 3),
            "pred_goals_gp":    round(goals, 3),
            "goals_gp_error":   round(actual_goals - goals, 3),
            "pred_gs_per_game": round(gs, 3),
            "seasons_used":     " → ".join(season_label(s) for s in seasons),
        })
    return pd.DataFrame(rows)


def build_player_insertion(player_id, team_code, bundle):
    """
    Rank the searched forward alongside the team's active forwards by
    predicted Points/GP in that team's context, and assign lineup slots.

    Returns (DataFrame, error). Columns: player_id, player_name, position,
    pred_points_gp, pred_goals_gp, is_searched_player, rank, lineup_slot, slot_color.
    """
    roster_df, err = nhl_api.fetch_team_forwards(team_code)
    if err or roster_df is None:
        return None, err or "Could not fetch roster."
    if player_id not in bundle.profiles:
        return None, "Player profile not found in model data."

    searched_profile, _ = bundle.profiles[player_id]
    position = searched_profile.get("position", "C")

    latest_ctx = get_latest_team_contexts(bundle.df, bundle.team_ctx)
    team_row = latest_ctx[(latest_ctx["player_team"] == team_code) & (latest_ctx["position"] == position)]
    if team_row.empty:
        return None, f"No team context found for {team_code}."
    team_row   = team_row.iloc[0]
    league_env = get_latest_league_env(bundle.df)

    def entry(pid, name, pos, profile, is_searched):
        preds = predict_row(with_context(profile, team_row, league_env), bundle.fit_models, bundle.has_age)
        return {"player_id": pid, "player_name": name, "position": pos,
                "pred_points_gp": preds["points_per_game"], "pred_goals_gp": preds["goals_per_game"],
                "is_searched_player": is_searched}

    rows = [entry(player_id, searched_profile.get("player_name", "Selected Player"), position, searched_profile, True)]
    for _, rp in roster_df.iterrows():
        pid = int(rp["player_id"])
        if pid != player_id and pid in bundle.profiles:
            rows.append(entry(pid, rp["player_name"], rp.get("position", position), bundle.profiles[pid][0], False))

    result = pd.DataFrame(rows).sort_values("pred_points_gp", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    result["lineup_slot"] = result["rank"].apply(lambda r: FWD_SLOT_MAP.get(r, "Extra"))
    result["slot_color"]  = result["lineup_slot"].map(SLOT_COLORS).fillna("#888888")
    result["pred_points_gp"] = result["pred_points_gp"].round(3)
    result["pred_goals_gp"]  = result["pred_goals_gp"].round(3)
    return result, None
