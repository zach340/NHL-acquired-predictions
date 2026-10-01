"""
defense.py
==========
Defenseman model: feature engineering, training and prediction of Hits,
Takeaways, xGA/60 (5v5) and PIM per game on any of the 32 team contexts.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import nhl_api
from .config import (
    AGES_FILE, DEF_AGE_FEATURES, DEF_BASELINE_FEATURES, DEF_LOWER_IS_BETTER, DEF_NONLINEAR_FEATURES,
    DEF_OUTCOME_FEATURES, DEF_PLAYER_FEATURES, DEF_SCORE_WEIGHTS, DEF_TARGET_LABELS, DEF_TARGETS,
    DEF_TEAM_FEATURES, DEF_TRAJECTORY_FEATURES, MIN_GP, season_label,
)
from .data_io import latest_known_age, load_ages, safe_read_csv
from .features import (
    add_career_curve_features, compute_baseline, design_matrix,
    latest_team_contexts, per_player, prior_mean, prior_rolling_mean, prior_slope, safe_div,
    weighted_recent_mean,
)
from .grading import classify_defenseman_type, grade_defensive_defenseman, grade_offensive_defenseman
from .training import ModelBundle, StreamlitProgress, total_training_steps, train_residual_models

# Source column → (history name, slope name), e.g. prev_season_hits_pg / recent_3yr_hits_slope
_HISTORY_STATS = {
    "ind_hits_pg":            ("hits_pg",      "hits"),
    "ind_takeaways_pg":       ("takeaways_pg", "takeaways"),
    "xg_against_per60_5v5":   ("xga_pg",       "xga"),
    "pk_ice_pct":             ("pk_pct",       "pk"),
    "ind_penalty_minutes_pg": ("pim_pg",       "pim"),
}

# Empirical league-wide (p5, p95) ranges, used when no season frame is given
_LEAGUE_RANGES = {
    "ind_hits_pg":          (0.0, 3.5),
    "ind_takeaways_pg":     (0.0, 0.65),
    "xg_against_per60_5v5": (1.8, 3.5),
    "pim_pg":               (0.0, 1.2),
}


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df):
    """League environment, career peaks to date and age interactions."""
    d = df.sort_values(["player_id", "season"]).copy()
    d["league_avg_hits_pg"] = d.groupby("season")["ind_hits_pg"].transform("mean")
    d["league_avg_pk_pct"]  = d.groupby("season")["pk_ice_pct"].transform("mean")

    # Peak to date only — never future seasons
    d["career_peak_hits_pg"]      = d.groupby("player_id")["ind_hits_pg"].cummax()
    d["career_peak_takeaways_pg"] = d.groupby("player_id")["ind_takeaways_pg"].cummax()
    d["career_peak_pk_pct"]       = d.groupby("player_id")["pk_ice_pct"].cummax()
    d["pct_of_peak_hits"]      = safe_div(d["ind_hits_pg"],      d["career_peak_hits_pg"])
    d["pct_of_peak_takeaways"] = safe_div(d["ind_takeaways_pg"], d["career_peak_takeaways_pg"])

    if "xg_against_per60_5v5" not in d.columns and "on_ice_against_expected_goals" in d.columns:
        fv5_hours = (d["fv5_ice_time"] / 3600).replace(0, np.nan)
        d["xg_against_per60_5v5"] = d["on_ice_against_expected_goals"] / fv5_hours

    if "age" in d.columns:
        d["age_x_hits"]      = d["age"] * d["ind_hits_per60"]
        d["age_x_takeaways"] = d["age"] * d["ind_takeaways_per60"]
        d["age_x_pk"]        = d["age"] * d["pk_ice_pct"]
    return d


def engineer_career_history(df):
    """Leakage-safe prior-season history features and YoY deltas."""
    d = df.sort_values(["player_id", "season"]).copy()
    g = d.groupby("player_id", sort=False)

    d["career_seasons_prior"] = g.cumcount().astype(float)
    d["career_year"]          = d["career_seasons_prior"] + 1
    d["games_played_pct"]     = d["games_played"] / 82.0

    for col, (name, slope_name) in _HISTORY_STATS.items():
        d[f"prev_season_{name}"]      = g[col].shift(1)
        d[f"recent_3yr_mean_{name}"]  = per_player(g, col, prior_rolling_mean)
        d[f"career_prev_mean_{name}"] = per_player(g, col, prior_mean)
        d[f"recent_3yr_{slope_name}_slope"] = per_player(g, col, lambda s: prior_slope(s, window=3))

    # 3-2-1 weighted recent form, including this season (Next Season model only)
    for col, (name, _) in _HISTORY_STATS.items():
        d[f"wavg_{name}"] = weighted_recent_mean(d, col, weight_col="games_played")

    d["yoy_hits_delta"]      = g["ind_hits_pg"].diff()
    d["yoy_takeaways_delta"] = g["ind_takeaways_pg"].diff()
    d["yoy_xga_delta"]       = g["xg_against_per60_5v5"].diff()
    return d


def engineer_nonlinear_trajectory_features(df):
    """Career-arc features: physical skills decline non-linearly with age."""
    return add_career_curve_features(
        df,
        curve_stats={"hits": "ind_hits_pg", "takeaways": "ind_takeaways_pg", "xga": "xg_against_per60_5v5"},
        peak_stats={"hits", "takeaways"},
        pct_peak_stats={"hits": "pct_of_peak_hits", "takeaways": "pct_of_peak_takeaways"},
        age_slope_pairs=[
            ("recent_3yr_hits_slope",      "def_age_x_3yr_hits_slope"),
            ("recent_3yr_takeaways_slope", "def_age_x_3yr_takeaways_slope"),
            ("recent_3yr_xga_slope",       "def_age_x_3yr_xga_slope"),
        ],
        prefix="def_",
    )


def build_team_context(df):
    """Team-level defensive context per season."""
    return (
        df.groupby(["player_team", "season"])
        .agg(
            team_avg_hits_pg          = ("ind_hits_pg",            "mean"),
            team_avg_takeaways_pg     = ("ind_takeaways_pg",       "mean"),
            team_avg_xga_per60        = ("xg_against_per60_5v5",   "mean"),
            team_avg_pk_pct           = ("pk_ice_pct",             "mean"),
            team_avg_pim_pg           = ("ind_penalty_minutes_pg", "mean"),
            team_avg_toi_pg           = ("pk_toi_per_game",        "mean"),
            team_avg_d_zone_start_pct = ("d_zone_start_pct",       "mean"),
        )
        .reset_index()
    )


def get_latest_team_contexts(df, team_ctx):
    return latest_team_contexts(df, team_ctx, keys=[])


def build_player_profile(player_rows):
    latest_season = player_rows["season"].max()
    return player_rows[player_rows["season"] == latest_season].iloc[0].copy(), [latest_season]


# ── Feature matrices ──────────────────────────────────────────────────────────

def feature_columns(df, has_age, next_season=False):
    """Model inputs present in `df`. Current Fit excludes same-season outcome features."""
    age  = (DEF_AGE_FEATURES + DEF_NONLINEAR_FEATURES) if has_age else []
    traj = DEF_TRAJECTORY_FEATURES if next_season else []
    excluded = set() if next_season else set(DEF_OUTCOME_FEATURES)
    return [f for f in DEF_PLAYER_FEATURES + age + traj + DEF_TEAM_FEATURES
            if f in df.columns and f not in excluded]


def compute_target_baseline(df_like, target):
    return compute_baseline(df_like, DEF_BASELINE_FEATURES.get(target, []))


def make_elite_sample_weights(y, target):
    """Tiered weights for the best 25/10/5% (lowest values when lower is better)."""
    arr = np.asarray(y, dtype=float)
    if target in DEF_LOWER_IS_BETTER:
        arr = -arr
    q75, q90, q95 = np.quantile(arr, [0.75, 0.90, 0.95])
    return 1.0 + 0.5 * (arr >= q75) + 1.0 * (arr >= q90) + 1.5 * (arr >= q95)


def make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=0.1,
        objective="huber", random_state=42, verbose=-1,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def load_and_train(def_path, ages_path):
    progress = StreamlitProgress(total_training_steps(len(DEF_TARGETS)))

    progress.status("⚙️ **Loading defensive data...**")
    df = safe_read_csv(def_path)
    df = df[df["games_played"] >= MIN_GP].copy()
    df = df.merge(load_ages(ages_path), on=["player_id", "season"], how="left")
    has_age = df["age"].notna().mean() > 0.5
    progress.advance(f"Data loaded — {len(df):,} defenseman-seasons | age matched: {df['age'].notna().sum():,}")

    progress.status("⚙️ **Engineering features...**")
    df = engineer_features(df)
    df = engineer_career_history(df)
    df = engineer_nonlinear_trajectory_features(df)
    df["pim_pg"] = df["ind_penalty_minutes_pg"]   # model target name
    team_ctx = build_team_context(df)
    df = df.merge(team_ctx, on=["player_team", "season"], how="left")
    progress.advance("Features engineered")

    progress.status("⚙️ **Building player profiles...**")
    profiles = {pid: build_player_profile(group) for pid, group in df.groupby("player_id")}
    progress.advance(f"Profiles built — {len(profiles):,} defensemen")

    common = dict(labels=DEF_TARGET_LABELS, make_model=make_lgbm, baseline_fn=compute_target_baseline,
                  weight_fn=make_elite_sample_weights, progress=progress)

    progress.status("⚙️ **Training Current Fit models...**")
    fit_cols = feature_columns(df, has_age)
    fit_models, fit_metrics = train_residual_models(
        design_matrix(df, fit_cols), df, DEF_TARGETS, {t: t for t in DEF_TARGETS},
        label_prefix="Current Fit", **common)

    progress.status("⚙️ **Training Next Season models...**")
    df_next = df.sort_values(["player_id", "season"]).copy()
    next_targets = df_next.groupby("player_id")[DEF_TARGETS].shift(-1)
    next_targets.columns = [f"next_{t}" for t in DEF_TARGETS]
    df_next = pd.concat([df_next, next_targets], axis=1).dropna(subset=list(next_targets.columns))
    next_cols = feature_columns(df_next, has_age, next_season=True)
    next_models, next_metrics = train_residual_models(
        design_matrix(df_next, next_cols), df_next, DEF_TARGETS, {t: f"next_{t}" for t in DEF_TARGETS},
        label_prefix="Next Season", **common)

    progress.finish("✅ All defensive models trained!")
    return ModelBundle(df, team_ctx, has_age, profiles,
                       fit_models, fit_metrics, fit_cols,
                       next_models, next_metrics, next_cols)


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_for_team(profile, team_row, models, feature_names):
    """{target: value} for a player on one team's context."""
    row = profile.copy()
    for col in DEF_TEAM_FEATURES:
        if col in team_row.index:
            row[col] = team_row[col]
    pred_df = pd.DataFrame([row])
    X = design_matrix(pred_df, feature_names)
    return {
        target: float(np.clip(compute_target_baseline(pred_df, target).values[0] + m["global"].predict(X)[0], 0, None))
        for target, m in models.items()
    }


def predict_all_teams(profile, all_teams, models, feature_names):
    rows = []
    for _, team_row in all_teams.iterrows():
        preds = predict_for_team(profile, team_row, models, feature_names)
        preds["player_team"] = team_row["player_team"]
        rows.append(preds)
    return pd.DataFrame(rows)


def add_defensive_score(df_preds, season_df=None):
    """
    Composite defensive score 0-100 per row. Each metric is normalised to the
    league's p5–p95 range (from `season_df` if given, else fixed ranges).
    """
    result = df_preds.copy()
    score  = np.zeros(len(result))
    for target, weight in DEF_SCORE_WEIGHTS.items():
        if target not in result.columns:
            continue
        vals = result[target].values.astype(float)
        if season_df is not None and target in season_df.columns:
            lo = np.nanpercentile(season_df[target], 5)
            hi = np.nanpercentile(season_df[target], 95)
        else:
            lo, hi = _LEAGUE_RANGES.get(target, (vals.min(), vals.max()))
        norm = np.full(len(vals), 0.5) if hi == lo else np.clip((vals - lo) / (hi - lo), 0, 1)
        if target in DEF_LOWER_IS_BETTER:
            norm = 1 - norm
        score += norm * weight
    result["defensive_score"] = np.round(score * 100, 1)
    return result


def _ranked(results, actual_team, season_df):
    results = add_defensive_score(results, season_df=season_df)
    results = results.sort_values("defensive_score", ascending=False).reset_index(drop=True)
    results.index += 1
    results["is_actual"] = results["player_team"] == actual_team
    return results


def predict_defenseman(player_id, bundle):
    """Current Fit and Next Season rankings across all teams for one D-man (None if unknown)."""
    if player_id not in bundle.profiles:
        return None
    pid = player_id
    profile, seasons = bundle.profiles[pid]
    actual_team = profile["player_team"]
    all_teams = get_latest_team_contexts(bundle.df, bundle.team_ctx)

    fit = predict_all_teams(profile, all_teams, bundle.fit_models,  bundle.fit_feature_names)
    nxt = predict_all_teams(profile, all_teams, bundle.next_models, bundle.next_feature_names)
    return {
        "pid":          pid,
        "matched":      nhl_api.fetch_player_display_name(int(pid)) or profile["player_name"],
        "actual_team":  actual_team,
        "seasons":      seasons,
        "fit_results":  _ranked(fit, actual_team, bundle.df),
        "next_results": _ranked(nxt, actual_team, bundle.df),
        "profile":      profile,
        "age":          _profile_age(profile, bundle, pid),
    }


def _profile_age(profile, bundle, pid):
    """Age in the profile season, falling back to the ages file."""
    age = profile.get("age")
    if age is None or pd.isna(age):
        try:
            age = latest_known_age(AGES_FILE, pid, int(bundle.df["season"].max()))
        except Exception:
            age = None
    return None if age is None or pd.isna(age) else float(age)


def build_validation(actual_df, bundle):
    """Compare Next Season predictions (from each D-man's latest profile) against the following season's NHL API stats."""
    latest_ctx = get_latest_team_contexts(bundle.df, bundle.team_ctx)
    rows = []
    for _, actual in actual_df.iterrows():
        pid = int(actual["player_id"])
        if pid not in bundle.profiles:
            continue
        profile, seasons = bundle.profiles[pid]
        team = profile["player_team"]
        team_row = latest_ctx[latest_ctx["player_team"] == team]
        if team_row.empty:
            continue
        preds = predict_for_team(profile, team_row.iloc[0], bundle.next_models, bundle.next_feature_names)

        row = {"player_name": actual["player_name"], "team": team, "games_played": actual["games_played"]}
        for short, actual_col, target in (("hits", "hits_pg", "ind_hits_pg"),
                                          ("tk", "takeaways_pg", "ind_takeaways_pg"),
                                          ("pim", "pim_pg", "pim_pg")):
            a, p = float(actual.get(actual_col, 0)), preds.get(target, 0)
            row[f"actual_{short}_pg"] = round(a, 3)
            row[f"pred_{short}_pg"]   = round(p, 3)
            row[f"{short}_error"]     = round(a - p, 3)
        row["seasons_used"] = " → ".join(season_label(s) for s in seasons)
        rows.append(row)
    return pd.DataFrame(rows)


def type_and_scores(preds, profile, off_stats, off_df, season_def_df):
    """(def_pct, off_pct, d_type, d_desc) for a set of predictions + profile."""
    _, def_pct, _, _ = grade_defensive_defenseman(preds, season_def_df=season_def_df)
    _, off_pct, _, _ = grade_offensive_defenseman(off_stats, season_off_df=off_df)
    d_type, d_desc = classify_defenseman_type(dict(profile), def_score=def_pct, off_score=off_pct)
    return def_pct, off_pct, d_type, d_desc
