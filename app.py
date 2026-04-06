"""
NHL Player Team Fit Predictor — Streamlit App
=============================================
Run with:  python -m streamlit run app.py

Tabs:
  1. Team Fit       — current skill profile projected across all 32 teams
  2. Next Season    — next-season forecast projected across all 32 teams
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_FILE      = "offensive_performance_by_season_per60_renamed.csv"
AGES_FILE      = "player_ages.csv"
TARGETS        = ["game_score_per_game", "ind_points_per60", "ind_goals_per60"]
MIN_GP         = 20
MIN_ICE        = 300
CV_FOLDS       = 5
N_SEASONS      = 3
SEASON_WEIGHTS = [3, 2, 1]

TARGET_LABELS = {
    "game_score_per_game": "Game Score / Game",
    "ind_points_per60":    "Points / 60",
    "ind_goals_per60":     "Goals / 60",
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
]

# Age features — only added when age data is available
AGE_FEATURES = ["age", "age_sq"]

TEAM_FEATURES = [
    "team_median_toi_pg",
    "team_avg_hd_share",
    "team_avg_adj_xg_per60",
    "team_adj_ratio",
    "team_avg_primary_rate",
    "team_avg_on_target",
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


# ── Team context ───────────────────────────────────────────────────────────────

def build_team_context(df):
    team_ctx = (
        df.groupby(["player_team", "season", "position"])
        .agg(
            team_median_toi_pg    = ("toi_per_game",                                    "median"),
            team_avg_hd_share     = ("hd_shot_share",                                   "mean"),
            team_avg_adj_xg_per60 = ("ind_flurry_score_venue_adj_expected_goals_per60", "mean"),
            _team_avg_raw_xg      = ("ind_expected_goals_per60",                        "mean"),
            team_avg_primary_rate = ("primary_assist_share",                            "mean"),
            team_avg_on_target    = ("on_target_rate",                                  "mean"),
        )
        .reset_index()
    )
    team_ctx["team_adj_ratio"] = safe_div(
        team_ctx["team_avg_adj_xg_per60"], team_ctx["_team_avg_raw_xg"], fill=1.0
    )
    return team_ctx.drop(columns=["_team_avg_raw_xg"])


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


# ── Weighted player profile ────────────────────────────────────────────────────

def build_weighted_player_profile(player_rows, has_age):
    seasons = sorted(player_rows["season"].unique(), reverse=True)[:N_SEASONS]
    feats   = PLAYER_FEATURES + (AGE_FEATURES if has_age else [])

    weighted_rows = []
    for i, season in enumerate(seasons):
        season_rows   = player_rows[player_rows["season"] == season]
        season_weight = SEASON_WEIGHTS[i]
        if len(season_rows) == 1:
            weighted_rows.append((season_rows.iloc[0], season_weight))
        else:
            total_ice = season_rows["ice_time"].sum()
            for _, row in season_rows.iterrows():
                ice_share   = row["ice_time"] / total_ice if total_ice > 0 else 1.0 / len(season_rows)
                weighted_rows.append((row, season_weight * ice_share))

    profile = weighted_rows[0][0].copy()
    for feat in feats:
        vals = [(row[feat], w) for row, w in weighted_rows
                if feat in row.index and not pd.isna(row[feat])]
        if vals:
            profile[feat] = sum(v * w for v, w in vals) / sum(w for _, w in vals)

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


# ── Training ───────────────────────────────────────────────────────────────────

def train_models_with_progress(X, df, targets, target_col_map, label_prefix, status, bar, step, total_steps):
    kf      = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    models  = {}
    metrics = {}

    for target in targets:
        label = TARGET_LABELS[target]
        y     = df[target_col_map[target]].values

        base_model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbm", GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42,
            )),
        ])

        fold_maes, fold_rmses = [], []
        for fold, (tr, val) in enumerate(kf.split(X), 1):
            status.markdown(f"🔁 **{label_prefix} — {label}** fold {fold}/{CV_FOLDS}")
            m = clone(base_model)
            m.fit(X.iloc[tr], y[tr])
            preds = m.predict(X.iloc[val])
            fold_maes.append(mean_absolute_error(y[val], preds))
            fold_rmses.append(np.sqrt(mean_squared_error(y[val], preds)))
            step += 1
            bar.progress(min(step / total_steps, 1.0),
                         text=f"{label_prefix} {label}: fold {fold}/{CV_FOLDS} — MAE {np.mean(fold_maes):.3f}")

        status.markdown(f"✅ **{label_prefix} — {label}** fitting final model...")
        base_model.fit(X, y)
        models[target]  = base_model
        metrics[target] = {
            "mae":  (float(np.mean(fold_maes)),  float(np.std(fold_maes))),
            "rmse": (float(np.mean(fold_rmses)), float(np.std(fold_rmses))),
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
    raw_targets = ["game_score", "ind_points_per60", "ind_goals_per60"]
    df   = df[(df["games_played"] >= MIN_GP) & (df["ice_time"] >= MIN_ICE)].dropna(subset=raw_targets).copy()
    ages = pd.read_csv(ages_path)[["player_id", "season", "age", "age_sq"]]
    df   = df.merge(ages, on=["player_id", "season"], how="left")
    has_age = df["age"].notna().mean() > 0.5
    advance(f"Data loaded — {len(df):,} rows  |  age matched: {df['age'].notna().sum():,}")

    # ── Engineer ───────────────────────────────────────────────────────────────
    status.markdown("⚙️ **Engineering features...**")
    df       = engineer_player_features(df)
    df       = engineer_trajectory_features(df)
    team_ctx = build_team_context(df)
    df       = df.merge(team_ctx, on=["player_team", "season", "position"], how="left")
    advance("Features engineered")

    # ── Weighted profiles ──────────────────────────────────────────────────────
    status.markdown("⚙️ **Building weighted player profiles...**")
    player_profiles = {}
    for pid, group in df.groupby("player_id"):
        profile, seasons = build_weighted_player_profile(group, has_age)
        player_profiles[pid] = (profile, seasons)
    advance(f"Profiles built — {len(player_profiles):,} players")

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

def _build_team_predictions(profile, position, all_teams, models, has_age, use_next_features=False, df=None):
    records = []
    for _, team_row in all_teams.iterrows():
        row = profile.copy()
        for col in TEAM_FEATURES:
            row[col] = team_row[col]
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
    for target, model in models.items():
        try:
            results[f"pred_{target}"] = model.predict(X_pred)
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
    fit_results = _build_team_predictions(profile, position, all_teams, fit_models, has_age)
    fit_results = fit_results.sort_values("pred_game_score_per_game", ascending=False).reset_index(drop=True)
    fit_results.index += 1
    fit_results["is_actual"] = fit_results["player_team"] == actual_team

    # Next season predictions
    next_results = _build_team_predictions(profile, position, all_teams, next_models, has_age,
                                           use_next_features=True, df=df)
    next_results = next_results.sort_values("pred_game_score_per_game", ascending=False).reset_index(drop=True)
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
    metric_cols   = ["pred_game_score_per_game", "pred_ind_points_per60", "pred_ind_goals_per60"]
    metric_labels = ["Game Score / Game", "Points / 60", "Goals / 60"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.patch.set_facecolor("#0e1117")
    fig.suptitle(title, color="white", fontsize=14, y=1.01)

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        ax.set_facecolor("#0e1117")
        sr = results.sort_values(col)
        bar_colors = ["#c8102e" if t == actual_team else "#4a90d9" for t in sr["player_team"]]
        ax.barh(sr["player_team"], sr[col], color=bar_colors)
        actual_val = results.loc[results["player_team"] == actual_team, col].values[0]
        ax.axvline(actual_val, color="#c8102e", linestyle="--", linewidth=1.2)
        ax.set_xlabel(label, color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    actual_patch = mpatches.Patch(color="#c8102e", label=f"Actual team ({actual_team})")
    other_patch  = mpatches.Patch(color="#4a90d9", label="Other teams")
    axes[-1].legend(handles=[actual_patch, other_patch], facecolor="#1a1a2e", labelcolor="white", loc="lower right")
    plt.tight_layout()
    return fig


def make_importance_chart(models, feature_names, top_n=15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0e1117")
    for ax, target in zip(axes, TARGETS):
        ax.set_facecolor("#0e1117")
        imp = models[target].named_steps["gbm"].feature_importances_
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
        "pred_ind_points_per60", "pred_ind_goals_per60", "is_actual"
    ]].copy()
    display.columns = ["Team", "GS/GP", "Points/60", "Goals/60", "Actual Team"]
    for col in ["GS/GP", "Points/60", "Goals/60"]:
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
    st.markdown(f"**{label} model quality (5-fold CV)**")
    st.caption("MAE = avg absolute error in same units as stat. RMSE penalises large errors more. Lower is better.")
    for target in TARGETS:
        mae_mean,  mae_std  = metrics[target]["mae"]
        rmse_mean, rmse_std = metrics[target]["rmse"]
        st.markdown(f"*{TARGET_LABELS[target]}*")
        c1, c2, _ = st.columns(3)
        c1.metric("MAE",  f"{mae_mean:.3f}", f"± {mae_std:.3f}")
        c2.metric("RMSE", f"{rmse_mean:.3f}", f"± {rmse_std:.3f}")


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="NHL Player Predictor", page_icon="🏒", layout="wide")
st.title("🏒 NHL Player Predictor")

if "fit_models" not in st.session_state:
    st.info("Training models on first load — this takes 5–8 minutes. Grab a coffee ☕")
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
    ) = load_and_train_with_progress(DATA_FILE, AGES_FILE)
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
tab1, tab2, tab3 = st.tabs(["📊 Team Fit", "🔮 Next Season", "🔬 Model Info"])

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
    with st.expander("Team Fit model quality", expanded=True):
        show_metrics(fit_metrics, "Team Fit")
    with st.expander("Next Season model quality", expanded=True):
        show_metrics(next_metrics, "Next Season")
    with st.expander("Team Fit — feature importance"):
        st.pyplot(make_importance_chart(fit_models, fit_feature_names))
    with st.expander("Next Season — feature importance"):
        st.pyplot(make_importance_chart(next_models, next_feature_names))