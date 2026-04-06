"""
Player Team Fit Predictor
=========================
Trains a gradient boosting model on historical NHL per-60 data to predict
how a player would perform if they played for every other team.

Usage:
    python player_team_predictor.py
    # Then enter a player name when prompted, e.g. "Connor McDavid"
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILE   = "offensive_performance_by_season_per60_renamed.csv"
TARGETS     = ["game_score", "ind_points_per60", "ind_goals_per60"]
MIN_GP      = 20        # minimum games played to include a row in training
MIN_ICE     = 300       # minimum ice time (minutes) to include a row

# ── 1. Load & clean ───────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["games_played"] >= MIN_GP) & (df["ice_time"] >= MIN_ICE)].copy()
    df = df.dropna(subset=TARGETS)
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────

def safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Division that returns fill where denominator is 0 or NaN."""
    return np.where(b == 0, fill, a / b)


def engineer_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive skill-signal features from raw per-60 columns."""
    d = df.copy()

    # -- Finishing skill (persistently >1.0 = real scorer) --------------------
    d["finishing_skill"] = safe_div(
        d["ind_goals_per60"], d["ind_expected_goals_per60"]
    )

    # -- Rebound / flurry specialist flag -------------------------------------
    d["flurry_reliance"] = safe_div(
        d["ind_expected_goals_per60"], d["ind_flurry_adj_expected_goals_per60"]
    )

    # -- High danger shot share -----------------------------------------------
    d["hd_shot_share"] = safe_div(
        d["ind_high_danger_shots_per60"], d["ind_shots_on_goal_per60"]
    )

    # -- High danger finishing precision --------------------------------------
    d["hd_finishing"] = safe_div(
        d["ind_high_danger_goals_per60"], d["ind_high_danger_shots_per60"]
    )

    # -- High danger xG outperformance ----------------------------------------
    d["hd_xg_outperformance"] = safe_div(
        d["ind_high_danger_goals_per60"], d["ind_high_danger_expected_goals_per60"]
    )

    # -- Playmaking quality (primary > secondary = better skill signal) --------
    d["primary_assist_share"] = safe_div(
        d["ind_primary_assists_per60"], d["ind_points_per60"]
    )
    d["primary_vs_secondary"] = safe_div(
        d["ind_primary_assists_per60"], d["ind_secondary_assists_per60"]
    )

    # -- Shot quality per attempt ---------------------------------------------
    d["xg_per_attempt"] = safe_div(
        d["ind_expected_goals_per60"], d["ind_shot_attempts_per60"]
    )

    # -- Shot accuracy / discipline -------------------------------------------
    d["on_target_rate"] = safe_div(
        d["ind_shots_on_goal_per60"], d["ind_shot_attempts_per60"]
    )

    # -- Context sensitivity (how much score/venue adjustments shift xG) ------
    d["context_sensitivity"] = safe_div(
        d["ind_score_venue_adj_expected_goals_per60"], d["ind_expected_goals_per60"]
    )

    # -- Full adjustment sensitivity (raw vs fully cleaned xG) ----------------
    # How much a player's xG changes once flurry + score state + venue removed
    # High = genuine producer; low = numbers inflate from context
    d["full_adj_sensitivity"] = safe_div(
        d["ind_flurry_score_venue_adj_expected_goals_per60"],
        d["ind_expected_goals_per60"],
        fill=1.0,
    )

    # -- Finishing skill on fully adjusted xG (cleanest finishing signal) ------
    d["finishing_skill_adj"] = safe_div(
        d["ind_goals_per60"],
        d["ind_flurry_score_venue_adj_expected_goals_per60"],
    )

    # -- Deployment proxy: TOI per game ---------------------------------------
    d["toi_per_game"] = d["ice_time"] / d["games_played"]

    return d


def build_team_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team, season, position), aggregate context features that capture:
      - how much ice time the team grants by position
      - the team's offensive style using fully adjusted xG
    """
    team_ctx = (
        df.groupby(["player_team", "season", "position"])
        .agg(
            team_median_toi_pg    = ("toi_per_game",                                    "median"),
            team_avg_hd_share     = ("hd_shot_share",                                   "mean"),
            # Fully adjusted xG — removes flurry, score state, and venue bias
            team_avg_adj_xg_per60 = ("ind_flurry_score_venue_adj_expected_goals_per60", "mean"),
            # Raw xG for computing the adjustment ratio below
            _team_avg_raw_xg      = ("ind_expected_goals_per60",                        "mean"),
            team_avg_primary_rate = ("primary_assist_share",                            "mean"),
            team_avg_on_target    = ("on_target_rate",                                  "mean"),
        )
        .reset_index()
    )
    # Ratio: how much the team's raw xG shrinks once fully adjusted
    # Low ratio = team's numbers are inflated by score/venue/flurry effects
    team_ctx["team_adj_ratio"] = safe_div(
        team_ctx["team_avg_adj_xg_per60"],
        team_ctx["_team_avg_raw_xg"],
        fill=1.0,
    )
    team_ctx = team_ctx.drop(columns=["_team_avg_raw_xg"])
    return team_ctx


def merge_team_context(df: pd.DataFrame, team_ctx: pd.DataFrame) -> pd.DataFrame:
    return df.merge(team_ctx, on=["player_team", "season", "position"], how="left")


# ── 3. Build feature matrix ───────────────────────────────────────────────────

PLAYER_FEATURES = [
    "finishing_skill",
    "finishing_skill_adj",          # goals/60 ÷ flurry+score+venue adj xG
    "flurry_reliance",
    "hd_shot_share",
    "hd_finishing",
    "hd_xg_outperformance",
    "primary_assist_share",
    "primary_vs_secondary",
    "xg_per_attempt",
    "on_target_rate",
    "context_sensitivity",
    "full_adj_sensitivity",         # flurry+score+venue adj xG ÷ raw xG
    "toi_per_game",
    "shifts_per60",
    # raw volume features
    "ind_expected_goals_per60",
    "ind_flurry_score_venue_adj_expected_goals_per60",  # cleanest individual xG
    "ind_shots_on_goal_per60",
    "ind_shot_attempts_per60",
    "ind_primary_assists_per60",
    "ind_secondary_assists_per60",
    "ind_high_danger_shots_per60",
    "ind_medium_danger_shots_per60",
    "ind_low_danger_shots_per60",
]

TEAM_FEATURES = [
    "team_median_toi_pg",
    "team_avg_hd_share",
    "team_avg_adj_xg_per60",   # flurry + score state + venue adjusted
    "team_adj_ratio",           # how much raw xG shrinks after adjustment
    "team_avg_primary_rate",
    "team_avg_on_target",
]

POSITION_DUMMIES = ["pos_C", "pos_D", "pos_L", "pos_R"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pos_dummies = pd.get_dummies(df["position"], prefix="pos")
    for col in POSITION_DUMMIES:
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0

    feature_cols = PLAYER_FEATURES + TEAM_FEATURES
    X = pd.concat([df[feature_cols].reset_index(drop=True),
                   pos_dummies[POSITION_DUMMIES].reset_index(drop=True)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


# ── 4. Train models ───────────────────────────────────────────────────────────

def train_models(X: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Train one GBM per target metric. Returns dict of fitted models."""
    models = {}
    print("\n── Training models ──────────────────────────────────────")
    for target in TARGETS:
        y = df[target].values
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbm",    GradientBoostingRegressor(
                n_estimators    = 300,
                max_depth       = 4,
                learning_rate   = 0.05,
                subsample       = 0.8,
                min_samples_leaf= 10,
                random_state    = 42,
            )),
        ])
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        model.fit(X, y)
        models[target] = model
        print(f"  {target:<25} R² = {scores.mean():.3f}  (±{scores.std():.3f})")
    return models


# ── 5. Predict for one player on all teams ────────────────────────────────────

def get_latest_team_contexts(df: pd.DataFrame, team_ctx: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent season's context for every team × position."""
    latest_season = df["season"].max()
    ctx = team_ctx[team_ctx["season"] == latest_season].copy()
    # Fall back to prior season for any team/position missing in latest season
    if ctx["player_team"].nunique() < df["player_team"].nunique():
        fallback = (
            team_ctx.sort_values("season", ascending=False)
            .groupby(["player_team", "position"])
            .first()
            .reset_index()
        )
        present = set(zip(ctx["player_team"], ctx["position"]))
        missing = fallback[
            ~fallback.apply(lambda r: (r["player_team"], r["position"]) in present, axis=1)
        ]
        ctx = pd.concat([ctx, missing], ignore_index=True)
    return ctx


def predict_player_on_all_teams(
    player_name : str,
    df          : pd.DataFrame,
    team_ctx    : pd.DataFrame,
    models      : dict,
) -> pd.DataFrame:
    """
    Find the player's most recent season, build a prediction row for every
    team by swapping in that team's context features, and return a ranked table.
    """
    # Locate the player
    mask = df["player_name"].str.lower() == player_name.strip().lower()
    player_rows = df[mask]
    if player_rows.empty:
        # Fuzzy fallback
        mask = df["player_name"].str.lower().str.contains(player_name.strip().lower())
        player_rows = df[mask]
        if player_rows.empty:
            raise ValueError(f"No player found matching '{player_name}'.")
        matched = player_rows["player_name"].iloc[0]
        print(f"  → Matched to: {matched}")

    # Most recent season
    latest = player_rows.sort_values("season").iloc[-1]
    position = latest["position"]
    actual_team = latest["player_team"]
    season = latest["season"]

    print(f"\n  Player  : {latest['player_name']}")
    print(f"  Position: {position}")
    print(f"  Season  : {season}  |  Team: {actual_team}")

    # Get all-team contexts for this position
    all_teams = get_latest_team_contexts(df, team_ctx)
    all_teams = all_teams[all_teams["position"] == position].copy()

    # Build one prediction row per team
    records = []
    for _, team_row in all_teams.iterrows():
        row = latest.copy()
        for col in TEAM_FEATURES:
            row[col] = team_row[col]
        records.append(row)

    pred_df = pd.DataFrame(records)

    # Position dummies
    pos_dummies = pd.get_dummies(pred_df["position"], prefix="pos")
    for col in POSITION_DUMMIES:
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0

    feature_cols = PLAYER_FEATURES + TEAM_FEATURES
    X_pred = pd.concat(
        [pred_df[feature_cols].reset_index(drop=True),
         pos_dummies[POSITION_DUMMIES].reset_index(drop=True)],
        axis=1
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict all targets
    results = all_teams[["player_team"]].reset_index(drop=True).copy()
    for target, model in models.items():
        results[f"pred_{target}"] = model.predict(X_pred)

    results = results.sort_values("pred_game_score", ascending=False).reset_index(drop=True)
    results.index += 1  # 1-based rank

    # Tag the player's actual team
    results["actual_team"] = results["player_team"] == actual_team

    return results, latest["player_name"], actual_team, season


# ── 6. Feature importance ─────────────────────────────────────────────────────

def plot_feature_importance(models: dict, feature_names: list, top_n: int = 15):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 6))
    for ax, target in zip(axes, TARGETS):
        gbm = models[target].named_steps["gbm"]
        importance = gbm.feature_importances_
        idx = np.argsort(importance)[-top_n:]
        ax.barh([feature_names[i] for i in idx], importance[idx])
        ax.set_title(f"Feature importance\n{target}")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    print("\n  Saved: feature_importance.png")


# ── 7. Results chart ──────────────────────────────────────────────────────────

def plot_predictions(results: pd.DataFrame, player_name: str,
                     actual_team: str, season: int):
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [
        "#e63946" if t == actual_team else "#457b9d"
        for t in results["player_team"]
    ]

    bars = ax.barh(
        results["player_team"][::-1],
        results["pred_game_score"][::-1],
        color=colors[::-1],
    )

    ax.set_xlabel("Predicted Game Score", fontsize=12)
    ax.set_title(
        f"{player_name} — Predicted Game Score by Team\n"
        f"(based on {season} season skill profile)",
        fontsize=13,
    )
    ax.axvline(
        results.loc[results["player_team"] == actual_team, "pred_game_score"].values[0],
        color="#e63946", linestyle="--", linewidth=1.2, label=f"Actual team ({actual_team})"
    )
    ax.legend()
    plt.tight_layout()
    fname = f"{player_name.replace(' ', '_')}_predictions.png"
    plt.savefig(fname, dpi=120)
    print(f"  Saved: {fname}")


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    print("── Loading data ─────────────────────────────────────────")
    raw = load_data(DATA_FILE)
    print(f"  {len(raw):,} rows after filtering  |  {raw['season'].nunique()} seasons  |  {raw['player_team'].nunique()} teams")

    print("\n── Engineering features ─────────────────────────────────")
    df = engineer_player_features(raw)
    team_ctx = build_team_context(df)
    df = merge_team_context(df, team_ctx)

    X = build_feature_matrix(df)
    feature_names = X.columns.tolist()
    print(f"  {len(feature_names)} features ready")

    models = train_models(X, df)

    plot_feature_importance(models, feature_names)

    # ── Interactive loop ──────────────────────────────────────────────────────
    print("\n── Player lookup ────────────────────────────────────────")
    while True:
        player_name = input("\nEnter player name (or 'q' to quit): ").strip()
        if player_name.lower() == "q":
            break
        try:
            results, matched_name, actual_team, season = predict_player_on_all_teams(
                player_name, df, team_ctx, models
            )

            print(f"\n{'Rank':<6}{'Team':<8}{'Game Score':>12}{'Points/60':>12}{'Goals/60':>12}{'Actual?':>10}")
            print("─" * 60)
            for rank, row in results.iterrows():
                marker = " ◄" if row["player_team"] == actual_team else ""
                print(
                    f"{rank:<6}{row['player_team']:<8}"
                    f"{row['pred_game_score']:>12.3f}"
                    f"{row['pred_ind_points_per60']:>12.3f}"
                    f"{row['pred_ind_goals_per60']:>12.3f}"
                    f"{marker}"
                )

            plot_predictions(results, matched_name, actual_team, season)

            # Save CSV
            out_file = f"{matched_name.replace(' ', '_')}_team_predictions.csv"
            results.drop(columns="actual_team").to_csv(out_file, index_label="rank")
            print(f"  Saved: {out_file}")

        except ValueError as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()