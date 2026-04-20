import pandas as pd
import numpy as np

# ── Load your files (adjust paths if needed) ──────────────────────────────────
DEF_FILE  = "defensive_dataset.csv"
AGES_FILE = "player_ages.csv"

df = pd.read_csv(DEF_FILE)
df["season"] = df["season"].astype(int)

# Merge ages
ages = pd.read_csv(AGES_FILE)
df = df.merge(ages[["player_id", "season", "age"]], on=["player_id", "season"], how="left")

# ── Sort so history features compute correctly ─────────────────────────────────
df = df.sort_values(["player_id", "season"]).copy()
g  = df.groupby("player_id", sort=False)

# ── Build leakage-safe history features for xGA ───────────────────────────────
df["prev_season_xga_pg"]       = g["xg_against_per60_5v5"].shift(1)
df["recent_3yr_mean_xga_pg"]   = (
    g["xg_against_per60_5v5"]
    .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    .reset_index(level=0, drop=True)
)
df["career_prev_mean_xga_pg"]  = (
    g["xg_against_per60_5v5"]
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)

# ── Team context ───────────────────────────────────────────────────────────────
team_ctx = (
    df.groupby(["player_team", "season"])["xg_against_per60_5v5"]
    .mean()
    .reset_index()
    .rename(columns={"xg_against_per60_5v5": "team_avg_xga_per60"})
)
df = df.merge(team_ctx, on=["player_team", "season"], how="left")

# ── Pull McAvoy rows ───────────────────────────────────────────────────────────
macavoy = df[df["player_name"].str.contains("MacAvoy|McAvoy|Mcavoy", case=False, na=False)]

if macavoy.empty:
    print("Player not found — check the spelling in your dataset:")
    print(df["player_name"].drop_duplicates().sort_values().to_string())
else:
    cols = [
        "season", "player_team", "games_played",
        "xg_against_per60_5v5",       # actual that season
        "prev_season_xga_pg",          # what model's baseline leans on
        "recent_3yr_mean_xga_pg",      # 3yr rolling prior average
        "career_prev_mean_xga_pg",     # full career prior average
        "team_avg_xga_per60",          # Boston team context
    ]
    available = [c for c in cols if c in macavoy.columns]
    print(macavoy[available].to_string(index=False))

    # ── Compute the naive baseline (what the model starts from) ───────────────
    latest = macavoy.iloc[-1]
    baseline_cols = ["prev_season_xga_pg", "recent_3yr_mean_xga_pg", "career_prev_mean_xga_pg"]
    available_baseline = [c for c in baseline_cols if c in latest.index and pd.notna(latest[c])]
    
    if available_baseline:
        baseline_val = latest[available_baseline[0]]  # model picks first non-null
        print(f"\n── Diagnosis ─────────────────────────────────────────────────")
        print(f"  Baseline anchor (what model starts from): {baseline_val:.3f}")
        print(f"  Actual xGA/60 this season:                {latest.get('xg_against_per60_5v5', 'N/A')}")
        print(f"  Team avg xGA/60:                          {latest.get('team_avg_xga_per60', 'N/A'):.3f}")
        print(f"  Gap model must correct via residual:      {3.76 - baseline_val:.3f}")
        print(f"\n  If gap > 0.5, the baseline anchor is your main problem.")