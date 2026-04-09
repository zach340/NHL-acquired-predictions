"""
build_season_dataset.py
=======================
Builds a clean season-level dataset from MoneyPuck game-level data.
Uses unique game_id count for true games_played, derives per-game targets
from raw counting stats, and keeps per-60 rate features for the model.

Usage:
    python build_season_dataset.py

Input:  2008_to_2024_cleaned.csv         (MoneyPuck game-level, all situations)
Output: season_dataset.csv               (clean season-level training data)
"""

import pandas as pd
import numpy as np

INPUT_FILE  = "2008_to_2024_cleaned.csv"
OUTPUT_FILE = "season_dataset.csv"
CHUNK_SIZE  = 100_000

# Danger zone weights (empirically derived)
HD_WEIGHT = 11.2
MD_WEIGHT = 4.1
LD_WEIGHT = 1.0

# ── Load and filter to all-situation ──────────────────────────────────────────

print("── Loading game-level data ──────────────────────────────────")
chunks = []
for i, chunk in enumerate(pd.read_csv(
    INPUT_FILE, low_memory=False, chunksize=CHUNK_SIZE
)):
    chunk["situation"] = chunk["situation"].astype(str).str.strip().str.lower()
    chunks.append(chunk[chunk["situation"] == "all"])
    print(f"  Chunk {i+1} processed", end="\r")

df = pd.concat(chunks, ignore_index=True)
df = df.fillna(0)
print(f"\n  {len(df):,} all-situation game-level rows")

# ── Aggregate to season level ──────────────────────────────────────────────────

print("\n── Aggregating to season level ──────────────────────────────")

GROUP = ["player_id", "player_name", "season", "player_team", "position"]

# Count unique game_ids for true games_played
game_counts = (
    df.groupby(GROUP)["game_id"]
    .nunique()
    .reset_index()
    .rename(columns={"game_id": "games_played"})
)

# Sum all counting stats
COUNT_COLS = [
    "ice_time",
    "shifts",
    "game_score",
    "ind_goals",
    "ind_primary_assists",
    "ind_secondary_assists",
    "ind_points",
    "ind_shots_on_goal",
    "ind_missed_shots",
    "ind_blocked_shot_attempts",
    "ind_shot_attempts",
    "ind_unblocked_shot_attempts",
    "ind_expected_goals",
    "ind_flurry_adj_expected_goals",
    "ind_score_venue_adj_expected_goals",
    "ind_flurry_score_venue_adj_expected_goals",
    "ind_expected_on_goal",
    "ind_low_danger_shots",
    "ind_medium_danger_shots",
    "ind_high_danger_shots",
    "ind_low_danger_goals",
    "ind_medium_danger_goals",
    "ind_high_danger_goals",
    "ind_low_danger_expected_goals",
    "ind_medium_danger_expected_goals",
    "ind_high_danger_expected_goals",
]

# Only sum columns that exist in the file
COUNT_COLS = [c for c in COUNT_COLS if c in df.columns]

season = (
    df.groupby(GROUP)[COUNT_COLS]
    .sum()
    .reset_index()
)

# Join true games_played
season = season.merge(game_counts, on=GROUP, how="left")

print(f"  {len(season):,} player-seasons")
print(f"  games_played range: {season['games_played'].min()} – {season['games_played'].max()}")

# ── Per-game targets (clean, no per-60 conversion needed) ─────────────────────

print("\n── Computing per-game targets ───────────────────────────────")

gp = season["games_played"].replace(0, np.nan)

# Primary targets
season["game_score_per_game"]   = season["game_score"]   / gp
season["points_per_game"]       = season["ind_points"]   / gp
season["goals_per_game"]        = season["ind_goals"]    / gp

# Danger-zone weighted shot rate — same formula as EDGE, no unit mismatch
season["weighted_shots_pg"]     = (
    season["ind_high_danger_shots"] * HD_WEIGHT +
    season["ind_medium_danger_shots"] * MD_WEIGHT +
    season["ind_low_danger_shots"]  * LD_WEIGHT
) / gp

# ── Per-60 rate features (kept for model features, not targets) ───────────────

print("── Computing per-60 rate features ───────────────────────────")

# ice_time is in seconds — convert to hours for per-60
ice_hours = season["ice_time"] / 3600

def per60(col):
    return np.where(ice_hours > 0, season[col] / ice_hours, 0)

season["toi_per_game"]                              = (season["ice_time"] / 60) / gp
season["shifts_per60"]                              = per60("shifts")
season["ind_goals_per60"]                           = per60("ind_goals")
season["ind_primary_assists_per60"]                 = per60("ind_primary_assists")
season["ind_secondary_assists_per60"]               = per60("ind_secondary_assists")
season["ind_points_per60"]                          = per60("ind_points")
season["ind_shots_on_goal_per60"]                   = per60("ind_shots_on_goal")
season["ind_missed_shots_per60"]                    = per60("ind_missed_shots")
season["ind_blocked_shot_attempts_per60"]           = per60("ind_blocked_shot_attempts")
season["ind_shot_attempts_per60"]                   = per60("ind_shot_attempts")
season["ind_expected_goals_per60"]                  = per60("ind_expected_goals")
season["ind_flurry_adj_expected_goals_per60"]       = per60("ind_flurry_adj_expected_goals")
season["ind_score_venue_adj_expected_goals_per60"]  = per60("ind_score_venue_adj_expected_goals")
season["ind_flurry_score_venue_adj_expected_goals_per60"] = per60("ind_flurry_score_venue_adj_expected_goals")
season["ind_low_danger_shots_per60"]                = per60("ind_low_danger_shots")
season["ind_medium_danger_shots_per60"]             = per60("ind_medium_danger_shots")
season["ind_high_danger_shots_per60"]               = per60("ind_high_danger_shots")
season["ind_low_danger_goals_per60"]                = per60("ind_low_danger_goals")
season["ind_medium_danger_goals_per60"]             = per60("ind_medium_danger_goals")
season["ind_high_danger_goals_per60"]               = per60("ind_high_danger_goals")
season["ind_low_danger_expected_goals_per60"]       = per60("ind_low_danger_expected_goals")
season["ind_medium_danger_expected_goals_per60"]    = per60("ind_medium_danger_expected_goals")
season["ind_high_danger_expected_goals_per60"]      = per60("ind_high_danger_expected_goals")

# ice_time_rank — use rank within season
season["ice_time_rank"] = season.groupby("season")["ice_time"].rank(ascending=False)

# ── Apply ATL→WPG and ARI→UTA renames ─────────────────────────────────────────

season["player_team"] = season["player_team"].replace({"ATL": "WPG", "ARI": "UTA"})

# ── Sanity checks ──────────────────────────────────────────────────────────────

print(f"\n── Sanity checks ────────────────────────────────────────────")
print(f"  Seasons: {sorted(season['season'].unique())}")
print(f"  Teams: {sorted(season['player_team'].unique())}")
print(f"  Positions: {sorted(season['position'].unique())}")
print(f"  games_played: min={season['games_played'].min()} max={season['games_played'].max()} mean={season['games_played'].mean():.1f}")

print(f"\n  Per-game target stats (forwards only):")
fwd = season[season["position"].isin(["C","L","R"])]
for col in ["game_score_per_game", "goals_per_game", "points_per_game", "weighted_shots_pg"]:
    print(f"    {col:<28} mean={fwd[col].mean():.3f}  max={fwd[col].max():.3f}")

print(f"\n  Top 10 forwards by weighted_shots_pg (2024):")
top = (
    fwd[fwd["season"] == 2024]
    .sort_values("weighted_shots_pg", ascending=False)
    .head(10)
)[["player_name", "player_team", "games_played", "weighted_shots_pg", "points_per_game"]]
print(top.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────

season.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved {len(season):,} rows to {OUTPUT_FILE}")
print(f"  Columns: {len(season.columns)}")