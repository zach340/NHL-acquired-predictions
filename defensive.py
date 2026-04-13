"""
build_defensive_dataset.py
==========================
Aggregates defensive stats from the full MoneyPuck game-level file
to season level per player.

Pulls from three situations:
  - "all"  → physical stats (hits, takeaways, giveaways, blocked shots,
              penalty minutes, faceoffs)
  - "4on5" → penalty kill ice time
  - "5on5" → on-ice defensive impact (xG against, corsi%, shot suppression)

Usage:
    python build_defensive_dataset.py

Input:  2008_to_2024_cleaned.csv   (full MoneyPuck game-level file)
Output: defensive_dataset.csv      (season-level defensive stats per player)
"""

import pandas as pd
import numpy as np

INPUT_FILE  = "2008_to_2024_cleaned.csv"
OUTPUT_FILE = "defensive_dataset.csv"
CHUNK_SIZE  = 150_000

GROUP = ["player_id", "player_name", "season", "player_team", "position"]

# ── Column sets ────────────────────────────────────────────────────────────────

ALL_COLS = [
    "ind_hits",
    "ind_takeaways",
    "ind_giveaways",
    "ind_d_zone_giveaways",
    "ind_penalty_minutes",
    "penalties",
    "penalty_minutes_drawn",
    "penalties_drawn",
    "shots_blocked_by_player",
    "ind_blocked_shot_attempts",
    "ind_faceoffs_won",
    "faceoffs_won",
    "faceoffs_lost",
    "ind_d_zone_shift_starts",
    "ind_o_zone_shift_starts",
    "ind_neutral_zone_shift_starts",
    "ice_time",
]

PK_COLS = ["ice_time"]  # 4on5 — penalty kill ice time only

FIVEONFIVE_COLS = [
    "ice_time",
    "on_ice_expected_goals_pct",
    "on_ice_corsi_pct",
    "on_ice_fenwick_pct",
    "on_ice_against_expected_goals",
    "on_ice_against_high_danger_shots",
    "on_ice_against_shots_on_goal",
    "on_ice_against_goals",
    "on_ice_for_expected_goals",
]

# ── Helper: load chunks filtered to a specific situation ──────────────────────

def load_situation(situation, usecols_extra):
    needed = GROUP + ["situation", "game_id"] + usecols_extra
    # Only keep columns that exist in file
    chunks = []
    first  = True
    avail  = None

    for chunk in pd.read_csv(INPUT_FILE, low_memory=False, chunksize=CHUNK_SIZE):
        if first:
            avail  = set(chunk.columns)
            needed = [c for c in needed if c in avail]
            first  = False
        chunk  = chunk[needed]
        # Filter to defensemen first — discard all forward/goalie rows immediately
        chunk  = chunk[chunk["position"] == "D"]
        if chunk.empty:
            continue
        chunk["situation"] = chunk["situation"].astype(str).str.strip().str.lower()
        chunk  = chunk[chunk["situation"] == situation]
        if chunk.empty:
            continue
        chunk  = chunk.fillna(0)
        chunks.append(chunk)
        print(f"  [{situation}] chunk processed — {len(chunks)} so far", end="\r")

    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    print(f"\n  [{situation}] {len(df):,} rows loaded")
    return df


# ── Step 1: Load all-situation data ───────────────────────────────────────────

print("── Loading all-situation data ────────────────────────────────")
df_all = load_situation("all", ALL_COLS)

# Defensemen only
df_all = df_all[df_all["position"] == "D"].copy()
print(f"  Filtered to defensemen: {len(df_all):,} rows")

# True games_played from unique game_ids
gp = (
    df_all.groupby(GROUP)["game_id"]
    .nunique()
    .reset_index()
    .rename(columns={"game_id": "games_played"})
)

# Aggregate counting stats
all_agg_cols = [c for c in ALL_COLS if c in df_all.columns]
all_agg = df_all.groupby(GROUP)[all_agg_cols].sum().reset_index()
all_agg = all_agg.merge(gp, on=GROUP, how="left")

# ── Step 2: Load penalty kill (4on5) data ────────────────────────────────────

print("\n── Loading penalty kill (4on5) data ─────────────────────────")
df_pk = load_situation("4on5", PK_COLS)

if not df_pk.empty:
    df_pk = df_pk[df_pk["position"] == "D"].copy()
    pk_agg = (
        df_pk.groupby(GROUP)["ice_time"]
        .sum()
        .reset_index()
        .rename(columns={"ice_time": "pk_ice_time"})
    )
else:
    pk_agg = pd.DataFrame(columns=GROUP + ["pk_ice_time"])

# ── Step 3: Load 5on5 on-ice defensive data ───────────────────────────────────

print("\n── Loading 5on5 on-ice data ──────────────────────────────────")
df_5v5 = load_situation("5on5", FIVEONFIVE_COLS)

if not df_5v5.empty:
    df_5v5 = df_5v5[df_5v5["position"] == "D"].copy()
    fv5_agg_cols = [c for c in FIVEONFIVE_COLS if c in df_5v5.columns]
    fv5_agg = df_5v5.groupby(GROUP)[fv5_agg_cols].sum().reset_index()
    # Rename to avoid collision with all-situation ice_time
    fv5_agg = fv5_agg.rename(columns={"ice_time": "fv5_ice_time"})
else:
    fv5_agg = pd.DataFrame(columns=GROUP + ["fv5_ice_time"])

# ── Step 4: Merge everything ──────────────────────────────────────────────────

print("\n── Merging datasets ─────────────────────────────────────────")
season = all_agg.merge(pk_agg,  on=GROUP, how="left")
season = season.merge(fv5_agg, on=GROUP, how="left")
season = season.fillna(0)

# ── Step 5: Compute per-game and per-60 rates ──────────────────────────────────

print("── Computing rates ──────────────────────────────────────────")

gp_safe    = season["games_played"].replace(0, np.nan)
ice_hours  = season["ice_time"] / 3600
fv5_hours  = season["fv5_ice_time"] / 3600 if "fv5_ice_time" in season.columns else ice_hours
ice_safe   = ice_hours.replace(0, np.nan)
fv5_safe   = fv5_hours.replace(0, np.nan)

# PK percentage of total ice time
season["pk_ice_pct"] = np.where(
    season["ice_time"] > 0,
    season["pk_ice_time"] / season["ice_time"],
    0
)
season["pk_toi_per_game"] = season["pk_ice_time"] / 60 / gp_safe

# Per-game physical stats
for col in ["ind_hits", "ind_takeaways", "ind_giveaways", "ind_d_zone_giveaways",
            "shots_blocked_by_player", "ind_blocked_shot_attempts",
            "penalties", "ind_penalty_minutes", "penalties_drawn"]:
    if col in season.columns:
        season[f"{col}_pg"] = season[col] / gp_safe

# Per-60 physical stats (rate-adjusted for TOI)
for col in ["ind_hits", "ind_takeaways", "ind_giveaways",
            "shots_blocked_by_player", "penalties", "ind_penalty_minutes"]:
    if col in season.columns:
        season[f"{col}_per60"] = season[col] / ice_safe

# Faceoff win percentage
season["faceoff_win_pct"] = np.where(
    (season["faceoffs_won"] + season["faceoffs_lost"]) > 0,
    season["faceoffs_won"] / (season["faceoffs_won"] + season["faceoffs_lost"]),
    np.nan
)

# Zone start % (defensive zone starts as % of non-neutral starts)
if "ind_d_zone_shift_starts" in season.columns:
    total_zone = season["ind_d_zone_shift_starts"] + season["ind_o_zone_shift_starts"]
    season["d_zone_start_pct"] = np.where(total_zone > 0,
        season["ind_d_zone_shift_starts"] / total_zone, np.nan)

# Takeaway/giveaway ratio
season["take_give_ratio"] = np.where(
    season["ind_giveaways"] > 0,
    season["ind_takeaways"] / season["ind_giveaways"],
    np.nan
)

# On-ice 5v5 defensive metrics (weighted average by 5v5 ice time)
for col in ["on_ice_expected_goals_pct", "on_ice_corsi_pct", "on_ice_fenwick_pct"]:
    if col in season.columns:
        # Already summed — convert back to weighted average using 5v5 ice time
        season[f"{col}_avg"] = season[col] / np.maximum(1, (season["fv5_ice_time"] / season["fv5_ice_time"].max() * 100))

# xG against per 60 (5v5)
if "on_ice_against_expected_goals" in season.columns:
    season["xg_against_per60_5v5"] = season["on_ice_against_expected_goals"] / fv5_safe

# HD shots against per 60 (5v5)
if "on_ice_against_high_danger_shots" in season.columns:
    season["hd_shots_against_per60_5v5"] = season["on_ice_against_high_danger_shots"] / fv5_safe

# Goals against per 60 (5v5)
if "on_ice_against_goals" in season.columns:
    season["goals_against_per60_5v5"] = season["on_ice_against_goals"] / fv5_safe

# Team rename
season["player_team"] = season["player_team"].replace({"ATL": "WPG", "ARI": "UTA"})

# ── Step 6: Filter minimums and save ──────────────────────────────────────────

MIN_GP = 20
season = season[season["games_played"] >= MIN_GP].copy()

print(f"\n── Output ───────────────────────────────────────────────────")
print(f"  {len(season):,} player-seasons (≥{MIN_GP} GP)")
print(f"  Seasons: {sorted(season['season'].unique())}")

print(f"\n  Top 10 hitters (2024):")
if "ind_hits_pg" in season.columns:
    top = season[season["season"] == 2024].sort_values("ind_hits_pg", ascending=False).head(10)
    print(top[["player_name", "player_team", "games_played", "ind_hits_pg", "pk_ice_pct"]].to_string(index=False))

print(f"\n  Top 10 by takeaways/game (2024):")
if "ind_takeaways_pg" in season.columns:
    top = season[season["season"] == 2024].sort_values("ind_takeaways_pg", ascending=False).head(10)
    print(top[["player_name", "player_team", "games_played", "ind_takeaways_pg", "take_give_ratio"]].to_string(index=False))

season.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved to {OUTPUT_FILE}")
print(f"  Columns: {list(season.columns)}")
