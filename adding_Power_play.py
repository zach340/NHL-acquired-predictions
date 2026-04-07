"""
extract_pp_features.py
======================
Extracts powerplay and zone start features from MoneyPuck game-level data
and aggregates to season level for joining onto the main training dataset.

Usage:
    python extract_pp_features.py

Input:  2008_to_2024_cleaned.csv  (raw MoneyPuck game-level data)
Output: pp_features.csv           (season-level PP + zone start features)
"""

import pandas as pd
import numpy as np

INPUT_CSV  = "2008_to_2024_cleaned.csv"
OUTPUT_CSV = "pp_features.csv"
CHUNK_SIZE = 100_000

# ── Columns we need ────────────────────────────────────────────────────────────

KEEP_COLS = [
    "player_id", "player_name", "season", "position", "situation",
    "ice_time",
    "ind_goals",
    "ind_primary_assists",
    "ind_secondary_assists",
    "ind_points",
    "ind_shots_on_goal",
    "ind_shot_attempts",
    "ind_o_zone_shift_starts",
    "ind_d_zone_shift_starts",
    "ind_neutral_zone_shift_starts",
    "ind_expected_goals",
    "ind_flurry_score_venue_adj_expected_goals",
]

SITUATIONS = ["5on4", "all"]

print("── Reading and filtering data ───────────────────────────────")
chunks_all = []
chunks_pp  = []

for i, chunk in enumerate(pd.read_csv(INPUT_CSV, low_memory=False, chunksize=CHUNK_SIZE, usecols=KEEP_COLS)):
    chunk.fillna(0, inplace=True)
    chunk["situation"] = chunk["situation"].astype(str).str.strip().str.lower()

    chunks_all.append(chunk[chunk["situation"] == "all"])
    chunks_pp.append(chunk[chunk["situation"] == "5on4"])

    print(f"  Chunk {i+1} processed", end="\r")

print("\n── Concatenating ────────────────────────────────────────────")
df_all = pd.concat(chunks_all, ignore_index=True)
df_pp  = pd.concat(chunks_pp,  ignore_index=True)
print(f"  All situation rows:  {len(df_all):,}")
print(f"  5on4 situation rows: {len(df_pp):,}")

# ── Aggregate to season level ──────────────────────────────────────────────────

GROUP = ["player_id", "player_name", "season", "position"]

def agg_season(df, suffix=""):
    agg = df.groupby(GROUP, as_index=False).agg(
        **{f"ice_time{suffix}":              ("ice_time",                                   "sum"),
           f"goals{suffix}":                 ("ind_goals",                                  "sum"),
           f"primary_assists{suffix}":       ("ind_primary_assists",                        "sum"),
           f"secondary_assists{suffix}":     ("ind_secondary_assists",                      "sum"),
           f"points{suffix}":               ("ind_points",                                  "sum"),
           f"shots_on_goal{suffix}":         ("ind_shots_on_goal",                          "sum"),
           f"shot_attempts{suffix}":         ("ind_shot_attempts",                          "sum"),
           f"xg{suffix}":                    ("ind_expected_goals",                         "sum"),
           f"xg_adj{suffix}":               ("ind_flurry_score_venue_adj_expected_goals",   "sum"),
           f"o_zone_starts{suffix}":         ("ind_o_zone_shift_starts",                    "sum"),
           f"d_zone_starts{suffix}":         ("ind_d_zone_shift_starts",                    "sum"),
           f"neutral_zone_starts{suffix}":   ("ind_neutral_zone_shift_starts",              "sum"),
        }
    )
    return agg

print("\n── Aggregating to season level ──────────────────────────────")
season_all = agg_season(df_all, suffix="_all")
season_pp  = agg_season(df_pp,  suffix="_pp")

# ── Merge all and pp ──────────────────────────────────────────────────────────

merged = season_all.merge(season_pp, on=GROUP, how="left")
merged = merged.fillna(0)

# ── Engineer features ──────────────────────────────────────────────────────────

def safe_div(a, b, fill=0.0):
    return np.where(b == 0, fill, a / b)

ice_hours_all = merged["ice_time_all"] / 3600
ice_hours_pp  = merged["ice_time_pp"]  / 3600

# PP rate stats per 60
merged["pp_goals_per60"]          = safe_div(merged["goals_pp"],           ice_hours_pp)
merged["pp_primary_assists_per60"]= safe_div(merged["primary_assists_pp"], ice_hours_pp)
merged["pp_points_per60"]         = safe_div(merged["points_pp"],          ice_hours_pp)
merged["pp_shots_per60"]          = safe_div(merged["shots_on_goal_pp"],   ice_hours_pp)
merged["pp_xg_per60"]             = safe_div(merged["xg_pp"],              ice_hours_pp)

# PP deployment — what fraction of total ice time is on PP
merged["pp_icetime_pct"]          = safe_div(merged["ice_time_pp"], merged["ice_time_all"])

# PP production share — what fraction of total points come on PP
merged["pp_points_share"]         = safe_div(merged["points_pp"],   merged["points_all"])

# Zone start features (from all-situation)
total_zone_starts = merged["o_zone_starts_all"] + merged["d_zone_starts_all"] + merged["neutral_zone_starts_all"]
merged["o_zone_start_pct"]        = safe_div(merged["o_zone_starts_all"], total_zone_starts)
merged["d_zone_start_pct"]        = safe_div(merged["d_zone_starts_all"], total_zone_starts)
merged["zone_start_diff"]         = merged["o_zone_start_pct"] - merged["d_zone_start_pct"]

# ── Select output columns ──────────────────────────────────────────────────────

OUTPUT_COLS = GROUP + [
    "pp_goals_per60",
    "pp_primary_assists_per60",
    "pp_points_per60",
    "pp_shots_per60",
    "pp_xg_per60",
    "pp_icetime_pct",
    "pp_points_share",
    "o_zone_start_pct",
    "d_zone_start_pct",
    "zone_start_diff",
]

result = merged[OUTPUT_COLS]

print(f"\n── Output ───────────────────────────────────────────────────")
print(f"  {len(result):,} player-seasons")
print(f"  {result['player_id'].nunique():,} unique players")
print(f"  Seasons: {sorted(result['season'].unique())}")
print(f"\n  Sample PP features:")
print(result[result["pp_icetime_pct"] > 0.05].sort_values("pp_points_per60", ascending=False).head(5)[
    ["player_name", "season", "pp_points_per60", "pp_icetime_pct", "o_zone_start_pct"]
].to_string(index=False))

result.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Saved to {OUTPUT_CSV}")