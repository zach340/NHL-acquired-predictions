"""
extract_linemate_features.py
============================
Extracts linemate quality features from MoneyPuck lines data and aggregates
to season level for joining onto the main training dataset.

Usage:
    python extract_linemate_features.py

Input:  lines.csv                          (raw MoneyPuck lines data)
        offensive_performance_by_season_per60_renamed.csv  (main dataset)
Output: linemate_features.csv             (season-level linemate quality)
"""

import pandas as pd
import numpy as np

LINES_FILE  = "2008_to_2024_lines.csv"
MAIN_FILE   = "offensive_performance_by_season_per60_renamed.csv"
OUTPUT_FILE = "linemate_features.csv"
CHUNK_SIZE  = 100_000

# ── Config ─────────────────────────────────────────────────────────────────────

# Only use 5on5 — cleanest signal for true linemate quality
SITUATION = "5on5"

# Line quality stats to aggregate
QUALITY_COLS = [
    "flurryScoreVenueAdjustedxGoalsFor",  # cleanest xG — main quality signal
    "xGoalsFor",                           # raw xG for reference
    "xGoalsPercentage",                    # possession quality
    "corsiPercentage",                     # shot attempt share
    "goalsFor",                            # actual goals
    "shotsOnGoalFor",                      # shots
    "highDangerxGoalsFor",                 # high danger chance generation
    "icetime",                             # for weighting
]

# ── Step 1: Load and filter lines data ────────────────────────────────────────

print("── Loading lines data ───────────────────────────────────────")
chunks = []
for i, chunk in enumerate(pd.read_csv(
        LINES_FILE, low_memory=False, chunksize=CHUNK_SIZE,
        usecols=["lineId", "name", "season", "playerTeam", "situation", "icetime"] + 
                [c for c in QUALITY_COLS if c != "icetime"])):
    chunk["situation"] = chunk["situation"].astype(str).str.strip().str.lower()
    chunks.append(chunk[chunk["situation"] == SITUATION])
    print(f"  Chunk {i+1} processed", end="\r")

lines = pd.concat(chunks, ignore_index=True)
lines.fillna(0, inplace=True)
print(f"\n  {len(lines):,} 5on5 line-game rows")

# ── Step 2: Aggregate to season level per line ────────────────────────────────

print("\n── Aggregating to season level ──────────────────────────────")
agg_dict = {col: "sum" for col in QUALITY_COLS if col != "icetime"}
agg_dict["icetime"] = "sum"

line_season = lines.groupby(["lineId", "name", "season", "playerTeam"], as_index=False).agg(agg_dict)

# Per-60 rates for line quality
ice_hours = line_season["icetime"] / 3600
line_season["line_xg_per60"]        = np.where(ice_hours > 0, line_season["xGoalsFor"] / ice_hours, 0)
line_season["line_adj_xg_per60"]    = np.where(ice_hours > 0, line_season["flurryScoreVenueAdjustedxGoalsFor"] / ice_hours, 0)
line_season["line_hd_xg_per60"]     = np.where(ice_hours > 0, line_season["highDangerxGoalsFor"] / ice_hours, 0)
line_season["line_goals_per60"]     = np.where(ice_hours > 0, line_season["goalsFor"] / ice_hours, 0)
line_season["line_xg_pct"]          = line_season["xGoalsPercentage"]
line_season["line_corsi_pct"]       = line_season["corsiPercentage"]
line_season["line_icetime"]         = line_season["icetime"]

print(f"  {len(line_season):,} unique line-seasons")

# ── Step 3: Explode lines to individual players ───────────────────────────────

print("\n── Exploding lines to individual players ────────────────────")
line_season["last_names"] = line_season["name"].str.split("-")
exploded = line_season.explode("last_names")
exploded = exploded.rename(columns={"last_names": "last_name", "playerTeam": "player_team"})
exploded["last_name"] = exploded["last_name"].str.strip()

print(f"  {len(exploded):,} player-line-season rows")
print(f"  {exploded['last_name'].nunique():,} unique last names")

# ── Step 4: Aggregate to player-season level ──────────────────────────────────
# Weight each line's quality by how much time the player spent on that line

print("\n── Aggregating to player-season level ───────────────────────")

LINE_QUALITY_COLS = [
    "line_adj_xg_per60", "line_xg_per60", "line_hd_xg_per60",
    "line_goals_per60", "line_xg_pct", "line_corsi_pct"
]

player_season_rows = []
for (last_name, season, player_team), grp in exploded.groupby(["last_name", "season", "player_team"]):
    total_ice = grp["line_icetime"].sum()
    if total_ice == 0:
        continue

    row = {
        "last_name":   last_name,
        "season":      season,
        "player_team": player_team,
        "total_line_icetime": total_ice,
    }

    for col in LINE_QUALITY_COLS:
        # Weighted average by ice time
        row[col] = (grp[col] * grp["line_icetime"]).sum() / total_ice

    # Number of distinct lines — more lines = less settled role
    row["n_distinct_lines"] = grp["lineId"].nunique()

    player_season_rows.append(row)

player_season = pd.DataFrame(player_season_rows)
print(f"  {len(player_season):,} player-season rows")

# ── Step 5: Join to main dataset via last_name + season + team ────────────────

print("\n── Joining to main dataset ──────────────────────────────────")
main = pd.read_csv(MAIN_FILE)[["player_id", "player_name", "player_team", "season"]]
main["last_name"] = main["player_name"].str.split().str[-1]

merged = main.merge(
    player_season,
    on=["last_name", "season", "player_team"],
    how="left"
)

matched   = merged[merged["line_adj_xg_per60"].notna()]
unmatched = merged[merged["line_adj_xg_per60"].isna()]

print(f"  Matched:   {len(matched):,} ({len(matched)/len(merged)*100:.1f}%)")
print(f"  Unmatched: {len(unmatched):,} ({len(unmatched)/len(merged)*100:.1f}%)")

# Check for duplicate matches (same last name + team + season = ambiguous)
dupes = merged.groupby(["player_id", "season"]).size()
dupes = dupes[dupes > 1]
if len(dupes) > 0:
    print(f"\n  ⚠️  {len(dupes)} ambiguous matches (same last name + team + season)")
    print("  These will be averaged across matches.")
    merged = merged.groupby(["player_id", "season"], as_index=False)[
        ["player_id", "season"] + LINE_QUALITY_COLS + ["n_distinct_lines", "total_line_icetime"]
    ].mean()

# ── Step 6: Save output ───────────────────────────────────────────────────────

OUTPUT_COLS = ["player_id", "season"] + LINE_QUALITY_COLS + ["n_distinct_lines", "total_line_icetime"]
result = merged[OUTPUT_COLS].drop_duplicates(subset=["player_id", "season"])
result = result.fillna(0)

print(f"\n── Output ───────────────────────────────────────────────────")
print(f"  {len(result):,} player-season rows saved")
print(f"\n  Sample top line quality players:")
print(result.sort_values("line_adj_xg_per60", ascending=False).head(10)[
    ["player_id", "season", "line_adj_xg_per60", "line_xg_pct", "n_distinct_lines"]
].to_string(index=False))

result.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved to {OUTPUT_FILE}")