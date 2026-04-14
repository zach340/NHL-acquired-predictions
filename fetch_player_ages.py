"""
fetch_player_ages.py
====================
Fetches player birth dates from the NHL API for all seasons in your dataset
and outputs a CSV with player_id, birth_date, and age at the start of each season.

Usage:
    python fetch_player_ages.py

Output:
    player_ages.csv  — join this onto your main dataset by player_id + season
"""

import requests
import pandas as pd
import time
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────

SEASONS = [
    "20072008", "20082009", "20092010", "20102011", "20112012",
    "20122013", "20132014", "20142015", "20152016", "20162017",
    "20172018", "20182019", "20192020", "20202021", "20212022",
    "20222023", "20232024", "20242025", "20252026",
]

# NHL season starts in October — use Oct 1 as age reference date
SEASON_START_MONTH = 10
SEASON_START_DAY   = 1

OUTPUT_FILE = "player_ages.csv"

# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_skater_bios(season: str) -> pd.DataFrame:
    url = (
        f"https://api.nhle.com/stats/rest/en/skater/bios"
        f"?limit=-1&start=0&cayenneExp=seasonId={season}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        df   = pd.json_normalize(data)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"skaterFullName": "player_name"})
        df["season_str"] = season
        return df[["playerId", "player_name", "birthDate", "season_str"]]
    except Exception as e:
        print(f"  Warning: Failed for season {season}: {e}")
        return pd.DataFrame()


def season_str_to_year(season_str: str) -> int:
    """Convert '20232024' -> 2024 (the season label used in your dataset)."""
    return int(season_str[4:])


def calc_age(birth_date_str: str, season_year: int) -> float:
    """Age in years as of Oct 1 of the season start year (season_year - 1)."""
    try:
        birth = datetime.strptime(birth_date_str, "%Y-%m-%d")
        ref   = datetime(season_year - 1, SEASON_START_MONTH, SEASON_START_DAY)
        return round((ref - birth).days / 365.25, 1)
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("── Fetching player bios from NHL API ────────────────────")
    all_rows = []

    for season in SEASONS:
        print(f"  Season {season}...", end=" ", flush=True)
        df = fetch_skater_bios(season)
        if not df.empty:
            all_rows.append(df)
            print(f"{len(df)} players")
        else:
            print("no data")
        time.sleep(0.3)  # be polite to the API

    if not all_rows:
        print("No data fetched — check your internet connection.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Add season year and age
    combined["season"] = combined["season_str"].apply(season_str_to_year)
    combined["age"]    = combined.apply(
        lambda r: calc_age(r["birthDate"], r["season"]), axis=1
    )
    combined["age_sq"] = combined["age"] ** 2

    # Rename to match your dataset
    combined = combined.rename(columns={"playerId": "player_id"})

    # Keep one row per player per season
    result = combined[["player_id", "player_name", "season", "birthDate", "age", "age_sq"]].drop_duplicates(
        subset=["player_id", "season"]
    ).sort_values(["player_id", "season"])

    # ── Fill any remaining missing ages via birthDate interpolation ────────────
    # For players with a known birthDate but missing age (e.g. new season),
    # compute age directly from birthDate rather than leaving NaN.
    missing_mask = result["age"].isna() & result["birthDate"].notna()
    if missing_mask.any():
        result.loc[missing_mask, "age"] = result.loc[missing_mask].apply(
            lambda r: calc_age(r["birthDate"], r["season"]), axis=1
        )
        result.loc[result["age"].notna(), "age_sq"] = result.loc[result["age"].notna(), "age"] ** 2
        print(f"  Filled {missing_mask.sum()} missing ages from birthDate.")

    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\n── Done ─────────────────────────────────────────────────")
    print(f"  {len(result):,} player-season rows saved to {OUTPUT_FILE}")
    print(f"  {result['player_id'].nunique():,} unique players")
    print(f"  Age range: {result['age'].min()} to {result['age'].max()}")
    missing_ages = result["age"].isna().sum()
    if missing_ages:
        print(f"  Warning: {missing_ages} rows still missing age (no birthDate in API).")


if __name__ == "__main__":
    main()