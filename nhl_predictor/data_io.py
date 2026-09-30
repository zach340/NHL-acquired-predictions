"""
data_io.py
==========
CSV loading helpers shared by both models.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from .config import CURRENT_SEASON_START, DATA_FILE


def safe_read_csv(path, **kwargs):
    """
    Read a CSV trying cp1252 first (Windows default), then UTF-8 variants.
    Handles special characters like ä, é, ö, å that appear in player names.
    """
    for enc in ("cp1252", "utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, ValueError):
            continue
    # Last resort — replace bad bytes rather than crash
    return pd.read_csv(path, encoding="latin-1", encoding_errors="replace", **kwargs)


def age_on_season_start(birth_date, season):
    """Age in years on Oct 1 of the season's start year (2024 = the 2024-25 season)."""
    try:
        birth = datetime.strptime(str(birth_date), "%Y-%m-%d")
        ref   = datetime(int(season), 10, 1)
        return round((ref - birth).days / 365.25, 1)
    except Exception:
        return None


def load_ages(ages_path):
    """
    Load player_ages.csv and compute any missing ages from birthDate.
    Returns a DataFrame with player_id, season, age, age_sq.
    """
    ages = safe_read_csv(ages_path)

    if "birthDate" in ages.columns and ages["age"].isna().any():
        ages["age"] = ages.apply(
            lambda r: r["age"] if pd.notna(r["age"])
            else (None if pd.isna(r.get("birthDate")) else age_on_season_start(r["birthDate"], r["season"])),
            axis=1,
        )
        ages["age_sq"] = ages["age"] ** 2

    keep = ["player_id", "season", "age", "age_sq"]
    return ages[[c for c in keep if c in ages.columns]]


def latest_known_age(ages_path, player_id, as_of_season):
    """
    Most recent age on file for a player, rolled forward to `as_of_season`.
    Returns None if the player has no age on file.
    """
    ages = safe_read_csv(ages_path)
    rows = ages[ages["player_id"] == player_id].sort_values("season", ascending=False)
    if rows.empty or pd.isna(rows.iloc[0].get("age")):
        return None
    base_age    = float(rows.iloc[0]["age"])
    base_season = int(rows.iloc[0]["season"])
    return base_age + max(0, int(as_of_season) - base_season)


def current_age(ages_path, player_id):
    """Player's age rolled forward to the current season, or None."""
    try:
        return latest_known_age(ages_path, player_id, CURRENT_SEASON_START)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_defensive_offensive_stats():
    """
    Offensive stats for defensemen (most recent season per player) from
    season_dataset.csv.

    Returns (stats_by_pid, dmen_df, error):
      - stats_by_pid: player_id -> {points_pg, goals_pg, game_score_pg, pp_icetime_pct, season}
      - dmen_df: one row per D-man, used for live percentile ranking
    """
    try:
        df = safe_read_csv(DATA_FILE)
        df = df[df["position"] == "D"].copy()
        df = df.sort_values("season", ascending=False).groupby("player_id").first().reset_index()
        result = {}
        for _, row in df.iterrows():
            result[int(row["player_id"])] = {
                "points_pg":      float(row.get("points_per_game",     0) or 0),
                "goals_pg":       float(row.get("goals_per_game",      0) or 0),
                "game_score_pg":  float(row.get("game_score_per_game", 0) or 0),
                "pp_icetime_pct": float(row.get("pp_icetime_pct",      0) or 0),
                "season":         int(row.get("season", 2024)),
            }
        return result, df, None
    except Exception as e:
        return {}, None, str(e)
