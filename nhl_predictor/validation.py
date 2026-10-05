"""
validation.py
=============
The Validation tab's comparison as plain functions, plus the weekly history
file (validation_history.csv) that scripts/validation_snapshot.py appends to.
"""

import os

import pandas as pd
from sklearn.metrics import mean_absolute_error

from . import defense, nhl_api, offense
from .config import VALIDATION_HISTORY_FILE, season_id, season_label

HISTORY_COLS = ["date", "season", "fwd_matched", "points_mae", "goals_mae",
                "def_matched", "hits_mae", "takeaways_mae", "pim_mae"]


def validation_season(fwd):
    """The first season the models never saw."""
    return int(fwd.df["season"].max()) + 1


def forward_results(fwd, season):
    """(validation frame, error). Empty frame when no forward has 10+ games yet."""
    actual, err = nhl_api.fetch_season_skaters(season_id(season))
    if err or actual is None or actual.empty:
        return pd.DataFrame(), err
    return offense.build_validation_results(actual, fwd), None


def defense_results(dfn, season):
    """(validation frame, error). Empty frame when no defenseman has 10+ games yet."""
    actual, err = nhl_api.fetch_defensive_stats(season_id(season))
    if err or actual is None or actual.empty:
        return pd.DataFrame(), err
    actual = actual[actual["player_id"].isin(dfn.profiles.keys())]
    return defense.build_validation(actual, dfn), None


def _mae(df, short):
    return round(mean_absolute_error(df[f"actual_{short}"], df[f"pred_{short}"]), 4)


def snapshot(fwd, dfn, date):
    """One history row for today, or None if no player has 10+ games yet."""
    season = validation_season(fwd)
    f_val, _ = forward_results(fwd, season)
    d_val, _ = defense_results(dfn, season)
    if f_val.empty and d_val.empty:
        return None
    row = {"date": str(date), "season": season_label(season),
           "fwd_matched": len(f_val), "def_matched": len(d_val)}
    if not f_val.empty:
        row.update(points_mae=_mae(f_val, "points_gp"), goals_mae=_mae(f_val, "goals_gp"))
    if not d_val.empty:
        row.update(hits_mae=_mae(d_val, "hits_pg"), takeaways_mae=_mae(d_val, "tk_pg"), pim_mae=_mae(d_val, "pim_pg"))
    return row


def load_history(path=VALIDATION_HISTORY_FILE):
    if not os.path.exists(path):
        return pd.DataFrame(columns=HISTORY_COLS)
    return pd.read_csv(path)


def append_history(row, path=VALIDATION_HISTORY_FILE):
    """Add `row`, replacing any earlier row for the same date and season (e.g. a manual re-run)."""
    hist = load_history(path)
    hist = hist[~((hist["date"].astype(str) == row["date"]) & (hist["season"] == row["season"]))]
    new = pd.DataFrame([row])
    hist = (pd.concat([hist, new], ignore_index=True) if len(hist) else new).reindex(columns=HISTORY_COLS)
    hist = hist.sort_values(["date", "season"]).reset_index(drop=True)
    hist.to_csv(path, index=False)
    return hist
