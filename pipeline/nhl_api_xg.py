"""
nhl_api_xg.py
=============
Expected-goals model trained on the parsed NHL API shots
(raw_data/nhl_api_parsed/<season>/shots.parquet). Adds to every shots file:

  xg             goal probability of each unblocked attempt (0 for blocked)
  xg_flurry      flurry-adjusted: later shots in a rapid sequence are discounted
                 by the chance an earlier one already scored
  danger         low / medium / high  (cut-offs in DANGER_CUTS)

xG is cross-fitted by season: each season is scored by a model that never saw
it, so a season's xG never contains its own goals.

Incremental by default: only shots files without xG (seasons that are new or
were just re-parsed) are scored, by one model trained on every other season.
The first run, or --full, refits all seasons with 4 season-group folds.

    python pipeline/nhl_api_xg.py           # score new / re-parsed seasons only
    python pipeline/nhl_api_xg.py --full    # refit and rescore every season
"""

import argparse
import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhl_api_parse import PARSED_DIR  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

UNBLOCKED   = {"goal", "shot-on-goal", "missed-shot"}
N_FOLDS     = 4
FLURRY_GAP  = 5          # seconds between same-team shots to count as one flurry
# Chosen so the low/medium/high shares of unblocked shots match MoneyPuck's
# (75% / 19% / 7% over 2010-2024) — MoneyPuck's own labels aren't xG cut-offs.
DANGER_CUTS = (0.090, 0.182)

NUMERIC = ["dist", "angle", "x", "abs_y", "own_skaters", "opp_skaters", "opp_empty_net", "own_goalie_pulled",
           "score_diff", "is_home", "period", "prev_same_team", "prev_dt", "prev_dist", "prev_xn",
           "rebound", "rush", "speed"]
CATEGORICAL = ["shot_type", "prev_kind"]


def load_shots():
    frames = []
    for path in sorted(glob.glob(os.path.join(PARSED_DIR, "*", "shots.parquet"))):
        frames.append(pd.read_parquet(path).assign(_path=path))
    return pd.concat(frames, ignore_index=True)


def features(s):
    f = pd.DataFrame(index=s.index)
    f["dist"], f["angle"], f["x"] = s["dist"], s["angle"], s["x"]
    f["abs_y"] = s["y"].abs()
    f["own_skaters"], f["opp_skaters"] = s["own_skaters"], s["opp_skaters"]
    f["opp_empty_net"] = (s["opp_goalie"] == 0).astype(int)
    f["own_goalie_pulled"] = (s["own_goalie"] == 0).astype(int)
    f["score_diff"] = s["score_diff"].clip(-3, 3)
    f["is_home"] = s["is_home"].astype(int)
    f["period"] = s["period"].clip(upper=4)
    f["prev_same_team"] = s["prev_same_team"].astype(float)
    f["prev_dt"] = s["prev_dt"].clip(0, 120)
    f["prev_dist"] = s["prev_dist"]
    f["prev_xn"] = s["prev_xn"]
    shot_kinds = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
    f["rebound"] = (s["prev_kind"].isin(shot_kinds) & s["prev_same_team"].fillna(False).astype(bool)
                    & (s["prev_dt"] <= 3)).astype(int)
    f["rush"] = ((s["prev_xn"] < 25) & (s["prev_dt"] <= 4)).astype(int)
    f["speed"] = s["prev_dist"] / s["prev_dt"].clip(lower=1)
    for c in CATEGORICAL:
        f[c] = s[c].fillna("none").astype("category")
    return f[NUMERIC + CATEGORICAL]


def make_model():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42, verbose=-1)


def trainable(s):
    """Unblocked, located, non-penalty-shot attempts."""
    return (s["kind"].isin(UNBLOCKED) & s["dist"].notna()
            & ~((s["own_skaters"] <= 1) & (s["opp_skaters"] <= 1)))


def fold_ids(s):
    """Cross-fit fold per shot: by season (interleaved), or by game while fewer seasons than folds."""
    seasons = np.array(sorted(s["season"].unique()))
    if len(seasons) >= N_FOLDS:
        return s["season"].map({y: i % N_FOLDS for i, y in enumerate(seasons)})
    return s["game_id"] % N_FOLDS


def flurry_adjust(s):
    """Discount each shot in a same-team sequence (gap ≤ FLURRY_GAP s) by P(no earlier shot scored)."""
    s = s.sort_values(["game_id", "t", "event_id"])
    new_seq = ((s["game_id"] != s["game_id"].shift()) | (s["team_id"] != s["team_id"].shift())
               | (s["t"] - s["t"].shift() > FLURRY_GAP) | (s["period"] != s["period"].shift()))
    seq = new_seq.cumsum()
    miss = 1 - s["xg"]
    prior_miss = miss.groupby(seq).cumprod() / miss.where(miss > 0, 1)
    return (s["xg"] * prior_miss).reindex(s.index)


def danger_labels(s):
    return np.where(~s["kind"].isin(UNBLOCKED), "",
                    np.where(s["xg"] >= DANGER_CUTS[1], "high",
                             np.where(s["xg"] >= DANGER_CUTS[0], "medium", "low")))


def needs_xg(path):
    return "xg" not in pq.read_schema(path).names


def report(p, yy):
    base = np.full(len(yy), yy.mean())
    print(f"overall: log loss {log_loss(yy, p):.4f} (no-skill {log_loss(yy, base):.4f})  AUC {roc_auc_score(yy, p):.4f}")
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    print("calibration by decile (predicted vs actual goal rate):")
    print(pd.DataFrame({"pred": p.groupby(dec).mean(), "actual": yy.groupby(dec).mean()}).round(4).to_string())


def main(full=False):
    paths = sorted(glob.glob(os.path.join(PARSED_DIR, "*", "shots.parquet")))
    todo_paths = paths if full else [p for p in paths if needs_xg(p)]
    if not todo_paths:
        print("xG already up to date for every season")
        return

    s = load_shots()
    mask = trainable(s)
    X, y = features(s), s["is_goal"].astype(int)
    todo = s["_path"].isin(todo_paths)
    refit_all = len(todo_paths) == len(paths)
    print(f"{mask.sum():,} unblocked attempts across {s['season'].nunique()} seasons; goal rate {y[mask].mean():.3%}")
    print(f"scoring {len(todo_paths)} of {len(paths)} seasons"
          + (" (4-fold refit)" if refit_all else " with a model trained on the others"), flush=True)

    s.loc[todo, "xg"] = 0.0
    if refit_all:
        folds = fold_ids(s)
        for k in range(N_FOLDS):
            tr, te = mask & (folds != k), mask & (folds == k)
            m = make_model().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            s.loc[te, "xg"] = p
            print(f"  fold {k} (seasons {sorted(s.loc[te, 'season'].unique())}): log loss {log_loss(y[te], p):.4f}  "
                  f"AUC {roc_auc_score(y[te], p):.4f}  xG/goals {p.sum() / y[te].sum():.3f}", flush=True)
    else:
        tr, te = mask & ~todo, mask & todo
        m = make_model().fit(X[tr], y[tr])
        s.loc[te, "xg"] = m.predict_proba(X[te])[:, 1]
    report(s.loc[mask & todo, "xg"], y[mask & todo])

    part = s[todo].copy()
    part["xg_flurry"] = flurry_adjust(part)
    part["danger"] = danger_labels(part)
    for path, rows in part.groupby("_path"):
        rows.drop(columns="_path").to_parquet(path, index=False)
    print(f"xG written to {len(todo_paths)} shots file(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="refit and rescore every season")
    main(ap.parse_args().full)
