"""
nhl_api_rapm.py
===============
Regularised adjusted plus-minus (RAPM) from the parsed NHL API shift data.

Each 5v5 stretch with no line change is two observations — home attacking
and away attacking — of xG generated per 60, weighted by its length. A ridge
regression on "who was on offense / who was on defense" (plus home ice and
score state) separates each skater's impact from his linemates' and
opponents':

  rapm_off   xG for per 60 the player adds on offense      (higher = better)
  rapm_def   xG against per 60 he allows on defense        (lower  = better)

Also, per player-season, the TOI-weighted RAPM of his 5v5 teammates
(mate_rapm_off / mate_rapm_def) — how much help he had. Fits use a rolling
3-season window ending at each season (more stable than a single season).

Output: rapm_features.csv  (player_id, season, rapm_off, rapm_def, rapm_toi,
                            mate_rapm_off, mate_rapm_def)

    python pipeline/nhl_api_rapm.py              # all parsed seasons
    python pipeline/nhl_api_rapm.py --alpha 2000 # override the ridge penalty
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhl_api_datasets import UNBLOCKED, load, parsed_seasons  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

WINDOW  = 3        # seasons per fit (ending at the season being described)
ALPHA   = 10.0     # ridge penalty (weights are hours); chosen with --tune
MIN_SECS = 1       # drop zero-length stretches
SKATER_COLS = [f"{s}{k}" for s in "ha" for k in range(1, 6)]


def stints(season):
    """5v5 stretches with home/away skater ids and xG for each side."""
    seg = load("segments", season)
    seg = seg[(seg["h_skaters"] == 5) & (seg["a_skaters"] == 5) & seg["h_goalie"] & seg["a_goalie"]
              & (seg["duration"] >= MIN_SECS)].copy()
    shots = load("shots", season)
    shots = shots[shots["kind"].isin(UNBLOCKED) & (shots["seg"] >= 0)]
    xg = shots.groupby(["game_id", "seg", "is_home"])["xg"].sum().unstack(fill_value=0)
    xg = xg.reindex(columns=[False, True], fill_value=0)
    xg.columns = ["xg_away", "xg_home"]
    seg = seg.merge(xg, left_on=["game_id", "seg"], right_index=True, how="left").fillna({"xg_away": 0, "xg_home": 0})

    # Score state at the stretch, from the home side: goals before its start
    goals = load("shots", season)
    goals = goals[goals["is_goal"]][["game_id", "t", "is_home"]]
    seg["score_home"] = _score_before(seg, goals)
    seg["season"] = season
    # Occasional shift-chart gaps leave a slot empty; RAPM needs all ten skaters
    return seg[(seg[SKATER_COLS] > 0).all(axis=1)].reset_index(drop=True)


def _score_before(seg, goals):
    out = np.zeros(len(seg))
    for gid, idx in seg.groupby("game_id").groups.items():
        g = goals[goals["game_id"] == gid]
        if g.empty:
            continue
        starts = seg.loc[idx, "start"].values
        home_t = np.sort(g.loc[g["is_home"], "t"].values)
        away_t = np.sort(g.loc[~g["is_home"], "t"].values)
        out[seg.index.get_indexer(idx)] = (np.searchsorted(home_t, starts, side="right")
                                           - np.searchsorted(away_t, starts, side="right"))
    return out


def design(st, players):
    """Sparse X (offense ids | defense ids | home | score state), y = xG/60, w = hours."""
    index = {p: i for i, p in enumerate(players)}
    n, P = len(st), len(players)
    home = [st[f"h{k}"].values for k in range(1, 6)]
    away = [st[f"a{k}"].values for k in range(1, 6)]
    rows, cols = [], []
    for obs, (off, dfn) in enumerate(((home, away), (away, home))):
        base = obs * n
        for k in range(5):
            rows.append(base + np.arange(n)); cols.append(np.array([index[p] for p in off[k]]))
            rows.append(base + np.arange(n)); cols.append(P + np.array([index[p] for p in dfn[k]]))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, 2 * P))

    is_home_off = np.r_[np.ones(n), np.zeros(n)]
    state = np.clip(np.r_[st["score_home"].values, -st["score_home"].values], -2, 2)
    extra = np.column_stack([is_home_off] + [(state == s).astype(float) for s in (-2, -1, 1, 2)])
    X = sparse.hstack([X, sparse.csr_matrix(extra)]).tocsr()

    hours = st["duration"].values / 3600
    y = np.r_[st["xg_home"].values, st["xg_away"].values] / np.r_[hours, hours]
    w = np.r_[hours, hours]
    return X, y, w


def fit_window(st, alpha):
    players = np.unique(st[SKATER_COLS].values)
    X, y, w = design(st, players)
    mean = np.average(y, weights=w)
    m = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg", max_iter=2000)
    m.fit(X, y - mean, sample_weight=w)
    P = len(players)
    toi = (np.asarray(abs(X[: len(st), :P]).T @ w[: len(st)]).ravel()
           + np.asarray(abs(X[len(st):, :P]).T @ w[len(st):]).ravel())
    return pd.DataFrame({"player_id": players, "rapm_off": m.coef_[:P], "rapm_def": m.coef_[P:2 * P],
                         "rapm_toi": toi * 60})


def teammate_rapm(st, coefs):
    """TOI-weighted RAPM of each player's 5v5 teammates in this season's stints."""
    off = coefs.set_index("player_id")["rapm_off"]
    dfn = coefs.set_index("player_id")["rapm_def"]
    rows = []
    for side in "ha":
        ids = st[[f"{side}{k}" for k in range(1, 6)]].values
        o = np.vectorize(lambda p: off.get(p, 0.0))(ids)
        d = np.vectorize(lambda p: dfn.get(p, 0.0))(ids)
        for k in range(5):
            others = [j for j in range(5) if j != k]
            rows.append(pd.DataFrame({"player_id": ids[:, k], "w": st["duration"].values,
                                      "mo": o[:, others].mean(axis=1), "md": d[:, others].mean(axis=1)}))
    long = pd.concat(rows, ignore_index=True)
    long = long[long["player_id"] > 0]
    agg = long.assign(mo=long["mo"] * long["w"], md=long["md"] * long["w"]).groupby("player_id")[["w", "mo", "md"]].sum()
    return pd.DataFrame({"mate_rapm_off": agg["mo"] / agg["w"], "mate_rapm_def": agg["md"] / agg["w"]}).reset_index()


def tune(seasons, alphas):
    """Pick alpha by predicting held-out games' stint xG (game-level split, last window)."""
    st = pd.concat([stints(s) for s in seasons[-WINDOW:]], ignore_index=True)
    test = st["game_id"] % 5 == 0
    players = np.unique(st[SKATER_COLS].values)
    Xtr, ytr, wtr = design(st[~test], players)
    Xte, yte, wte = design(st[test], players)
    mean = np.average(ytr, weights=wtr)
    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=False, solver="sparse_cg", max_iter=2000).fit(Xtr, ytr - mean, sample_weight=wtr)
        err = np.average((yte - mean - m.predict(Xte)) ** 2, weights=wte)
        base = np.average((yte - mean) ** 2, weights=wte)
        print(f"  alpha {a:>7g}: weighted MSE {err:.3f}  (no-player baseline {base:.3f})", flush=True)


def main(alpha, out_path):
    seasons = parsed_seasons()
    cache = {}
    out = []
    for s in seasons:
        window = [y for y in seasons if s - WINDOW < y <= s]
        for y in window:
            if y not in cache:
                cache[y] = stints(y)
        for y in list(cache):
            if y not in window:
                del cache[y]
        coefs = fit_window(pd.concat([cache[y] for y in window], ignore_index=True), alpha)
        mates = teammate_rapm(cache[s], coefs)
        played = np.unique(cache[s][SKATER_COLS].values)
        res = coefs[coefs["player_id"].isin(played)].merge(mates, on="player_id", how="left")
        res.insert(1, "season", s)
        out.append(res)
        print(f"  {s}: {len(res):,} skaters (window {window[0]}–{window[-1]})", flush=True)
    pd.concat(out, ignore_index=True).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--tune", action="store_true", help="print held-out error for a grid of alphas")
    ap.add_argument("--out", default="rapm_features.csv")
    a = ap.parse_args()
    if a.tune:
        tune(parsed_seasons(), [100, 300, 1000, 3000, 10000, 30000])
    else:
        main(a.alpha, a.out)
