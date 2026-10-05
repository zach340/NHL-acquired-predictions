"""
yearly_retrain.py
=================
Yearly refresh + retrain (run by scripts/yearly_retrain.bat from Task
Scheduler each July, after the season ends):

  1. Score the current saved models on the season that just finished — the
     first season they never saw.
  2. refresh_and_retrain.py: fetch new games, parse / score only new or
     changed seasons, rebuild the CSVs.
  3. Sanity check: train the refreshed pipeline on the same seasons as the
     old models and score it on that same holdout season. A big gap from
     step 1 means the data or code changed something unexpectedly.
  4. Train the final models on every season and save the caches.
  5. Log old vs new metrics. Nothing is committed or pushed.

    python scripts/yearly_retrain.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402
import streamlit.logger  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

# Training and the NHL API fetchers use Streamlit widgets and caches; outside the
# app they only log "no runtime" warnings, which would bury the comparison in the log
streamlit.logger.set_log_level("error")

from nhl_predictor import config, defense, offense, validation  # noqa: E402
from nhl_predictor.training import load_bundle, save_bundle  # noqa: E402

DATA_FILES = (config.DATA_FILE, config.DEF_FILE, config.PP_FILE, config.LINEMATE_FILE)
WORSE_FLAG = 0.05     # flag a metric that got more than 5% worse

METRICS = [  # (label, frame, actual column, predicted column)
    ("Points/GP MAE",    "fwd", "actual_points_gp", "pred_points_gp"),
    ("Goals/GP MAE",     "fwd", "actual_goals_gp",  "pred_goals_gp"),
    ("Hits/GP MAE",      "def", "actual_hits_pg",   "pred_hits_pg"),
    ("Takeaways/GP MAE", "def", "actual_tk_pg",     "pred_tk_pg"),
    ("PIM/GP MAE",       "def", "actual_pim_pg",    "pred_pim_pg"),
]


def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}" if msg else "", flush=True)


def holdout_metrics(fwd, dfn, season):
    """{label: MAE} plus matched counts for both models on `season`."""
    frames = {"fwd": validation.forward_results(fwd, season)[0], "def": validation.defense_results(dfn, season)[0]}
    out = {"Forwards matched": len(frames["fwd"]), "Defensemen matched": len(frames["def"])}
    for label, which, actual, pred in METRICS:
        df = frames[which]
        out[label] = mean_absolute_error(df[actual], df[pred]) if len(df) else float("nan")
    return out


def cv_metrics(fwd, dfn):
    out = {f"F {k}": v["mae"][0] for k, v in fwd.next_metrics.items()}
    out.update({f"D {k}": v["mae"][0] for k, v in dfn.next_metrics.items()})
    return out


def train(data_dir=None):
    """Train both models on the CSVs in `data_dir` (default: the repo)."""
    here = os.getcwd()
    if data_dir:
        os.chdir(data_dir)
    try:
        return (offense.load_and_train(config.DATA_FILE, config.AGES_FILE),
                defense.load_and_train(config.DEF_FILE, config.AGES_FILE))
    finally:
        os.chdir(here)


def train_through(last_season):
    """Train on a copy of the refreshed CSVs restricted to seasons <= last_season."""
    with tempfile.TemporaryDirectory() as tmp:
        for f in DATA_FILES:
            d = pd.read_csv(f)
            d[d["season"] <= last_season].to_csv(os.path.join(tmp, f), index=False)
        for f in (config.AGES_FILE, config.NAMES_FILE):
            if os.path.exists(f):
                shutil.copy(f, tmp)
        return train(tmp)


def table(title, columns, rows, flag=True):
    """rows: {label: [value per column]}; flags the last column if > WORSE_FLAG worse than the first."""
    lines = [title, f"  {'':<22}" + "".join(f"{c:>16}" for c in columns)]
    for label, vals in rows.items():
        cells = "".join(f"{v:>16.4f}" if isinstance(v, float) else f"{v:>16}" for v in vals)
        first, last = vals[0], vals[-1]
        if isinstance(first, float) and isinstance(last, float) and first > 0 and "MAE" in label + title:
            change = (last - first) / first
            note = f"   {change:+.1%}" + ("   <-- CHECK" if flag and change > WORSE_FLAG else "")
        else:
            note = ""
        lines.append(f"  {label:<22}{cells}{note}")
    return "\n".join(lines)


def main():
    t0 = time.time()
    log(f"Yearly retrain started in {ROOT}")

    old_fwd, old_dfn = load_bundle(config.CACHE_FILE), load_bundle(config.DEF_CACHE_FILE)
    if old_fwd is None or old_dfn is None:
        log("No saved models found — old-vs-new comparison will be skipped.")
    else:
        old_last = int(old_fwd.df["season"].max())
        log(f"Old models trained through {config.season_label(old_last)}")

    log("Running refresh_and_retrain.py ...")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if subprocess.run([sys.executable, "refresh_and_retrain.py"], env=env).returncode != 0:
        log("refresh_and_retrain.py failed — stopping. The CSVs may be partly rebuilt; check `git status`.")
        sys.exit(1)

    new_last = int(pd.read_csv(config.DATA_FILE, usecols=["season"])["season"].max())
    log(f"Refreshed data runs through {config.season_label(new_last)}")

    report = []
    holdout = None if old_fwd is None else old_last + 1
    if holdout is not None and holdout > new_last:
        log(f"No new finished season since the old models were trained "
            f"({config.season_label(holdout)} isn't over yet) — skipping the holdout comparison.")
    elif holdout is not None:
        log(f"Scoring old models on {config.season_label(holdout)} ...")
        old_hold = holdout_metrics(old_fwd, old_dfn, holdout)
        log(f"Training refreshed pipeline on seasons through {config.season_label(old_last)} (same window) ...")
        chk_fwd, chk_dfn = train_through(old_last)
        new_hold = holdout_metrics(chk_fwd, chk_dfn, holdout)
        report.append(table(f"Holdout {config.season_label(holdout)}: old models vs refreshed pipeline trained on "
                            f"the same seasons (players with 10+ games)",
                            ["old models", "refreshed"],
                            {k: [old_hold[k], new_hold[k]] for k in old_hold}))

    log("Training final models on every season ...")
    fwd, dfn = train()
    save_bundle(fwd, config.CACHE_FILE)
    save_bundle(dfn, config.DEF_CACHE_FILE)
    log(f"Saved {config.CACHE_FILE} and {config.DEF_CACHE_FILE}")

    if old_fwd is not None:
        old_cv, new_cv = cv_metrics(old_fwd, old_dfn), cv_metrics(fwd, dfn)
        report.append(table("Next Season model, season-based CV MAE (reference only: each model is validated "
                            "on its own last 3 seasons, so these move with the seasons, not just the model)",
                            ["old models", "new models"],
                            {k: [old_cv[k], new_cv.get(k, float("nan"))] for k in old_cv}, flag=False))

    print()
    print("=" * 78)
    print("\n\n".join(report) if report else "No old models to compare against.")
    print("=" * 78)
    log(f"Done in {(time.time() - t0) / 60:.0f} min. Nothing was committed — review `git status` and the "
        "numbers above, then commit and push the CSVs and .joblib files if they look right.")


if __name__ == "__main__":
    main()
