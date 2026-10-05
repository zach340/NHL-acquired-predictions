"""
refresh_and_retrain.py
=======================
One-command "get the model as current as possible" pipeline:

  1. Refresh player_ages.csv from the NHL API.
  2. Download any new games from the NHL API, parse them, refit the
     expected-goals model and rebuild season_dataset.csv, pp_features.csv,
     defensive_dataset.csv and linemate_features.csv. The first run takes
     ~4-5 hours (raw games go to raw_data/nhl_api/); later runs only fetch
     new games.
  3. Delete the cached trained models so the app retrains on next launch.

Usage (from the repo root):
    python refresh_and_retrain.py

Then run `streamlit run app.py` once — it will notice the models are
missing, retrain (~15 min), and re-save the cache. Review the diffs and
commit/push when you're happy with the result; nothing here pushes to git
for you.
"""

import os
import subprocess
import sys

from nhl_predictor.config import CACHE_FILE, DEF_CACHE_FILE

# Each step reads the previous step's output.
STEPS = [
    ("Refreshing player ages",           "pipeline/fetch_player_ages.py"),
    ("Downloading new games",            "pipeline/nhl_api_download.py"),
    ("Parsing play-by-play and shifts",  "pipeline/nhl_api_parse.py"),
    ("Fitting the expected-goals model", "pipeline/nhl_api_xg.py"),
    ("Building the season-level CSVs",   "pipeline/nhl_api_datasets.py"),
]


def run_step(label, script):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        print(f"\n✗ {label} failed (exit code {result.returncode}) — stopping.")
        sys.exit(result.returncode)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for label, script in STEPS:
        run_step(label, script)

    print(f"\n{'=' * 70}\nClearing cached models so the app retrains\n{'=' * 70}")
    for f in (CACHE_FILE, DEF_CACHE_FILE):
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed {f}")
        else:
            print(f"  {f} not present, nothing to remove")

    print(
        "\nDone. Data files rebuilt and model cache cleared.\n"
        "Next: run `streamlit run app.py` once to retrain (~15 min) and "
        "re-save the model cache, then review and commit the changes."
    )


if __name__ == "__main__":
    main()
