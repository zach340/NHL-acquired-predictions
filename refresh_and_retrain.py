"""
refresh_and_retrain.py
=======================
One-command "get the model as current as possible" pipeline:

  1. Refresh player_ages.csv from the NHL API.
  2. Rebuild season_dataset.csv, pp_features.csv, defensive_dataset.csv,
     linemate_features.csv from either
       - the NHL API (default): downloads any new games, parses them and
         fits the expected-goals model. No manual downloads; the first run
         takes ~4-5 hours (raw games go to raw_data/nhl_api/), later runs
         only fetch new games, or
       - MoneyPuck (--source moneypuck): whatever raw exports are sitting in
         raw_data/game_level/ and raw_data/line_level/ (drop a new season's
         file in there beforehand — see pipeline/data_sources.py).
  3. Delete the cached trained models so the app retrains on next launch.

Usage (from the repo root):
    python refresh_and_retrain.py
    python refresh_and_retrain.py --source moneypuck

Then run `streamlit run app.py` once — it will notice the models are
missing, retrain (~15-20 min), and re-save the cache. Review the diffs and
commit/push when you're happy with the result; nothing here pushes to git
for you.
"""

import argparse
import os
import subprocess
import sys

from nhl_predictor.config import CACHE_FILE, DEF_CACHE_FILE

# Order matters within each source: linemates.py reads season_dataset.csv, and
# each NHL API step reads the previous step's output.
STEPS = {
    "moneypuck": [
        ("Refreshing player ages",         ["pipeline/fetch_player_ages.py"]),
        ("Building season_dataset.csv",    ["pipeline/season_dataset.py"]),
        ("Building pp_features.csv",       ["pipeline/power_play.py"]),
        ("Building defensive_dataset.csv", ["pipeline/defensive_dataset.py"]),
        ("Building linemate_features.csv", ["pipeline/linemates.py"]),
    ],
    "nhl-api": [
        ("Refreshing player ages",           ["pipeline/fetch_player_ages.py"]),
        ("Downloading new games",            ["pipeline/nhl_api_download.py"]),
        ("Parsing play-by-play and shifts",  ["pipeline/nhl_api_parse.py"]),
        ("Fitting the expected-goals model", ["pipeline/nhl_api_xg.py"]),
        ("Building the season-level CSVs",   ["pipeline/nhl_api_datasets.py", "--out", "."]),
    ],
}


def run_step(label, args):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, *args], env=env)
    if result.returncode != 0:
        print(f"\n✗ {label} failed (exit code {result.returncode}) — stopping.")
        sys.exit(result.returncode)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(STEPS), default="nhl-api")
    for label, args in STEPS[ap.parse_args().source]:
        run_step(label, args)

    print(f"\n{'=' * 70}\nClearing cached models so the app retrains\n{'=' * 70}")
    for f in (CACHE_FILE, DEF_CACHE_FILE):
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed {f}")
        else:
            print(f"  {f} not present, nothing to remove")

    print(
        "\nDone. Data files rebuilt and model cache cleared.\n"
        "Next: run `streamlit run app.py` once to retrain (~5-8 min) and "
        "re-save the model cache, then review and commit the changes."
    )


if __name__ == "__main__":
    main()
