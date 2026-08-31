"""
refresh_and_retrain.py
=======================
One-command "get the model as current as possible" pipeline:

  1. Refresh player_ages.csv from the NHL API.
  2. Rebuild season_dataset.csv, pp_features.csv, defensive_dataset.csv,
     linemate_features.csv from whatever raw MoneyPuck exports are sitting
     in raw_data/game_level/ and raw_data/line_level/ (drop a new season's
     file in there beforehand to include it — see data_sources.py).
  3. Delete the cached trained models so the app retrains on next launch.

Usage:
    python refresh_and_retrain.py

Then run `streamlit run app.py` once — it will notice the models are
missing, retrain (~5-8 min), and re-save the cache. Review the diffs and
commit/push when you're happy with the result; nothing here pushes to git
for you.
"""

import os
import subprocess
import sys

# Must match CACHE_FILE / DEF_CACHE_FILE in model_utils.py
MODEL_CACHE_FILES = [
    "trained_models_forwards_v5.joblib",
    "defensive_models.joblib",
]

# Order matters: lines.py reads season_dataset.csv, so combining_by_season.py
# must run first. The rest are independent of each other.
STEPS = [
    ("Refreshing player ages",            ["fetch_player_ages.py"]),
    ("Building season_dataset.csv",       ["combining_by_season.py"]),
    ("Building pp_features.csv",          ["adding_Power_play.py"]),
    ("Building defensive_dataset.csv",    ["defensive.py"]),
    ("Building linemate_features.csv",    ["lines.py"]),
]


def run_step(label, script_args):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable] + script_args, env=env)
    if result.returncode != 0:
        print(f"\n✗ {label} failed (exit code {result.returncode}) — stopping.")
        sys.exit(result.returncode)


def main():
    for label, script_args in STEPS:
        run_step(label, script_args)

    print(f"\n{'=' * 70}\nClearing cached models so the app retrains\n{'=' * 70}")
    for f in MODEL_CACHE_FILES:
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
