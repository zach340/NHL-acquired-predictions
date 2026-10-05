"""
validation_snapshot.py
======================
Weekly accuracy check (run by .github/workflows/weekly-validation-snapshot.yml):
load the saved models (no retraining), compare their Next Season predictions
with the current season's NHL API stats — the same comparison as the
Validation tab — and append one row to validation_history.csv.

Exits quietly without writing anything until some players have 10+ games.

    python scripts/validation_snapshot.py
"""

import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from nhl_predictor import validation  # noqa: E402
from nhl_predictor.config import CACHE_FILE, DEF_CACHE_FILE  # noqa: E402
from nhl_predictor.training import load_bundle  # noqa: E402


def main():
    fwd, dfn = load_bundle(CACHE_FILE), load_bundle(DEF_CACHE_FILE)
    if fwd is None or dfn is None:
        sys.exit(f"Saved models not found ({CACHE_FILE}, {DEF_CACHE_FILE}); nothing to validate.")

    row = validation.snapshot(fwd, dfn, date.today())
    if row is None:
        print("No players with 10+ games yet — skipping.")
        return
    validation.append_history(row)
    print("Appended:", row)


if __name__ == "__main__":
    main()
