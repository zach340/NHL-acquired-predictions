"""
data_sources.py
================
Shared helpers so the pipeline scripts (combining_by_season.py,
adding_Power_play.py, defensive.py, lines.py) can read from *multiple*
raw MoneyPuck exports instead of one hardcoded file.

MoneyPuck publishes one big historical export plus a separate file for
the current in-progress season. To add a new season: drop its CSV into
raw_data/game_level/ (game-level exports) or raw_data/line_level/
(line-level exports) and rerun the pipeline — no code changes needed.

Historical and current-season exports can overlap in coverage (e.g. a
season that was "in progress" in one download and "final" in another),
so downstream scripts should dedupe on a natural key (player_id/lineId +
game_id + situation) after concatenating chunks from every file.
"""

import glob
import os

import pandas as pd

GAME_LEVEL_DIR = "raw_data/game_level"
LINE_LEVEL_DIR = "raw_data/line_level"


def game_level_files():
    return sorted(glob.glob(os.path.join(GAME_LEVEL_DIR, "*.csv")))


def line_level_files():
    return sorted(glob.glob(os.path.join(LINE_LEVEL_DIR, "*.csv")))


def _require_files(file_paths, source_dir):
    if not file_paths:
        raise FileNotFoundError(
            f"No CSV files found in {source_dir}/ — download the MoneyPuck export(s) "
            f"and place them there before running this script."
        )


def iter_csv_chunks(file_paths, source_dir, usecols=None, chunksize=100_000, low_memory=False):
    """Yield chunks across one or more CSVs as if they were a single file.

    Each file's usecols is narrowed to whatever columns it actually has,
    so files from different MoneyPuck export vintages with slightly
    different column sets can still be read together.
    """
    _require_files(file_paths, source_dir)
    for path in file_paths:
        if usecols is None:
            file_usecols = None
        else:
            header_cols = pd.read_csv(path, nrows=0).columns
            file_usecols = [c for c in usecols if c in header_cols]
        for chunk in pd.read_csv(
            path, low_memory=low_memory, chunksize=chunksize, usecols=file_usecols
        ):
            yield chunk
