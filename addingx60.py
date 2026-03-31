"""
Convert raw counting stats to per-60 rates.

For each row, we scale numeric, non-percentage columns by 3600 / ice_time,
so that values represent production per 60 minutes. Rows with ice_time <= 0
are left as zeros to avoid inf/NaN explosions. Scaled columns are suffixed
with _per60 so you can distinguish raw counts from rates. The ice_time column
is also converted from seconds to minutes for readability.
"""

import argparse
import os
from typing import List, Set

import pandas as pd


DEFAULT_INPUT = "offensive_performance_by_season.csv"
DEFAULT_OUTPUT = "offensive_performance_by_season_per60.csv"
CHUNK_SIZE = 200_000

# Columns that should never be scaled (identifiers/context).
KEY_COLS: List[str] = [
    "player_id",
    "player_name",
    "player_team",
    "season",
    "position",
    "game_id",
    "opposing_team",
    "home_or_away",
    "game_date",
    "situation",
]

# Extra numeric columns that should remain as-is (not rates).
NUMERIC_EXCLUDE: Set[str] = {
    "ice_time",
    "ice_time_rank",
    "games_played",
    "game_score",
}

# Simple heuristic: treat any column name containing these tokens as a percent/rate
# that should not be rescaled.
PERCENT_TOKENS = ("pct", "percent", "percentage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize counting stats to per-60 rates by dividing by (ice_time / 3600)."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Rows per chunk to process",
    )
    return parser.parse_args()


def choose_rate_columns(df: pd.DataFrame) -> List[str]:
    """Pick numeric columns to scale, skipping identifiers and percent-like fields."""

    numeric_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]

    def is_percent(col: str) -> bool:
        low = col.lower()
        return any(token in low for token in PERCENT_TOKENS)

    exclude: Set[str] = set(KEY_COLS) | NUMERIC_EXCLUDE

    return [
        col
        for col in numeric_cols
        if col not in exclude and not is_percent(col)
    ]


def scale_chunk(chunk: pd.DataFrame, rate_cols: List[str]) -> pd.DataFrame:
    if "ice_time" not in chunk.columns:
        raise ValueError("Column 'ice_time' is required for per-60 scaling")

    # Avoid division by zero; rows with no ice time stay at 0 after scaling.
    factor = chunk["ice_time"].replace({0: pd.NA})
    hours = factor / 3600

    # Scale selected columns. Non-numeric columns in rate_cols are ignored by pandas.
    scaled = chunk.copy()
    scaled[rate_cols] = scaled[rate_cols].div(hours, axis=0)
    scaled[rate_cols] = scaled[rate_cols].fillna(0)

    # Convert ice_time from seconds to minutes for output clarity.
    scaled["ice_time"] = chunk["ice_time"] / 60

    return scaled


def normalize_file(input_path: str, output_path: str, chunk_size: int) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    tmp_path = f"{output_path}.tmp"
    first_chunk = True
    rows_written = 0
    rate_cols: List[str] = []
    rename_map = {}

    for chunk in pd.read_csv(input_path, low_memory=False, chunksize=chunk_size):
        if not rate_cols:
            rate_cols = choose_rate_columns(chunk)
            rename_map = {
                col: col if col.endswith("_per60") else f"{col}_per60"
                for col in rate_cols
            }

        scaled = scale_chunk(chunk, rate_cols)
        scaled = scaled.rename(columns=rename_map)
        rows_written += len(scaled)

        scaled.to_csv(
            tmp_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False

    if first_chunk:
        # No data was processed; write empty file with headers.
        pd.DataFrame(columns=[]).to_csv(tmp_path, index=False)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(tmp_path, output_path)

    print(f"Rows written: {rows_written:,}")
    print(f"Saved to {output_path}")


def main() -> None:
    args = parse_args()
    normalize_file(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()