
import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

CONTEXT_COLS = [
    "player_id",
    "player_name",
    "game_id",
    "season",
    "player_team",
    "opposing_team",
    "home_or_away",
    "game_date",
    "position",
    "situation",
    "ice_time",
    "shifts",
    "game_score",
    "ice_time_rank",
]

# Key production / shooting
PRODUCTION_COLS = [
    "ind_goals",
    "ind_primary_assists",
    "ind_secondary_assists",
    "ind_points",
    "ind_shots_on_goal",
    "ind_missed_shots",
    "ind_blocked_shot_attempts",
    "ind_shot_attempts",
    "ind_unblocked_shot_attempts",
]

# Expected-goals quality
XG_COLS = [
    "ind_expected_goals",
    "ind_flurry_adj_expected_goals",
    "ind_score_venue_adj_expected_goals",
    "ind_flurry_score_venue_adj_expected_goals",
    "ind_expected_on_goal",
]

# Shot danger breakdown
DANGER_COLS = [
    "ind_low_danger_shots",
    "ind_medium_danger_shots",
    "ind_high_danger_shots",
    "ind_low_danger_goals",
    "ind_medium_danger_goals",
    "ind_high_danger_goals",
    "ind_low_danger_expected_goals",
    "ind_medium_danger_expected_goals",
    "ind_high_danger_expected_goals",
]

# Puck-battle / playmaking / defensive
BATTLE_COLS = [
    "ind_rebounds",
    "ind_rebound_goals",
    "ind_hits",
    "ind_takeaways",
    "ind_giveaways",
    "ind_d_zone_giveaways",
    "ind_faceoffs_won",
    "faceoffs_won",
    "faceoffs_lost",
    "ind_freeze",
    "ind_play_stopped",
    "ind_play_continued_in_zone",
    "ind_play_continued_outside_zone",
]

# Penalties
PENALTY_COLS = [
    "penalties",
    "ind_penalty_minutes",
    "penalty_minutes",
    "penalty_minutes_drawn",
    "penalties_drawn",
]

# Zone-start context (lightweight on-ice context without raw shot counts)
ZONE_COLS = [
    "on_ice_expected_goals_pct",
    "off_ice_expected_goals_pct",
    "on_ice_corsi_pct",
    "on_ice_fenwick_pct",
    "ind_o_zone_shift_starts",
    "ind_d_zone_shift_starts",
    "ind_neutral_zone_shift_starts",
]

ALL_KEEP = CONTEXT_COLS + PRODUCTION_COLS + XG_COLS + DANGER_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice offensive performance columns into a new CSV."
    )
    parser.add_argument(
        "--input",
        default="2008_to_2024_cleaned.csv",
        help="Input CSV to trim (default: 2008_to_2024_cleaned.csv)",
    )
    parser.add_argument(
        "--output",
        default="offensive_performance_by_game.csv",
        help="Output CSV containing only ALL_KEEP columns",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk to process",
    )
    return parser.parse_args()


def slice_columns(input_path: str, output_path: str, chunk_size: int) -> None:
    if not os.path.exists(input_path):
        sys.stderr.write(f"Input CSV not found: {input_path}\n")
        sys.exit(1)

    tmp_path = f"{output_path}.tmp"
    first_chunk = True
    rows_written = 0
    missing_logged = False
    columns_kept_count = 0

    for chunk in pd.read_csv(input_path, low_memory=False, chunksize=chunk_size):
        missing = [col for col in ALL_KEEP if col not in chunk.columns]
        if missing and not missing_logged:
            sys.stderr.write(
                "Warning: missing columns will be skipped: "
                + ", ".join(missing)
                + "\n"
            )
            missing_logged = True

        keep_cols = [col for col in ALL_KEEP if col in chunk.columns]
        columns_kept_count = len(keep_cols)
        trimmed = chunk[keep_cols]
        rows_written += len(trimmed)

        trimmed.to_csv(
            tmp_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False

    if first_chunk:
        # No data read; create an empty file with headers.
        pd.DataFrame(columns=ALL_KEEP).to_csv(tmp_path, index=False)
        columns_kept_count = len(ALL_KEEP)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(tmp_path, output_path)

    print(f"Rows written: {rows_written:,}")
    print(f"Columns kept: {columns_kept_count}")
    print(f"Saved to {output_path}")


def main() -> None:
    args = parse_args()
    slice_columns(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()

 

