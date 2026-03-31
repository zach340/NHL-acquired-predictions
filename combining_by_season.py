import argparse
import os
from typing import List

import pandas as pd


# Columns that identify a unique player-season with a specific team
GROUP_KEYS: List[str] = [
	"player_id",
	"player_name",
	"player_team",
	"season",
	"position",
]

# Game-level columns we do not need in the season aggregate
DROP_COLUMNS: List[str] = [
	"game_id",
	"opposing_team",
	"home_or_away",
	"game_date",
	"situation",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Aggregate per-player stats by season while the team remains the same "
			"(sums numeric columns, counts games)."
		)
	)
	parser.add_argument(
		"--input",
		default="offensive_performance_by_game.csv",
		help="Input CSV with per-game stats",
	)
	parser.add_argument(
		"--output",
		default="offensive_performance_by_season.csv",
		help="Output CSV with per-season aggregates",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=200_000,
		help="Rows per chunk when reading input",
	)
	return parser.parse_args()


def aggregate_file(input_path: str, output_path: str, chunk_size: int) -> None:
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input CSV not found: {input_path}")

	tmp_path = f"{output_path}.tmp"
	first_write = True
	total_rows = 0
	total_groups = 0

	# Prepare writers outside the loop; we will write incrementally per chunk
	for chunk in pd.read_csv(input_path, low_memory=False, chunksize=chunk_size):
		total_rows += len(chunk)

		# Drop columns that do not make sense in a season aggregate
		for col in DROP_COLUMNS:
			if col in chunk.columns:
				chunk = chunk.drop(columns=col)

		missing_keys = [col for col in GROUP_KEYS if col not in chunk.columns]
		if missing_keys:
			raise ValueError(f"Missing required key columns: {', '.join(missing_keys)}")

		# Add a games_played counter so we retain game counts
		chunk["games_played"] = 1

		# Determine numeric columns to sum (exclude grouping keys)
		numeric_cols = [
			col
			for col in chunk.columns
			if col not in GROUP_KEYS and pd.api.types.is_numeric_dtype(chunk[col])
		]

		# Coerce non-numeric columns that should be numeric (in case of mixed dtypes)
		for col in chunk.columns:
			if col in GROUP_KEYS or col in numeric_cols:
				continue
			coerced = pd.to_numeric(chunk[col], errors="coerce")
			if coerced.notna().any():
				chunk[col] = coerced.fillna(0)
				numeric_cols.append(col)

		# Safety: ensure all numeric columns have no NaNs
		chunk[numeric_cols] = chunk[numeric_cols].fillna(0)

		agg_dict = {col: "sum" for col in numeric_cols}
		grouped = chunk.groupby(GROUP_KEYS, dropna=False).agg(agg_dict).reset_index()
		total_groups += len(grouped)

		grouped.to_csv(
			tmp_path,
			mode="w" if first_write else "a",
			header=first_write,
			index=False,
		)
		first_write = False

	if first_write:
		# No data was processed; write an empty file with headers.
		pd.DataFrame(columns=GROUP_KEYS + ["games_played"]).to_csv(tmp_path, index=False)

	# Second pass: load the concatenated per-chunk aggregates and reduce again
	final_df = pd.read_csv(tmp_path)
	numeric_cols = [col for col in final_df.columns if col not in GROUP_KEYS]
	final_df = (
		final_df.groupby(GROUP_KEYS, dropna=False)[numeric_cols]
		.sum()
		.reset_index()
	)

	if os.path.exists(output_path):
		os.remove(output_path)
	os.replace(tmp_path, output_path)

	print(f"Rows read: {total_rows:,}")
	print(f"Intermediate groups written: {total_groups:,}")
	print(f"Final groups: {len(final_df):,}")
	print(f"Saved to {output_path}")


def main() -> None:
	args = parse_args()
	aggregate_file(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
	main()
