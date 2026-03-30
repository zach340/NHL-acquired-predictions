import os

import pandas as pd

INPUT_CSV = "2008_to_2024_cleaned.csv"
OUTPUT_CSV = "2008_to_2024_cleaned.csv"
TEMP_OUTPUT_CSV = "2008_to_2024_cleaned.tmp.csv"
CHUNK_SIZE = 100000

rows_before_filter = 0
rows_after_filter = 0
column_count = None
write_header = True

# Read from cleaned CSV, clean/filter chunks, write to temp file.
for chunk in pd.read_csv(INPUT_CSV, low_memory=False, chunksize=CHUNK_SIZE):
    chunk.fillna(0, inplace=True)

    rows_before_filter += len(chunk)
    filtered_chunk = chunk[chunk["situation"].astype(str).str.strip().str.lower() == "all"]
    rows_after_filter += len(filtered_chunk)

    if column_count is None:
        column_count = len(filtered_chunk.columns)

    filtered_chunk.to_csv(
        TEMP_OUTPUT_CSV,
        index=False,
        mode="w" if write_header else "a",
        header=write_header,
    )
    write_header = False

# Atomic-ish replace on Windows: remove target, then move temp into place.
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
os.replace(TEMP_OUTPUT_CSV, OUTPUT_CSV)

print(f"Rows before situation filter: {rows_before_filter:,}")
print(f"Rows after situation filter: {rows_after_filter:,}")
print(f"Columns: {column_count if column_count is not None else 0}")
print(f"Saved to {OUTPUT_CSV}")