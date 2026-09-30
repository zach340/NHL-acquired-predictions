import pandas as pd

input_file = "offensive_performance_by_season_per60.csv"
output_file = "offensive_performance_by_season_per60_renamed.csv"

df = pd.read_csv(input_file)

df["player_team"] = df["player_team"].replace({"ATL": "WPG", "ARI": "UTA"})

df.to_csv(output_file, index=False)

print(f"Done! Saved to {output_file}")