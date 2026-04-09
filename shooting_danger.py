import pandas as pd
df = pd.read_csv("2008_to_2024_cleaned.csv", nrows=5)
df = df[df["situation"] == "all"]
print(list(df.columns))
print(df[["player_name", "season", "games_played", "ice_time",
          "ind_goals", "ind_points", "ind_primary_assists",
          "ind_secondary_assists", "ind_shots_on_goal",
          "ind_high_danger_shots", "ind_medium_danger_shots",
          "ind_low_danger_shots", "game_score"]].to_string())