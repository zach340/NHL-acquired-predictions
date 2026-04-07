import pandas as pd
df = pd.read_csv("2008_to_2024_cleaned.csv", nrows=5000)
print(list(df.columns))
print("\nSituations available:")
print(df["situation"].value_counts())
print("\nSample 5on4 row:")
pp = df[df["situation"] == "5on4"]
if not pp.empty:
    print(pp.iloc[0][["player_name", "season", "situation", 
                       "ice_time", "ind_goals", "ind_primary_assists"]].to_dict())