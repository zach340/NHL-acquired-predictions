macavoy_rows = df[df["player_name"].str.contains("MacAvoy", case=False)]
print(macavoy_rows[["season", "xg_against_per60_5v5", 
                     "prev_season_xga_pg", "recent_3yr_mean_xga_pg",
                     "team_avg_xga_per60"]].tail(5))