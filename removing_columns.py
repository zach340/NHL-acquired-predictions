
import argparse
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
 
ALL_KEEP = (
    CONTEXT_COLS
    + PRODUCTION_COLS
    + XG_COLS
    + DANGER_COLS


)
 
 

 