"""
config.py
=========
Project-wide constants: file locations, teams, model settings and the
feature lists for both the forward (offensive) and defenseman models.
"""

# ── Files ──────────────────────────────────────────────────────────────────────

DATA_FILE      = "season_dataset.csv"      # forwards + D-men, season level
AGES_FILE      = "player_ages.csv"
PP_FILE        = "pp_features.csv"
LINEMATE_FILE  = "linemate_features.csv"
DEF_FILE       = "defensive_dataset.csv"
NAMES_FILE     = "player_names.csv"        # persistent NHL API name/headshot cache

CACHE_FILE     = "trained_models_forwards_v7.joblib"
DEF_CACHE_FILE = "defensive_models_v3.joblib"

# Shift pair data is cached on disk so it survives app restarts.
# Each file: shifts_cache/{TEAM}_{N_GAMES}.json — refreshed when > TTL hours old.
SHIFTS_CACHE_DIR   = "shifts_cache"
SHIFTS_CACHE_TTL_H = 6

# ── Season ─────────────────────────────────────────────────────────────────────
# Seasons are labelled by their START year everywhere in the data (MoneyPuck
# convention): 2024 = the 2024-25 season. The NHL API uses "20242025" ids.

def season_id(start_year):
    """2024 -> '20242025' (NHL API season id)."""
    return f"{int(start_year)}{int(start_year) + 1}"


def season_label(start_year):
    """2024 -> '2024-25'."""
    return f"{int(start_year)}-{str(int(start_year) + 1)[-2:]}"


def _current_season_start(today=None):
    """New-season rosters appear after the draft / free agency, so roll over in July."""
    from datetime import date
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


CURRENT_SEASON_START = _current_season_start()
CURRENT_SEASON       = season_id(CURRENT_SEASON_START)   # rosters, shift charts, bios
FIRST_SEASON_START   = 2008                              # first MoneyPuck season

# ── Training ───────────────────────────────────────────────────────────────────

MIN_GP         = 20
MIN_ICE        = 300
CV_FOLDS       = 3
ELITE_QUANTILE = 0.90

# ── Teams ──────────────────────────────────────────────────────────────────────

NHL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

# Primary brand colours for each NHL team — used to highlight the actual team in charts.
# Secondary colour used as the bar outline; primary as the fill / vline.
TEAM_COLORS = {
    "ANA": {"primary": "#F47A38", "secondary": "#B9975B"},
    "BOS": {"primary": "#FCB514", "secondary": "#000000"},
    "BUF": {"primary": "#003087", "secondary": "#FFB81C"},
    "CAR": {"primary": "#CC0000", "secondary": "#000000"},
    "CBJ": {"primary": "#CE1126", "secondary": "#002654"},
    "CGY": {"primary": "#C8102E", "secondary": "#F1BE48"},
    "CHI": {"primary": "#CF0A2C", "secondary": "#FF671B"},
    "COL": {"primary": "#6F263D", "secondary": "#236192"},
    "DAL": {"primary": "#006847", "secondary": "#8F8F8C"},
    "DET": {"primary": "#CE1126", "secondary": "#FFFFFF"},
    "EDM": {"primary": "#FC4C02", "secondary": "#041E42"},
    "FLA": {"primary": "#C8102E", "secondary": "#041E42"},
    "LAK": {"primary": "#A2AAAD", "secondary": "#111111"},
    "MIN": {"primary": "#154734", "secondary": "#A6192E"},
    "MTL": {"primary": "#AF1E2D", "secondary": "#192168"},
    "NJD": {"primary": "#CE1126", "secondary": "#003087"},
    "NSH": {"primary": "#FFB81C", "secondary": "#041E42"},
    "NYI": {"primary": "#003087", "secondary": "#FC4C02"},
    "NYR": {"primary": "#0038A8", "secondary": "#CE1126"},
    "OTT": {"primary": "#C52032", "secondary": "#C69214"},
    "PHI": {"primary": "#F74902", "secondary": "#000000"},
    "PIT": {"primary": "#FCB514", "secondary": "#000000"},
    "SEA": {"primary": "#99D9D9", "secondary": "#001628"},
    "SJS": {"primary": "#006D75", "secondary": "#EA7200"},
    "STL": {"primary": "#002F87", "secondary": "#FCB514"},
    "TBL": {"primary": "#002868", "secondary": "#FFFFFF"},
    "TOR": {"primary": "#003E7E", "secondary": "#FFFFFF"},
    "UTA": {"primary": "#6CAEDF", "secondary": "#010101"},
    "VAN": {"primary": "#00843D", "secondary": "#00205B"},
    "VGK": {"primary": "#B4975A", "secondary": "#333F42"},
    "WPG": {"primary": "#004C97", "secondary": "#041E42"},
    "WSH": {"primary": "#C8102E", "secondary": "#041E42"},
}


def get_team_color(team: str, key: str = "primary") -> str:
    """Return a team's brand colour, defaulting to a neutral red if not found."""
    return TEAM_COLORS.get(team, {}).get(key, "#c8102e")


# ── Forward (offensive) model ─────────────────────────────────────────────────

FORWARD_POSITIONS = ["C", "L", "R"]

TARGETS = ["game_score_per_game", "points_per_game", "goals_per_game"]

TARGET_LABELS = {
    "game_score_per_game": "Game Score / Game",
    "points_per_game":     "Points / Game",
    "goals_per_game":      "Goals / Game",
}

# Team Fit (same-season) baselines: leakage-safe prior seasons only. The model
# predicts the residual on top of the first non-null candidate.
BASELINE_FEATURES = {
    "game_score_per_game": [
        "recent_3yr_mean_gamescore_pg",
        "career_prev_mean_gamescore_pg",
        "prev_season_gamescore_pg",
    ],
    "points_per_game": [
        "recent_3yr_mean_points_pg",
        "career_prev_mean_points_pg",
        "prev_season_points_pg",
        "league_avg_points_pg",
    ],
    "goals_per_game": [
        "recent_3yr_mean_goals_pg",
        "career_prev_mean_goals_pg",
        "prev_season_goals_pg",
        "league_avg_goals_pg",
    ],
}

PLAYER_FEATURES = [
    "finishing_skill",
    "finishing_skill_adj",
    "flurry_reliance",
    "hd_shot_share",
    "hd_finishing",
    "hd_xg_outperformance",
    "xg_per_attempt",
    "on_target_rate",
    "primary_assist_share",
    "primary_vs_secondary",
    "ind_shot_attempts_per60",
    "ind_high_danger_shots_per60",
    "ind_medium_danger_shots_per60",
    "ind_low_danger_shots_per60",
    "shifts_per60",
    # Scoring environment — captures league-wide trends by season
    "league_avg_points_pg",
    "league_avg_goals_pg",
    # Career peak features — player ceiling signal
    "career_peak_points_pg",
    "career_peak_goals_pg",
    "pct_of_peak_points",
    "pct_of_peak_goals",
    # Powerplay & zone start features — key deployment signals
    "pp_icetime_pct",
    "pp_points_per60",
    "pp_goals_per60",
    "pp_xg_per60",
    "pp_points_share",
    "o_zone_start_pct",
    "zone_start_diff",
    # Career history features — prior seasons only
    "career_seasons_prior",
    "prev_season_points_pg",
    "prev_season_goals_pg",
    "prev_season_gamescore_pg",
    "career_prev_mean_points_pg",
    "career_prev_mean_goals_pg",
    "career_prev_mean_gamescore_pg",
    "career_prev_peak_points_pg",
    "career_prev_peak_goals_pg",
    "recent_3yr_mean_points_pg",
    "recent_3yr_mean_goals_pg",
    "recent_3yr_mean_gamescore_pg",
    # Explicit trend features — slope of prior seasons only
    "recent_3yr_points_slope",
    "recent_3yr_goals_slope",
    "recent_3yr_gamescore_slope",
    "career_points_slope",
    "career_goals_slope",
    "career_gamescore_slope",
]

# Only used when age data is available
AGE_FEATURES = ["age", "age_sq", "age_x_shot_attempts", "age_x_finishing", "age_x_hd_share"]

TEAM_FEATURES = [
    "team_median_toi_pg",
    "team_avg_hd_share",
    "team_avg_adj_xg_per60",
    "team_adj_ratio",
    "team_avg_primary_rate",
    "team_avg_on_target",
    # Team-level line quality — swapped per team at prediction time
    "team_avg_line_adj_xg_per60",
    "team_avg_line_xg_pct",
    "team_avg_line_hd_xg_per60",
    "team_avg_line_corsi_pct",
]

# Next-season model also uses trajectory (YoY delta) features and the
# 3-2-1 weighted recent means (which include the current season)
TRAJECTORY_FEATURES = [
    "wavg_points_pg",
    "wavg_goals_pg",
    "wavg_gamescore_pg",
    "wavg_toi",
    "wavg_p60",
    "wavg_g60",
    "yoy_points_delta",
    "yoy_goals_delta",
    "yoy_gamescore_delta",
    "games_played_pct",
    "career_year",
]

# Non-linear career curve features — only used when age data is available
NONLINEAR_FEATURES = [
    "curve_accel_points",
    "curve_accel_goals",
    "curve_accel_gs",
    "curve_local_deriv_points",
    "curve_local_deriv_goals",
    "curve_local_deriv_gs",
    "seasons_from_est_peak_points",
    "seasons_from_est_peak_goals",
    "seasons_from_est_peak_gs",
    "pct_peak_points_slope",
    "pct_peak_goals_slope",
    "age_x_3yr_pts_slope",
    "age_x_3yr_goals_slope",
    "age_x_career_pts_slope",
]

POSITION_DUMMIES = ["pos_C", "pos_D", "pos_L", "pos_R"]

# Features computed from the SAME season's goals / assists / points. The Team
# Fit model predicts that same season's production, so these would hand it the
# answer (e.g. pct_of_peak × peak = points/GP). They are only used by the Next
# Season model, where they are genuinely prior information.
OUTCOME_FEATURES = [
    "finishing_skill", "finishing_skill_adj", "hd_finishing", "hd_xg_outperformance",
    "primary_assist_share", "primary_vs_secondary",
    "pp_points_per60", "pp_goals_per60", "pp_points_share",
    "career_peak_points_pg", "career_peak_goals_pg", "pct_of_peak_points", "pct_of_peak_goals",
    "age_x_finishing",
]

# ── Defenseman model ──────────────────────────────────────────────────────────

DEF_TARGETS = [
    "ind_hits_pg",
    "ind_takeaways_pg",
    "xg_against_per60_5v5",
    "pim_pg",
]

DEF_TARGET_LABELS = {
    "ind_hits_pg":          "Hits / Game",
    "ind_takeaways_pg":     "Takeaways / Game",
    "xg_against_per60_5v5": "xGA Against / 60 (5v5)",
    "pim_pg":               "PIM / Game",
}

DEF_LOWER_IS_BETTER = {"xg_against_per60_5v5", "pim_pg"}

# Composite defensive score weights
DEF_SCORE_WEIGHTS = {
    "ind_hits_pg":          0.25,
    "ind_takeaways_pg":     0.25,
    "xg_against_per60_5v5": 0.30,
    "pim_pg":               0.20,
}

# Forward Next Season baselines: 3-2-1 weighted mean of this season and the
# two before it; Points/Goals as ice-time-weighted TOI/GP × rate/60.
NEXT_BASELINE_RATES    = {"points_per_game": "wavg_p60", "goals_per_game": "wavg_g60"}
NEXT_BASELINE_FEATURES = {"game_score_per_game": "wavg_gamescore_pg"}

# Defensemen use prior seasons for both models (a 3-2-1 baseline tested worse
# on the 2025-26 holdout; the 3-2-1 means help as Next Season features instead).
DEF_BASELINE_FEATURES = {
    "ind_hits_pg":          ["prev_season_hits_pg",      "recent_3yr_mean_hits_pg",      "career_prev_mean_hits_pg"],
    "ind_takeaways_pg":     ["prev_season_takeaways_pg", "recent_3yr_mean_takeaways_pg", "career_prev_mean_takeaways_pg"],
    "xg_against_per60_5v5": ["prev_season_xga_pg",       "recent_3yr_mean_xga_pg",       "career_prev_mean_xga_pg"],
    "pim_pg":               ["prev_season_pim_pg",       "recent_3yr_mean_pim_pg",       "career_prev_mean_pim_pg"],
}

DEF_PLAYER_FEATURES = [
    # Physical skill signals
    "ind_hits_per60",
    "ind_takeaways_per60",
    "ind_giveaways_per60",
    "shots_blocked_by_player_per60",
    "ind_penalty_minutes_per60",
    "take_give_ratio",
    "d_zone_start_pct",
    "faceoff_win_pct",
    # On-ice defensive impact (xg_against_per60_5v5 is a target, not a feature)
    "hd_shots_against_per60_5v5",
    "on_ice_corsi_pct",
    "on_ice_fenwick_pct",
    # Career peak signals
    "career_peak_hits_pg",
    "career_peak_takeaways_pg",
    "career_peak_pk_pct",
    "pct_of_peak_hits",
    "pct_of_peak_takeaways",
    # Career history (prior seasons only)
    "prev_season_hits_pg",
    "prev_season_takeaways_pg",
    "prev_season_xga_pg",
    "prev_season_pk_pct",
    "prev_season_pim_pg",
    "recent_3yr_mean_hits_pg",
    "recent_3yr_mean_takeaways_pg",
    "recent_3yr_mean_xga_pg",
    "recent_3yr_mean_pk_pct",
    "recent_3yr_mean_pim_pg",
    "career_prev_mean_hits_pg",
    "career_prev_mean_takeaways_pg",
    "career_prev_mean_xga_pg",
    "career_prev_mean_pk_pct",
    "career_prev_mean_pim_pg",
    "career_seasons_prior",
    # Slopes
    "recent_3yr_hits_slope",
    "recent_3yr_takeaways_slope",
    "recent_3yr_xga_slope",
    "recent_3yr_pk_slope",
    "recent_3yr_pim_slope",
    # League environment
    "league_avg_hits_pg",
    "league_avg_pk_pct",
]

DEF_AGE_FEATURES = ["age", "age_sq", "age_x_hits", "age_x_takeaways", "age_x_pk"]

# Same-season values of (or direct inputs to) the Current Fit targets — see
# OUTCOME_FEATURES. Only the Next Season model uses them.
DEF_OUTCOME_FEATURES = [
    "ind_hits_per60", "ind_takeaways_per60", "ind_penalty_minutes_per60", "take_give_ratio",
    "hd_shots_against_per60_5v5",
    "career_peak_hits_pg", "career_peak_takeaways_pg", "pct_of_peak_hits", "pct_of_peak_takeaways",
    "age_x_hits", "age_x_takeaways",
]

DEF_TEAM_FEATURES = [
    "team_avg_hits_pg",
    "team_avg_takeaways_pg",
    "team_avg_xga_per60",
    "team_avg_pk_pct",
    "team_avg_pim_pg",
    "team_avg_toi_pg",
    "team_avg_d_zone_start_pct",
]

DEF_TRAJECTORY_FEATURES = [
    "wavg_hits_pg",
    "wavg_takeaways_pg",
    "wavg_xga_pg",
    "wavg_pim_pg",
    "yoy_hits_delta",
    "yoy_takeaways_delta",
    "yoy_xga_delta",
    "games_played_pct",
    "career_year",
]

DEF_NONLINEAR_FEATURES = [
    "def_curve_accel_hits",
    "def_curve_accel_takeaways",
    "def_curve_accel_xga",
    "def_curve_local_deriv_hits",
    "def_curve_local_deriv_takeaways",
    "def_curve_local_deriv_xga",
    "def_seasons_from_est_peak_hits",
    "def_seasons_from_est_peak_takeaways",
    "def_pct_peak_hits_slope",
    "def_pct_peak_takeaways_slope",
    "def_age_x_3yr_hits_slope",
    "def_age_x_3yr_takeaways_slope",
    "def_age_x_3yr_xga_slope",
]

# ── Lineup slots ──────────────────────────────────────────────────────────────

SLOT_COLORS = {
    "1st Line": "#FFD700", "1st Pair": "#FFD700",
    "2nd Line": "#4a90d9", "2nd Pair": "#4a90d9",
    "3rd Line": "#57a85a", "3rd Pair": "#57a85a",
    "4th Line": "#888888", "3rd Pair (extra)": "#888888",
    "4th Pair": "#888888",
}

# 3 forwards per line (C, LW, RW) across 4 lines = 12 forwards
FWD_SLOT_MAP = {rank: f"{('1st', '2nd', '3rd', '4th')[(rank - 1) // 3]} Line" for rank in range(1, 13)}
DEF_SLOT_MAP = {rank: f"{('1st', '2nd', '3rd')[(rank - 1) // 2]} Pair" for rank in range(1, 7)}

PAIR_SLOT_NAMES = ["1st Pair", "2nd Pair", "3rd Pair", "4th Pair"]
