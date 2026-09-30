"""
contract.py
===========
Multi-year contract projections: age players forward along empirical NHL age
curves, re-predict each year, and assess CBA limits and risk.
"""

from . import defense, offense
from .config import DEF_PLAYER_FEATURES, DEF_TEAM_FEATURES, PLAYER_FEATURES, TEAM_FEATURES
from .data_io import load_defensive_offensive_stats
from .grading import grade_defensive_defenseman, grade_offensive_defenseman

# Annual production multiplier by age bracket (inclusive)
OFF_AGE_CURVE = {
    (18, 22): 1.05,   # rapid development
    (23, 25): 1.02,   # continued growth
    (26, 28): 1.00,   # peak years
    (29, 30): 0.97,   # early decline
    (31, 32): 0.94,   # moderate decline
    (33, 34): 0.90,   # significant decline
    (35, 99): 0.85,   # steep decline
}

DEF_AGE_CURVE = {
    (18, 22): 1.04,
    (23, 25): 1.02,
    (26, 29): 1.00,   # defensemen peak slightly later
    (30, 31): 0.97,
    (32, 33): 0.94,
    (34, 35): 0.90,
    (36, 99): 0.85,
}

# Per-metric D-man aging: [(min_age, multiplier), ...] checked oldest first, then default.
# Physical play declines fastest; hockey sense and positioning hold longer.
_DEF_METRIC_AGING = {
    "ind_hits_per60":             ([(34, 0.91), (31, 0.95), (28, 0.98)], 1.00),
    "ind_takeaways_per60":        ([(35, 0.94), (32, 0.97), (29, 0.99)], 1.00),
    "hd_shots_against_per60_5v5": ([(36, 0.95), (33, 0.98)],             1.00),
    "on_ice_corsi_pct":           ([(36, 0.95), (33, 0.98)],             1.00),
    # Decision-making improves until ~30 then slowly declines
    "ind_giveaways_per60":        ([(34, 1.02), (30, 1.00)],             0.99),
}
_DEF_SKILL_COLS = ["ind_hits_per60", "ind_takeaways_per60", "ind_giveaways_per60",
                   "shots_blocked_by_player_per60", "hd_shots_against_per60_5v5", "on_ice_corsi_pct"]
_FWD_SKILL_COLS = ["finishing_skill", "finishing_skill_adj", "ind_shot_attempts_per60",
                   "ind_high_danger_shots_per60", "ind_medium_danger_shots_per60",
                   "ind_points_per60", "ind_goals_per60"]

# Confidence decay per contract year
CONFIDENCE_DECAY = {1: 1.0, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.40, 6: 0.30, 7: 0.25}

# Recompute these age interactions after aging: column → source feature
_AGE_INTERACTIONS = {
    "age_x_shot_attempts": "ind_shot_attempts_per60",
    "age_x_finishing":     "finishing_skill_adj",
    "age_x_hd_share":      "hd_shot_share",
    "age_x_hits":          "ind_hits_per60",
    "age_x_takeaways":     "ind_takeaways_per60",
    "age_x_pk":            "pk_ice_pct",
}


def get_age_adjusted_confidence(year, age):
    """Time decay, steepened for older players (late-career volatility)."""
    base = CONFIDENCE_DECAY.get(year, 0.25)
    rate = 0.82 if age >= 34 else 0.90 if age >= 31 else 0.95 if age >= 28 else 1.00
    return round(min(base * rate ** (year - 1), 1.0), 3)


def get_age_multiplier(age, is_defenseman=False):
    for (lo, hi), mult in (DEF_AGE_CURVE if is_defenseman else OFF_AGE_CURVE).items():
        if lo <= age <= hi:
            return mult
    return 0.85


def _def_metric_multiplier(col, age):
    if col not in _DEF_METRIC_AGING:
        return get_age_multiplier(age, is_defenseman=True)
    steps, default = _DEF_METRIC_AGING[col]
    for min_age, mult in steps:
        if age >= min_age:
            return mult
    return default


def _pim_multiplier(age):
    """PIM rises slightly into the early 30s, then falls as ice time drops."""
    return 1.01 if age <= 30 else 1.02 if age <= 33 else 1.00 if age <= 35 else 0.97


def _compound(fn, start_age, years):
    mult = 1.0
    for y in range(years):
        mult *= fn(start_age + y)
    return mult


def age_profile(profile, years_ahead, is_defenseman=False):
    """Copy of a profile aged `years_ahead` seasons along the age curves."""
    p = profile.copy()
    current_age = float(p.get("age", 28))
    new_age = current_age + years_ahead
    p["age"], p["age_sq"] = new_age, new_age ** 2

    def scale(cols, mult):
        for col in cols:
            if col in p.index:
                p[col] = float(p[col]) * mult

    if is_defenseman:
        for col in _DEF_SKILL_COLS:
            scale([col], _compound(lambda a: _def_metric_multiplier(col, a), current_age, years_ahead))
        scale(["ind_penalty_minutes_pg", "pim_pg", "prev_season_pim_pg", "recent_3yr_mean_pim_pg"],
              _compound(_pim_multiplier, current_age, years_ahead))
        scale(["prev_season_hits_pg", "recent_3yr_mean_hits_pg", "prev_season_takeaways_pg",
               "recent_3yr_mean_takeaways_pg", "prev_season_xga_pg", "recent_3yr_mean_xga_pg"],
              get_age_multiplier(current_age, is_defenseman=True) ** years_ahead)
    else:
        mult = _compound(lambda a: get_age_multiplier(a), current_age, years_ahead)
        scale(_FWD_SKILL_COLS, mult)
        scale(["prev_season_points_pg", "recent_3yr_mean_points_pg", "prev_season_goals_pg",
               "recent_3yr_mean_goals_pg", "career_prev_mean_points_pg", "career_prev_mean_goals_pg"], mult)

    for col, default in (("career_seasons_prior", 5), ("career_year", 5)):
        if col in p.index:
            p[col] = float(p.get(col, default)) + years_ahead

    for col, src in _AGE_INTERACTIONS.items():
        if col in p.index:
            p[col] = new_age * float(p.get(src, 0))

    # Ensure every non-age feature the model expects exists
    for col in (DEF_PLAYER_FEATURES + DEF_TEAM_FEATURES if is_defenseman else PLAYER_FEATURES + TEAM_FEATURES):
        if col not in p.index:
            p[col] = 0.0
    return p


def build_contract_projection(pred, dpred, fwd, dfn, team, n_years, curr_age=None):
    """
    Year-by-year projection for `pred` (forward) or `dpred` (defenseman, when
    pred["position"] == "D") signing with `team`. `fwd` / `dfn` are the model
    bundles. Returns (rows, error).
    """
    is_d = pred is not None and pred.get("position") == "D"

    if is_d:
        if dpred is None:
            return None, "Defensive model not loaded."
        pid, profile = dpred["pid"], dpred["profile"]
        age = float(curr_age) if curr_age else float(profile.get("age", 28))
        ctx = defense.get_latest_team_contexts(dfn.df, dfn.team_ctx)
        team_row = ctx[ctx["player_team"] == team]
        if team_row.empty:
            return None, f"No defensive team context for {team}."
        off_stats, off_df, _ = load_defensive_offensive_stats()
    else:
        if pred is None or pred.get("fit_results") is None:
            return None, "Offensive model not loaded or player not found."
        pid = pred["pid"]
        profile = fwd.profiles[pid][0]
        age = float(curr_age) if curr_age else float(pred.get("age") or profile.get("age", 28) or 28)
        ctx = offense.get_latest_team_contexts(fwd.df, fwd.team_ctx)
        team_row = ctx[(ctx["player_team"] == team) & (ctx["position"] == profile["position"])]
        if team_row.empty:
            return None, f"No offensive team context for {team}."
        league_env = offense.get_latest_league_env(fwd.df)
    team_row = team_row.iloc[0]

    rows = []
    for year in range(1, n_years + 1):
        aged  = age_profile(profile, year - 1, is_defenseman=is_d)
        age_y = age + year - 1
        row   = {"year": year, "age": round(age_y, 0), "confidence": get_age_adjusted_confidence(year, age_y)}

        if is_d:
            preds = defense.predict_for_team(aged, team_row, dfn.fit_models, dfn.fit_feature_names)
            _, def_pct, _, _ = grade_defensive_defenseman(preds, season_def_df=dfn.df)
            _, off_pct, _, _ = grade_offensive_defenseman(off_stats.get(pid, {}), season_off_df=off_df)
            row.update({
                "hits_pg":          round(max(preds.get("ind_hits_pg", 0), 0), 2),
                "takeaways_pg":     round(max(preds.get("ind_takeaways_pg", 0), 0), 3),
                "goals_against_pg": round(max(preds.get("xg_against_per60_5v5", 0), 0), 3),
                "pim_pg":           round(max(preds.get("pim_pg", 0), 0), 3),
                "def_score":        round(def_pct, 1),
                "off_score":        round(off_pct, 1),
            })
        else:
            aged_row = offense.with_context(aged, team_row, league_env)
            if "position" not in aged_row.index:
                aged_row["position"] = profile.get("position", "C")
            preds = offense.predict_row(aged_row, fwd.fit_models, fwd.has_age)
            pts = preds["points_per_game"]
            row.update({
                "points_pg": round(pts, 3),
                "goals_pg":  round(preds["goals_per_game"], 3),
                "gs_pg":     round(preds["game_score_per_game"], 3),
                "pts_pct":   round(float((fwd.df["points_per_game"].dropna() < pts).mean() * 100), 1),
            })
        rows.append(row)
    return rows, None


def get_cba_limits(current_age, actual_team, signing_team):
    """
    CBA contract-length limits: 7 years re-signing with the same team, 6 with a
    new team, plus an age-curve recommendation. The 35+ rule applies only to
    contracts signed at age 35 or older (its cap hit stays on the books even if
    the player retires).
    """
    is_same_team = actual_team == signing_team
    max_years    = 7 if is_same_team else 6
    if current_age >= 35:
        recommended = 1
    elif current_age >= 33:
        recommended = 2
    elif current_age >= 31:
        recommended = 3
    elif current_age >= 28:
        recommended = 4
    else:
        recommended = max_years
    return {
        "max_years":     max_years,
        "is_same_team":  is_same_team,
        "recommended":   min(recommended, max_years),
        "is_35_signing": current_age >= 35,
    }


def contract_risk_rating(rows, is_d):
    """(label, colour, explanation) from signing age and projected percentile decline."""
    if not rows:
        return "Unknown", "#888888", ""

    first, last, n = rows[0], rows[-1], len(rows)
    age = first["age"]
    if is_d:
        decline = first.get("def_score", 50) - last.get("def_score", 50)
    else:
        decline = first.get("pts_pct", first.get("points_pg", 0)) - last.get("pts_pct", last.get("points_pg", 0))

    if age >= 35:
        return "Very High Risk", "#c8102e", f"Age {age:.0f} at signing — steep decline likely. 35+ rule applies."
    if age >= 33:
        return "High Risk", "#e8622a", f"Age {age:.0f} at signing — physical decline likely in later years of contract."
    if age >= 32 and decline > 8:
        return "High Risk", "#e8622a", f"Age {age:.0f} — projected {decline:.0f} percentile point decline over {n} years."
    if age >= 32:
        return "Moderate Risk", "#FFD700", f"Age {age:.0f} — entering decline window, monitor later contract years closely."
    if age >= 30 and decline > 8:
        return "Moderate Risk", "#FFD700", f"Age {age:.0f} — some decline expected but manageable."
    if decline < -5 and age < 30:
        return "Low Risk", "#57a85a", f"Age {age:.0f} — ascending player, percentile rank expected to improve."
    return "Low Risk", "#57a85a", f"Age {age:.0f} — stable production expected through contract."
