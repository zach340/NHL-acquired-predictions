"""
grading.py
==========
Letter grades and defenseman archetypes from league percentile ranks.

Grade cutoffs (percentile):  A ≥ 90 · B+ ≥ 75 · B ≥ 50 · C+ ≥ 35 · C ≥ 20 · D below.
"""

import numpy as np

_GRADE_CUTOFFS = [(90, "A"), (75, "B+"), (50, "B"), (35, "C+"), (20, "C")]


def score_to_grade(pct):
    for cutoff, grade in _GRADE_CUTOFFS:
        if pct >= cutoff:
            return grade
    return "D"


def percentile_rank(val, values, lower_is_better=False):
    """Share (0-100) of `values` that `val` beats."""
    values = values.dropna()
    return float(((values > val) if lower_is_better else (values < val)).mean() * 100)


def _interp_percentile(val, breakpoints):
    """Piecewise-linear percentile from (value, percentile) pairs, clamped at the ends."""
    pts = sorted(breakpoints)
    return float(np.interp(val, [v for v, _ in pts], [p for _, p in pts]))


# Fallback distributions (2024 D-men) used only when no league frame is available
_OFF_BREAKPOINTS = {
    "points_pg": [(0.0, 0), (0.10, 20), (0.17, 35), (0.24, 50), (0.36, 75), (0.56, 90), (1.15, 99)],
    "goals_pg":  [(0.0, 0), (0.03, 20), (0.06, 35), (0.08, 50), (0.13, 75), (0.20, 90), (0.40, 99)],
}
_DEF_BREAKPOINTS = {
    "ind_hits_pg":          [(0.3, 10), (0.7, 25), (1.1, 50), (1.8, 75), (2.5, 90), (4.0, 99)],
    "ind_takeaways_pg":     [(0.10, 10), (0.18, 25), (0.28, 50), (0.40, 75), (0.52, 90), (0.70, 99)],
    "xg_against_per60_5v5": [(1.8, 99), (2.2, 90), (2.6, 75), (3.0, 50), (3.3, 25), (3.8, 10)],
    "pim_pg":               [(0.1, 99), (0.2, 90), (0.35, 75), (0.5, 50), (0.7, 25), (1.2, 10)],
}

_OFF_DESC = {
    "A":  "Elite offensive D-man — top 10% in the league",
    "B+": "Above average offensively — top 25%",
    "B":  "Solid offensive production — above median",
    "C+": "Average offensive output for a defenseman",
    "C":  "Below average offensively — bottom third",
    "D":  "Minimal offensive contribution — bottom 20%",
}
_DEF_DESC = {
    "A":  "Elite defensive D-man — top 10% in the league",
    "B+": "Above average defensively — top 25%",
    "B":  "Solid defensive contributor — above median",
    "C+": "Average defensive production",
    "C":  "Below average defensively — bottom third",
    "D":  "Minimal defensive contribution — bottom 20%",
}


def grade_offensive_defenseman(off_stats, season_off_df=None):
    """
    Offensive grade for a D-man: 70% Points/GP + 30% Goals/GP percentile among
    all D-men. Returns (grade, composite_pct, description, breakdown).
    """
    pts   = off_stats.get("points_pg", 0)
    goals = off_stats.get("goals_pg",  0)

    if season_off_df is not None and len(season_off_df) > 10:
        def pct(val, col):
            return percentile_rank(val, season_off_df[col]) if col in season_off_df.columns else 50.0
        pts_pct, goals_pct = pct(pts, "points_per_game"), pct(goals, "goals_per_game")
    else:
        pts_pct   = _interp_percentile(pts,   _OFF_BREAKPOINTS["points_pg"])
        goals_pct = _interp_percentile(goals, _OFF_BREAKPOINTS["goals_pg"])

    composite = pts_pct * 0.70 + goals_pct * 0.30
    grade = score_to_grade(composite)
    breakdown = {
        "Points/GP": (round(pts,   3), round(pts_pct,   1)),
        "Goals/GP":  (round(goals, 3), round(goals_pct, 1)),
    }
    return grade, round(composite, 1), _OFF_DESC[grade], breakdown


def grade_defensive_defenseman(def_stats, season_def_df=None):
    """
    Defensive grade for a D-man from percentile ranks among all D-men:
    xGA/60 (5v5, lower better) 30% · Takeaways/GP 25% · Hits/GP 25% · PIM/GP (lower better) 20%.
    Returns (grade, composite_pct, description, breakdown).
    """
    hits = def_stats.get("ind_hits_pg",          0)
    tka  = def_stats.get("ind_takeaways_pg",     0)
    xga  = def_stats.get("xg_against_per60_5v5", 2.5)
    pim  = def_stats.get("pim_pg",               0)

    def pct(val, col, lower_is_better=False):
        if season_def_df is not None and col in season_def_df.columns and season_def_df[col].notna().sum() > 10:
            return percentile_rank(val, season_def_df[col], lower_is_better)
        return _interp_percentile(val, _DEF_BREAKPOINTS[col])

    hits_pct = pct(hits, "ind_hits_pg")
    tka_pct  = pct(tka,  "ind_takeaways_pg")
    xga_pct  = pct(xga,  "xg_against_per60_5v5", lower_is_better=True)
    pim_pct  = pct(pim,  "pim_pg",               lower_is_better=True)

    composite = xga_pct * 0.30 + tka_pct * 0.25 + hits_pct * 0.25 + pim_pct * 0.20
    grade = score_to_grade(composite)
    breakdown = {
        "xGA/60 (5v5)": (round(xga,  2), round(xga_pct,  1)),
        "Takeaways/GP": (round(tka,  3), round(tka_pct,  1)),
        "Hits/GP":      (round(hits, 2), round(hits_pct, 1)),
        "PIM/GP":       (round(pim,  2), round(pim_pct,  1)),
    }
    return grade, round(composite, 1), _DEF_DESC[grade], breakdown


def classify_defenseman_type(scores, def_score=None, off_score=None,
                             season_def_df=None, season_off_df=None):
    """
    Offensive D / Defensive D / Two-Way D from the gap between offensive and
    defensive percentile scores (|gap| ≤ 30 → Two-Way D).
    Scores are computed from `scores` if not passed in.
    """
    if def_score is None:
        _, def_score, _, _ = grade_defensive_defenseman(scores, season_def_df)
    if off_score is None:
        off_stats = {
            "points_pg": scores.get("points_per_game", scores.get("points_pg", 0)),
            "goals_pg":  scores.get("goals_per_game",  scores.get("goals_pg",  0)),
        }
        _, off_score, _, _ = grade_offensive_defenseman(off_stats, season_off_df)

    gap = off_score - def_score
    if gap > 30:
        return "Offensive D", (
            "Offensive defenceman — exceptional skating and puck-handling, "
            "creates scoring opportunities but may be vulnerable defensively."
        )
    if gap < -30:
        return "Defensive D", (
            "Defensive defenceman — physical, blocks shots, clears the zone. "
            "Strong defensively but may struggle generating offense."
        )
    return "Two-Way D", (
        "Two-way defenceman — well-rounded with contributions at both ends. "
        "Versatile and reliable in any situation."
    )


def combined_score(def_pct, off_pct, d_type):
    """Blend defensive and offensive percentiles, weighted by archetype."""
    w_def, w_off = {"Two-Way D": (0.5, 0.5), "Offensive D": (0.3, 0.7)}.get(d_type, (0.8, 0.2))
    return round(def_pct * w_def + off_pct * w_off, 1)
