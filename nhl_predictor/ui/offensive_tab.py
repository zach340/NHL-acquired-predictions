"""Offensive tab: Team Fit, Next Season and Roster Insertion for forwards."""

import pandas as pd
import streamlit as st

from .. import charts, nhl_api, offense
from ..config import NHL_TEAMS, season_label
from ..grading import percentile_rank
from ..theme import update_team_colors
from . import state
from .components import (
    csv_download, grade_item, grade_metrics, pct_bar, player_header, rankings_table, slug, traded_banner,
)

RANKING_COLUMNS = {
    "player_team":              "Team",
    "pred_game_score_per_game": "GS/GP",
    "pred_points_per_game":     "Points/GP",
    "pred_goals_per_game":      "Goals/GP",
}

SKILL_PROFILE = [
    ("finishing_skill_adj",  "Finishing Skill (adj xG)"),
    ("hd_shot_share",        "High-Danger Shot Share"),
    ("hd_finishing",         "High-Danger Finishing"),
    ("primary_assist_share", "Primary Assist Share"),
    ("on_target_rate",       "On-Target Rate"),
    ("pp_icetime_pct",       "PP Ice Time %"),
]


def resolve_forward(player_id, bundle, radio_key, container=None):
    """
    predict_player for `player_id`; if they were traded mid-season, ask which
    team is current (banner + radio rendered into `container`). Returns (pred, first).
    """
    first = offense.predict_player(player_id, bundle)
    if first is None or not first["traded_teams"]:
        return first, first
    banner_team = st.session_state.get(radio_key, first["traded_teams"][-1])
    with (container.container() if container is not None else st.container()):
        traded_banner(banner_team, f"🔁 <strong>{first['matched']}</strong> was traded during "
                                   f"{season_label(first['seasons'][0])}. Select their current team:")
        current = st.radio("Current team", options=first["traded_teams"], horizontal=True, key=radio_key)
    return offense.predict_player(player_id, bundle, override_team=current), first


def render(fwd, player_ids, fmt):
    st.caption("Search for a forward to see offensive predictions.")
    pid = st.selectbox("Search for a forward", options=[None] + player_ids, index=0,
                       key="off_player_input", format_func=fmt)
    if pid is None:
        state.clear_player_background()

    # Conditional pre-tab content goes through one st.empty() slot so st.tabs
    # keeps a fixed position in the widget tree (otherwise Streamlit resets the
    # active sub-tab when a player is selected).
    above_tabs = st.empty()
    t_fit, t_next, t_insert = st.tabs(["Team Fit", "Next Season", "Roster Insertion"])

    pred = None
    if pid is not None:
        pred, first = resolve_forward(pid, fwd, "off_team_override", container=above_tabs)
        if first is None:
            above_tabs.error(f"No model data for {fmt(pid)}.")
        elif pred and state.track_player("offensive", pred["pid"], pred["actual_team"]):
            update_team_colors(player_team=pred["actual_team"], override_team=state.override_team())

    with t_fit:
        if pred:
            _render_projection(pred, fwd, "fit_results", next_season=False)
        else:
            st.info("Search for a forward above to see predictions.")
    with t_next:
        if pred:
            _render_projection(pred, fwd, "next_results", next_season=True)
        else:
            st.info("Search for a forward above to see predictions.")
    with t_insert:
        _render_insertion(pred, fwd)


def _render_projection(pred, fwd, results_key, next_season):
    df, results = fwd.df, pred[results_key]
    profile = fwd.profiles[pred["pid"]][0] if pred["pid"] in fwd.profiles else None
    has_age = pred.get("age") is not None and pd.notna(pred.get("age"))

    if next_season:
        age_str = f"  |  Age {pred['age']:.0f} → {pred['age'] + 1:.0f}" if has_age else ""
        player_header(pred["pid"], f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}",
                      "Predicted next-season performance across all 32 teams.")
    else:
        seasons_str = " → ".join(season_label(s) for s in pred["seasons"])
        age_str = f"  |  Age {pred['age']:.0f}" if has_age else ""
        player_header(pred["pid"],
                      f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}  |  Seasons: {seasons_str}",
                      "Predicted performance based on current weighted skill profile across all 32 teams.")

    actual_row = results[results["is_actual"]]
    row = actual_row.iloc[0] if not actual_row.empty else results.iloc[0]
    stats = [("Points / Game", "points_per_game", "Points/GP Grade"),
             ("Goals / Game", "goals_per_game", "Goals/GP Grade"),
             ("Game Score / Game", "game_score_per_game", "Game Score Grade")]
    pcts = {col: percentile_rank(row[f"pred_{col}"], df[col]) for _, col, _ in stats}
    overall = pcts["points_per_game"] * 0.5 + pcts["goals_per_game"] * 0.3 + pcts["game_score_per_game"] * 0.2
    grade_metrics([grade_item(grade_label, pcts[col]) for _, col, grade_label in stats] +
                  [grade_item("Overall Grade", overall)])

    st.markdown("#### Category Breakdown")
    st.markdown("**Next-Season Projection (on actual team)**" if next_season else "**Production (on actual team)**")
    for label, col, _ in stats:
        pct_bar(label, f"{row[f'pred_{col}']:.3f}", pcts[col])

    if profile is not None and not next_season:
        st.markdown("**Skill Profile**")
        for col, label in SKILL_PROFILE:
            if col in profile.index and not pd.isna(profile[col]) and col in df.columns:
                value = float(profile[col])
                pct_bar(label, f"{value:.3f}", percentile_rank(value, df[col]))

    if profile is not None and next_season and fwd.has_age and has_age:
        _render_age_trajectory(profile)

    title = (f"{pred['matched']}  |  Next season forecast" if next_season else
             f"{pred['matched']}  |  Current skill profile  |  Seasons: {' → '.join(season_label(s) for s in pred['seasons'])}")
    st.plotly_chart(charts.forward_bar_chart(results, pred["actual_team"], title), width="stretch")

    st.markdown("#### Rankings Table")
    display = rankings_table(results, pred["actual_team"], RANKING_COLUMNS)
    csv_download("Download CSV", display.drop(columns="_is_actual"),
                 f"{slug(pred['matched'])}_{'next_season' if next_season else 'team_fit'}.csv", index_label="rank")


def _render_age_trajectory(profile):
    st.markdown("**Age Trajectory**")
    peak = float(profile.get("career_peak_points_pg", 0) or 0)
    if peak > 0:
        pct_of_peak = float(profile.get("pct_of_peak_points", 0) or 0)
        st.markdown(f"**Career Peak** &nbsp; `{peak:.3f} pts/gp` &nbsp; — &nbsp; "
                    f"Currently at `{pct_of_peak * 100:.0f}%` of peak", unsafe_allow_html=True)
    slope = float(profile.get("recent_3yr_points_slope", 0) or 0)
    trend_line(slope, "pts/gp")


def trend_line(slope, unit):
    color = "#57a85a" if slope >= 0 else "#c8102e"
    label = "▲ ascending" if slope > 0.01 else ("▼ declining" if slope < -0.01 else "→ stable")
    st.markdown(f"**3-Year Trend** &nbsp; <span style='color:{color}'>**{label}**</span> "
                f"(`{slope:+.3f}` {unit} per season)", unsafe_allow_html=True)


def _render_insertion(pred, fwd):
    st.caption("Select a team to see where the searched player would slot into their active roster.")
    if not pred:
        st.info("Search for a forward above to use this tab.")
        return

    c1, c2 = st.columns([1, 1])
    team = c1.selectbox(
        "Select team to insert player into", options=NHL_TEAMS,
        index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
        key="insertion_team", on_change=state.team_select_callback("insertion_team"),
    )
    if c2.button("Refresh roster"):
        nhl_api.fetch_team_forwards.clear()

    with st.spinner(f"Building {team} roster with {pred['matched']} inserted..."):
        insertion, err = offense.build_player_insertion(pred["pid"], team, fwd)
    if err:
        st.error(err)
        return
    if insertion is None or insertion.empty:
        return

    searched = insertion[insertion["is_searched_player"]].iloc[0]
    color    = searched["slot_color"]
    slot_text = ("an <b>extra / scratch</b>" if searched["lineup_slot"] == "Extra"
                 else f"a <b>{searched['lineup_slot']}</b>")
    st.markdown(
        f"<h3 style='color:{color}'>{pred['matched']} projects as {slot_text} player on {team} "
        f"(rank {int(searched['rank'])} of {len(insertion)} forwards)</h3>",
        unsafe_allow_html=True,
    )

    display = insertion[["rank", "player_name", "position", "lineup_slot", "pred_points_gp", "pred_goals_gp"]].copy()
    display.columns = ["Rank", "Player", "Pos", "Line/Pair", "Points/GP", "Goals/GP"]
    highlight = f"background-color:{color}22;font-weight:bold;border-left:3px solid {color}"
    st.dataframe(
        display.style.apply(lambda r: [highlight if insertion.loc[r.name, "is_searched_player"] else ""] * len(r), axis=1),
        width="stretch", height=min(50 + len(display) * 35, 600),
    )

    slot_counts = display["Line/Pair"].value_counts()
    st.markdown("**Roster slot breakdown after insertion:**")
    for col, label in zip(st.columns(5), ["1st Line", "2nd Line", "3rd Line", "4th Line", "Extra"]):
        col.metric(label, int(slot_counts.get(label, 0)))

    csv_download("Download roster insertion CSV", insertion.drop(columns=["slot_color"]),
                 f"{slug(pred['matched'])}_{team}_insertion.csv", index=False)
