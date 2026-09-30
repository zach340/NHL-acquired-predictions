"""Defensive tab: Team Fit, Next Season and Pairing for defensemen."""

import pandas as pd
import streamlit as st

from .. import defense, nhl_api, pairing
from ..config import NHL_TEAMS, SLOT_COLORS, season_label
from ..data_io import load_defensive_offensive_stats
from ..grading import classify_defenseman_type, grade_defensive_defenseman, grade_offensive_defenseman
from ..theme import update_team_colors
from . import state
from .components import csv_download, grade_item, grade_metrics, pct_bar, player_header, rankings_table, slug
from .offensive_tab import trend_line

RANKING_COLUMNS = {
    "player_team":          "Team",
    "ind_hits_pg":          "Hits/GP",
    "ind_takeaways_pg":     "TK/GP",
    "xg_against_per60_5v5": "xGA/60",
    "pim_pg":               "PEN/GP",
    "defensive_score":      "Def Score",
}
LOWER_IS_BETTER_CATEGORIES = {"xGA/60 (5v5)", "PIM/GP"}


def dpred_key(pid):
    """Session key under which a D-man's predictions are shared with the Contract tab."""
    return f"dpred_{pid}"


def render(dfn, player_ids, fmt):
    if dfn is None:
        st.warning("Defensive model not loaded. Ensure defensive_dataset.csv is present.")
        return

    st.caption("Search for a defenseman to see defensive predictions.")
    pid = st.selectbox("Search for a defenseman", options=[None] + player_ids, index=0,
                       key="def_player_input", format_func=fmt)

    # Keep st.tabs at a fixed widget-tree position: anything conditional goes
    # through this placeholder, and the background update comes after the tabs.
    above_tabs = st.empty()
    dpred = defense.predict_defenseman(pid, dfn) if pid is not None else None
    if pid is not None and dpred is None:
        above_tabs.error(f"No model data for {fmt(pid)}.")

    t_fit, t_next, t_pair = st.tabs(["Team Fit", "Next Season", "Pairing"])

    if dpred:
        st.session_state[dpred_key(dpred["pid"])] = dpred
        if state.track_player("defensive", dpred["pid"], dpred["actual_team"]):
            update_team_colors(player_team=dpred["actual_team"], override_team=state.override_team())

    off_stats_by_pid, off_df, _ = load_defensive_offensive_stats()
    off_stats = off_stats_by_pid.get(dpred["pid"], {}) if dpred else {}

    for tab, render_tab in ((t_fit, _render_fit), (t_next, _render_next), (t_pair, _render_pairing)):
        with tab:
            if not dpred:
                st.info("Search for a defenseman above.")
            elif render_tab is _render_pairing:
                render_tab(dpred, dfn)
            else:
                render_tab(dpred, dfn, off_stats, off_df)


def _grades(dpred, dfn, results, off_stats, off_df):
    """Defensive/offensive grades and archetype for the actual-team row of `results`."""
    actual_row = results[results["is_actual"]]
    preds = (actual_row if not actual_row.empty else results).iloc[0].to_dict()
    def_grade, def_pct, _, def_breakdown = grade_defensive_defenseman(preds, season_def_df=dfn.df)
    off_grade, off_pct, _, off_breakdown = (
        grade_offensive_defenseman(off_stats, season_off_df=off_df) if off_stats else ("—", 0, "", {}))
    profile = dict(dfn.profiles[dpred["pid"]][0]) if dpred["pid"] in dfn.profiles else preds
    d_type, d_desc = classify_defenseman_type(profile, def_score=def_pct, off_score=off_pct)

    def_item = ("Defensive Grade", def_grade, def_pct)
    off_item = ("Offensive Grade", off_grade if off_stats else "—", off_pct if off_stats else None)
    combined = grade_item("Combined Grade", def_pct * 0.5 + off_pct * 0.5)
    items = [off_item, def_item, combined] if d_type == "Offensive D" else [def_item, off_item, combined]
    return d_type, d_desc, items, def_breakdown, off_breakdown if off_stats else {}


def _breakdown_bars(title, def_breakdown, off_breakdown):
    st.markdown("#### Category Breakdown")
    st.markdown(title)
    for cat, (val, pct) in def_breakdown.items():
        pct_bar(cat, val, pct, lower_is_better=cat in LOWER_IS_BETTER_CATEGORIES)
    if off_breakdown:
        st.markdown("**Offensive**")
        for cat, (val, pct) in off_breakdown.items():
            pct_bar(cat, val, pct)


def _rankings(dpred, results, suffix):
    st.markdown("#### Rankings Table")
    display = rankings_table(results, dpred["actual_team"], RANKING_COLUMNS)
    csv_download("Download CSV", display.drop(columns="_is_actual"), f"{slug(dpred['matched'])}_{suffix}.csv",
                 index_label="rank")


def _render_fit(dpred, dfn, off_stats, off_df):
    d_type, d_desc, items, def_bd, off_bd = _grades(dpred, dfn, dpred["fit_results"], off_stats, off_df)
    seasons_str = " → ".join(season_label(s) for s in dpred["seasons"])
    player_header(dpred["pid"], f"{dpred['matched']}  —  {d_type}  |  {dpred['actual_team']}  |  Seasons: {seasons_str}",
                  d_desc)
    grade_metrics(items)
    _breakdown_bars("**Defensive**", def_bd, off_bd)
    _rankings(dpred, dpred["fit_results"], "def_fit")


def _render_next(dpred, dfn, off_stats, off_df):
    age = dpred.get("age")
    has_age = age is not None and pd.notna(age)
    player_header(dpred["pid"],
                  f"{dpred['matched']}  —  D  |  {dpred['actual_team']}" + (f"  |  Age {age:.0f} → {age + 1:.0f}" if has_age else ""),
                  "Next-season defensive forecast based on current profile and trajectory.")
    _, _, items, def_bd, off_bd = _grades(dpred, dfn, dpred["next_results"], off_stats, off_df)
    grade_metrics(items)
    _breakdown_bars("**Next-Season Defensive Projection**", def_bd, off_bd)

    profile = dfn.profiles.get(dpred["pid"], (None,))[0]
    if profile is not None and dfn.has_age and has_age:
        st.markdown("**Age Trajectory**")
        peak = float(profile.get("career_peak_hits_pg", 0) or 0)
        if peak > 0:
            st.markdown(f"**Career Peak (Hits/GP)** &nbsp; `{peak:.3f}` &nbsp; — &nbsp; "
                        f"Currently at `{float(profile.get('pct_of_peak_hits', 0) or 0) * 100:.0f}%` of peak",
                        unsafe_allow_html=True)
        trend_line(float(profile.get("recent_3yr_hits_slope", 0) or 0), "hits/gp")

    _rankings(dpred, dpred["next_results"], "def_next")


# ── Pairing ────────────────────────────────────────────────────────────────────

def _render_pairing(dpred, dfn):
    c1, c2 = st.columns([1, 1])
    team = c1.selectbox(
        "Select team", options=NHL_TEAMS,
        index=NHL_TEAMS.index(dpred["actual_team"]) if dpred["actual_team"] in NHL_TEAMS else 0,
        key="pair_team_sel", on_change=state.team_select_callback("pair_team_sel"),
    )
    n_games = st.slider("Games to include in pairing data", min_value=10, max_value=82, value=25, step=5,
                        help="More games = more stable pairs but slower load. 25 games reflects recent pairings well.",
                        key="pair_games_slider")

    cache_key = f"_pairs_{team}_{n_games}"
    if c2.button("Refresh roster & shifts", key="pair_refresh"):
        nhl_api.fetch_team_defensemen.clear()
        nhl_api.clear_shifts_cache(team, n_games)
        st.session_state.pop(cache_key, None)

    if cache_key not in st.session_state:
        label, bar = st.empty(), st.progress(0)

        def on_progress(done, total):
            label.markdown(f"<small style='color:#aaa'>Fetching {team} shifts — game {done} of {total}</small>",
                           unsafe_allow_html=True)
            bar.progress(int(done / total * 100))

        st.session_state[cache_key] = nhl_api.fetch_shift_pairs(team, n_games=n_games, on_progress=on_progress)
        bar.empty()
        label.empty()

    with st.spinner("Building pairing model..."):
        depth_pairs, scratched, scores, cascade_log, unmodeled, info = pairing.build_pairing_insertion(
            dpred["pid"], team, dfn, n_games=n_games, prefetched_pairs=st.session_state[cache_key])

    if not depth_pairs and not scores:
        st.error(info.get("pair_err", "Could not build pairings. Try refreshing."))
        return

    name = dpred["matched"]
    if info.get("pair_err"):
        st.caption(f"Note: Could not fetch shift data ({info['pair_err']}). Using model-ranked order.")
    st.markdown(f"### {name} — {scores.get(dpred['pid'], {}).get('d_type', '—')} | "
                f"Combined Score: {info['searched_score']:.0f}")
    if info["partner_name"] != "—":
        st.success(f"Projected pair: **{name}** with **{info['partner_name']}** ({info['partner_slot']})")

    st.divider()
    if info["is_returning"]:
        st.markdown(f"#### {team} Defensive Depth Chart — Current Pairs")
        st.caption("Gold = highlighted player. Season-long shift pairs shown as-is.")
    else:
        st.markdown(f"#### {team} Defensive Depth Chart — After Insertion")
        st.caption("Gold = new player. Pairs anchored from season-long shift data. "
                   "Cascade ripples down — weakest player scratched.")
    for pair in depth_pairs:
        _pair_row(pair, scores, dpred["pid"])
    st.caption("Score = combined grade (defensive + offensive weighted by player type).")

    if scratched:
        st.divider()
        st.markdown("#### Scratched / Excess D-men")
        for col, pid in zip(st.columns(min(len(scratched), 4)), scratched):
            s = scores.get(pid, {})
            col.metric(s.get("player_name", str(pid)), f"Score: {s.get('combined_score', 0):.0f}", s.get("d_type", ""))

    if cascade_log:
        st.divider()
        st.markdown("#### Displacement Cascade")
        st.caption("Step-by-step ripple: each displaced player finds their next best slot.")
        for entry in cascade_log:
            if entry["action"] == "scratched":
                st.markdown(f"<span style='color:#c8102e'>✗</span> **{entry['player']}** could not improve "
                            "any pair → **Scratched**", unsafe_allow_html=True)
            else:
                color = "#FFD700" if entry["player"] == name else "#4a90d9"
                displaced = f" (displaces {entry['displaced']})" if entry["displaced"] != "—" else ""
                st.markdown(f"<span style='color:{color}'>↓</span> **{entry['player']}** → **{entry['slot']}**{displaced}",
                            unsafe_allow_html=True)

    if unmodeled:
        with st.expander(f"{len(unmodeled)} rostered D-men not in model data"):
            for pid in unmodeled:
                st.caption(f"• {info['roster_names'].get(pid, pid)} — not enough historical seasons in training data")

    with st.expander("Full model scores for all D-men"):
        table = pd.DataFrame([{
            "Player":        s["player_name"],
            "Type":          s.get("d_type", "—"),
            "Combined":      s.get("combined_score", s["defensive_score"]),
            "Def Score":     s["defensive_score"],
            "Hits/GP":       round(s.get("ind_hits_pg", 0), 2),
            "TK/GP":         round(s.get("ind_takeaways_pg", 0), 3),
            "xGA/60":        round(s.get("xg_against_per60_5v5", 0), 3),
            "PIM/GP":        round(s.get("pim_pg", 0), 2),
            "Is New Player": s.get("is_searched_player", False),
        } for s in scores.values()])
        ranked = table.sort_values("Combined", ascending=False).reset_index(drop=True)
        is_new = ranked["Is New Player"].values
        st.dataframe(
            ranked.drop(columns="Is New Player").style.apply(
                lambda r: ["background-color:#FFD70022;font-weight:bold" if is_new[r.name] else ""] * len(r), axis=1),
            width="stretch", hide_index=True,
        )
        csv_download("Download pairing CSV", table, f"{slug(name)}_{team}_pairing.csv", index=False)


def _pair_row(pair, scores, searched_pid):
    color = SLOT_COLORS.get(pair["slot"], "#888888")
    hand1, hand2 = pair.get("shoots1", ""), pair.get("shoots2", "")
    hand_icon = "🤝" if pair.get("hand_match") else ("⚠️" if (hand1 and hand2) else "")

    col_slot, col_p1, col_vs, col_p2, col_score = st.columns([1.2, 2.5, 0.3, 2.5, 1.0])
    col_slot.markdown(f"<span style='color:{color};font-weight:bold'>{pair['slot']}</span><br><small>{hand_icon}</small>",
                      unsafe_allow_html=True)
    for col, n in ((col_p1, "1"), (col_p2, "2")):
        pid, hand = pair[f"pid{n}"], pair[f"shoots{n}"]
        style = "color:#FFD700;font-weight:bold" if pid == searched_pid else ""
        col.markdown(
            f"<span style='{style}'>**{pair[f'name{n}']}**{f' ({hand})' if hand else ''} ({pair[f'score{n}']:.0f})</span>"
            f"  \n*{scores.get(pid, {}).get('d_type', '')}*",
            unsafe_allow_html=True,
        )
    col_vs.markdown("—")
    col_score.markdown(f"Avg: **{pair['pair_score']:.0f}**")
