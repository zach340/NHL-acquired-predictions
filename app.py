"""
app.py
======
NHL Player Predictor — Streamlit entry point.
Model logic lives in the nhl_predictor package; each tab in nhl_predictor/ui.

Run with:  python -m streamlit run app.py
"""

import streamlit as st
from streamlit_option_menu import option_menu

from nhl_predictor.theme import apply_team_theme, asset, inject_script
from nhl_predictor.ui import (
    contract_tab, defensive_tab, models_tab, offensive_tab, paper, state, validation_tab,
)

st.set_page_config(page_title="NHL Player Predictor", page_icon="🏒", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(asset("layout.html"), unsafe_allow_html=True)
# player_base_team drives the background everywhere; active_team only on the
# Roster Insertion / Pairing / Contract tabs once the user picks a team there.
apply_team_theme(player_team=st.session_state.get("player_base_team"), override_team=state.override_team())
st.title("NHL Player Predictor")

fwd, dfn = state.load_models()

if fwd.has_age:
    st.markdown(
        '<div style="display:block;background:#1a4a2e;border:1px solid #2ecc71;'
        'border-radius:20px;padding:6px 18px;font-size:13px;">'
        '<span style="color:#2ecc71 !important;font-weight:700;">● Age data loaded</span>'
        '<span style="color:#cccccc !important;"> — next-season forecasting active</span>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="display:block;background:#4a3000;border:1px solid #f39c12;'
        'border-radius:20px;padding:6px 18px;font-size:13px;">'
        '<span style="color:#f39c12 !important;font-weight:700;">⚠ Age data not found</span>'
        '<span style="color:#cccccc !important;"> — next-season model running without age features</span>'
        '</div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    active_page = option_menu(
        menu_title=None,
        options=["NHL Predictor"] + list(paper.PAGES),
        icons=[""] * (len(paper.PAGES) + 1),
        default_index=0,
        styles={"icon": {"display": "none"}},
    )


def active_players(df, cutoff_offset=2):
    """
    {player_id: (name, position)} for players whose latest season is within the
    last 3 seasons, keyed by id so players who share a name stay distinct.
    """
    latest = df.groupby("player_id")["season"].max()
    active = latest[latest >= int(df["season"].max()) - cutoff_offset].index
    rows = (df[df["player_id"].isin(active)]
            .sort_values("season", ascending=False)
            .drop_duplicates("player_id"))
    return {int(r.player_id): (r.player_name, r.position) for r in rows.itertuples()}


def picker(players):
    """(options sorted by name, format_func) for a player selectbox."""
    ids = sorted(players, key=lambda pid: players[pid][0])
    return ids, lambda pid: "" if pid is None else f"{players[pid][0]}  ({players[pid][1]})"


if active_page == "NHL Predictor":
    fwd_players = active_players(fwd.df)
    def_players = active_players(dfn.df) if dfn is not None else {}
    all_players = {**fwd_players, **def_players}

    t_off, t_def, t_contract, t_models, t_val = st.tabs(
        ["Offensive", "Defensive", "Contract Evaluator", "Models", "Validation"])

    # Spotlight tour + outer-tab persistence (the tab can reset to "Offensive"
    # when a conditional element shifts the widget tree; JS restores it).
    inject_script("tour.html")
    inject_script("tab_persist.html")

    with t_off:
        offensive_tab.render(fwd, *picker(fwd_players))
    with t_def:
        defensive_tab.render(dfn, *picker(def_players))
    with t_contract:
        contract_tab.render(fwd, dfn, *picker(all_players), positions={p: v[1] for p, v in all_players.items()})
    with t_models:
        models_tab.render(fwd, dfn)
    with t_val:
        validation_tab.render(fwd, dfn)
else:
    paper.PAGES[active_page]()
