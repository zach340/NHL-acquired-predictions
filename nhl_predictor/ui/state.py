"""
state.py
========
Session-state helpers: loading/training the model bundles and tracking which
player and team drive the page background.

Background keys:
  player_base_team   the selected player's real team (non-override tabs)
  active_team        insertion / pairing / contract team (override tabs)
  _team_override     True once the user picks a team in one of those tabs
"""

import os

import streamlit as st

from .. import defense, offense
from ..config import AGES_FILE, CACHE_FILE, DATA_FILE, DEF_CACHE_FILE, DEF_FILE
from ..training import load_bundle, save_bundle

FWD_KEY = "fwd_bundle"
DEF_KEY = "def_bundle"

# Selectboxes whose team choice drives --team-bg-override
TEAM_SELECT_KEYS = ("insertion_team", "pair_team_sel", "contract_team")


def _load_or_train(state_key, cache_file, train, training_msg, loading_msg):
    """Load a bundle from its joblib cache, or train + cache it and rerun."""
    if os.path.exists(cache_file):
        with st.spinner(loading_msg):
            st.session_state[state_key] = load_bundle(cache_file)
    else:
        st.info(training_msg)
        save_bundle(train(), cache_file)
        st.rerun()   # reload from the fresh cache on a clean page


def load_models():
    """(forward_bundle, defense_bundle_or_None), training and caching on first run."""
    if FWD_KEY not in st.session_state:
        _load_or_train(
            FWD_KEY, CACHE_FILE, lambda: offense.load_and_train(DATA_FILE, AGES_FILE),
            "Training models for the first time — this takes 5–8 minutes. Won't happen again until you retrain.",
            "Loading saved models from disk...",
        )
    if DEF_KEY not in st.session_state:
        if os.path.exists(DEF_CACHE_FILE) or os.path.exists(DEF_FILE):
            _load_or_train(
                DEF_KEY, DEF_CACHE_FILE, lambda: defense.load_and_train(DEF_FILE, AGES_FILE),
                "Training defensive models for the first time — takes 3-5 minutes.",
                "Loading saved defensive models...",
            )
        else:
            st.session_state[DEF_KEY] = None
    return st.session_state[FWD_KEY], st.session_state[DEF_KEY]


def forget_models(state_key):
    st.session_state.pop(state_key, None)


def track_player(source, pid, actual_team):
    """
    Record that `source` ("offensive" / "defensive" / "contract") selected a
    player. A new player resets any team overrides. Returns True if this
    source currently owns the page background.
    """
    ss = st.session_state
    pid_key = f"_{source}_pid"
    if ss.get(pid_key) != pid:
        ss[pid_key] = pid
        ss["_last_player_source"] = source
        ss["_team_override"] = False
        for k in TEAM_SELECT_KEYS:
            ss.pop(k, None)
    if ss.get("_last_player_source") != source:
        return False
    ss["player_base_team"] = actual_team
    if not ss.get("_team_override"):
        ss["active_team"] = actual_team
    return True


def override_team():
    """The override team, or None if the user hasn't picked one."""
    return st.session_state.get("active_team") if st.session_state.get("_team_override") else None


def team_select_callback(select_key):
    """on_change handler: a team selectbox now drives the override background."""
    def _cb():
        st.session_state["active_team"]    = st.session_state.get(select_key)
        st.session_state["_team_override"] = True
    return _cb


def clear_player_background():
    st.session_state.pop("active_team", None)
    st.session_state.pop("player_base_team", None)
