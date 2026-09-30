"""
theme.py
========
Dark theme and team-coloured page background.

How the background switching works:
  • Two CSS custom properties on :root are written every rerun:
        --team-bg-base     (player's real team gradient)
        --team-bg-override (insertion / pairing / contract team gradient)
  • .stApp uses var(--team-bg-base) directly.
  • assets/team_background.html (a zero-height iframe) watches which sub-tab
    and team selectbox are active and paints the matching gradient instantly,
    without a Streamlit rerun.
"""

import json
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from .config import TEAM_COLORS

ASSETS = Path(__file__).parent / "assets"


@lru_cache(maxsize=None)
def asset(name):
    return (ASSETS / name).read_text(encoding="utf-8")


def inject_script(name):
    """Render an HTML/JS asset in an invisible iframe."""
    components.html(asset(name), height=0)


def _brighten_for_gradient(hex_color, min_luminance=0.22):
    """
    Mix a too-dark hex colour toward white until it clears `min_luminance`
    so team gradients stay visible on the dark background (e.g. VGK, DAL).
    """
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    if luminance >= min_luminance:
        return hex_color
    for _ in range(5):
        r = min(255, int(r + (255 - r) * 0.45))
        g = min(255, int(g + (255 - g) * 0.45))
        b = min(255, int(b + (255 - b) * 0.45))
        if (0.299 * r + 0.587 * g + 0.114 * b) / 255 >= min_luminance:
            break
    return f"#{r:02x}{g:02x}{b:02x}"


def team_gradient(team):
    """CSS gradient string for a team, or 'none' if unknown."""
    if team and team in TEAM_COLORS:
        p = _brighten_for_gradient(TEAM_COLORS[team]["primary"])
        s = _brighten_for_gradient(TEAM_COLORS[team]["secondary"])
        # 99 ≈ 60% opacity primary, 70 ≈ 44% secondary
        return f"linear-gradient(135deg, {p}99 0%, #141414 50%, {s}70 100%)"
    return "none"


def team_background_html():
    grads = json.dumps({team: team_gradient(team) for team in TEAM_COLORS})
    return asset("team_background.html").replace("__GRADS_JSON__", grads)


def _gradients(player_team, override_team):
    base = team_gradient(player_team)
    return base, (team_gradient(override_team) if override_team else base)


def apply_team_theme(player_team=None, override_team=None):
    """Inject the dark-mode CSS (with live team vars) and the background switcher."""
    base, override = _gradients(player_team, override_team)
    css = asset("theme.html").replace("__BASE_GRAD__", base).replace("__OVERRIDE_GRAD__", override)
    st.markdown(css, unsafe_allow_html=True)
    components.html(team_background_html(), height=0)


def update_team_colors(player_team=None, override_team=None):
    """Re-set just the two CSS vars so the background follows the current player/team."""
    base, override = _gradients(player_team, override_team)
    st.markdown(
        f"\n<style>:root{{--team-bg-base:{base};--team-bg-override:{override};}}</style>",
        unsafe_allow_html=True,
    )
