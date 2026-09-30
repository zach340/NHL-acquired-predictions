"""Contract Evaluator tab: multi-year projections for forwards and defensemen."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .. import defense
from ..config import AGES_FILE, NHL_TEAMS
from ..contract import build_contract_projection, contract_risk_rating, get_cba_limits
from ..data_io import current_age
from ..theme import update_team_colors
from . import state
from .components import csv_download, slug
from .defensive_tab import dpred_key
from .offensive_tab import resolve_forward


def _find_player(pid, position, fwd, dfn):
    """(pred, dpred) for a forward or a defenseman; pred is None if not found."""
    if position != "D":
        pred, first = resolve_forward(pid, fwd, "contract_team_override")
        return (pred, None) if first is not None else (None, None)
    if dfn is None:
        return None, None
    dpred = defense.predict_defenseman(pid, dfn)
    if dpred is None:
        return None, None
    st.session_state[dpred_key(dpred["pid"])] = dpred
    return {"pid": dpred["pid"], "matched": dpred["matched"], "actual_team": dpred["actual_team"],
            "position": "D", "seasons": dpred["seasons"], "traded_teams": [],
            "fit_results": None, "next_results": None, "age": None}, dpred


def _resolve_age(pred, dfn):
    """Current age: ages file (rolled forward), else D-man profile, else prediction, else 28."""
    age = current_age(AGES_FILE, pred["pid"])
    if not age and pred["position"] == "D" and dfn is not None:
        profile = dfn.profiles.get(pred["pid"], (None,))[0]
        if profile is not None:
            age = profile.get("age")
    if not age:
        age = pred.get("age")
    return float(age) if age and not pd.isna(age) else 28.0


def render(fwd, dfn, player_ids, fmt, positions):
    st.subheader("Contract Evaluator")
    st.caption(
        "Projects a player's production across multiple seasons using empirical age curves. "
        "Works for both forwards (offensive stats) and defensemen (defensive stats). "
        "Confidence decreases in later years — use ranges rather than exact numbers."
    )
    pid = st.selectbox("Search for a player", options=[None] + player_ids, index=0,
                       key="contract_player_input", format_func=fmt)
    if pid is None:
        st.info("Search for a player above to use the contract evaluator.")
        return

    pred, dpred = _find_player(pid, positions.get(pid), fwd, dfn)
    if not pred:
        st.error(f"No model data for {fmt(pid)}.")
        st.info("Search for a player above to use the contract evaluator.")
        return

    if state.track_player("contract", pred["pid"], pred["actual_team"]):
        # The background switcher only uses --team-bg-override on the Roster
        # Insertion / Pairing sub-tabs, so put the display team (signing team
        # when overridden) in both vars.
        shown = st.session_state.get("active_team", pred["actual_team"])
        update_team_colors(player_team=shown, override_team=shown)

    is_d = pred["position"] == "D"
    c1, c2, c3 = st.columns(3)
    team = c1.selectbox("Team signing the player", options=NHL_TEAMS,
                        index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
                        key="contract_team", on_change=state.team_select_callback("contract_team"))
    age = _resolve_age(pred, dfn)
    cba = get_cba_limits(age, pred["actual_team"], team)
    n_years = c2.slider("Contract length (years)", min_value=1, max_value=cba["max_years"],
                        value=min(cba["recommended"], cba["max_years"]))
    c3.metric("Current Age", f"{age:.0f}")

    _render_cba_bar(pred, cba, age, n_years)

    if is_d and dpred is None:
        dpred = st.session_state.get(dpred_key(pred["pid"]))
        if dpred is None and dfn is not None:
            with st.spinner("Loading defensive predictions for contract..."):
                dpred = defense.predict_defenseman(pred["pid"], dfn)
                if dpred:
                    st.session_state[dpred_key(pred["pid"])] = dpred

    with st.spinner("Projecting contract years..."):
        rows, err = build_contract_projection(pred, dpred if is_d else None, fwd, dfn, team, n_years, curr_age=age)
    if err:
        st.error(err)
    elif rows:
        _render_projection(pred, rows, is_d, team, n_years, cba, age)


def _render_cba_bar(pred, cba, age, n_years):
    cols = st.columns(4)
    cols[0].metric("Signing Type", "Re-signing (same team)" if cba["is_same_team"] else "New signing (different team)")
    cols[1].metric("CBA Max Length", f"{cba['max_years']} years")
    cols[2].metric("Recommended Max", f"{cba['recommended']} years")
    cols[3].metric("Age at Expiry", f"{age + n_years:.0f}")

    if cba["is_35_signing"]:
        st.error(f"35+ Rule: {pred['matched']} is {age:.0f} at signing. "
                 "The cap hit counts against your team even if the player retires early. "
                 "This creates significant cap recapture risk.")
    elif age + n_years > 35:
        st.warning(f"{pred['matched']} will be {age + n_years:.0f} when this contract ends. "
                   "The 35+ rule doesn't apply (signed before 35), but the final years carry decline risk.")
    if n_years > cba["recommended"]:
        st.warning(f"This contract is longer than the recommended maximum of {cba['recommended']} years "
                   f"based on the player's age curve. Years {cba['recommended'] + 1}+ carry high uncertainty.")


def _pct_label(pct):
    return f"Top {100 - pct:.0f}%" if pct > 10 else "Elite"


def _render_projection(pred, rows, is_d, team, n_years, cba, age):
    risk_label, risk_color, risk_explanation = contract_risk_rating(rows, is_d)
    st.markdown(f"<h3>{pred['matched']} on {team} — <span style='color:{risk_color}'>{risk_label}</span></h3>",
                unsafe_allow_html=True)
    if risk_explanation:
        st.caption(risk_explanation)

    st.markdown("#### Year-by-Year Projection")
    st.caption("Confidence reflects uncertainty compounding over time. Use wider mental ranges in later years.")
    if is_d:
        table = pd.DataFrame([{
            "Year":         f"Year {r['year']} (Age {r['age']:.0f})",
            "Hits/GP":      r["hits_pg"],
            "Takeaways/GP": r["takeaways_pg"],
            "xGA/60 (5v5)": r["goals_against_pg"],
            "PIM/GP":       r["pim_pg"],
            "Def %ile":     _pct_label(r["def_score"]),
            "Off %ile":     _pct_label(r["off_score"]),
            "Confidence":   f"{r['confidence'] * 100:.0f}%",
        } for r in rows])
    else:
        table = pd.DataFrame([{
            "Year":       f"Year {r['year']} (Age {r['age']:.0f})",
            "Points/GP":  r["points_pg"],
            "Goals/GP":   r["goals_pg"],
            "Pts %ile":   _pct_label(r.get("pts_pct", 50)),
            "Confidence": f"{r['confidence'] * 100:.0f}%",
        } for r in rows])
    st.dataframe(table, width="stretch", hide_index=True)

    st.markdown("#### Production Trend")
    _trend_chart(rows, is_d)

    st.divider()
    st.markdown("#### Contract Recommendation")
    first, last = rows[0], rows[-1]
    if is_d:
        st.markdown(
            f"- **Year 1 Defensive Percentile:** {first['def_score']:.0f}th% among all D-men  \n"
            f"- **Year {n_years} Defensive Percentile:** {last['def_score']:.0f}th%  \n"
            f"- **Projected percentile drop:** {max(first['def_score'] - last['def_score'], 0):.0f} points  \n"
            f"- **Risk rating:** {risk_label}"
        )
    else:
        st.markdown(
            f"- **Year 1 Points/GP:** {first['points_pg']:.3f}  \n"
            f"- **Year {n_years} Points/GP:** {last['points_pg']:.3f}  \n"
            f"- **Projected total points:** ~{sum(r['points_pg'] * 82 for r in rows):.0f} over {n_years} years  \n"
            f"- **Risk rating:** {risk_label}"
        )

    st.markdown("**CBA Summary:**")
    c1, c2 = st.columns(2)
    c1.info(
        f"Same-team max: **7 years**  \n"
        f"New-team max: **6 years**  \n"
        f"Signing type: **{'Same team (re-signing)' if cba['is_same_team'] else 'New team'}**  \n"
        f"CBA max for this deal: **{cba['max_years']} years**"
    )
    rule35 = ("35+ rule applies — cap hit stays on the books if the player retires early."
              if cba["is_35_signing"] else "35+ rule does not apply.")
    c2.success(f"Model recommendation: **{cba['recommended']} years**  \n"
               f"Based on age {age:.0f} trajectory.  \n{rule35}")

    csv_download("Download contract projection CSV", table,
                 f"{slug(pred['matched'])}_{team}_{n_years}yr_contract.csv", index=False)


def _trend_chart(rows, is_d):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    color = "#4a90d9"
    years = [r["year"] for r in rows]
    vals  = [r["def_score"] if is_d else r.get("pts_pct", 50) for r in rows]
    confs = [r["confidence"] for r in rows]

    ax.plot(years, vals, color=color, linewidth=2.5, marker="o", markersize=8, zorder=3)
    # Asymmetric band — late-contract risk is mostly downside (25% up / 75% down)
    spread = [v * (1 - c) * 0.4 for v, c in zip(vals, confs)]
    ax.fill_between(years, [max(v - s * 0.75, 0) for v, s in zip(vals, spread)],
                    [v + s * 0.25 for v, s in zip(vals, spread)], alpha=0.2, color=color, label="Confidence range")
    ax.set_xticks(years)
    ax.set_xticklabels([f"Yr {r['year']} (Age {r['age']:.0f})" for r in rows], color="white", fontsize=9)
    ax.set_ylabel("Defensive Percentile" if is_d else "Points Percentile", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
