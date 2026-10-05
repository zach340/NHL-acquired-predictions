"""Validation tab: model predictions vs live current-season NHL API stats."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error

from .. import charts, defense, nhl_api, offense, validation
from ..config import ELITE_QUANTILE, season_id, season_label
from .components import csv_download

FWD_TABLE_COLS = ["player_name", "team", "games_played", "actual_points_gp", "pred_points_gp",
                  "points_gp_error", "actual_goals_gp", "pred_goals_gp", "seasons_used"]
DEF_TABLE_COLS = ["player_name", "team", "games_played", "actual_hits_pg", "pred_hits_pg", "hits_error",
                  "actual_tk_pg", "pred_tk_pg", "tk_error", "actual_pim_pg", "pred_pim_pg", "seasons_used"]


def render(fwd, dfn):
    # Validate on the first season the models never saw
    season = int(fwd.df["season"].max()) + 1
    t_off, t_def = st.tabs(["Offensive", "Defensive"])
    with t_off:
        _render_offensive(fwd, season)
    with t_def:
        _render_defensive(dfn, season)


def _mae(df, actual, pred):
    return f"{mean_absolute_error(df[actual], df[pred]):.3f}"


def _misses_and_best(val_df, error_col, cols, title_suffix=""):
    ranked = val_df[error_col].abs()
    st.markdown(f"#### Biggest Misses{title_suffix}")
    st.dataframe(val_df.reindex(ranked.nlargest(15).index)[cols], width="stretch")
    st.markdown(f"#### Best Predictions{title_suffix}")
    st.dataframe(val_df.reindex(ranked.nsmallest(15).index)[cols], width="stretch")


def _render_history(metrics, title):
    """Weekly MAE snapshots (validation_history.csv, written by scripts/validation_snapshot.py)."""
    hist = validation.load_history()
    hist = hist.dropna(subset=[col for col, _ in metrics], how="all")
    if hist.empty:
        st.caption("Accuracy history: no weekly snapshots yet — a GitHub Action adds one every Monday "
                   "once players reach 10 games.")
        return
    st.plotly_chart(charts.validation_history_chart(hist, metrics, title), width="stretch")


def _render_offensive(fwd, season):
    label = season_label(season)
    st.subheader(f"{label} Offensive Validation")
    st.caption(f"Next Season predictions made from each forward's {season_label(season - 1)} profile, compared "
               f"with their actual {label} regular-season stats from the NHL API (per-game rates). "
               f"{label} was never used in training.")
    if st.button("Refresh NHL API stats"):
        st.cache_data.clear()
        st.rerun()
    _render_history([("points_mae", "Points/GP MAE"), ("goals_mae", "Goals/GP MAE")],
                    "Forward accuracy over time (weekly snapshots)")

    actual_df, err = nhl_api.fetch_season_skaters(season_id(season))
    if err:
        st.error(f"Could not fetch NHL API data: {err}")
        return
    if actual_df is None:
        return
    st.success(f"Fetched {len(actual_df):,} skaters with 10+ games played.")

    with st.spinner("Comparing predictions to actual stats..."):
        val_df = offense.build_validation_results(actual_df, fwd)
    if val_df.empty:
        st.warning("No players matched between NHL API and model profiles.")
        return

    st.markdown(f"**{len(val_df):,} players matched**")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(charts.BG)
    charts.scatter(val_df, "actual_points_gp", "pred_points_gp", "Points / Game", axes[0])
    charts.scatter(val_df, "actual_goals_gp",  "pred_goals_gp",  "Goals / Game",  axes[1])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Points/GP MAE", _mae(val_df, "actual_points_gp", "pred_points_gp"))
    c2.metric("Goals/GP MAE",  _mae(val_df, "actual_goals_gp",  "pred_goals_gp"))
    c3.metric("Players matched", f"{len(val_df):,}")

    actual_range = val_df["actual_points_gp"].max() - val_df["actual_points_gp"].min()
    spread_ratio = ((val_df["pred_points_gp"].max() - val_df["pred_points_gp"].min()) / actual_range
                    if actual_range > 0 else np.nan)
    slope = charts.calibration_slope(val_df, "actual_points_gp", "pred_points_gp")
    c4, c5 = st.columns(2)
    c4.metric("Prediction Spread Ratio", f"{spread_ratio:.2%}" if not pd.isna(spread_ratio) else "n/a")
    c5.metric("Calibration Slope",       f"{slope:.2f}"        if not pd.isna(slope)        else "n/a")

    top_pct = int(round((1 - ELITE_QUANTILE) * 100))
    elite = {stat: charts.elite_segment_stats(val_df, f"actual_{stat}_gp", f"pred_{stat}_gp", quantile=ELITE_QUANTILE)
             for stat in ("points", "goals")}
    for col, (stat, label) in zip(st.columns(2), (("points", "Points"), ("goals", "Goals"))):
        mae, bias, _ = elite[stat]
        col.metric(f"Elite {label}/GP MAE (top {top_pct}%)",
                   f"{mae:.3f}" if not pd.isna(mae) else "n/a",
                   f"bias {bias:+.3f}" if not pd.isna(bias) else None)
    st.caption(f"Elite sample sizes: Points {elite['points'][2]}, Goals {elite['goals'][2]}")

    st.divider()
    _misses_and_best(val_df, "points_gp_error", FWD_TABLE_COLS)
    csv_download("Download full validation CSV", val_df, f"validation_{label.replace('-', '_')}.csv", index=False)


def _render_defensive(dfn, season):
    label = season_label(season)
    st.subheader(f"{label} Defensive Validation")
    st.caption(f"Next Season predictions made from each defenseman's {season_label(season - 1)} profile, "
               f"compared with their actual {label} regular-season hits, takeaways and PIM per game.")
    if st.button("Refresh defensive stats"):
        nhl_api.fetch_defensive_stats.clear()
    _render_history([("hits_mae", "Hits/GP MAE"), ("takeaways_mae", "Takeaways/GP MAE"), ("pim_mae", "PIM/GP MAE")],
                    "Defenseman accuracy over time (weekly snapshots)")

    if dfn is None:
        st.warning("Defensive model not loaded.")
        return
    actual, err = nhl_api.fetch_defensive_stats(season_id(season))
    if err:
        st.error(f"Could not fetch NHL API data: {err}")
        return
    if actual is None:
        return

    actual = actual[actual["player_id"].isin(dfn.profiles.keys())].copy()
    st.success(f"Fetched {len(actual):,} defensemen with 10+ games played.")
    with st.spinner("Comparing predictions to actual stats..."):
        val_df = defense.build_validation(actual, dfn)
    if val_df.empty:
        st.warning("No defensemen matched between NHL API and model profiles.")
        return

    st.markdown(f"**{len(val_df):,} defensemen matched**")
    has_pim = "actual_pim_pg" in val_df.columns and val_df["actual_pim_pg"].sum() > 0

    cols = st.columns(4)
    cols[0].metric("Hits/GP MAE",      _mae(val_df, "actual_hits_pg", "pred_hits_pg"))
    cols[1].metric("Takeaways/GP MAE", _mae(val_df, "actual_tk_pg",   "pred_tk_pg"))
    if has_pim:
        cols[2].metric("PIM/GP MAE",   _mae(val_df, "actual_pim_pg",  "pred_pim_pg"))
    cols[3].metric("Defensemen matched", f"{len(val_df):,}")
    st.caption("PIM/GP: actual = penaltyMinutes/GP from the NHL API stats reports; the model is trained on "
               "PIM/GP built from NHL API play-by-play.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(charts.BG)
    charts.scatter(val_df, "actual_hits_pg", "pred_hits_pg", "Hits / Game",      axes[0][0])
    charts.scatter(val_df, "actual_tk_pg",   "pred_tk_pg",   "Takeaways / Game", axes[0][1])
    if has_pim:
        charts.scatter(val_df, "actual_pim_pg", "pred_pim_pg", "PIM / Game", axes[1][0])
    else:
        charts.blank_panel(axes[1][0], "PIM / Game", "PIM data not available\nfrom NHL API")
    charts.blank_panel(axes[1][1], "xGA Against / 60", "xGA validation not shown\n(needs on-ice xG for the season)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    _misses_and_best(val_df, "hits_error", [c for c in DEF_TABLE_COLS if c in val_df.columns], " (Hits/GP)")
    csv_download("Download defensive validation CSV", val_df, f"def_validation_{label.replace('-', '_')}.csv", index=False)
