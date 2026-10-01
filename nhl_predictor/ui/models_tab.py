"""Models tab: cross-validation quality, feature importance and cache resets."""

import os

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts
from ..config import (
    CACHE_FILE, CV_FOLDS, DEF_CACHE_FILE, DEF_LOWER_IS_BETTER, DEF_TARGET_LABELS, DEF_TARGETS,
    TARGET_LABELS, TARGETS,
)
from . import state


def show_metrics(metrics, label, targets, labels, directions=None):
    st.markdown(f"**{label} model quality (season-based CV: each of the last {CV_FOLDS} seasons predicted from earlier seasons only)**")
    if directions is None:
        st.caption("MAE = avg absolute error in same units as stat. RMSE penalises large errors more. Lower is better.")
    for target in targets:
        mae_mean,  mae_std  = metrics[target]["mae"]
        rmse_mean, rmse_std = metrics[target]["rmse"]
        st.markdown(f"*{labels[target]}*" + (f" — {directions(target)}" if directions else ""))
        c1, c2, _ = st.columns(3)
        c1.metric("MAE",  f"{mae_mean:.3f}",  f"± {mae_std:.3f}")
        c2.metric("RMSE", f"{rmse_mean:.3f}", f"± {rmse_std:.3f}")
        elite_mean, elite_std = metrics[target].get("elite_mae", (np.nan, np.nan))
        if not pd.isna(elite_mean):
            st.caption(f"Elite MAE (top 10% actual): {elite_mean:.3f} ± {elite_std:.3f}")


def _def_direction(target):
    return "↓ lower is better" if target in DEF_LOWER_IS_BETTER else "↑ higher is better"


def render(fwd, dfn):
    st.markdown("#### Offensive Model Cache")
    if st.button("Retrain offensive model (deletes cache)"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        state.forget_models(state.FWD_KEY)
        st.success("Offensive cache cleared — refresh the page to retrain.")

    st.divider()
    with st.expander("Team Fit model quality", expanded=True):
        show_metrics(fwd.fit_metrics, "Team Fit", TARGETS, TARGET_LABELS)
    with st.expander("Next Season model quality"):
        show_metrics(fwd.next_metrics, "Next Season", TARGETS, TARGET_LABELS)
    with st.expander("Team Fit — feature importance"):
        st.pyplot(charts.importance_chart(fwd.fit_models, fwd.fit_feature_names))
    with st.expander("Next Season — feature importance"):
        st.pyplot(charts.importance_chart(fwd.next_models, fwd.next_feature_names))

    if dfn is None:
        return
    st.divider()
    st.markdown("#### Defensive Model Cache")
    if st.button("Retrain defensive models"):
        if os.path.exists(DEF_CACHE_FILE):
            os.remove(DEF_CACHE_FILE)
        state.forget_models(state.DEF_KEY)
        for k in [k for k in st.session_state if k.startswith("dpred_")]:
            del st.session_state[k]
        st.success("Defensive cache cleared — refresh to retrain.")
    with st.expander("Defensive Current Fit quality", expanded=True):
        show_metrics(dfn.fit_metrics, "Defensive Current Fit", DEF_TARGETS, DEF_TARGET_LABELS, _def_direction)
    with st.expander("Defensive Next Season quality"):
        show_metrics(dfn.next_metrics, "Defensive Next Season", DEF_TARGETS, DEF_TARGET_LABELS, _def_direction)
