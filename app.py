"""
app.py
======
NHL Player Predictor — Streamlit UI entry point.
All model logic lives in model_utils.py.

Run with:  python -m streamlit run app.py
"""

import streamlit as st
from model_utils import *

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="NHL Player Predictor", page_icon="🏒", layout="wide")
st.title("NHL Player Predictor")
st.caption("Forwards-only mode: defensemen are excluded from training and predictions.")

if "fit_models" not in st.session_state:
    if os.path.exists(CACHE_FILE):
        with st.spinner("Loading saved models from disk..."):
            cached = joblib.load(CACHE_FILE)
            (
                st.session_state["df"],
                st.session_state["team_ctx"],
                st.session_state["has_age"],
                st.session_state["player_profiles"],
                st.session_state["fit_models"],
                st.session_state["fit_metrics"],
                st.session_state["fit_feature_names"],
                st.session_state["next_models"],
                st.session_state["next_metrics"],
                st.session_state["next_feature_names"],
            ) = cached
    else:
        st.info("Training models for the first time — this takes 5–8 minutes. Won't happen again until you retrain.")
        results = load_and_train_with_progress(DATA_FILE, AGES_FILE)
        joblib.dump(results, CACHE_FILE)
        (
            st.session_state["df"],
            st.session_state["team_ctx"],
            st.session_state["has_age"],
            st.session_state["player_profiles"],
            st.session_state["fit_models"],
            st.session_state["fit_metrics"],
            st.session_state["fit_feature_names"],
            st.session_state["next_models"],
            st.session_state["next_metrics"],
            st.session_state["next_feature_names"],
        ) = results
        st.rerun()

df                 = st.session_state["df"]
team_ctx           = st.session_state["team_ctx"]
has_age            = st.session_state["has_age"]
player_profiles    = st.session_state["player_profiles"]
fit_models         = st.session_state["fit_models"]
fit_metrics        = st.session_state["fit_metrics"]
fit_feature_names  = st.session_state["fit_feature_names"]
next_models        = st.session_state["next_models"]
next_metrics       = st.session_state["next_metrics"]
next_feature_names = st.session_state["next_feature_names"]

# ── Load / train defensive model ───────────────────────────────────────────────
if "def_fit_models" not in st.session_state:
    if os.path.exists(DEF_CACHE_FILE):
        with st.spinner("Loading saved defensive models..."):
            def_cached = joblib.load(DEF_CACHE_FILE)
            (
                st.session_state["def_df"],
                st.session_state["def_team_ctx"],
                st.session_state["def_has_age"],
                st.session_state["def_player_profiles"],
                st.session_state["def_fit_models"],
                st.session_state["def_fit_metrics"],
                st.session_state["def_fit_feature_names"],
                st.session_state["def_next_models"],
                st.session_state["def_next_metrics"],
                st.session_state["def_next_feature_names"],
            ) = def_cached

    elif os.path.exists(DEF_FILE):
        st.info("Training defensive models for the first time — takes 3-5 minutes.")
        def_results = def_load_and_train(DEF_FILE, AGES_FILE)
        joblib.dump(def_results, DEF_CACHE_FILE)
        (
            st.session_state["def_df"],
            st.session_state["def_team_ctx"],
            st.session_state["def_has_age"],
            st.session_state["def_player_profiles"],
            st.session_state["def_fit_models"],
            st.session_state["def_fit_metrics"],
            st.session_state["def_fit_feature_names"],
            st.session_state["def_next_models"],
            st.session_state["def_next_metrics"],
            st.session_state["def_next_feature_names"],
        ) = def_results
        st.rerun()
    else:
        st.session_state["def_fit_models"] = None

def_models_loaded = st.session_state.get("def_fit_models") is not None

if def_models_loaded:
    def_df                = st.session_state["def_df"]
    def_team_ctx          = st.session_state["def_team_ctx"]
    def_has_age           = st.session_state["def_has_age"]
    def_player_profiles   = st.session_state["def_player_profiles"]
    def_fit_models        = st.session_state["def_fit_models"]
    def_fit_metrics       = st.session_state["def_fit_metrics"]
    def_fit_feature_names = st.session_state["def_fit_feature_names"]
    def_next_models       = st.session_state["def_next_models"]
    def_next_metrics      = st.session_state["def_next_metrics"]
    def_next_feature_names= st.session_state["def_next_feature_names"]

if has_age:
    st.caption("Age data loaded ✅ — next-season forecasting active.")
else:
    st.caption("⚠️ Age data not found — next-season model running without age features.")

# ── Player search (shared across tabs) ────────────────────────────────────────
st.divider()
# Combine forwards (from offensive df) and defensemen (from defensive df)
_fwd_names = set(df["player_name"].unique())
_def_names = set(def_df["player_name"].unique()) if def_models_loaded else set()
all_players = sorted(_fwd_names | _def_names)
player_input = st.selectbox("Search for a player", options=[""] + all_players, index=0)

pred = None
if player_input:
    # Try offensive model first (forwards)
    first = predict_player(player_input, df, team_ctx, fit_models, next_models,
                           player_profiles, has_age)

    if first is None and def_models_loaded:
        # Try defensive model (defensemen)
        def_first = def_predict_defenseman(
            player_input, def_df, def_team_ctx,
            def_fit_models, def_next_models, def_player_profiles, def_has_age,
            fit_feature_names=def_fit_feature_names,
            next_feature_names=def_next_feature_names
        )
        if def_first is not None:
            # Create a minimal pred dict so tabs know a defenseman is searched
            pred = {
                "pid":          def_first["pid"],
                "matched":      def_first["matched"],
                "actual_team":  def_first["actual_team"],
                "position":     "D",
                "seasons":      def_first["seasons"],
                "traded_teams": [],
                "fit_results":  None,
                "next_results": None,
                "age":          None,
            }
        else:
            st.error(f"No player found matching '{player_input}'.")
    elif first is None:
        st.error(f"No player found matching '{player_input}'.")
    else:
        override_team = None
        if first["traded_teams"]:
            st.warning(
                f"{first['matched']} was traded in {first['seasons'][0]}. "
                f"Select their current team:"
            )
            override_team = st.radio(
                "Current team", options=first["traded_teams"],
                horizontal=True, key="team_override"
            )
        pred = predict_player(player_input, df, team_ctx, fit_models, next_models,
                              player_profiles, has_age, override_team=override_team)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_off, tab_def, tab_contract, tab_model, tab_val = st.tabs([
    "Offensive",
    "Defensive",
    "Contract Evaluator",
    "Model Info",
    "Validation",
])

# ── Offensive (nested) ────────────────────────────────────────────────────────
with tab_off:
    off_t1, off_t2, off_t3 = st.tabs(["Team Fit", "Next Season", "Roster Insertion"])

    with off_t1:
        if pred and pred.get("fit_results") is not None:
            seasons_str = " → ".join(str(s) for s in pred["seasons"])
            age_str     = f"  |  Age {pred['age']:.0f}" if pred.get("age") else ""
            st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}  |  Seasons: {seasons_str}")
            st.caption("Predicted performance based on current weighted skill profile across all 32 teams.")
            st.pyplot(make_bar_chart(pred["fit_results"], pred["matched"], pred["actual_team"],
                                     f"{pred['matched']}  |  Current skill profile  |  Seasons: {seasons_str}"))
            st.markdown("#### Rankings Table")
            display = show_results_table(pred["fit_results"], pred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{pred['matched'].replace(' ','_')}_team_fit.csv",
                               mime="text/csv")
        else:
            st.info("Search for a forward above to see predictions.")

    with off_t2:
        if pred and pred.get("next_results") is not None:
            age_str = f"  |  Age {pred['age']:.0f} → {pred['age']+1:.0f}" if pred.get("age") else ""
            st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}")
            st.caption("Predicted next-season performance across all 32 teams.")
            st.pyplot(make_bar_chart(pred["next_results"], pred["matched"], pred["actual_team"],
                                     f"{pred['matched']}  |  Next season forecast"))
            st.markdown("#### Rankings Table")
            display = show_results_table(pred["next_results"], pred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{pred['matched'].replace(' ','_')}_next_season.csv",
                               mime="text/csv")
        else:
            st.info("Search for a forward above to see predictions.")

    with off_t3:
        st.caption("Select a team to see where the searched player would slot into their active roster.")
        if not pred or pred.get("position") == "D":
            st.info("Search for a forward above to use this tab.")
        else:
            c1, c2 = st.columns([1, 1])
            insertion_team = c1.selectbox(
                "Select team to insert player into",
                options=NHL_TEAMS,
                index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
                key="insertion_team"
            )
            if c2.button("Refresh roster"):
                fetch_active_team_roster.clear()

            with st.spinner(f"Building {insertion_team} roster with {pred['matched']} inserted..."):
                insertion_df, insertion_err = build_player_insertion(
                    pred["pid"], insertion_team, df, team_ctx,
                    fit_models, player_profiles, has_age
                )

            if insertion_err:
                st.error(insertion_err)
            elif insertion_df is not None and not insertion_df.empty:
                searched_row = insertion_df[insertion_df["is_searched_player"]].iloc[0]
                slot   = searched_row["lineup_slot"]
                rank   = int(searched_row["rank"])
                color  = searched_row["slot_color"]
                total  = len(insertion_df)
                st.markdown(
                    f"<h3 style='color:{color}'>"
                    f"{pred['matched']} projects as a <b>{slot}</b> player on {insertion_team} "
                    f"(rank {rank} of {total} {'forwards' if pred['position'] in ('C','L','R') else 'defensemen'})"
                    f"</h3>",
                    unsafe_allow_html=True
                )
                display = insertion_df[[
                    "rank","player_name","position","lineup_slot","pred_points_gp","pred_goals_gp"
                ]].copy()
                display.columns = ["Rank","Player","Pos","Line/Pair","Points/GP","Goals/GP"]
                def _hi_searched(row):
                    m = insertion_df.loc[insertion_df["player_name"]==row["Player"],"is_searched_player"].values
                    if len(m)>0 and m[0]:
                        return [f"background-color:{color}22;font-weight:bold;border-left:3px solid {color}"]*len(row)
                    return [""]*len(row)
                st.dataframe(display.style.apply(_hi_searched,axis=1),
                             use_container_width=True,
                             height=min(50+len(display)*35,600))
                slot_counts = display["Line/Pair"].value_counts()
                is_fwd = pred["position"] in ("C","L","R")
                slot_labels = (["1st Line","2nd Line","3rd Line","4th Line"] if is_fwd
                               else ["1st Pair","2nd Pair","3rd Pair","3rd Pair (extra)"])
                st.markdown("**Roster slot breakdown after insertion:**")
                cols_slots = st.columns(4)
                for col_s, lbl in zip(cols_slots, slot_labels):
                    col_s.metric(lbl, int(slot_counts.get(lbl,0)))
                csv = insertion_df.drop(columns=["slot_color"]).to_csv(index=False)
                st.download_button("Download roster insertion CSV", data=csv,
                                   file_name=f"{pred['matched'].replace(' ','_')}_{insertion_team}_insertion.csv",
                                   mime="text/csv")

# ── Defensive (nested) ────────────────────────────────────────────────────────
with tab_def:
    is_d = pred is not None and pred.get("position") == "D"

    dpred = None
    if is_d and def_models_loaded:
        _dpred_key = f"dpred_{pred['pid']}"
        if _dpred_key not in st.session_state:
            with st.spinner("Computing defensive predictions..."):
                st.session_state[_dpred_key] = def_predict_defenseman(
                    pred["matched"], def_df, def_team_ctx,
                    def_fit_models, def_next_models, def_player_profiles, def_has_age,
                    fit_feature_names=def_fit_feature_names,
                    next_feature_names=def_next_feature_names
                )
        dpred = st.session_state[_dpred_key]

    def_t1, def_t2, def_t3 = st.tabs(["Team Fit", "Next Season", "Pairing"])

    with def_t1:
        if not def_models_loaded:
            st.warning("Defensive model not loaded. Ensure defensive_dataset.csv is present.")
        elif not pred:
            st.info("Search for a defenseman above.")
        elif not is_d:
            st.info(f"{pred['matched']} is a forward. Search for a defenseman to use defensive tabs.")
        elif dpred:
            seasons_str = " → ".join(str(s) for s in dpred["seasons"])
            st.subheader(f"{dpred['matched']}  —  D  |  {dpred['actual_team']}  |  Seasons: {seasons_str}")
            st.caption("Defensive Score: xGA suppression 30%, takeaways 20%, PK usage 20%, hits 15%, discipline 15%.")
            st.markdown("#### Rankings Table")
            display = def_show_results_table(dpred["fit_results"], dpred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{dpred['matched'].replace(' ','_')}_def_fit.csv",
                               mime="text/csv")

    with def_t2:
        if not def_models_loaded:
            st.warning("Defensive model not loaded.")
        elif not pred or not is_d:
            st.info("Search for a defenseman above.")
        elif dpred:
            st.subheader(f"{dpred['matched']}  —  D  |  {dpred['actual_team']}")
            st.caption("Next-season defensive forecast based on current profile and trajectory.")
            st.markdown("#### Rankings Table")
            display = def_show_results_table(dpred["next_results"], dpred["actual_team"])
            csv = display.drop(columns="Actual Team").to_csv(index_label="rank")
            st.download_button("Download CSV", data=csv,
                               file_name=f"{dpred['matched'].replace(' ','_')}_def_next.csv",
                               mime="text/csv")

    with def_t3:
        if not def_models_loaded:
            st.warning("Defensive model not loaded.")
        elif not pred or not is_d:
            st.info("Search for a defenseman above.")
        elif dpred:
            pc1, pc2 = st.columns([1, 1])
            pair_team = pc1.selectbox(
                "Select team",
                options=NHL_TEAMS,
                index=NHL_TEAMS.index(dpred["actual_team"]) if dpred["actual_team"] in NHL_TEAMS else 0,
                key="pair_team_sel"
            )
            if pc2.button("Refresh roster & shifts", key="pair_refresh"):
                def_fetch_team_roster_d.clear()
                fetch_actual_pairs.clear()

            with st.spinner(f"Fetching {pair_team} shifts and building pairings..."):
                pair_result = def_build_pairing_insertion(
                    dpred["pid"], pair_team, def_df, def_team_ctx,
                    def_fit_models, def_player_profiles, def_has_age,
                    feature_names=def_fit_feature_names
                )

            # New return format: (current_pairs, unassigned, player_scores, insertion_info)
            if isinstance(pair_result, tuple) and len(pair_result) == 4:
                current_pairs, unassigned, player_scores, insertion = pair_result

                if isinstance(current_pairs, str):
                    # Error string returned as first element
                    st.error(current_pairs)
                else:
                    searched_name  = dpred["matched"]
                    searched_score = insertion["searched_score"]
                    partner_name   = insertion["partner_name"]
                    partner_slot   = insertion["partner_slot"]
                    pushed_out     = insertion["pushed_out_name"]
                    pair_err       = insertion.get("pair_err")
                    missing_model_players = insertion.get("missing_model_players", [])

                    if pair_err:
                        st.caption(f"Note: Could not fetch shift data ({pair_err}). Showing model-based ranking.")

                    # ── Insertion summary ──────────────────────────────────────
                    st.markdown(
                        f"### {searched_name} — Defensive Score: {searched_score:.1f}",
                    )
                    if partner_name != "—":
                        st.success(
                            f"Best fit: **{partner_slot}** alongside **{partner_name}**  \n"
                            f"{'Replaces: **' + pushed_out + '**' if pushed_out != '—' else ''}"
                        )

                    # ── Current pairs table ────────────────────────────────────
                    st.divider()
                    st.markdown(f"#### {pair_team} Current Pairings (with model scores)")
                    source_note = "from recent shift data" if any(p.get("from_shifts") for p in current_pairs) else "model-ranked"
                    st.caption(f"Pairs derived {source_note}. Defensive Score: 0-100 composite (higher = better).")

                    SLOT_COLORS_PAIRS = {
                        "1st Pair": "#FFD700",
                        "2nd Pair": "#4a90d9",
                        "3rd Pair": "#57a85a",
                        "4th Pair": "#888888",
                    }

                    for i, pair in enumerate(current_pairs):
                        slot_label = pair.get("slot", f"Pair {i+1}")
                        color      = SLOT_COLORS_PAIRS.get(slot_label.split()[0] + " " + slot_label.split()[1] if len(slot_label.split()) >= 2 else slot_label, "#888888")
                        is_target_pair = (pair.get("pid1") == (insertion.get("best_partner_pid")) or
                                          pair.get("pid2") == (insertion.get("best_partner_pid")))

                        col_slot, col_p1, col_vs, col_p2, col_score = st.columns([1.2, 2, 0.3, 2, 1.2])
                        col_slot.markdown(f"<span style='color:{color};font-weight:bold'>{slot_label}</span>", unsafe_allow_html=True)
                        col_p1.markdown(f"**{pair['name1']}** ({pair['score1']:.0f})")
                        col_vs.markdown("—")
                        col_p2.markdown(f"**{pair['name2']}** ({pair['score2']:.0f})")
                        col_score.markdown(f"Pair avg: **{pair['pair_score']:.0f}**")

                    # ── Searched player row ────────────────────────────────────
                    st.divider()
                    st.markdown("#### Where the searched player fits")
                    sp_col1, sp_col2, sp_col3 = st.columns([2, 2, 2])
                    sp_col1.metric("Player", searched_name)
                    sp_col2.metric("Defensive Score", f"{searched_score:.1f}")
                    sp_col3.metric("Best Pair Slot", partner_slot)

                    if partner_name != "—":
                        st.info(
                            f"**{searched_name}** (score {searched_score:.0f}) would pair best with "
                            f"**{partner_name}** in the **{partner_slot}**."
                            + (f"  \nThis would push **{pushed_out}** to a different role." if pushed_out != "—" else "")
                        )

                    # ── Coverage diagnostics ───────────────────────────────────
                    if missing_model_players:
                        with st.expander(f"{len(missing_model_players)} rostered D-men not in model data"):
                            for p in missing_model_players:
                                st.caption(f"• {p.get('player_name', str(p.get('player_id')))} — not enough historical seasons in training data")

                    if unassigned:
                        with st.expander(f"{len(unassigned)} rostered D-men in model but not found in recent shift pair data"):
                            for pid in unassigned:
                                name = player_scores.get(pid, {}).get("player_name", str(pid))
                                st.caption(f"• {name} — no recent shared-shift pairing found")

                    # ── Full scores table ──────────────────────────────────────
                    with st.expander("Full model scores for all D-men"):
                        rows_table = []
                        for pid, info in player_scores.items():
                            rows_table.append({
                                "Player":    info["player_name"],
                                "Def Score": info["defensive_score"],
                                "Hits/GP":   round(info.get("ind_hits_pg", 0), 2),
                                "TK/GP":     round(info.get("ind_takeaways_pg", 0), 3),
                                "GA/GP":     round(info.get("goals_against_per_game", 0), 3),
                                "PK%":       round(info.get("pk_ice_pct", 0) * 100, 1),
                                "PIM/GP":    round(info.get("pim_pg", 0), 2),
                                "Is New Player": info.get("is_searched_player", False),
                            })
                        scores_df = pd.DataFrame(rows_table).sort_values("Def Score", ascending=False)
                        display_scores_df = scores_df.drop(columns=["Is New Player"], errors="ignore")

                        def _hi_new(row):
                            is_new = False
                            if "Is New Player" in scores_df.columns:
                                is_new = bool(scores_df.loc[row.name, "Is New Player"])
                            if is_new:
                                return ["background-color:#FFD70022;font-weight:bold"] * len(row)
                            return [""] * len(row)

                        st.dataframe(
                            display_scores_df.style.apply(_hi_new, axis=1),
                            use_container_width=True,
                            hide_index=True,
                        )

                        csv_rows = pd.DataFrame(rows_table).to_csv(index=False)
                        st.download_button("Download pairing CSV", data=csv_rows,
                                           file_name=f"{dpred['matched'].replace(' ','_')}_{pair_team}_pairing.csv",
                                           mime="text/csv")
            else:
                # Fallback for unexpected return format
                st.error("Unexpected result from pairing function. Try refreshing.")

# ── Contract Evaluator ────────────────────────────────────────────────────────
with tab_contract:
    st.subheader("Contract Evaluator")
    st.caption(
        "Projects a player's production across multiple seasons using empirical age curves. "
        "Works for both forwards (offensive stats) and defensemen (defensive stats). "
        "Confidence decreases in later years — use ranges rather than exact numbers."
    )

    if not pred:
        st.info("Search for a player above to use the contract evaluator.")
    else:
        is_d_contract = pred.get("position") == "D"

        # ── Controls ───────────────────────────────────────────────────────────
        cc1, cc2, cc3 = st.columns(3)
        contract_team = cc1.selectbox(
            "Team signing the player",
            options=NHL_TEAMS,
            index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
            key="contract_team"
        )
        # Resolve current age — try pred dict, then profile, then ages CSV directly
        _pid = pred.get("pid")
        curr_age = pred.get("age")

        if not curr_age and is_d_contract and def_models_loaded:
            # Defensive profiles may not carry age if model trained without age features
            _prof = def_player_profiles.get(_pid, (None,))[0]
            if _prof is not None:
                curr_age = _prof.get("age")

        if not curr_age and os.path.exists(AGES_FILE):
            # Direct lookup from player_ages.csv — most reliable source
            try:
                _ages_df = pd.read_csv(AGES_FILE)
                _row = _ages_df[_ages_df["player_id"] == _pid].sort_values("season", ascending=False)
                if not _row.empty and pd.notna(_row.iloc[0].get("age")):
                    _latest_season = int(_row.iloc[0]["season"])
                    _base_age = float(_row.iloc[0]["age"])
                    # Adjust to current season if the latest row isn't current
                    curr_age = _base_age + max(0, 2026 - _latest_season)
            except Exception:
                pass

        curr_age = float(curr_age) if curr_age and not (isinstance(curr_age, float) and curr_age != curr_age) else 28.0

        # CBA limits based on signing team
        cba = get_cba_limits(curr_age, pred["actual_team"], contract_team)

        n_years = cc2.slider(
            "Contract length (years)",
            min_value=1,
            max_value=cba["max_years"],
            value=min(cba["recommended"], cba["max_years"]),
        )
        cc3.metric("Current Age", f"{curr_age:.0f}")

        # ── CBA Info bar ───────────────────────────────────────────────────────
        cba_cols = st.columns(4)
        signing_type = "Re-signing (same team)" if cba["is_same_team"] else "New signing (different team)"
        cba_cols[0].metric("Signing Type",    signing_type)
        cba_cols[1].metric("CBA Max Length",  f"{cba['max_years']} years")
        cba_cols[2].metric("Recommended Max", f"{cba['recommended']} years")
        cba_cols[3].metric("Age at Expiry",   f"{cba['age_at_expiry']:.0f}")

        # 35+ rule warning
        if cba["is_35_signing"]:
            st.error(
                f"35+ Rule: {pred['matched']} is {curr_age:.0f} at signing. "
                "The cap hit counts against your team even if the player retires early. "
                "This creates significant cap recapture risk."
            )
        elif cba["hits_35_rule"]:
            st.warning(
                f"35+ Rule alert: {pred['matched']} will reach age 35 during this contract. "
                "If they retire before expiry, the cap hit remains on your books."
            )

        if n_years > cba["recommended"]:
            st.warning(
                f"This contract is longer than the recommended maximum of {cba['recommended']} years "
                f"based on the player's age curve. Years {cba['recommended']+1}+ carry high uncertainty."
            )

        # ── Run projection ─────────────────────────────────────────────────────
        with st.spinner("Projecting contract years..."):
            proj_rows, proj_err = build_contract_projection(
                pred["matched"], pred, dpred if is_d_contract else None,
                df, team_ctx, fit_models, next_models, player_profiles, has_age,
                def_df if def_models_loaded else pd.DataFrame(),
                def_team_ctx if def_models_loaded else pd.DataFrame(),
                def_fit_models if def_models_loaded else {},
                def_player_profiles if def_models_loaded else {},
                def_has_age if def_models_loaded else False,
                contract_team, n_years,
                def_fit_feature_names=def_fit_feature_names if def_models_loaded else None
            )

        if proj_err:
            st.error(proj_err)
        elif proj_rows:
            risk_label, risk_color, risk_explanation = contract_risk_rating(proj_rows, is_d_contract)

            # ── Risk header ────────────────────────────────────────────────────
            st.markdown(
                f"<h3>{pred['matched']} on {contract_team} — "
                f"<span style='color:{risk_color}'>{risk_label}</span></h3>",
                unsafe_allow_html=True
            )
            if risk_explanation:
                st.caption(risk_explanation)

            # ── Year-by-year table ─────────────────────────────────────────────
            st.markdown("#### Year-by-Year Projection")
            st.caption("Confidence reflects uncertainty compounding over time. Use wider mental ranges in later years.")

            if is_d_contract:
                proj_df = pd.DataFrame([{
                    "Year":          f"Year {r['year']} (Age {r['age']:.0f})",
                    "Hits/GP":       r["hits_pg"],
                    "Takeaways/GP":  r["takeaways_pg"],
                    "GA/GP (5v5)":   r["goals_against_pg"],
                    "PK%":           f"{r['pk_pct']*100:.1f}%",
                    "PIM/GP":        r["pim_pg"],
                    "Def Score":     r["def_score"],
                    "Confidence":    f"{r['confidence']*100:.0f}%",
                } for r in proj_rows])
            else:
                proj_df = pd.DataFrame([{
                    "Year":         f"Year {r['year']} (Age {r['age']:.0f})",
                    "Points/GP":    r["points_pg"],
                    "Goals/GP":     r["goals_pg"],
                    "Game Score/GP":r["gs_pg"],
                    "Confidence":   f"{r['confidence']*100:.0f}%",
                } for r in proj_rows])

            st.dataframe(proj_df, use_container_width=True, hide_index=True)

            # ── Trend chart ────────────────────────────────────────────────────
            st.markdown("#### Production Trend")
            fig_c, ax_c = plt.subplots(figsize=(10, 4))
            fig_c.patch.set_facecolor("#0e1117")
            ax_c.set_facecolor("#0e1117")

            years  = [r["year"] for r in proj_rows]
            labels = [f"Yr {r['year']} (Age {r['age']:.0f})" for r in proj_rows]

            if is_d_contract:
                vals   = [r["def_score"] for r in proj_rows]
                ylabel = "Defensive Score (0-100)"
                color  = "#4a90d9"
            else:
                vals   = [r["points_pg"] for r in proj_rows]
                ylabel = "Points / Game"
                color  = "#4a90d9"

            confs = [r["confidence"] for r in proj_rows]

            # Plot line with confidence band
            ax_c.plot(years, vals, color=color, linewidth=2.5, marker="o", markersize=8, zorder=3)
            # Confidence band — widen with lower confidence
            spread = [v * (1 - c) * 0.5 for v, c in zip(vals, confs)]
            ax_c.fill_between(years,
                              [v - s for v, s in zip(vals, spread)],
                              [v + s for v, s in zip(vals, spread)],
                              alpha=0.2, color=color, label="Confidence range")
            ax_c.set_xticks(years)
            ax_c.set_xticklabels(labels, color="white", fontsize=9)
            ax_c.set_ylabel(ylabel, color="white", fontsize=10)
            ax_c.tick_params(colors="white")
            ax_c.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
            for spine in ax_c.spines.values():
                spine.set_edgecolor("#333")
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close()

            # ── Summary recommendation ─────────────────────────────────────────
            st.divider()
            st.markdown("#### Contract Recommendation")
            yr1     = proj_rows[0]
            yr_last = proj_rows[-1]

            if is_d_contract:
                y1_score    = yr1["def_score"]
                yn_score    = yr_last["def_score"]
                decline     = max(y1_score - yn_score, 0)
                decline_pct = max((y1_score - yn_score) / max(y1_score, 1) * 100, 0)
                st.markdown(
                    f"- **Year 1 Defensive Score:** {y1_score:.1f} / 100  \n"
                    f"- **Year {n_years} Defensive Score:** {yn_score:.1f} / 100  \n"
                    f"- **Projected decline:** {decline:.1f} pts ({decline_pct:.0f}%)  \n"
                    f"- **Risk rating:** {risk_label}"
                )
            else:
                y1_pts    = yr1["points_pg"]
                yn_pts    = yr_last["points_pg"]
                total_pts = sum(r["points_pg"] * 82 for r in proj_rows)
                st.markdown(
                    f"- **Year 1 Points/GP:** {y1_pts:.3f}  \n"
                    f"- **Year {n_years} Points/GP:** {yn_pts:.3f}  \n"
                    f"- **Projected total points:** ~{total_pts:.0f} over {n_years} years  \n"
                    f"- **Risk rating:** {risk_label}"
                )

            # CBA recommendation box
            st.markdown("**CBA Summary:**")
            rec_col1, rec_col2 = st.columns(2)
            signing_lbl = "Same team (re-signing)" if cba["is_same_team"] else "New team"
            rec_col1.info(
                f"Same-team max: **7 years**  \n"
                f"New-team max: **6 years**  \n"
                f"Signing type: **{signing_lbl}**  \n"
                f"CBA max for this deal: **{cba['max_years']} years**"
            )
            rule35_note = "35+ rule applies — cap recapture risk if player retires early." if cba["hits_35_rule"] else "No 35+ rule concerns."
            rec_col2.success(
                f"Model recommendation: **{cba['recommended']} years**  \n"
                f"Based on age {curr_age:.0f} trajectory.  \n"
                f"{rule35_note}"
            )

            # Download
            csv = proj_df.to_csv(index=False)
            st.download_button(
                "Download contract projection CSV", data=csv,
                file_name=f"{pred['matched'].replace(' ','_')}_{contract_team}_{n_years}yr_contract.csv",
                mime="text/csv"
            )

# ── Model Info ────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("#### Offensive Model Cache")
    if st.button("Retrain offensive model (deletes cache)"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        for key in ["df","team_ctx","has_age","player_profiles","fit_models",
                    "fit_metrics","fit_feature_names","next_models","next_metrics","next_feature_names"]:
            st.session_state.pop(key, None)
        st.success("Offensive cache cleared — refresh the page to retrain.")

    st.divider()
    with st.expander("Team Fit model quality", expanded=True):
        show_metrics(fit_metrics, "Team Fit")
    with st.expander("Next Season model quality"):
        show_metrics(next_metrics, "Next Season")
    with st.expander("Team Fit — feature importance"):
        st.pyplot(make_importance_chart(fit_models, fit_feature_names))
    with st.expander("Next Season — feature importance"):
        st.pyplot(make_importance_chart(next_models, next_feature_names))

    if def_models_loaded:
        st.divider()
        st.markdown("#### Defensive Model Cache")
        if st.button("Retrain defensive models"):
            if os.path.exists(DEF_CACHE_FILE):
                os.remove(DEF_CACHE_FILE)
            for k in list(st.session_state.keys()):
                if k.startswith("def_") or k.startswith("dpred_"):
                    del st.session_state[k]
            st.success("Defensive cache cleared — refresh to retrain.")
        with st.expander("Defensive Current Fit quality", expanded=True):
            def_show_metrics(def_fit_metrics, "Defensive Current Fit")
        with st.expander("Defensive Next Season quality"):
            def_show_metrics(def_next_metrics, "Defensive Next Season")

# ── Validation (nested) ───────────────────────────────────────────────────────
with tab_val:
    val_t1, val_t2 = st.tabs(["Offensive", "Defensive"])

    with val_t1:
        st.subheader("2025-26 Offensive Validation")
        st.caption(
            "Pulls live 2025-26 stats from the NHL API and compares against "
            "the model predictions. Points and Goals converted to per-game rates."
        )
        if st.button("Refresh NHL API stats"):
            st.cache_data.clear()
            for k in ["edge_weighted_shots"]:
                st.session_state.pop(k, None)
            st.rerun()

        actual_df, err = fetch_nhl_current_season()
        if err:
            st.error(f"Could not fetch NHL API data: {err}")
        elif actual_df is not None:
            st.success(f"Fetched {len(actual_df):,} skaters with 10+ games played.")
            # weighted_shots_pg not used in validation — points and goals only

            with st.spinner("Comparing predictions to actual stats..."):
                val_df = build_validation_results(
                    actual_df, df, team_ctx, fit_models, player_profiles, has_age
                )
            if val_df.empty:
                st.warning("No players matched between NHL API and model profiles.")
            else:
                st.markdown(f"**{len(val_df):,} players matched**")
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.patch.set_facecolor("#0e1117")
                make_scatter(val_df, "actual_points_gp", "pred_points_gp", "Points / Game",  axes[0])
                make_scatter(val_df, "actual_goals_gp",  "pred_goals_gp",  "Goals / Game",   axes[1])
                plt.tight_layout()
                st.pyplot(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("Points/GP MAE",
                          f"{mean_absolute_error(val_df['actual_points_gp'], val_df['pred_points_gp']):.3f}")
                c2.metric("Goals/GP MAE",
                          f"{mean_absolute_error(val_df['actual_goals_gp'], val_df['pred_goals_gp']):.3f}")
                c3.metric("Players matched", f"{len(val_df):,}")

                pred_range    = val_df["pred_points_gp"].max() - val_df["pred_points_gp"].min()
                actual_range  = val_df["actual_points_gp"].max() - val_df["actual_points_gp"].min()
                compression   = pred_range / actual_range if actual_range > 0 else np.nan
                slope_val     = calibration_slope(val_df, "actual_points_gp", "pred_points_gp")
                c3b, c4b = st.columns(2)
                c3b.metric("Prediction Spread Ratio", f"{compression:.2%}" if not pd.isna(compression) else "n/a")
                c4b.metric("Calibration Slope",       f"{slope_val:.2f}"   if not pd.isna(slope_val)   else "n/a")

                elite_pts_mae, elite_pts_bias, elite_pts_n = elite_segment_stats(
                    val_df, "actual_points_gp", "pred_points_gp", quantile=ELITE_QUANTILE)
                elite_goals_mae, elite_goals_bias, elite_goals_n = elite_segment_stats(
                    val_df, "actual_goals_gp", "pred_goals_gp", quantile=ELITE_QUANTILE)
                e1, e2 = st.columns(2)
                e1.metric(f"Elite Points/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                          f"{elite_pts_mae:.3f}" if not pd.isna(elite_pts_mae) else "n/a",
                          f"bias {elite_pts_bias:+.3f}" if not pd.isna(elite_pts_bias) else None)
                e2.metric(f"Elite Goals/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                          f"{elite_goals_mae:.3f}" if not pd.isna(elite_goals_mae) else "n/a",
                          f"bias {elite_goals_bias:+.3f}" if not pd.isna(elite_goals_bias) else None)
                st.caption(f"Elite sample sizes: Points {elite_pts_n}, Goals {elite_goals_n}")
                st.divider()
                st.markdown("#### Biggest Misses")
                misses = val_df.reindex(val_df["points_gp_error"].abs().nlargest(15).index)[
                    ["player_name","team","games_played","actual_points_gp",
                     "pred_points_gp","points_gp_error","actual_goals_gp","pred_goals_gp","seasons_used"]]
                st.dataframe(misses, use_container_width=True)
                st.markdown("#### Best Predictions")
                best = val_df.reindex(val_df["points_gp_error"].abs().nsmallest(15).index)[
                    ["player_name","team","games_played","actual_points_gp",
                     "pred_points_gp","points_gp_error","actual_goals_gp","pred_goals_gp","seasons_used"]]
                st.dataframe(best, use_container_width=True)
                csv = val_df.to_csv(index=False)
                st.download_button("Download full validation CSV", data=csv,
                                   file_name="validation_2025_26.csv", mime="text/csv")

    with val_t2:
        st.subheader("2025-26 Defensive Validation")
        st.caption(
            "Compares defensive model predictions against 2025-26 actual stats "
            "from the NHL API realtime endpoint (hits and takeaways per game)."
        )
        if st.button("Refresh defensive stats"):
            fetch_nhl_defensive_stats.cache_clear()

        if not def_models_loaded:
            st.warning("Defensive model not loaded.")
        else:
            def_actual, def_err = fetch_nhl_defensive_stats()
            if def_err:
                st.error(f"Could not fetch NHL API data: {def_err}")
            elif def_actual is not None:
                def_actual_d = def_actual[
                    def_actual["player_id"].isin(def_player_profiles.keys())
                ].copy()
                st.success(f"Fetched {len(def_actual_d):,} defensemen with 10+ games played.")
                with st.spinner("Comparing predictions to actual stats..."):
                    def_val_df = build_defensive_validation(
                        def_actual_d, def_df, def_team_ctx,
                        def_fit_models, def_player_profiles, def_has_age,
                        feature_names=def_fit_feature_names
                    )
                if def_val_df.empty:
                    st.warning("No defensemen matched between NHL API and model profiles.")
                else:
                    st.markdown(f"**{len(def_val_df):,} defensemen matched**")

                    has_pk  = "actual_pk_pct" in def_val_df.columns and def_val_df["actual_pk_pct"].sum() > 0
                    has_pim = "actual_pim_pg" in def_val_df.columns and def_val_df["actual_pim_pg"].sum() > 0

                    metric_cols = st.columns(5)
                    metric_cols[0].metric("Hits/GP MAE",
                              f"{mean_absolute_error(def_val_df['actual_hits_pg'], def_val_df['pred_hits_pg']):.3f}")
                    metric_cols[1].metric("Takeaways/GP MAE",
                              f"{mean_absolute_error(def_val_df['actual_tk_pg'], def_val_df['pred_tk_pg']):.3f}")
                    if has_pk:
                        metric_cols[2].metric("PK% MAE",
                                  f"{mean_absolute_error(def_val_df['actual_pk_pct'], def_val_df['pred_pk_pct']):.4f}")
                    if has_pim:
                        metric_cols[3].metric("PIM/GP MAE",
                                  f"{mean_absolute_error(def_val_df['actual_pim_pg'], def_val_df['pred_pim_pg']):.3f}")
                    metric_cols[4].metric("Defensemen matched", f"{len(def_val_df):,}")
                    st.caption("PIM/GP: actual = penaltyMinutes/GP from NHL API, predicted = ind_penalty_minutes_pg from MoneyPuck.")

                    # Always show 2x2 grid — all 4 metrics
                    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                    fig.patch.set_facecolor("#0e1117")
                    make_scatter(def_val_df, "actual_hits_pg", "pred_hits_pg", "Hits / Game",      axes[0][0])
                    make_scatter(def_val_df, "actual_tk_pg",   "pred_tk_pg",   "Takeaways / Game", axes[0][1])
                    if has_pk:
                        make_scatter(def_val_df, "actual_pk_pct", "pred_pk_pct", "PK Ice %",   axes[1][0])
                    else:
                        axes[1][0].set_facecolor("#0e1117")
                        axes[1][0].text(0.5, 0.5, "PK% data not available\nfrom NHL API",
                                        ha="center", va="center", color="white", fontsize=12,
                                        transform=axes[1][0].transAxes)
                        axes[1][0].set_title("PK Ice %", color="white")
                    if has_pim:
                        make_scatter(def_val_df, "actual_pim_pg", "pred_pim_pg", "PIM / Game", axes[1][1])
                    else:
                        axes[1][1].set_facecolor("#0e1117")
                        axes[1][1].text(0.5, 0.5, "PIM data not available\nfrom NHL API",
                                        ha="center", va="center", color="white", fontsize=12,
                                        transform=axes[1][1].transAxes)
                        axes[1][1].set_title("PIM / Game", color="white")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.divider()
                    st.markdown("#### Biggest Misses (Hits/GP)")
                    miss_cols = ["player_name","team","games_played",
                                 "actual_hits_pg","pred_hits_pg","hits_error",
                                 "actual_tk_pg","pred_tk_pg","tk_error",
                                 "actual_pk_pct","pred_pk_pct",
                                 "actual_pim_pg","pred_pim_pg","seasons_used"]
                    misses = def_val_df.reindex(def_val_df["hits_error"].abs().nlargest(15).index)[
                        [c for c in miss_cols if c in def_val_df.columns]]
                    st.dataframe(misses, use_container_width=True)

                    st.markdown("#### Best Predictions (Hits/GP)")
                    best = def_val_df.reindex(def_val_df["hits_error"].abs().nsmallest(15).index)[
                        [c for c in miss_cols if c in def_val_df.columns]]
                    st.dataframe(best, use_container_width=True)

                    csv = def_val_df.to_csv(index=False)
                    st.download_button("Download defensive validation CSV", data=csv,
                                       file_name="def_validation_2025_26.csv", mime="text/csv")