"""Unit tests for pure logic (no network, no trained models). Run: python -m pytest tests"""

import numpy as np
import pandas as pd
import pytest

from nhl_predictor import contract, features, grading, nhl_api, pairing


# ── features ──────────────────────────────────────────────────────────────────

def test_prior_slope_uses_only_previous_seasons():
    s = pd.Series([1.0, 2.0, 3.0, 10.0])
    out = features.prior_slope(s, window=3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])   # < 2 prior values
    assert out.iloc[2] == pytest.approx(1.0)                  # slope of [1, 2]
    assert out.iloc[3] == pytest.approx(1.0)                  # slope of [1, 2, 3]; 10 not used


def test_compute_baseline_takes_first_non_null_candidate():
    df = pd.DataFrame({"a": [np.nan, 1.0, np.nan], "b": [2.0, 5.0, np.nan]})
    assert features.compute_baseline(df, ["a", "b", "missing"]).tolist() == [2.0, 1.0, 0.0]


def test_weighted_recent_mean_is_3_2_1_within_each_player():
    d = pd.DataFrame({"player_id": [1, 1, 1, 1, 2], "season": [2020, 2021, 2022, 2023, 2023],
                      "x": [9.0, 1.0, 2.0, 3.0, 5.0], "toi": [1.0, 1.0, 1.0, 3.0, 1.0]})
    out = features.weighted_recent_mean(d, "x")
    assert out.iloc[1] == pytest.approx((3 * 1 + 2 * 9) / 5)        # only one prior season
    assert out.iloc[3] == pytest.approx((3 * 3 + 2 * 2 + 1 * 1) / 6)  # 9.0 is outside the window
    assert out.iloc[4] == 5.0                                          # player 2 doesn't see player 1
    weighted = features.weighted_recent_mean(d, "x", weight_col="toi")
    assert weighted.iloc[3] == pytest.approx((9 * 3 + 2 * 2 + 1 * 1) / (9 + 2 + 1))


def test_season_folds_never_train_on_the_validation_season_or_later():
    from nhl_predictor.training import season_folds
    seasons = np.array([2020, 2021, 2022, 2023, 2021, 2023])
    folds = season_folds(seasons, n_folds=2)
    assert [sorted(set(seasons[v])) for _, v in folds] == [[2022], [2023]]
    for tr, val in folds:
        assert seasons[tr].max() < seasons[val].min()


def test_latest_team_contexts_falls_back_for_missing_teams():
    df = pd.DataFrame({"season": [2023, 2024]})
    ctx = pd.DataFrame({"player_team": ["AAA", "AAA", "BBB"], "season": [2023, 2024, 2023], "x": [1, 2, 3]})
    out = features.latest_team_contexts(df, ctx, keys=[]).set_index("player_team")
    assert out.loc["AAA", "x"] == 2 and out.loc["BBB", "x"] == 3


# ── grading ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pct,grade", [(95, "A"), (90, "A"), (80, "B+"), (50, "B"), (40, "C+"), (20, "C"), (5, "D")])
def test_score_to_grade(pct, grade):
    assert grading.score_to_grade(pct) == grade


def test_defensive_fallback_percentiles_are_monotonic():
    """Without a league frame, better xGA must never grade worse (lower is better)."""
    good = grading.grade_defensive_defenseman({"xg_against_per60_5v5": 1.5})[3]["xGA/60 (5v5)"][1]
    bad  = grading.grade_defensive_defenseman({"xg_against_per60_5v5": 4.5})[3]["xGA/60 (5v5)"][1]
    assert good > bad


@pytest.mark.parametrize("off,defn,expected", [(90, 20, "Offensive D"), (20, 90, "Defensive D"), (60, 50, "Two-Way D")])
def test_classify_defenseman_type(off, defn, expected):
    assert grading.classify_defenseman_type({}, def_score=defn, off_score=off)[0] == expected


# ── contract ──────────────────────────────────────────────────────────────────

def test_cba_limits():
    same = contract.get_cba_limits(27, "TOR", "TOR")
    new  = contract.get_cba_limits(34, "TOR", "BOS")
    assert (same["max_years"], same["recommended"]) == (7, 7)
    assert (new["max_years"], new["recommended"], new["is_35_signing"]) == (6, 2, False)
    assert contract.get_cba_limits(35, "TOR", "TOR")["is_35_signing"]


def test_age_profile_forward_applies_compounded_curve():
    profile = pd.Series({"age": 30.0, "position": "C", "ind_goals_per60": 1.0, "career_year": 8.0})
    aged = contract.age_profile(profile, 2)
    assert aged["age"] == 32.0
    assert aged["ind_goals_per60"] == pytest.approx(0.97 * 0.94)   # ages 30 then 31
    assert aged["career_year"] == 10.0


def test_confidence_decays_faster_for_older_players():
    assert contract.get_age_adjusted_confidence(3, 25) > contract.get_age_adjusted_confidence(3, 35)


# ── pairing ───────────────────────────────────────────────────────────────────

def _scores(**players):
    return {pid: {"player_name": pid, "combined_score": s, "shoots": ""} for pid, s in players.items()}


def test_cascade_displaces_down_the_chart_and_scratches_the_last():
    scores = _scores(a=90, b=50, c=70, d=40, new=80)
    pairs = [["a", "b"], ["c", "d"]]
    log, scratched = pairing._cascade(pairs, "new", scores)
    assert pairs == [["a", "new"], ["c", "b"]]
    assert scratched == ["d"]
    assert [e["action"] for e in log] == ["moved in", "moved in", "scratched"]


def test_greedy_shift_pairs_prefers_most_shared_toi():
    actual = [("a", "b", 100), ("a", "c", 90), ("c", "d", 80)]
    pairs, assigned = pairing._greedy_shift_pairs(actual, {"a", "b", "c", "d"})
    assert pairs == [["a", "b"], ["c", "d"]] and assigned == {"a", "b", "c", "d"}


# ── shift overlap parsing ────────────────────────────────────────────────────

def test_to_secs():
    assert nhl_api._to_secs("12:34") == 754
    assert nhl_api._to_secs(90) == 90
    assert nhl_api._to_secs("bad") == 0


# ── regression tests for fixed bugs ───────────────────────────────────────────

def test_season_helpers_and_rollover():
    from datetime import date
    from nhl_predictor import config
    assert config.season_id(2024) == "20242025" and config.season_label(2024) == "2024-25"
    assert config._current_season_start(date(2026, 9, 30)) == 2026
    assert config._current_season_start(date(2026, 3, 1)) == 2025


def test_age_is_measured_at_the_start_of_the_labelled_season():
    from nhl_predictor.data_io import age_on_season_start
    # 2015 = the 2015-16 season; McDavid (born 1997-01-13) was 18 that October
    assert age_on_season_start("1997-01-13", 2015) == pytest.approx(18.7)


def test_current_fit_models_exclude_same_season_outcomes():
    from nhl_predictor import config, defense, offense
    fit_cols = offense.feature_columns(has_age=True, next_season=False)
    assert not set(fit_cols) & set(config.OUTCOME_FEATURES)
    assert set(config.OUTCOME_FEATURES) <= set(offense.feature_columns(has_age=True, next_season=True))
    df = pd.DataFrame(columns=config.DEF_PLAYER_FEATURES + config.DEF_TEAM_FEATURES)
    assert not set(defense.feature_columns(df, has_age=False)) & set(config.DEF_OUTCOME_FEATURES)


def test_career_peak_never_looks_ahead():
    from nhl_predictor import defense
    df = pd.DataFrame({"player_id": [1, 1, 1], "season": [2020, 2021, 2022],
                       "ind_hits_pg": [1.0, 3.0, 2.0], "ind_takeaways_pg": [0.1, 0.2, 0.3],
                       "pk_ice_pct": [0.1, 0.1, 0.1]})
    out = defense.engineer_features(df)
    assert out["career_peak_hits_pg"].tolist() == [1.0, 3.0, 3.0]   # 2020 must not see 2021's 3.0


def test_defensive_slope_features_are_produced_under_their_configured_names():
    from nhl_predictor import config, defense
    df = pd.DataFrame({"player_id": [1] * 3, "season": [2020, 2021, 2022], "games_played": [80] * 3,
                       "ind_hits_pg": [1.0, 2.0, 3.0], "ind_takeaways_pg": [0.1] * 3,
                       "xg_against_per60_5v5": [2.5] * 3, "pk_ice_pct": [0.1] * 3,
                       "ind_penalty_minutes_pg": [0.5] * 3})
    out = defense.engineer_career_history(df)
    slopes = [f for f in config.DEF_PLAYER_FEATURES if f.endswith("_slope")]
    assert slopes and all(f in out.columns for f in slopes)
    assert out["recent_3yr_hits_slope"].iloc[2] == pytest.approx(1.0)


# ── NHL API pipeline ──────────────────────────────────────────────────────────

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))


def _toy_shifts():
    """Home team 1 (skaters 11-16, goalie 10), away team 2 (21-25, goalie 20); 16 replaces 15 at t=30."""
    rows = [(10, 1, 0, 60, True), (20, 2, 0, 60, True)]
    rows += [(p, 1, 0, 60, False) for p in (11, 12, 13, 14)] + [(15, 1, 0, 30, False), (16, 1, 30, 60, False)]
    rows += [(p, 2, 0, 60, False) for p in (21, 22, 23, 24, 25)]
    return pd.DataFrame(rows, columns=["player_id", "team_id", "start", "end", "is_goalie"])


def test_segments_split_at_line_changes_and_events_land_on_the_right_side():
    import nhl_api_parse as P
    seg, _ = P.build_segments(_toy_shifts(), home_id=1, away_id=2)
    assert seg[["start", "end"]].values.tolist() == [[0, 30], [30, 60]]
    assert sorted(seg.loc[0, ["h1", "h2", "h3", "h4", "h5"]]) == [11, 12, 13, 14, 15]
    assert sorted(seg.loc[1, ["h1", "h2", "h3", "h4", "h5"]]) == [11, 12, 13, 14, 16]
    assert (seg["h_skaters"] == 5).all() and seg["h_goalie"].all() and seg["a_goalie"].all()
    s, e = seg["start"].values, seg["end"].values
    assert P._segment_index(s, e, 30) == 0                  # a goal at the change: outgoing players
    assert P._segment_index(s, e, 30, faceoff=True) == 1    # a faceoff at the change: incoming players


def test_flurry_adjustment_discounts_follow_up_shots():
    import nhl_api_xg as X
    s = pd.DataFrame({"game_id": 1, "team_id": [1, 1, 1], "period": 1, "t": [10, 12, 30],
                      "event_id": [1, 2, 3], "xg": [0.3, 0.4, 0.2]})
    out = X.flurry_adjust(s)
    assert out.tolist() == pytest.approx([0.3, 0.4 * 0.7, 0.2])   # third shot starts a new sequence


def test_unit_key_ignores_order():
    import nhl_api_datasets as D
    keys = D._unit_key(np.array([[3, 1, 2, 0, 0], [2, 3, 1, 0, 0], [1, 2, 4, 0, 0]]))
    assert keys[0] == keys[1] != keys[2]


def test_untracked_games_get_toi_split_by_tracked_shares():
    import nhl_api_datasets as D
    pg = pd.DataFrame({"player_id": [1, 1], "has_shifts": [True, False], "toi_all": [1000.0, 500.0],
                       "toi_5v5": [800.0, np.nan], "toi_5v4": [100.0, np.nan], "toi_4v5": [50.0, np.nan]})
    out = D.impute_strength_toi(pg)
    assert out["toi_5v5"].tolist() == [800.0, 400.0]
    assert out["toi_5v5_tracked"].tolist() == [800.0, 0.0]   # on-ice rates only use tracked time


# ── models ────────────────────────────────────────────────────────────────────

def test_blend_predicts_reports_importances_and_survives_a_cache_round_trip(tmp_path):
    import joblib
    from nhl_predictor import defense, offense
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = 2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=300)
    for make in (offense.make_model, defense.make_model):
        m = make().fit(X, y, sample_weight=np.ones(300))
        imp = m.feature_importances_
        assert imp.shape == (4,) and imp.sum() == pytest.approx(1.0) and imp[:2].sum() > imp[2:].sum()
        joblib.dump(m, tmp_path / "m.joblib")
        again = joblib.load(tmp_path / "m.joblib")
        assert np.allclose(again.predict(X), m.predict(X))
        assert np.corrcoef(m.predict(X), y)[0, 1] > 0.95
