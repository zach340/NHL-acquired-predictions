"""
nhl_api_datasets.py
===================
Build the model's season-level CSVs from the parsed NHL API data (after
nhl_api_parse.py and nhl_api_xg.py), with the same columns the MoneyPuck
pipeline produces:

  season_dataset.csv      forward/skater offense      (season_dataset.py)
  defensive_dataset.csv   defenseman stats            (defensive_dataset.py)
  pp_features.csv         5v4 power play + zone starts (power_play.py)
  linemate_features.csv   5v5 linemate quality         (linemates.py)

Written to OUT_DIR (default nhl_api_datasets/) so they can be compared
with the MoneyPuck files before replacing them.

    python pipeline/nhl_api_datasets.py
    python pipeline/nhl_api_datasets.py --out .      # overwrite the app's CSVs
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhl_api_parse import PARSED_DIR  # noqa: E402
from nhl_api_xg import danger_labels  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

TEAM_RENAMES = {"ATL": "WPG", "PHX": "UTA", "ARI": "UTA"}
UNBLOCKED    = {"goal", "shot-on-goal", "missed-shot"}
KEY          = ["player_id", "season", "player_team"]
MIN_LINE_SECS = 60       # a forward trio / D pair counts as a "line" after a minute together

# Danger-zone weights used by weighted_shots_pg (same as season_dataset.py)
HD_WEIGHT, MD_WEIGHT, LD_WEIGHT = 11.2, 4.1, 1.0


def safe_div(a, b, fill=0.0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(b == 0, fill, a / np.where(b == 0, 1, b))


def parsed_seasons():
    return sorted(int(os.path.basename(os.path.dirname(p)))
                  for p in glob.glob(os.path.join(PARSED_DIR, "*", "shots.parquet")))


def load(name, season):
    return pd.read_parquet(os.path.join(PARSED_DIR, str(season), f"{name}.parquet"))


# ── Shots: score/venue adjustment ──────────────────────────────────────────────

def score_venue_weights(seasons):
    """
    {(score diff clipped ±3, is_home): weight} = 0.5 / (league-wide xG share in
    that score state and venue), so trailing teams' extra shooting and home-ice
    effects net out.
    """
    cols = ["kind", "score_diff", "is_home", "xg"]
    s = pd.concat([pd.read_parquet(os.path.join(PARSED_DIR, str(y), "shots.parquet"), columns=cols)
                   for y in seasons], ignore_index=True)
    s = s[s["kind"].isin(UNBLOCKED)]
    xg_by = s.groupby([s["score_diff"].clip(-3, 3), s["is_home"]])["xg"].sum()
    weights = {}
    for (d, home), xg in xg_by.items():
        opp = xg_by.get((-d, not home), np.nan)
        weights[(d, home)] = 0.5 / (xg / (xg + opp)) if xg > 0 and opp > 0 else 1.0
    return weights


def add_score_venue_adjustment(shots, weights):
    w = pd.Series([weights.get((d, h), 1.0) for d, h in zip(shots["score_diff"].clip(-3, 3), shots["is_home"])],
                  index=shots.index)
    shots["xg_sv"] = shots["xg"] * w
    shots["xg_flurry_sv"] = shots["xg_flurry"] * w
    return shots


# ── Individual shooting ────────────────────────────────────────────────────────

def individual_shooting(shots, team_of_game):
    s = shots.merge(team_of_game, left_on=["shooter", "game_id"], right_on=["player_id", "game_id"], how="inner")
    unb = s["kind"].isin(UNBLOCKED)
    s = s.assign(
        ind_goals=s["is_goal"].astype(int),
        ind_shots_on_goal=s["kind"].isin({"goal", "shot-on-goal"}).astype(int),
        ind_missed_shots=(s["kind"] == "missed-shot").astype(int),
        ind_blocked_shot_attempts=(s["kind"] == "blocked-shot").astype(int),
        ind_shot_attempts=1,
        ind_unblocked_shot_attempts=unb.astype(int),
        ind_expected_goals=s["xg"],
        ind_flurry_adj_expected_goals=s["xg_flurry"],
        ind_score_venue_adj_expected_goals=s["xg_sv"],
        ind_flurry_score_venue_adj_expected_goals=s["xg_flurry_sv"],
    )
    for level in ("low", "medium", "high"):
        hit = unb & (s["danger"] == level)
        s[f"ind_{level}_danger_shots"] = hit.astype(int)
        s[f"ind_{level}_danger_goals"] = (hit & s["is_goal"]).astype(int)
        s[f"ind_{level}_danger_expected_goals"] = np.where(hit, s["xg"], 0.0)
    cols = [c for c in s.columns if c.startswith("ind_")]
    return s.groupby(["player_id", "game_id"])[cols].sum().reset_index()


def assists(shots):
    goals = shots[shots["is_goal"]]
    a1 = goals.groupby(["assist1", "game_id"]).size().rename("ind_primary_assists")
    a2 = goals.groupby(["assist2", "game_id"]).size().rename("ind_secondary_assists")
    a1.index.names = a2.index.names = ["player_id", "game_id"]
    return pd.concat([a1, a2], axis=1).fillna(0).reset_index()


# ── On-ice (5v5) ───────────────────────────────────────────────────────────────

def on_ice_5v5(shots, segments):
    """Long table of every 5v5 shot attempt × skater on ice, flagged for/against."""
    seg = segments[(segments["h_skaters"] == 5) & (segments["a_skaters"] == 5)
                   & segments["h_goalie"] & segments["a_goalie"]]
    s = shots[shots["seg"] >= 0].merge(seg, on=["game_id", "seg"], how="inner")
    rows = []
    for k in range(1, 6):
        for side in ("h", "a"):
            pid = s[f"{side}{k}"]
            is_for = s["is_home"] == (side == "h")
            rows.append(pd.DataFrame({"player_id": pid.values, "game_id": s["game_id"].values,
                                      "is_for": is_for.values, "kind": s["kind"].values,
                                      "is_goal": s["is_goal"].values, "xg": s["xg"].values,
                                      "xg_adj": s["xg_flurry_sv"].values, "danger": s["danger"].values}))
    long = pd.concat(rows, ignore_index=True)
    long = long[long["player_id"] > 0]
    unb = long["kind"].isin(UNBLOCKED)
    sog = long["kind"].isin({"goal", "shot-on-goal"})
    out = pd.DataFrame({"player_id": long["player_id"], "game_id": long["game_id"]})
    for side, mask in (("for", long["is_for"]), ("against", ~long["is_for"])):
        out[f"on_ice_{side}_corsi"] = mask.astype(int)
        out[f"on_ice_{side}_fenwick"] = (mask & unb).astype(int)
        out[f"on_ice_{side}_shots_on_goal"] = (mask & sog).astype(int)
        out[f"on_ice_{side}_goals"] = (mask & long["is_goal"]).astype(int)
        out[f"on_ice_{side}_expected_goals"] = np.where(mask, long["xg"], 0.0)
        out[f"on_ice_{side}_adj_expected_goals"] = np.where(mask, long["xg_adj"], 0.0)
        out[f"on_ice_{side}_high_danger_shots"] = (mask & unb & (long["danger"] == "high")).astype(int)
        out[f"on_ice_{side}_hd_expected_goals"] = np.where(mask & (long["danger"] == "high"), long["xg"], 0.0)
    return out.groupby(["player_id", "game_id"]).sum().reset_index()


# ── Power play (5v4) individual ────────────────────────────────────────────────

def power_play(shots, team_of_game):
    pp = shots[(shots["own_skaters"] == 5) & (shots["opp_skaters"] == 4)
               & (shots["own_goalie"] == 1) & (shots["opp_goalie"] == 1)]
    ind = pp.assign(goals_pp=pp["is_goal"].astype(int),
                    shots_on_goal_pp=pp["kind"].isin({"goal", "shot-on-goal"}).astype(int),
                    shot_attempts_pp=1, xg_pp=pp["xg"], xg_adj_pp=pp["xg_flurry_sv"])
    ind = ind.groupby(["shooter", "game_id"])[["goals_pp", "shots_on_goal_pp", "shot_attempts_pp",
                                               "xg_pp", "xg_adj_pp"]].sum()
    ind.index.names = ["player_id", "game_id"]
    g = pp[pp["is_goal"]]
    a1 = g.groupby(["assist1", "game_id"]).size().rename("primary_assists_pp")
    a2 = g.groupby(["assist2", "game_id"]).size().rename("secondary_assists_pp")
    a1.index.names = a2.index.names = ["player_id", "game_id"]
    out = pd.concat([ind, a1, a2], axis=1).fillna(0).reset_index()
    out["points_pp"] = out["goals_pp"] + out["primary_assists_pp"] + out["secondary_assists_pp"]
    return out


# ── Linemates (5v5) ────────────────────────────────────────────────────────────

def _unit_key(ids):
    """Order-independent int64 key for each row's set of player ids (0 = empty slot)."""
    ids = np.sort(ids, axis=1).astype(np.int64)
    key = np.zeros(len(ids), dtype=np.int64)
    with np.errstate(over="ignore"):
        for k in range(ids.shape[1]):
            key = key * np.int64(1_000_003) + ids[:, k]
    return key


def linemate_units(segments, positions, shots):
    """
    Per player-season: TOI-weighted quality of the 5v5 units they played in
    (forward trio for forwards, pair for defensemen) and how many distinct
    units they had at least MIN_LINE_SECS with.
    """
    seg = segments[(segments["h_skaters"] == 5) & (segments["a_skaters"] == 5)
                   & segments["h_goalie"] & segments["a_goalie"]].copy()
    s5 = shots[shots["seg"] >= 0].merge(seg[["game_id", "seg"]], on=["game_id", "seg"])
    unb = s5["kind"].isin(UNBLOCKED)
    s5 = s5.assign(xg_u=np.where(unb, s5["xg"], 0), adj_u=np.where(unb, s5["xg_flurry_sv"], 0),
                   hd_u=np.where(unb & (s5["danger"] == "high"), s5["xg"], 0), g=s5["is_goal"].astype(int), c=1)
    agg = s5.groupby(["game_id", "seg", "is_home"])[["xg_u", "adj_u", "hd_u", "g", "c"]].sum().unstack(fill_value=0)

    rows = []
    f_for = agg.reindex(pd.MultiIndex.from_arrays([seg["game_id"], seg["seg"]]), fill_value=0)
    is_d = np.vectorize(lambda p: positions.get(p) == "D")
    for side, home in (("h", True), ("a", False)):
        ids = seg[[f"{side}{k}" for k in range(1, 6)]].values
        d_mask = is_d(ids)
        fwd_unit, d_unit = _unit_key(np.where(d_mask, 0, ids)), _unit_key(np.where(d_mask, ids, 0))
        stat = {}
        for col in ("xg_u", "adj_u", "hd_u", "g", "c"):
            stat[f"{col}_for"] = f_for[(col, home)].values if (col, home) in f_for.columns else 0
            stat[f"{col}_against"] = f_for[(col, not home)].values if (col, not home) in f_for.columns else 0
        for k in range(5):
            df = pd.DataFrame({"player_id": ids[:, k], "game_id": seg["game_id"].values,
                               "duration": seg["duration"].values,
                               "unit": np.where(d_mask[:, k], d_unit, fwd_unit),
                               **stat})
            rows.append(df)
    long = pd.concat(rows, ignore_index=True)
    long = long[long["player_id"] > 0]
    long["season"] = (long["game_id"] // 1_000_000).astype(int)   # 2024020001 → 2024

    per_unit = long.groupby(["player_id", "season", "unit"])["duration"].sum()
    n_units = (per_unit >= MIN_LINE_SECS).groupby(["player_id", "season"]).sum().rename("n_distinct_lines")

    tot = long.groupby(["player_id", "season"])[["duration", "xg_u_for", "xg_u_against", "adj_u_for", "hd_u_for",
                                                 "g_for", "c_for", "c_against"]].sum()
    hours = tot["duration"] / 3600
    out = pd.DataFrame({
        "line_adj_xg_per60": safe_div(tot["adj_u_for"], hours),
        "line_xg_per60":     safe_div(tot["xg_u_for"], hours),
        "line_hd_xg_per60":  safe_div(tot["hd_u_for"], hours),
        "line_goals_per60":  safe_div(tot["g_for"], hours),
        "line_xg_pct":       safe_div(tot["xg_u_for"], tot["xg_u_for"] + tot["xg_u_against"]),
        "line_corsi_pct":    safe_div(tot["c_for"], tot["c_for"] + tot["c_against"]),
        "total_line_icetime": tot["duration"],
    }, index=tot.index).join(n_units)
    return out.reset_index()


# ── Build ──────────────────────────────────────────────────────────────────────

def impute_strength_toi(pg):
    """
    Games without a shift chart only have total TOI (from the boxscore). Split it
    by the player's 5v5 / 5v4 / 4v5 shares in his tracked games that season.
    On-ice 5v5 rates use tracked time only (toi_5v5_tracked), since those games
    have no on-ice events either.
    """
    pg = pg.copy()
    tracked = pg["has_shifts"]
    pg["toi_5v5_tracked"] = np.where(tracked, pg["toi_5v5"], 0.0)
    totals = pg[tracked].groupby("player_id")[["toi_all", "toi_5v5", "toi_5v4", "toi_4v5"]].sum()
    for col in ("toi_5v5", "toi_5v4", "toi_4v5"):
        share = (totals[col] / totals["toi_all"]).reindex(pg["player_id"]).values
        fallback = pg.loc[tracked, col].sum() / pg.loc[tracked, "toi_all"].sum()
        est = pg["toi_all"] * np.where(np.isnan(share), fallback, share)
        pg[col] = np.where(tracked, pg[col], est)
    return pg


def build_season(season, weights):
    shots = add_score_venue_adjustment(load("shots", season), weights)
    shots["danger"] = danger_labels(shots)
    segments = load("segments", season)
    pg = load("player_games", season)
    pg = pg[pg["position"] != "G"].copy()
    pg["player_team"] = pg["player_team"].replace(TEAM_RENAMES)
    pg = impute_strength_toi(pg)
    team_of_game = pg[["player_id", "game_id"]]

    per_game = (pg.merge(individual_shooting(shots, team_of_game), on=["player_id", "game_id"], how="left")
                  .merge(assists(shots), on=["player_id", "game_id"], how="left")
                  .merge(on_ice_5v5(shots, segments), on=["player_id", "game_id"], how="left")
                  .merge(power_play(shots, team_of_game), on=["player_id", "game_id"], how="left"))
    num = per_game.select_dtypes("number").columns.difference(["player_id", "game_id", "season", "team_id"])
    per_game[num] = per_game[num].fillna(0)

    names = pg.groupby("player_id")[["player_name", "position"]].agg(lambda x: x.mode().iloc[-1])
    totals = per_game.groupby(KEY)[list(num)].sum()
    totals["games_played"] = per_game.groupby(KEY)["game_id"].nunique()
    totals = totals.reset_index().merge(names, left_on="player_id", right_index=True)

    positions = names["position"].to_dict()
    lines = linemate_units(segments, positions, shots)
    print(f"  {season}: {len(totals):,} player-team rows", flush=True)
    return totals, lines


def build(out_dir):
    seasons = parsed_seasons()
    print(f"Seasons: {seasons}", flush=True)
    weights = score_venue_weights(seasons)
    parts = [build_season(y, weights) for y in seasons]
    season = pd.concat([p[0] for p in parts], ignore_index=True)
    lines = pd.concat([p[1] for p in parts], ignore_index=True)
    season["ind_points"] = season["ind_goals"] + season["ind_primary_assists"] + season["ind_secondary_assists"]
    season["ice_time"] = season["toi_all"]
    season["fv5_ice_time"] = season["toi_5v5_tracked"]

    # Game score (Luszczyszyn), as MoneyPuck computes it
    season["game_score"] = (
        0.75 * season["ind_goals"] + 0.7 * season["ind_primary_assists"] + 0.55 * season["ind_secondary_assists"]
        + 0.075 * season["ind_shots_on_goal"] + 0.05 * season["shots_blocked"]
        + 0.15 * season["penalties_drawn"] - 0.15 * season["penalties"]
        + 0.01 * season["faceoffs_won"] - 0.01 * season["faceoffs_lost"]
        + 0.05 * (season["on_ice_for_corsi"] - season["on_ice_against_corsi"])
        + 0.15 * (season["on_ice_for_goals"] - season["on_ice_against_goals"])
    )
    os.makedirs(out_dir, exist_ok=True)
    write_offense(season, out_dir)
    write_defense(season, out_dir)
    write_power_play(season, out_dir)

    lines = season[["player_id", "season"]].drop_duplicates().merge(lines, on=["player_id", "season"], how="left")
    lines.fillna(0).to_csv(os.path.join(out_dir, "linemate_features.csv"), index=False)
    print(f"  linemate_features.csv: {len(lines):,} rows", flush=True)


def write_offense(season, out_dir):
    d = season.copy()
    gp = d["games_played"].replace(0, np.nan)
    d["game_score_per_game"] = d["game_score"] / gp
    d["points_per_game"] = d["ind_points"] / gp
    d["goals_per_game"] = d["ind_goals"] / gp
    d["weighted_shots_pg"] = (d["ind_high_danger_shots"] * HD_WEIGHT + d["ind_medium_danger_shots"] * MD_WEIGHT
                              + d["ind_low_danger_shots"] * LD_WEIGHT) / gp
    d["toi_per_game"] = (d["ice_time"] / 60) / gp
    hours = d["ice_time"] / 3600
    d["shifts_per60"] = safe_div(d["shifts"], hours)
    rate_cols = [c for c in d.columns if c.startswith("ind_") and c not in ("ind_expected_on_goal",)]
    for c in rate_cols:
        d[f"{c}_per60"] = safe_div(d[c], hours)
    d["ice_time_rank"] = d.groupby("season")["ice_time"].rank(ascending=False)
    cols = ["player_id", "player_name", "season", "player_team", "position", "ice_time", "shifts", "game_score",
            *rate_cols, "games_played", "game_score_per_game", "points_per_game", "goals_per_game",
            "weighted_shots_pg", "toi_per_game", "shifts_per60", *[f"{c}_per60" for c in rate_cols], "ice_time_rank"]
    d[cols].to_csv(os.path.join(out_dir, "season_dataset.csv"), index=False)
    print(f"  season_dataset.csv: {len(d):,} rows", flush=True)


def write_defense(season, out_dir):
    d = season[season["position"] == "D"].copy()
    d = d.rename(columns={
        "hits": "ind_hits", "takeaways": "ind_takeaways", "giveaways": "ind_giveaways",
        "d_zone_giveaways": "ind_d_zone_giveaways", "pim": "ind_penalty_minutes",
        "pim_drawn": "penalty_minutes_drawn", "shots_blocked": "shots_blocked_by_player",
        "o_zone_starts": "ind_o_zone_shift_starts", "d_zone_starts": "ind_d_zone_shift_starts",
        "n_zone_starts": "ind_neutral_zone_shift_starts", "toi_4v5": "pk_ice_time",
    })
    d["ind_faceoffs_won"] = d["faceoffs_won"]
    gp = d["games_played"].replace(0, np.nan)
    hours, fv5 = d["ice_time"] / 3600, (d["fv5_ice_time"] / 3600).replace(0, np.nan)
    d["on_ice_expected_goals_pct"] = safe_div(d["on_ice_for_expected_goals"],
                                              d["on_ice_for_expected_goals"] + d["on_ice_against_expected_goals"], np.nan)
    d["on_ice_corsi_pct"] = safe_div(d["on_ice_for_corsi"], d["on_ice_for_corsi"] + d["on_ice_against_corsi"], np.nan)
    d["on_ice_fenwick_pct"] = safe_div(d["on_ice_for_fenwick"], d["on_ice_for_fenwick"] + d["on_ice_against_fenwick"], np.nan)
    d["pk_ice_pct"] = safe_div(d["pk_ice_time"], d["ice_time"])
    d["pk_toi_per_game"] = d["pk_ice_time"] / 60 / gp
    for c in ("ind_hits", "ind_takeaways", "ind_giveaways", "ind_d_zone_giveaways", "shots_blocked_by_player",
              "ind_blocked_shot_attempts", "penalties", "ind_penalty_minutes", "penalties_drawn"):
        d[f"{c}_pg"] = d[c] / gp
    for c in ("ind_hits", "ind_takeaways", "ind_giveaways", "shots_blocked_by_player", "penalties",
              "ind_penalty_minutes"):
        d[f"{c}_per60"] = safe_div(d[c], hours)
    fo = d["faceoffs_won"] + d["faceoffs_lost"]
    d["faceoff_win_pct"] = safe_div(d["faceoffs_won"], fo, np.nan)
    zones = d["ind_d_zone_shift_starts"] + d["ind_o_zone_shift_starts"]
    d["d_zone_start_pct"] = safe_div(d["ind_d_zone_shift_starts"], zones, np.nan)
    d["take_give_ratio"] = safe_div(d["ind_takeaways"], d["ind_giveaways"], np.nan)
    d["xg_against_per60_5v5"] = d["on_ice_against_expected_goals"] / fv5
    d["hd_shots_against_per60_5v5"] = d["on_ice_against_high_danger_shots"] / fv5
    d["goals_against_per60_5v5"] = d["on_ice_against_goals"] / fv5
    d = d[d["games_played"] >= 20]
    cols = ["player_id", "player_name", "season", "player_team", "position", "ind_hits", "ind_takeaways",
            "ind_giveaways", "ind_d_zone_giveaways", "ind_penalty_minutes", "penalties", "penalty_minutes_drawn",
            "penalties_drawn", "shots_blocked_by_player", "ind_blocked_shot_attempts", "ind_faceoffs_won",
            "faceoffs_won", "faceoffs_lost", "ind_d_zone_shift_starts", "ind_o_zone_shift_starts",
            "ind_neutral_zone_shift_starts", "ice_time", "games_played", "pk_ice_time", "fv5_ice_time",
            "on_ice_expected_goals_pct", "on_ice_corsi_pct", "on_ice_fenwick_pct", "on_ice_against_expected_goals",
            "on_ice_against_high_danger_shots", "on_ice_against_shots_on_goal", "on_ice_against_goals",
            "on_ice_for_expected_goals", "pk_ice_pct", "pk_toi_per_game", "ind_hits_pg", "ind_takeaways_pg",
            "ind_giveaways_pg", "ind_d_zone_giveaways_pg", "shots_blocked_by_player_pg",
            "ind_blocked_shot_attempts_pg", "penalties_pg", "ind_penalty_minutes_pg", "penalties_drawn_pg",
            "ind_hits_per60", "ind_takeaways_per60", "ind_giveaways_per60", "shots_blocked_by_player_per60",
            "penalties_per60", "ind_penalty_minutes_per60", "faceoff_win_pct", "d_zone_start_pct",
            "take_give_ratio", "xg_against_per60_5v5", "hd_shots_against_per60_5v5", "goals_against_per60_5v5"]
    d[cols].to_csv(os.path.join(out_dir, "defensive_dataset.csv"), index=False)
    print(f"  defensive_dataset.csv: {len(d):,} rows", flush=True)


def write_power_play(season, out_dir):
    # power_play.py aggregates across teams (no player_team in its key)
    group = ["player_id", "season"]
    sums = ["toi_all", "toi_5v4", "goals_pp", "primary_assists_pp", "points_pp", "shots_on_goal_pp", "xg_pp",
            "ind_points", "o_zone_starts", "d_zone_starts", "n_zone_starts"]
    d = season.groupby(group)[sums].sum().reset_index()
    d = d.merge(season.groupby(group)[["player_name", "position"]].first().reset_index(), on=group)
    pp_hours = d["toi_5v4"] / 3600
    d["pp_goals_per60"] = safe_div(d["goals_pp"], pp_hours)
    d["pp_primary_assists_per60"] = safe_div(d["primary_assists_pp"], pp_hours)
    d["pp_points_per60"] = safe_div(d["points_pp"], pp_hours)
    d["pp_shots_per60"] = safe_div(d["shots_on_goal_pp"], pp_hours)
    d["pp_xg_per60"] = safe_div(d["xg_pp"], pp_hours)
    d["pp_icetime_pct"] = safe_div(d["toi_5v4"], d["toi_all"])
    d["pp_points_share"] = safe_div(d["points_pp"], d["ind_points"])
    zones = d["o_zone_starts"] + d["d_zone_starts"] + d["n_zone_starts"]
    d["o_zone_start_pct"] = safe_div(d["o_zone_starts"], zones)
    d["d_zone_start_pct"] = safe_div(d["d_zone_starts"], zones)
    d["zone_start_diff"] = d["o_zone_start_pct"] - d["d_zone_start_pct"]
    cols = ["player_id", "player_name", "season", "position", "pp_goals_per60", "pp_primary_assists_per60",
            "pp_points_per60", "pp_shots_per60", "pp_xg_per60", "pp_icetime_pct", "pp_points_share",
            "o_zone_start_pct", "d_zone_start_pct", "zone_start_diff"]
    d[cols].to_csv(os.path.join(out_dir, "pp_features.csv"), index=False)
    print(f"  pp_features.csv: {len(d):,} rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nhl_api_datasets")
    build(ap.parse_args().out)
