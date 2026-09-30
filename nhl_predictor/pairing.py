"""
pairing.py
==========
Defensive depth chart for a team: real pairs from shift-chart data, scored
by the defenseman model, with a searched player cascaded in.
"""

from . import nhl_api
from .config import PAIR_SLOT_NAMES, SLOT_COLORS
from .data_io import load_defensive_offensive_stats
from .defense import get_latest_team_contexts, predict_for_team, type_and_scores
from .grading import combined_score

MAX_DISPLAY     = 3   # pairs shown in the depth chart
MAX_BUILD_PAIRS = 4   # pairs built before the cascade (8 D-men)
HAND_BONUS      = 3.0 # cascade bonus for being opposite-handed to the stronger partner


def _error(msg):
    return [], [], {}, [], [], {"pair_err": msg, "partner_name": "—", "partner_slot": "—",
                                "searched_score": 0, "actual_pairs": []}


def _score_player(profile, team_row, bundle, off_stats, off_df):
    preds = predict_for_team(profile, team_row, bundle.fit_models, bundle.fit_feature_names)
    def_pct, off_pct, d_type, d_desc = type_and_scores(preds, profile, off_stats, off_df, bundle.df)
    return {**preds, "defensive_score": def_pct, "combined_score": combined_score(def_pct, off_pct, d_type),
            "d_type": d_type, "d_desc": d_desc}


def _greedy_shift_pairs(actual_pairs, eligible, limit=None):
    """Walk pairs by shared TOI, taking each pair whose players are both still free."""
    pairs, assigned = [], set()
    for pid1, pid2, _ in actual_pairs or []:
        if pid1 in eligible and pid2 in eligible and pid1 not in assigned and pid2 not in assigned:
            pairs.append([pid1, pid2])
            assigned.update([pid1, pid2])
            if limit is not None and len(pairs) == limit:
                break
    return pairs, assigned


def _fill_from_pool(pairs, pool, scores, limit):
    """Pair up remaining players best-score-first until `limit` pairs exist."""
    pool = sorted(pool, key=lambda p: scores[p]["combined_score"], reverse=True)
    while len(pairs) < limit and len(pool) >= 2:
        pairs.append([pool.pop(0), pool.pop(0)])
    return pool


def _pair_record(i, p1, p2, scores, actual_pairs):
    shoots1 = scores[p1].get("shoots", "")
    shoots2 = scores[p2].get("shoots", "")
    slot = PAIR_SLOT_NAMES[i] if i < len(PAIR_SLOT_NAMES) else f"Pair {i + 1}"
    return {
        "pid1": p1, "pid2": p2,
        "name1": scores[p1]["player_name"], "name2": scores[p2]["player_name"],
        "score1": scores[p1]["combined_score"], "score2": scores[p2]["combined_score"],
        "shoots1": shoots1, "shoots2": shoots2,
        "hand_match": bool(shoots1 and shoots2 and shoots1 != shoots2),
        "pair_score": round((scores[p1]["combined_score"] + scores[p2]["combined_score"]) / 2, 1),
        "from_shifts": any({p1, p2} == {a, b} for a, b, _ in actual_pairs or []),
        "slot": slot,
        "slot_color": SLOT_COLORS.get(slot, "#888888"),
    }


def _cascade(pairs, new_player, scores):
    """
    Insert `new_player` into `pairs` (mutated in place). Moving down the chart,
    they replace the weaker partner of the first pair they outscore (with a
    handedness bonus); the displaced player then continues from the next pair
    down. Whoever can't displace anyone is scratched.
    Returns (cascade_log, scratched).
    """
    log, scratched = [], []
    to_place, start_from = new_player, 0
    for _ in range(len(pairs) + 2):
        new_s      = scores[to_place]["combined_score"]
        new_shoots = scores[to_place].get("shoots", "")
        placed = False
        for i, pair in enumerate(pairs[start_from:], start=start_from):
            p1, p2 = pair
            s1, s2 = scores[p1]["combined_score"], scores[p2]["combined_score"]
            weaker, stronger = (p1, p2) if s1 <= s2 else (p2, p1)
            stronger_shoots  = scores[stronger].get("shoots", "")
            bonus = HAND_BONUS if (new_shoots and stronger_shoots and new_shoots != stronger_shoots) else 0.0
            if new_s + bonus > min(s1, s2):
                log.append({"player": scores[to_place]["player_name"], "action": "moved in",
                            "slot": PAIR_SLOT_NAMES[i] if i < len(PAIR_SLOT_NAMES) else f"Pair {i + 1}",
                            "displaced": scores[weaker]["player_name"]})
                pair[pair.index(weaker)] = to_place
                start_from, to_place, placed = i + 1, weaker, True
                break
        if not placed:
            log.append({"player": scores[to_place]["player_name"], "action": "scratched",
                        "slot": "—", "displaced": "—"})
            scratched.append(to_place)
            break
    return log, scratched


def build_pairing_insertion(player_id, team_code, bundle, n_games=25, prefetched_pairs=None):
    """
    Returns (depth_pairs, scratched, player_scores, cascade_log, unmodeled, info).

    If the player already plays for `team_code`, the real shift pairs are shown
    with them highlighted; otherwise they are cascaded into the depth chart.
    """
    roster = nhl_api.fetch_team_defensemen(team_code)
    if not roster:
        return _error("Could not fetch roster (empty response).")
    if "_error" in roster[0]:
        return _error(f"Could not fetch roster: {roster[0]['_error']}")

    roster_names = {p["player_id"]: p["player_name"] for p in roster}
    shoots = {p["player_id"]: p.get("shoots", "") for p in roster}
    # A player from another team isn't on this roster — look up handedness separately
    new_player_shoots = shoots.get(player_id, "") or nhl_api.fetch_shoots(player_id)

    all_teams = get_latest_team_contexts(bundle.df, bundle.team_ctx)
    team_row  = all_teams[all_teams["player_team"] == team_code]
    if team_row.empty:
        return _error(f"No team context for {team_code}.")
    team_row = team_row.iloc[0]
    if player_id not in bundle.profiles:
        return _error("Player not found in model data.")

    off_stats, off_df, _ = load_defensive_offensive_stats()
    scores = {}
    for pid, name in roster_names.items():
        if pid in bundle.profiles:
            scores[pid] = {**_score_player(bundle.profiles[pid][0], team_row, bundle, off_stats.get(pid, {}), off_df),
                           "player_name": name, "shoots": shoots.get(pid, "")}
    search_profile = bundle.profiles[player_id][0]
    scores[player_id] = {**_score_player(search_profile, team_row, bundle, off_stats.get(player_id, {}), off_df),
                         "player_name": search_profile.get("player_name", "Selected Player"),
                         "is_searched_player": True, "shoots": new_player_shoots}

    if prefetched_pairs is not None:
        actual_pairs, pair_err = prefetched_pairs
    else:
        actual_pairs, pair_err = nhl_api.fetch_shift_pairs(team_code, n_games, pids=set(roster_names) | {player_id})

    if player_id in roster_names:
        # Returning player: show the real pairs as-is, no cascade
        pairs, assigned = _greedy_shift_pairs(actual_pairs, set(scores), limit=MAX_DISPLAY)
        _fill_from_pool(pairs, [p for p in scores if p not in assigned and p in roster_names], scores, MAX_DISPLAY)
        cascade_log, scratched = [], []
    else:
        others = {p for p in scores if p != player_id}
        pairs, assigned = _greedy_shift_pairs(actual_pairs, others)
        _fill_from_pool(pairs, [p for p in others if p not in assigned and p in roster_names], scores, MAX_BUILD_PAIRS)
        cascade_log, scratched = _cascade(pairs, player_id, scores)

    depth_pairs = [_pair_record(i, p1, p2, scores, actual_pairs) for i, (p1, p2) in enumerate(pairs[:MAX_DISPLAY])]

    # Everyone modelled who didn't make the top pairs is an extra / scratch
    dressed = {p for pair in depth_pairs for p in (pair["pid1"], pair["pid2"])}
    extras  = sorted((p for p in scores if p not in dressed and p not in scratched),
                     key=lambda p: scores[p]["combined_score"], reverse=True)
    scratched = scratched + extras

    partner_name = partner_slot = "—"
    for pair in depth_pairs:
        if player_id in (pair["pid1"], pair["pid2"]):
            partner_name = pair["name2"] if pair["pid1"] == player_id else pair["name1"]
            partner_slot = pair["slot"]

    return depth_pairs, scratched, scores, cascade_log, [pid for pid in roster_names if pid not in scores], {
        "is_returning":   player_id in roster_names,
        "roster_names":   roster_names,
        "partner_name":   partner_name,
        "partner_slot":   partner_slot,
        "searched_score": scores[player_id]["combined_score"],
        "pair_err":       pair_err,
        "actual_pairs":   actual_pairs,
    }
