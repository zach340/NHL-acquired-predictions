"""
nhl_api.py
==========
Every call to the public NHL APIs, plus the on-disk caches for player
names/headshots (player_names.csv) and shift-chart pairings (shifts_cache/).
"""

import base64
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import streamlit as st

from .config import CURRENT_SEASON, NAMES_FILE, SHIFTS_CACHE_DIR, SHIFTS_CACHE_TTL_H

STATS_API = "https://api.nhle.com/stats/rest/en"
WEB_API   = "https://api-web.nhle.com/v1"


def _get_json(url, timeout=15):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _skater_report(report, season):
    """A regular-season stats API skater report (summary, realtime, timeonice) as a DataFrame."""
    url = f"{STATS_API}/skater/{report}?limit=-1&start=0&cayenneExp=seasonId={season} and gameTypeId=2"
    return pd.json_normalize(_get_json(url).get("data", []))


# ── Season stats (validation) ─────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_season_skaters(season):
    """
    Regular-season skater stats for `season` ("20252026"): player_id, player_name,
    goals, points, games_played (≥10), goals_per_game, points_per_game.
    Returns (df, error).
    """
    try:
        df = _skater_report("summary", season)
        if df.empty:
            return None, "No data returned from NHL API."
        df = df.rename(columns={"playerId": "player_id", "skaterFullName": "player_name",
                                "gamesPlayed": "games_played"})
        keep = ["player_id", "player_name", "goals", "points", "games_played"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df = df.dropna(subset=["player_id", "goals", "points", "games_played"])
        df = df[df["games_played"] >= 10]
        df["goals_per_game"]  = df["goals"]  / df["games_played"]
        df["points_per_game"] = df["points"] / df["games_played"]
        df["player_id"] = df["player_id"].astype(int)
        return df, None
    except Exception as e:
        return None, str(e)


def _defense_only(df, cols):
    df = df[[c for c in cols if c in df.columns]].copy()
    if "position" in df.columns:
        df = df[df["position"] == "D"]
    df["player_id"] = df["player_id"].astype(int)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_defensive_stats(season):
    """
    Regular-season defenseman stats for `season` (≥10 GP) merging the realtime (hits,
    blocks, takeaways, giveaways), summary (PIM) and time-on-ice (PK TOI)
    reports, with per-game rates. Returns (df, error).
    """
    try:
        rt = _skater_report("realtime", season)
        if rt.empty:
            return None, "No realtime data returned."
        rt = rt.rename(columns={"playerId": "player_id", "skaterFullName": "player_name",
                                "positionCode": "position", "blockedShots": "blocked_shots",
                                "gamesPlayed": "games_played"})
        rt = rt[[c for c in ["player_id", "player_name", "position", "hits", "blocked_shots",
                             "takeaways", "giveaways", "games_played"] if c in rt.columns]].copy()
        if "position" in rt.columns:
            rt = rt[rt["position"] == "D"]
        rt = rt.dropna(subset=["player_id", "games_played"])
        rt = rt[rt["games_played"] >= 10]
        rt["player_id"] = rt["player_id"].astype(int)

        summary = _skater_report("summary", season)
        if not summary.empty:
            summary = summary.rename(columns={"playerId": "player_id", "positionCode": "position",
                                              "penaltyMinutes": "penalty_minutes"})
            summary = _defense_only(summary, ["player_id", "position", "penalty_minutes"])
            rt = rt.merge(summary[["player_id", "penalty_minutes"]], on="player_id", how="left")
        else:
            rt["penalty_minutes"] = np.nan

        toi = _skater_report("timeonice", season)
        if not toi.empty:
            toi = toi.rename(columns={"playerId": "player_id", "positionCode": "position",
                                      "shTimeOnIce": "pk_time_on_ice", "timeOnIce": "total_time_on_ice"})
            toi = _defense_only(toi, ["player_id", "position", "pk_time_on_ice", "total_time_on_ice"])
            df = rt.merge(toi[["player_id", "pk_time_on_ice", "total_time_on_ice"]], on="player_id", how="left")
        else:
            df = rt.copy()
            df["pk_time_on_ice"]    = np.nan
            df["total_time_on_ice"] = np.nan

        gp = df["games_played"]
        df["hits_pg"]      = df["hits"]          / gp
        df["blocks_pg"]    = df["blocked_shots"] / gp
        df["takeaways_pg"] = df["takeaways"]     / gp
        df["giveaways_pg"] = df["giveaways"]     / gp
        df["pim_pg"]       = df["penalty_minutes"] / gp
        return df.fillna(0), None
    except Exception as e:
        return None, str(e)


def fetch_shoots(player_id, season=CURRENT_SEASON):
    """'L' / 'R' handedness from the bios report, or '' if unavailable."""
    try:
        url = f"{STATS_API}/skater/bios?limit=1&cayenneExp=playerId={player_id} and seasonId={season}"
        resp = requests.get(url, timeout=8)
        if resp.ok:
            data = resp.json().get("data", [])
            if data:
                return data[0].get("shootsCatches", "")
    except Exception:
        pass
    return ""


# ── Rosters ───────────────────────────────────────────────────────────────────

def normalize_forward_position(raw_pos):
    """C / L / R for forwards (LW/RW normalised); None for anything else."""
    return {"LW": "L", "RW": "R", "C": "C", "L": "L", "R": "R"}.get(str(raw_pos).upper())


def _name_part(p, key):
    v = p.get(key)
    return v.get("default") if isinstance(v, dict) else v


def parse_roster_entries(entries, team_code):
    rows = []
    for p in entries:
        pid = p.get("id") or p.get("playerId")
        pos = normalize_forward_position(p.get("positionCode") or p.get("position"))
        if pid is None or pos is None:
            continue
        full_name = p.get("fullName") or " ".join(
            [str(_name_part(p, "firstName") or "").strip(), str(_name_part(p, "lastName") or "").strip()]
        ).strip()
        rows.append({
            "player_id":   int(pid),
            "player_name": full_name or str(p.get("name", "Unknown Player")),
            "position":    pos,
            "nhl_team":    team_code,
        })
    return rows


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_forwards(team_code, season=CURRENT_SEASON):
    """Forwards on a team's current roster as (DataFrame, error)."""
    try:
        data = _get_json(f"{WEB_API}/roster/{team_code}/{season}")
        roster_df = pd.DataFrame(parse_roster_entries(data.get("forwards", []), team_code))
        if roster_df.empty:
            return None, f"No skater roster data found for {team_code}."
        return roster_df.drop_duplicates(subset=["player_id"]).reset_index(drop=True), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_defensemen(team_code):
    """Current defensemen as [{player_id, player_name, position, shoots}] or [{"_error": msg}]."""
    try:
        data = _get_json(f"{WEB_API}/roster/{team_code}/{CURRENT_SEASON}", timeout=10)
        return [{
            "player_id":   int(p["id"]),
            "player_name": f"{p['firstName']['default']} {p['lastName']['default']}",
            "position":    "D",
            "shoots":      p.get("shootsCatches", ""),
        } for p in data.get("defensemen", [])]
    except Exception as e:
        return [{"_error": str(e)}]


# ── Player names & headshots (persistent cache) ───────────────────────────────

_NAMES_CACHE = None   # {player_id: {"name": str, "headshot_b64": str}}, loaded lazily


def _names_cache():
    global _NAMES_CACHE
    if _NAMES_CACHE is None:
        _NAMES_CACHE = {}
        if os.path.exists(NAMES_FILE):
            try:
                ndf = pd.read_csv(NAMES_FILE, dtype={"player_id": int})
                for _, row in ndf.iterrows():
                    b64 = row.get("headshot_b64")
                    _NAMES_CACHE[int(row["player_id"])] = {
                        "name":         str(row.get("name", "")),
                        "headshot_b64": str(b64) if pd.notna(b64) else "",
                    }
            except Exception:
                pass
    return _NAMES_CACHE


def _save_names_cache():
    try:
        pd.DataFrame([
            {"player_id": pid, "name": v.get("name", ""), "headshot_b64": v.get("headshot_b64", "")}
            for pid, v in _names_cache().items()
        ]).to_csv(NAMES_FILE, index=False, encoding="utf-8")
    except Exception:
        pass


def _ensure_player_cached(pid):
    """Fetch name + headshot from the NHL API unless both are already cached."""
    cache = _names_cache()
    entry = cache.get(pid, {})
    if entry.get("name") and entry.get("headshot_b64"):
        return
    try:
        data = _get_json(f"{WEB_API}/player/{pid}/landing", timeout=8)
        name = f"{data.get('firstName', {}).get('default', '')} {data.get('lastName', {}).get('default', '')}".strip()

        headshot_b64 = entry.get("headshot_b64", "")
        hs_url = data.get("headshot", "")
        if hs_url and not headshot_b64:
            try:
                hs_resp = requests.get(hs_url, timeout=8)
                if hs_resp.status_code == 200:
                    headshot_b64 = base64.b64encode(hs_resp.content).decode("utf-8")
            except Exception:
                headshot_b64 = ""

        if name or headshot_b64:
            cache[pid] = {"name": name or entry.get("name", ""), "headshot_b64": headshot_b64}
            _save_names_cache()
    except Exception:
        pass


def fetch_player_display_name(player_id):
    """Correctly-accented player name (from cache or NHL API), or None."""
    pid = int(player_id)
    if not _names_cache().get(pid, {}).get("name"):
        _ensure_player_cached(pid)
    return _names_cache().get(pid, {}).get("name") or None


def fetch_headshot_b64(player_id):
    """Base64 PNG headshot (from cache or NHL API), or ''."""
    pid = int(player_id)
    if not _names_cache().get(pid, {}).get("headshot_b64"):
        _ensure_player_cached(pid)
    return _names_cache().get(pid, {}).get("headshot_b64", "")


# ── Shift charts → actual pairings (persistent cache) ─────────────────────────

def _shifts_cache_path(team_code, n_games):
    os.makedirs(SHIFTS_CACHE_DIR, exist_ok=True)
    return os.path.join(SHIFTS_CACHE_DIR, f"{team_code}_{n_games}.json")


def _load_shifts_disk_cache(team_code, n_games):
    """(pairs, err) from disk, or None if missing/stale/unreadable."""
    path = _shifts_cache_path(team_code, n_games)
    try:
        if not os.path.exists(path) or time.time() - os.path.getmtime(path) > SHIFTS_CACHE_TTL_H * 3600:
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        return [tuple(p) for p in payload["pairs"]], payload.get("err")
    except Exception:
        return None


def _save_shifts_disk_cache(team_code, n_games, pairs, err):
    try:
        with open(_shifts_cache_path(team_code, n_games), "w") as f:
            json.dump({"pairs": [list(p) for p in pairs], "err": err}, f)
    except Exception:
        pass   # disk write failure is non-fatal


def clear_shifts_cache(team_code, n_games):
    try:
        os.remove(_shifts_cache_path(team_code, n_games))
    except OSError:
        pass


def _to_secs(t):
    if isinstance(t, (int, float)):
        return int(t)
    try:
        minutes, seconds = str(t).split(":")[:2]
        return int(minutes) * 60 + int(seconds)
    except Exception:
        return 0


def _finished_regular_season_games(team_code, season):
    games = _get_json(f"{WEB_API}/club-schedule-season/{team_code}/{season}").get("games", [])
    return [g for g in games
            if g.get("gameType", 2) == 2 and g.get("gameState") not in ("FUT", "PRE", "PREVIEW")]


def _recent_finished_games(team_code, n_games):
    """
    The n most recent finished regular-season games, topped up from last
    season when the current one has barely started (e.g. in preseason).
    """
    start = int(CURRENT_SEASON[:4])
    finished = _finished_regular_season_games(team_code, CURRENT_SEASON)
    if len(finished) < n_games:
        finished += _finished_regular_season_games(team_code, f"{start - 1}{start}")
    return sorted(finished, key=lambda g: g.get("gameDate", ""), reverse=True)[:n_games]


def _pair_overlaps(games, team_code, pids, on_progress=None):
    """
    Seconds of shared ice time for every pair of `team_code` players (limited
    to `pids` if given) across `games`. Returns (pairs sorted by TOI desc, error).
    """
    pair_toi = defaultdict(int)
    errors   = []
    for idx, game in enumerate(games):
        game_id = game.get("id")
        if game_id:
            try:
                shifts = _get_json(f"{STATS_API}/shiftcharts?cayenneExp=gameId={game_id}").get("data", [])
                by_period = defaultdict(list)
                for s in shifts:
                    if (s.get("teamAbbrev") == team_code and s.get("detailCode") == 0
                            and (pids is None or s.get("playerId") in pids)):
                        by_period[s["period"]].append(s)

                for period_shifts in by_period.values():
                    period_shifts.sort(key=lambda x: _to_secs(x.get("startTime", 0)))
                    for i, si in enumerate(period_shifts):
                        start_i, end_i = _to_secs(si.get("startTime", 0)), _to_secs(si.get("endTime", 0))
                        for sj in period_shifts[i + 1:]:
                            start_j = _to_secs(sj.get("startTime", 0))
                            if start_j >= end_i:
                                break
                            if si.get("playerId") == sj.get("playerId"):
                                continue
                            overlap = min(end_i, _to_secs(sj.get("endTime", 0))) - max(start_i, start_j)
                            if overlap > 0:
                                pair_toi[tuple(sorted([si.get("playerId"), sj.get("playerId")]))] += overlap
            except Exception as e:
                errors.append(f"game {game_id}: {e}")
        if on_progress:
            on_progress(idx + 1, len(games))

    if not pair_toi:
        return [], (f"Shift data unavailable ({'; '.join(errors[:3])})." if errors
                    else "No shift overlap data found.")
    pairs = sorted(pair_toi.items(), key=lambda x: x[1], reverse=True)
    return [(p[0], p[1], toi) for p, toi in pairs], None


def fetch_shift_pairs(team_code, n_games=25, pids=None, on_progress=None):
    """
    Actual line-mate pairs from NHL shift charts over the last `n_games`:
    list of (pid1, pid2, shared_seconds) sorted by shared TOI, plus an error
    string. Served from the disk cache when fresh; on_progress(done, total)
    is called per game otherwise.
    """
    cached = _load_shifts_disk_cache(team_code, n_games)
    if cached is not None:
        return cached
    try:
        games = _recent_finished_games(team_code, n_games)
        if not games:
            return [], "No finished regular-season games found for this team."
        pairs, err = _pair_overlaps(games, team_code, set(pids) if pids else None, on_progress)
        _save_shifts_disk_cache(team_code, n_games, pairs, err)
        return pairs, err
    except Exception as e:
        return [], str(e)
