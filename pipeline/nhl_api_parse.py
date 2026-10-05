"""
nhl_api_parse.py
================
Turn the raw games downloaded by nhl_api_download.py into three per-season
tables under raw_data/nhl_api_parsed/<season>/:

  shots.parquet         every shot attempt (goal / shot-on-goal / missed / blocked)
                        with xG-model inputs, shooter/assists/blocker and the
                        on-ice segment it happened in
  segments.parquet      the game cut into stretches with no line change: start,
                        end, skaters on ice per team (h1..h6 / a1..a6), goalie
                        flags and score — used for TOI by strength and
                        on-ice stats
  player_games.parquet  per player per game: TOI (all / 5v5 / 5v4 / 4v5),
                        shifts, hits, takeaways, giveaways, penalties,
                        faceoffs, blocks and zone starts

Shootouts are dropped. All times are seconds from the start of the game.

    python pipeline/nhl_api_parse.py              # every downloaded season that changed
    python pipeline/nhl_api_parse.py 2023 2024    # just these seasons
    python pipeline/nhl_api_parse.py --force      # re-parse everything

Seasons whose raw games haven't changed since the last parse are skipped
(tracked in each parsed folder's _manifest.json).
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhl_api_download import RAW_DIR, load_game  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PARSED_DIR = os.environ.get("NHL_API_PARSED_DIR", "raw_data/nhl_api_parsed")
NET_X      = 89.0
SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot", "blocked-shot"}
MAX_SKATERS = 6


def _secs(mmss):
    if not mmss:
        return np.nan
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def _game_time(period, mmss):
    return (period - 1) * 1200 + _secs(mmss)


# ── Shifts → segments ──────────────────────────────────────────────────────────

def _shift_frame(shifts, roster_pos):
    rows = [(s["playerId"], s["teamId"], _game_time(s["period"], s["startTime"]),
             _game_time(s["period"], s["endTime"]))
            for s in shifts if s.get("typeCode") == 517 and s.get("startTime") and s.get("endTime")]
    sh = pd.DataFrame(rows, columns=["player_id", "team_id", "start", "end"])
    sh = sh[sh["end"] > sh["start"]].drop_duplicates()
    sh["is_goalie"] = sh["player_id"].map(roster_pos).eq("G")
    return sh.reset_index(drop=True)


def build_segments(sh, home_id, away_id):
    """Cut the game at every shift start/end; return (segments frame, membership matrix shifts×segments)."""
    bounds = np.unique(np.concatenate([sh["start"].values, sh["end"].values]))
    seg_start, seg_end = bounds[:-1], bounds[1:]
    member = (sh["start"].values[:, None] <= seg_start[None, :]) & (sh["end"].values[:, None] >= seg_end[None, :])

    seg = pd.DataFrame({"start": seg_start, "end": seg_end, "duration": seg_end - seg_start})
    for side, team in (("h", home_id), ("a", away_id)):
        team_mask = (sh["team_id"].values == team)
        skater = member & (team_mask & ~sh["is_goalie"].values)[:, None]
        goalie = member & (team_mask & sh["is_goalie"].values)[:, None]
        seg[f"{side}_skaters"] = skater.sum(axis=0)
        seg[f"{side}_goalie"]  = goalie.any(axis=0)
        pids = sh["player_id"].values
        cols = np.full((len(seg), MAX_SKATERS), -1, dtype=np.int64)
        for j in range(len(seg)):
            ids = np.sort(pids[skater[:, j]])[:MAX_SKATERS]
            cols[j, :len(ids)] = ids
        for k in range(MAX_SKATERS):
            seg[f"{side}{k + 1}"] = cols[:, k]
    return seg, member


def _segment_index(bounds_start, bounds_end, t, faceoff=False):
    """Segment holding event time t: (start, end] normally, [start, end) for faceoffs."""
    if not len(bounds_start):
        return -1
    side = "right" if faceoff else "left"
    i = np.searchsorted(bounds_end if not faceoff else bounds_start, t, side=side)
    i = i - 1 if faceoff else i
    return int(i) if 0 <= i < len(bounds_start) else -1


# ── Play-by-play ───────────────────────────────────────────────────────────────

def _attack_directions(plays, team_of):
    """{(period, team_id): +1/-1} — which end each team attacks, from its offensive-zone shots."""
    xs = {}
    for p in plays:
        d = p.get("details") or {}
        if p["typeDescKey"] not in ("goal", "shot-on-goal", "missed-shot"):
            continue
        team = team_of.get(d.get("scoringPlayerId") or d.get("shootingPlayerId"))
        if team is not None and d.get("zoneCode") == "O" and d.get("xCoord") is not None:
            xs.setdefault((p["periodDescriptor"]["number"], team), []).append(d["xCoord"])
    return {k: (1 if np.median(v) > 0 else -1) for k, v in xs.items() if np.median(v) != 0}


def _direction(dirs, period, team, other):
    for per in (period, period - 1, period + 1):
        flip = 1 if per == period else -1
        if (per, team) in dirs:
            return dirs[(per, team)] * flip
        if (per, other) in dirs:
            return -dirs[(per, other)] * flip
    return 1


def parse_game(path):
    g = load_game(path)
    pbp, shifts = g["pbp"], g["shifts"]
    gid, season = pbp["id"], int(str(pbp["season"])[:4])
    home, away = pbp["homeTeam"], pbp["awayTeam"]
    home_id, away_id = home["id"], away["id"]
    abbrev = {home_id: home["abbrev"], away_id: away["abbrev"]}
    roster = pbp.get("rosterSpots", [])
    team_of = {r["playerId"]: r["teamId"] for r in roster}
    pos_of  = {r["playerId"]: r["positionCode"] for r in roster}
    name_of = {r["playerId"]: f"{r['firstName']['default']} {r['lastName']['default']}" for r in roster}

    plays = [p for p in pbp["plays"] if p["periodDescriptor"].get("periodType") != "SO"]
    dirs = _attack_directions(plays, team_of)

    sh = _shift_frame(shifts, pos_of)
    has_shifts = len(sh) > 0 and sh["player_id"].nunique() >= 30
    if has_shifts:
        seg, member = build_segments(sh, home_id, away_id)
    else:
        seg, member = pd.DataFrame(columns=["start", "end", "duration"]), None
    s_start, s_end = seg["start"].values, seg["end"].values

    # ── shots + individual events ──
    shots, events = [], []
    home_score = away_score = 0
    prev = None
    for p in plays:
        kind, d = p["typeDescKey"], p.get("details") or {}
        period = p["periodDescriptor"]["number"]
        t = _game_time(period, p.get("timeInPeriod"))
        code = p.get("situationCode") or "1551"
        away_g, away_sk, home_sk, home_g = (int(c) for c in code) if len(code) == 4 else (1, 5, 5, 1)
        x, y = d.get("xCoord"), d.get("yCoord")

        if kind in SHOT_TYPES:
            shooter = d.get("scoringPlayerId") or d.get("shootingPlayerId")
            team = team_of.get(shooter, d.get("eventOwnerTeamId"))
            is_home = team == home_id
            other = away_id if is_home else home_id
            direction = _direction(dirs, period, team, other)
            own_sk, opp_sk = (home_sk, away_sk) if is_home else (away_sk, home_sk)
            own_g, opp_g = (home_g, away_g) if is_home else (away_g, home_g)
            xn = x * direction if x is not None else np.nan
            yn = y if y is not None else np.nan
            dx = NET_X - xn
            dist = float(np.hypot(dx, yn)) if x is not None and y is not None else np.nan
            angle = float(np.degrees(np.arctan2(abs(yn), dx))) if x is not None and y is not None else np.nan
            row = dict(
                game_id=gid, season=season, period=period, t=t, event_id=p["eventId"], kind=kind,
                team_id=team, is_home=is_home, shooter=shooter,
                goalie=d.get("goalieInNetId"), blocker=d.get("blockingPlayerId"),
                assist1=d.get("assist1PlayerId"), assist2=d.get("assist2PlayerId"),
                shot_type=d.get("shotType"), x=xn, y=yn, dist=dist, angle=angle,
                own_skaters=own_sk, opp_skaters=opp_sk, own_goalie=own_g, opp_goalie=opp_g,
                score_diff=(home_score - away_score) * (1 if is_home else -1),
                is_goal=kind == "goal",
                seg=_segment_index(s_start, s_end, t),
            )
            if prev is not None:
                row.update(prev_kind=prev["kind"], prev_same_team=prev["team"] == team,
                           prev_dt=t - prev["t"],
                           prev_dist=float(np.hypot(x - prev["x"], y - prev["y"]))
                           if None not in (x, y, prev["x"], prev["y"]) else np.nan,
                           prev_xn=prev["x"] * direction if prev["x"] is not None else np.nan)
            shots.append(row)
            if kind == "goal":
                if is_home:
                    home_score += 1
                else:
                    away_score += 1
        elif kind == "hit":
            events.append((d.get("hittingPlayerId"), "hits", 1))
        elif kind == "takeaway":
            events.append((d.get("playerId"), "takeaways", 1))
        elif kind == "giveaway":
            events.append((d.get("playerId"), "giveaways", 1))
            if d.get("zoneCode") == "D":
                events.append((d.get("playerId"), "d_zone_giveaways", 1))
        elif kind == "penalty":
            mins = d.get("duration") or 0
            if d.get("committedByPlayerId"):
                events += [(d["committedByPlayerId"], "penalties", 1), (d["committedByPlayerId"], "pim", mins)]
            if d.get("drawnByPlayerId"):
                events += [(d["drawnByPlayerId"], "penalties_drawn", 1), (d["drawnByPlayerId"], "pim_drawn", mins)]
        elif kind == "faceoff":
            events += [(d.get("winningPlayerId"), "faceoffs_won", 1), (d.get("losingPlayerId"), "faceoffs_lost", 1)]
        if kind == "blocked-shot" and d.get("blockingPlayerId"):
            events.append((d["blockingPlayerId"], "shots_blocked", 1))

        team_owner = d.get("eventOwnerTeamId")
        prev = dict(kind=kind, t=t, x=x, y=y,
                    team=team_of.get(d.get("scoringPlayerId") or d.get("shootingPlayerId"), team_owner))

    # ── player-game table ──
    players = set(team_of)
    pg = pd.DataFrame({"player_id": sorted(players)})
    pg["game_id"], pg["season"] = gid, season
    pg["team_id"] = pg["player_id"].map(team_of)
    pg["player_team"] = pg["team_id"].map(abbrev)
    pg["position"] = pg["player_id"].map(pos_of)
    pg["player_name"] = pg["player_id"].map(name_of)
    pg["is_home"] = pg["team_id"] == home_id
    pg["has_shifts"] = has_shifts

    ev = pd.DataFrame(events, columns=["player_id", "stat", "value"]).dropna()
    if len(ev):
        wide = ev.groupby(["player_id", "stat"])["value"].sum().unstack(fill_value=0)
        pg = pg.merge(wide, left_on="player_id", right_index=True, how="left")
    for col in ("hits", "takeaways", "giveaways", "d_zone_giveaways", "penalties", "pim",
                "penalties_drawn", "pim_drawn", "faceoffs_won", "faceoffs_lost", "shots_blocked"):
        pg[col] = pg[col].fillna(0) if col in pg.columns else 0

    if has_shifts:
        dur = seg["duration"].values.astype(float)
        both_g = seg["h_goalie"].values & seg["a_goalie"].values
        hs, as_ = seg["h_skaters"].values, seg["a_skaters"].values
        sit = {
            "toi_5v5": (both_g & (hs == 5) & (as_ == 5), both_g & (hs == 5) & (as_ == 5)),
            "toi_5v4": (both_g & (hs == 5) & (as_ == 4), both_g & (as_ == 5) & (hs == 4)),
            "toi_4v5": (both_g & (hs == 4) & (as_ == 5), both_g & (as_ == 4) & (hs == 5)),
        }
        pid_arr = sh["player_id"].values
        toi = {}
        for pid in np.unique(pid_arr):
            on = member[pid_arr == pid].any(axis=0)
            home_side = team_of.get(pid) == home_id
            rec = {"toi_all": float(dur[on].sum()), "shifts": int((pid_arr == pid).sum())}
            for name, (h_mask, a_mask) in sit.items():
                rec[name] = float(dur[on & (h_mask if home_side else a_mask)].sum())
            toi[pid] = rec
        toi = pd.DataFrame.from_dict(toi, orient="index")
        pg = pg.merge(toi, left_on="player_id", right_index=True, how="left")

        # Zone starts: a shift that begins exactly at a faceoff
        fo = [(_game_time(p["periodDescriptor"]["number"], p.get("timeInPeriod")),
               (p.get("details") or {}).get("zoneCode"), (p.get("details") or {}).get("eventOwnerTeamId"))
              for p in plays if p["typeDescKey"] == "faceoff"]
        fo = pd.DataFrame(fo, columns=["start", "zone", "owner"]).drop_duplicates("start")
        zs = sh.merge(fo, on="start")
        if len(zs):
            flip = {"O": "D", "D": "O", "N": "N"}
            zs["rel"] = np.where(zs["team_id"] == zs["owner"], zs["zone"], zs["zone"].map(flip))
            zc = zs.groupby(["player_id", "rel"]).size().unstack(fill_value=0)
            zc = zc.rename(columns={"O": "o_zone_starts", "D": "d_zone_starts", "N": "n_zone_starts"})
            pg = pg.merge(zc, left_on="player_id", right_index=True, how="left")
    elif g.get("boxscore"):
        # No shift chart: total TOI and shifts from the boxscore; strength splits stay NaN
        box = g["boxscore"]["playerByGameStats"]
        rec = {p["playerId"]: (_secs(p.get("toi")), p.get("shifts"))
               for side in ("homeTeam", "awayTeam") for grp in ("forwards", "defense") for p in box[side][grp]}
        pg["toi_all"] = pg["player_id"].map({k: v[0] for k, v in rec.items()})
        pg["shifts"] = pg["player_id"].map({k: v[1] for k, v in rec.items()})
    for col in ("toi_all", "toi_5v5", "toi_5v4", "toi_4v5", "shifts",
                "o_zone_starts", "d_zone_starts", "n_zone_starts"):
        if col not in pg.columns:
            pg[col] = np.nan if col.startswith("toi") or col == "shifts" else 0
        elif not col.startswith("toi") and col != "shifts":
            pg[col] = pg[col].fillna(0)
    # Dressed but never on the ice (backup goalies)
    if has_shifts or g.get("boxscore"):
        pg = pg[pg["toi_all"].fillna(0) > 0]

    seg.insert(0, "game_id", gid)
    seg["seg"] = np.arange(len(seg))
    return pd.DataFrame(shots), seg, pg


OUTPUTS = ("shots.parquet", "segments.parquet", "player_games.parquet")


def _raw_signature(files):
    """Game count + newest file time: changes whenever a game is added or rewritten (e.g. boxscore backfill)."""
    return {"games": len(files), "newest": max(os.path.getmtime(f) for f in files)}


def _up_to_date(out, signature):
    manifest = os.path.join(out, "_manifest.json")
    if not all(os.path.exists(os.path.join(out, f)) for f in OUTPUTS) or not os.path.exists(manifest):
        return False
    with open(manifest) as f:
        return json.load(f) == signature


def parse_season(season, workers=None, force=False):
    files = sorted(glob.glob(os.path.join(RAW_DIR, str(season), "*.json.gz")))
    if not files:
        print(f"{season}: no games downloaded")
        return
    out = os.path.join(PARSED_DIR, str(season))
    signature = _raw_signature(files)
    if not force and _up_to_date(out, signature):
        print(f"{season}-{(season + 1) % 100:02d}: unchanged, skipped", flush=True)
        return
    shots, segs, pgs, bad = [], [], [], 0
    with ProcessPoolExecutor(workers) as pool:
        for path, res in zip(files, pool.map(_safe_parse, files, chunksize=8)):
            if isinstance(res, str):
                bad += 1
                print(f"  {os.path.basename(path)}: {res}", flush=True)
                continue
            s, sg, pg = res
            shots.append(s); segs.append(sg); pgs.append(pg)
    os.makedirs(out, exist_ok=True)
    shots = pd.concat(shots, ignore_index=True)
    pgs = pd.concat(pgs, ignore_index=True)
    segs = pd.concat(segs, ignore_index=True)
    shots.to_parquet(os.path.join(out, "shots.parquet"), index=False)
    segs.to_parquet(os.path.join(out, "segments.parquet"), index=False)
    pgs.to_parquet(os.path.join(out, "player_games.parquet"), index=False)
    with open(os.path.join(out, "_manifest.json"), "w") as f:
        json.dump(signature, f)
    no_shifts = (~pgs.groupby("game_id")["has_shifts"].first()).sum()
    print(f"{season}-{(season + 1) % 100:02d}: {len(files) - bad} games, {len(shots):,} shot attempts, "
          f"{len(pgs):,} player-games, {no_shifts} games without shift data, {bad} failed", flush=True)


def _safe_parse(path):
    try:
        return parse_game(path)
    except Exception as e:  # report and continue; one bad game shouldn't sink a season
        return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="*", type=int, help="start years (default: every downloaded season)")
    ap.add_argument("--force", action="store_true", help="re-parse even if the raw games haven't changed")
    a = ap.parse_args()
    for s in a.seasons or sorted(int(d) for d in os.listdir(RAW_DIR) if d.isdigit()):
        parse_season(s, force=a.force)
