"""
nhl_api_download.py
===================
Download raw play-by-play + shift charts for every regular-season game from
the public NHL API (2010-11 onward — the first season with shot coordinates
and shift charts) into raw_data/nhl_api/<season>/<game_id>.json.gz.

Resumable: games already on disk are skipped, so rerun it any time (e.g.
during a season) to pick up newly finished games.

    python pipeline/nhl_api_download.py                 # every season 2010 → current
    python pipeline/nhl_api_download.py 2023 2024       # just these seasons (start years)
"""

import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nhl_predictor.config import CURRENT_SEASON  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR     = os.environ.get("NHL_API_RAW_DIR", "raw_data/nhl_api")
FIRST_SEASON = 2010
PBP_URL     = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
SHIFTS_URL  = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}"
BOX_URL     = "https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore"
WORKERS     = 4
MIN_GAP     = float(os.environ.get("NHL_API_MIN_GAP", "0.6"))   # seconds between requests, all threads
MAX_GAMES   = 1400          # 32 teams × 82 / 2 = 1312; stop well past it
FINISHED    = {"OFF", "FINAL"}

_session = requests.Session()


class _Throttle:
    """Shared pacing: at most one request per MIN_GAP, and a global pause on 429 Retry-After."""

    def __init__(self, gap):
        self.gap, self.next_at, self.lock = gap, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.gap
        time.sleep(max(0.0, start - now))

    def back_off(self, seconds):
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic() + seconds)


_throttle = _Throttle(MIN_GAP)


def game_path(season, gid):
    return os.path.join(RAW_DIR, str(season), f"{gid}.json.gz")


def load_game(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _get(url, tries=8):
    """GET JSON with backoff on throttling / server errors. Returns (status, json|None)."""
    for attempt in range(tries):
        _throttle.wait()
        try:
            r = _session.get(url, timeout=30)
        except requests.RequestException:
            _throttle.back_off(2 ** attempt)
            continue
        if r.status_code == 404:
            return 404, None
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 30))
            print(f"  throttled — pausing {wait:.0f}s", flush=True)
            _throttle.back_off(wait + 1)
            continue
        if r.status_code in (500, 502, 503, 504):
            _throttle.back_off(2 ** attempt)
            continue
        r.raise_for_status()
        return r.status_code, r.json()
    raise RuntimeError(f"giving up on {url}")


def fetch_game(season, gid):
    """'saved' | 'missing' (no such game) | 'unfinished'."""
    status, pbp = _get(PBP_URL.format(gid=gid))
    if status == 404:
        return "missing"
    if pbp.get("gameState") not in FINISHED:
        return "unfinished"
    _, shifts = _get(SHIFTS_URL.format(gid=gid))
    game = {"pbp": pbp, "shifts": (shifts or {}).get("data", [])}
    if not game["shifts"]:
        game["boxscore"] = _get(BOX_URL.format(gid=gid))[1]   # TOI fallback when no shift chart exists
    _save(game_path(season, gid), game)
    return "saved"


def _save(path, game):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(game, f, separators=(",", ":"))
    os.replace(tmp, path)


def backfill_boxscores(season):
    """Add the boxscore to saved games that have no shift chart (older downloads skipped it)."""
    added = 0
    for name in sorted(os.listdir(os.path.join(RAW_DIR, str(season)))):
        if not name.endswith(".json.gz"):
            continue
        path = os.path.join(RAW_DIR, str(season), name)
        game = load_game(path)
        if game["shifts"] or game.get("boxscore"):
            continue
        game["boxscore"] = _get(BOX_URL.format(gid=game["pbp"]["id"]))[1]
        _save(path, game)
        added += 1
    if added:
        print(f"  {season}: added boxscores to {added} games without shift charts", flush=True)


def download_season(season):
    gids = [int(f"{season}02{n:04d}") for n in range(1, MAX_GAMES + 1)]
    todo = [g for g in gids if not os.path.exists(game_path(season, g))]
    have = len(gids) - len(todo)
    counts = {"saved": 0, "missing": 0, "unfinished": 0, "error": 0}
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(fetch_game, season, g): g for g in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                counts[fut.result()] += 1
            except Exception as e:  # keep going; a rerun retries it
                counts["error"] += 1
                print(f"  {futures[fut]}: {e}", flush=True)
            if i % 200 == 0:
                print(f"  {season}: {i}/{len(todo)} checked, {counts['saved']} saved "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"{season}-{(season + 1) % 100:02d}: {have + counts['saved']} games on disk "
          f"(+{counts['saved']} new, {counts['unfinished']} unfinished, {counts['error']} errors)", flush=True)


def main(seasons):
    for season in seasons:
        download_season(season)
        backfill_boxscores(season)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    current = int(str(CURRENT_SEASON)[:4])
    main(args or range(FIRST_SEASON, current + 1))
