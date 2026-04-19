"""
test_api.py
===========
Tests all NHL API endpoints used by the app, with detailed shift data validation.
Run with:  python test_api.py [TEAM]

Example:   python test_api.py DAL
"""

import sys
import json
import requests
from collections import defaultdict

TEAM = sys.argv[1].upper() if len(sys.argv) > 1 else "DAL"

def ok(label):
    print(f"  PASS  {label}")

def fail(label, reason):
    print(f"  FAIL  {label}: {reason}")

def check(label, url, *, expect_key=None, print_sample=False):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if expect_key and expect_key not in data:
            fail(label, f"key '{expect_key}' not in response. Keys: {list(data.keys())}")
            return None
        if print_sample:
            items = data.get(expect_key, data) if expect_key else data
            if isinstance(items, list) and items:
                sample = items[0]
                print(f"         Sample fields: {list(sample.keys())}")
                print(f"         Sample values: {json.dumps(sample, default=str)[:300]}")
        ok(label)
        return data
    except Exception as e:
        fail(label, str(e))
        return None


def to_secs(t):
    if isinstance(t, (int, float)):
        return int(t)
    try:
        parts = str(t).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


print(f"\n{'='*60}")
print(f"  NHL API Test — Team: {TEAM}")
print(f"{'='*60}\n")

# ── 1. Roster ──────────────────────────────────────────────────────────────────
print("1. Roster")
roster_data = check(
    f"Roster ({TEAM})",
    f"https://api-web.nhle.com/v1/roster/{TEAM}/20252026",
)
d_pids = set()
d_names = {}
if roster_data:
    dmen = roster_data.get("defensemen", [])
    names = [f"{p['firstName']['default']} {p['lastName']['default']}" for p in dmen]
    d_pids = {p["id"] for p in dmen}
    d_names = {p["id"]: f"{p['firstName']['default']} {p['lastName']['default']}" for p in dmen}
    print(f"         Defensemen ({len(dmen)}): {names}")

# ── 2. Schedule ────────────────────────────────────────────────────────────────
print("\n2. Schedule — finding recent finished games")
import datetime

game_id = None
finished_games = []

for week_offset in range(6):
    check_date = (datetime.date.today() - datetime.timedelta(weeks=week_offset)).strftime("%Y-%m-%d")
    label = f"Schedule week -{week_offset} ({TEAM})"
    try:
        resp = requests.get(
            f"https://api-web.nhle.com/v1/club-schedule/{TEAM}/week/{check_date}",
            timeout=10
        )
        resp.raise_for_status()
        games = resp.json().get("games", [])
        done = [g for g in games
                if g.get("gameState") not in ("FUT", "PRE", "PREVIEW")
                and g.get("gameType", 2) == 2]
        if done:
            print(f"  PASS  {label} — {len(done)} finished game(s)")
            finished_games.extend(done)
        else:
            print(f"  SKIP  {label} — 0 finished games")
    except Exception as e:
        print(f"  FAIL  {label}: {e}")

    if len(finished_games) >= 5:
        break

# Deduplicate
seen = set()
unique = []
for g in finished_games:
    if g["id"] not in seen:
        seen.add(g["id"])
        unique.append(g)
unique.sort(key=lambda g: g.get("gameDate",""), reverse=True)
print(f"\n         Total unique finished games found: {len(unique)}")
if unique:
    print(f"         Most recent: {unique[0]['id']} on {unique[0].get('gameDate')}")
    game_id = unique[0]["id"]

# ── 3. Shift chart deep validation ────────────────────────────────────────────
print("\n3. Shift Chart — Deep Validation")

if not game_id:
    print("  SKIP  No game ID — cannot test shifts")
else:
    shift_data = check(
        f"Shiftcharts (game {game_id})",
        f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}",
        expect_key="data",
        print_sample=True,
    )
    if shift_data:
        shifts = shift_data["data"]
        team_all  = [s for s in shifts if s.get("teamAbbrev") == TEAM]
        team_reg  = [s for s in team_all if s.get("detailCode") == 0]
        d_shifts  = [s for s in team_reg if s.get("playerId") in d_pids]

        print(f"\n         Total shifts in game:        {len(shifts)}")
        print(f"         {TEAM} shifts (all):           {len(team_all)}")
        print(f"         {TEAM} regular shifts (dc=0):  {len(team_reg)}")
        print(f"         {TEAM} D-men shifts only:      {len(d_shifts)}")

        if not d_shifts:
            print(f"\n  WARN  No D-men shifts matched. D pids: {d_pids}")
            print(f"         Shift playerIds sample: {[s.get('playerId') for s in team_reg[:10]]}")
        else:
            # Compute pair TOI
            pair_toi = defaultdict(int)
            by_period = defaultdict(list)
            for s in d_shifts:
                by_period[s["period"]].append(s)

            for period, pshifts in by_period.items():
                pshifts.sort(key=lambda x: to_secs(x.get("startTime", 0)))
                n = len(pshifts)
                for i in range(n):
                    si = pshifts[i]
                    pid_i   = si["playerId"]
                    start_i = to_secs(si.get("startTime", 0))
                    end_i   = to_secs(si.get("endTime", 0))
                    for j in range(i + 1, n):
                        sj = pshifts[j]
                        start_j = to_secs(sj.get("startTime", 0))
                        if start_j >= end_i:
                            break
                        pid_j = sj["playerId"]
                        if pid_i == pid_j:
                            continue
                        end_j   = to_secs(sj.get("endTime", 0))
                        overlap = min(end_i, end_j) - max(start_i, start_j)
                        if overlap > 0:
                            key = tuple(sorted([pid_i, pid_j]))
                            pair_toi[key] += overlap

            print(f"\n         Pair TOI computed from {len(d_shifts)} D-men shifts:")
            if not pair_toi:
                print("  WARN  No pair overlap computed — check startTime/endTime parsing")
            else:
                pairs_sorted = sorted(pair_toi.items(), key=lambda x: x[1], reverse=True)
                print(f"         {'Pair':<40} {'Shared TOI':>12}")
                print(f"         {'-'*52}")
                for (p1, p2), toi in pairs_sorted[:8]:
                    n1 = d_names.get(p1, str(p1))
                    n2 = d_names.get(p2, str(p2))
                    mins = toi // 60
                    secs = toi % 60
                    print(f"         {n1 + ' — ' + n2:<40} {mins}:{secs:02d}")

        # Validate time parsing
        print(f"\n         Time format check:")
        sample_shift = d_shifts[0] if d_shifts else team_reg[0] if team_reg else None
        if sample_shift:
            st = sample_shift.get("startTime")
            et = sample_shift.get("endTime")
            print(f"           startTime raw: {st!r}  → {to_secs(st)}s")
            print(f"           endTime raw:   {et!r}  → {to_secs(et)}s")
            print(f"           duration:      {to_secs(et) - to_secs(st)}s")

# ── 4. Realtime stats ─────────────────────────────────────────────────────────
print("\n4. Realtime Stats (hits, takeaways, PIM)")
check("Realtime skater stats",
      "https://api.nhle.com/stats/rest/en/skater/realtime?limit=5&start=0&cayenneExp=seasonId=20252026",
      expect_key="data", print_sample=True)

# ── 5. Time on ice ────────────────────────────────────────────────────────────
print("\n5. Time on Ice (PK%)")
check("Skater time on ice",
      "https://api.nhle.com/stats/rest/en/skater/timeonice?limit=5&start=0&cayenneExp=seasonId=20252026",
      expect_key="data", print_sample=True)

# ── 6. Bios ───────────────────────────────────────────────────────────────────
print("\n6. Skater Bios (birth dates)")
check("Skater bios 2025-26",
      "https://api.nhle.com/stats/rest/en/skater/bios?limit=5&start=0&cayenneExp=seasonId=20252026",
      expect_key="data", print_sample=True)

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\n7. Skater Summary (points, goals)")
check("Skater summary",
      "https://api.nhle.com/stats/rest/en/skater/summary?limit=5&start=0&cayenneExp=seasonId=20252026",
      expect_key="data", print_sample=True)

print(f"\n{'='*60}")
print("  Done.")
print(f"{'='*60}\n")