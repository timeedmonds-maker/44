#!/usr/bin/env python3
"""Recover WNBA player potential assists from the hidden v3 matchup box score.

Method:
1. Pull official WNBA league game logs for calendar-year seasons.
2. For every game, fetch /stats/boxscorematchupsv3.
3. Each offensive player has opponent matchup rows with matchupPotentialAssists.
4. Sum matchupPotentialAssists across that player's defender rows to get a player-game total.
5. Aggregate player-game totals to season totals.

This intentionally requests the WNBA host with a Chrome TLS fingerprint via curl_cffi.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from curl_cffi import requests

BASE = "https://stats.wnba.com/stats"
SEASONS = ["2025", "2026"]
OUT = Path("outputs/wnba_potential_assists")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.wnba.com",
    "Referer": "https://www.wnba.com/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
}


def get_json(path: str, params: list[tuple[str, str]], tries: int = 5, timeout: int = 30):
    url = f"{BASE}/{path}"
    err = None
    for attempt in range(tries):
        try:
            r = requests.get(
                url,
                params=params,
                headers=HEADERS,
                impersonate="chrome",
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
            err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            err = e
        time.sleep((0.6 * (2**attempt)) + random.random() * 0.35)
    raise RuntimeError(f"GET {path} failed: {err}")


def first_result_set(payload: dict) -> tuple[list[str], list[list]]:
    sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(sets, dict):
        sets = [sets]
    if not sets:
        raise ValueError(f"No resultSets: keys={list(payload)[:20]}")
    rs = sets[0]
    return rs.get("headers", []), rs.get("rowSet", [])


def game_log(season: str) -> list[dict]:
    # Param order is deliberate. WNBA host has been observed to reject other orderings.
    params = [
        ("LeagueID", "10"),
        ("Season", season),
        ("SeasonType", "Regular Season"),
        ("PlayerOrTeam", "T"),
        ("Counter", "0"),
        ("Direction", "ASC"),
        ("Sorter", "DATE"),
        ("DateFrom", ""),
        ("DateTo", ""),
    ]
    p = get_json("leaguegamelog", params)
    h, rows = first_result_set(p)
    out = [dict(zip(h, r)) for r in rows]
    if not out:
        raise RuntimeError(f"leaguegamelog {season}: zero rows")
    return out


def unique_games(season: str) -> list[dict]:
    rows = game_log(season)
    games: dict[str, dict] = {}
    for r in rows:
        gid = str(r.get("GAME_ID") or r.get("Game_ID") or r.get("gameId") or "")
        if not gid:
            continue
        g = games.setdefault(gid, {"season": season, "game_id": gid, "game_date": r.get("GAME_DATE") or r.get("GAME_DATE_EST") or "", "teams": []})
        if r.get("TEAM_ABBREVIATION") and r.get("TEAM_ABBREVIATION") not in g["teams"]:
            g["teams"].append(r["TEAM_ABBREVIATION"])
    return sorted(games.values(), key=lambda x: (str(x["game_date"]), x["game_id"]))


def matchup_rows(game: dict) -> list[dict]:
    gid = game["game_id"]
    p = get_json("boxscorematchupsv3", [("GameID", gid)])
    root = p.get("boxScoreMatchups")
    if not isinstance(root, dict):
        return []
    out = []
    for side in ("homeTeam", "awayTeam"):
        team = root.get(side) or {}
        team_id = team.get("teamId")
        team_tri = team.get("teamTricode")
        for player in team.get("players") or []:
            player_id = player.get("personId")
            player_name = " ".join(x for x in [player.get("firstName"), player.get("familyName")] if x).strip()
            pa = 0
            ast = 0
            n_matchups = 0
            for m in player.get("matchups") or []:
                s = m.get("statistics") or {}
                v = s.get("matchupPotentialAssists")
                a = s.get("matchupAssists")
                if v is not None:
                    pa += int(v)
                if a is not None:
                    ast += int(a)
                n_matchups += 1
            out.append({
                "season": game["season"],
                "game_id": gid,
                "game_date": game.get("game_date", ""),
                "team_id": team_id,
                "team_abbreviation": team_tri,
                "player_id": player_id,
                "player_name": player_name,
                "potential_assists": pa,
                "matchup_assists": ast,
                "matchup_rows": n_matchups,
            })
    return out


def traditional_assists(game_id: str) -> dict[int, int]:
    # Small QA helper: sum-matchup assists should equal ordinary player assists for most/all players.
    try:
        p = get_json("boxscoretraditionalv3", [("GameID", game_id)], tries=3)
        root = p.get("boxScoreTraditional") or {}
        out = {}
        for side in ("homeTeam", "awayTeam"):
            for player in (root.get(side) or {}).get("players") or []:
                pid = player.get("personId")
                ast = (player.get("statistics") or {}).get("assists")
                if pid is not None and ast is not None:
                    out[int(pid)] = int(ast)
        return out
    except Exception:
        return {}


def parse_date(s: str):
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%b %d, %Y %I:%M:%S %p"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except Exception:
            pass
    return None


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    # Hard probe first: known 2026 PHX-GSV game.
    probe = {"season": "2026", "game_id": "1022600011", "game_date": "2026-05-10", "teams": ["PHX", "GSV"]}
    probe_rows = matchup_rows(probe)
    print(f"PROBE game={probe['game_id']} rows={len(probe_rows)} nonzero_PA={sum(r['potential_assists']>0 for r in probe_rows)} total_PA={sum(r['potential_assists'] for r in probe_rows)}")
    if not probe_rows:
        raise SystemExit("Matchup endpoint returned no WNBA player rows; aborting full crawl")

    games = []
    for season in SEASONS:
        gs = unique_games(season)
        print(f"SCHEDULE season={season} games={len(gs)}")
        games.extend(gs)

    # Crawl in parallel but conservatively; retry logic handles transient throttles.
    all_rows: list[dict] = []
    failures = []
    workers = int(os.environ.get("WNBA_WORKERS", "6"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(matchup_rows, g): g for g in games}
        done = 0
        for fut in as_completed(futs):
            g = futs[fut]
            try:
                rows = fut.result()
                if rows:
                    all_rows.extend(rows)
                else:
                    failures.append({**g, "error": "zero matchup rows"})
            except Exception as e:
                failures.append({**g, "error": str(e)[:500]})
            done += 1
            if done % 25 == 0 or done == len(games):
                print(f"PROGRESS {done}/{len(games)} player_game_rows={len(all_rows)} failures={len(failures)}")

    # De-dupe player-game rows defensively.
    dedup = {}
    for r in all_rows:
        dedup[(r["game_id"], r["player_id"])] = r
    all_rows = list(dedup.values())
    all_rows.sort(key=lambda r: (r["season"], str(r["game_date"]), r["game_id"], str(r["team_abbreviation"]), str(r["player_name"])))

    # QA sample: compare matchup assists vs traditional assists for first 12 successful games.
    qa = []
    by_game = defaultdict(list)
    for r in all_rows:
        by_game[r["game_id"]].append(r)
    for gid in list(by_game)[:12]:
        trad = traditional_assists(gid)
        for r in by_game[gid]:
            pid = int(r["player_id"]) if r["player_id"] is not None else -1
            if pid in trad:
                qa.append({
                    "game_id": gid,
                    "player_id": pid,
                    "player_name": r["player_name"],
                    "matchup_assists": r["matchup_assists"],
                    "traditional_assists": trad[pid],
                    "diff": r["matchup_assists"] - trad[pid],
                })

    # Aggregate season totals.
    ag = {}
    for r in all_rows:
        key = (r["season"], r["player_id"], r["player_name"], r["team_abbreviation"])
        x = ag.setdefault(key, {"season": r["season"], "player_id": r["player_id"], "player_name": r["player_name"], "team_abbreviation": r["team_abbreviation"], "games_with_matchup_data": 0, "potential_assists": 0, "matchup_assists": 0})
        x["games_with_matchup_data"] += 1
        x["potential_assists"] += r["potential_assists"]
        x["matchup_assists"] += r["matchup_assists"]
    totals = list(ag.values())
    for x in totals:
        gp = x["games_with_matchup_data"] or 1
        x["potential_assists_per_game"] = round(x["potential_assists"] / gp, 3)
    totals.sort(key=lambda x: (x["season"], -x["potential_assists"], x["player_name"]))

    # Alyssa Thomas 2026 validation through May 27 (Phoenix was 2-5 = 7 games).
    at_rows = []
    for r in all_rows:
        if r["season"] == "2026" and r["player_name"].lower() == "alyssa thomas":
            d = parse_date(r["game_date"])
            if d is None or d.isoformat() <= "2026-05-27":
                at_rows.append(r)
    at_sum = sum(r["potential_assists"] for r in at_rows)
    at_gp = len(at_rows)
    qa_summary = {
        "method": "sum matchupPotentialAssists across defender matchup rows in official WNBA boxscorematchupsv3",
        "seasons": SEASONS,
        "schedule_games": len(games),
        "successful_matchup_games": len(by_game),
        "failed_or_empty_games": len(failures),
        "player_game_rows": len(all_rows),
        "alyssa_thomas_2026_through_2026_05_27_games": at_gp,
        "alyssa_thomas_2026_through_2026_05_27_potential_assists": at_sum,
        "alyssa_thomas_external_validation_target": 118,
        "alyssa_thomas_validation_pass": (at_gp == 7 and at_sum == 118),
        "assist_qa_rows": len(qa),
        "assist_qa_exact_matches": sum(q["diff"] == 0 for q in qa),
    }

    write_csv(OUT / "wnba_potential_assists_player_game_2025_2026.csv", all_rows,
              ["season","game_id","game_date","team_id","team_abbreviation","player_id","player_name","potential_assists","matchup_assists","matchup_rows"])
    write_csv(OUT / "wnba_potential_assists_season_totals_2025_2026.csv", totals,
              ["season","player_id","player_name","team_abbreviation","games_with_matchup_data","potential_assists","potential_assists_per_game","matchup_assists"])
    write_csv(OUT / "wnba_potential_assists_assist_qa.csv", qa,
              ["game_id","player_id","player_name","matchup_assists","traditional_assists","diff"])
    write_csv(OUT / "wnba_potential_assists_failures.csv", failures,
              ["season","game_id","game_date","teams","error"])
    (OUT / "qa_summary.json").write_text(json.dumps(qa_summary, indent=2), encoding="utf-8")

    print("QA_SUMMARY", json.dumps(qa_summary, sort_keys=True))
    print("TOP_2026")
    for x in [x for x in totals if x["season"] == "2026"][:20]:
        print(x)
    print("TOP_2025")
    for x in [x for x in totals if x["season"] == "2025"][:20]:
        print(x)


if __name__ == "__main__":
    main()
