#!/usr/bin/env python3
"""Discover exact 2025-26 Adams OREB -> Durant made 3 events within 3.0s.

Derivative only. This file does not modify Screen Tracker 1.1 or its canonical
runtime. It produces exact event manifests for Screen Tracker - Kickout 3.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyreadr
import requests

PBP_URL = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds"
ADAMS_ID = 203500
DURANT_ID = 201142
MAX_SECONDS = 3.0
PLAYER_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def scalar(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def integer(v):
    try:
        if pd.isna(v) or v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def number(v):
    try:
        if pd.isna(v) or v == "":
            return None
        return float(v)
    except Exception:
        return None


def actor(v):
    s = scalar(v)
    if not s:
        return None, ""
    m = PLAYER_RE.match(s)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, s


def gid(v):
    s = scalar(v).replace(".0", "")
    return s.zfill(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    r = requests.get(PBP_URL, timeout=180)
    r.raise_for_status()
    p = a.out / "pbp_2025_26.rds"
    p.write_bytes(r.content)
    d = next(iter(pyreadr.read_r(str(p)).values())).reset_index(drop=True)
    p.unlink(missing_ok=True)

    required = {"game_id","period","event_num","msg_type","team_abb","off_team_abb",
                "player1_name","player2_name","shot_pts","description","secs_game"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise SystemExit(f"PBP missing required columns: {missing}")

    current = None
    pending_miss = None
    active = None
    candidates = []

    for row in d.itertuples(index=False):
        game = gid(getattr(row, "game_id"))
        period = integer(getattr(row, "period"))
        event = integer(getattr(row, "event_num"))
        msg = integer(getattr(row, "msg_type"))
        team = scalar(getattr(row, "team_abb")).upper() or scalar(getattr(row, "off_team_abb")).upper()
        p1_id, p1_name = actor(getattr(row, "player1_name"))
        p2_id, p2_name = actor(getattr(row, "player2_name"))
        pts = integer(getattr(row, "shot_pts"))
        desc = scalar(getattr(row, "description"))
        sec = number(getattr(row, "secs_game"))
        key = (game, period)
        if key != current:
            current = key
            pending_miss = None
            active = None

        # The first subsequent field-goal attempt after the OREB decides the chain.
        if msg in (1, 2) and active is not None:
            if team == active["team"]:
                dt = None if sec is None or active["secs_game"] is None else abs(sec - active["secs_game"])
                is_three = pts == 3 or "3PT" in desc.upper()
                if (p1_id == DURANT_ID and msg == 1 and is_three and dt is not None
                        and 0.0 <= dt <= MAX_SECONDS + 1e-9):
                    candidates.append({
                        "game_id": game,
                        "period": period,
                        "team": team,
                        "miss_event_num": active["miss_event_num"],
                        "rebound_event_num": active["rebound_event_num"],
                        "shot_event_num": event,
                        "rebounder_id": ADAMS_ID,
                        "rebounder_name": "Steven Adams",
                        "shooter_id": DURANT_ID,
                        "shooter_name": "Kevin Durant",
                        "elapsed_seconds": float(dt),
                        "credited_adams_assist": bool(p2_id == ADAMS_ID),
                        "assist_player_id": p2_id,
                        "assist_player_name": p2_name,
                        "shot_description": desc,
                        "shot_secs_game": sec,
                        "rebound_secs_game": active["secs_game"],
                        "filter": "individual Steven Adams OREB -> first subsequent HOU FGA -> Kevin Durant made 3PT, <=3.0s",
                    })
            active = None

        # Maintain exact immediate OREB chain.
        if msg == 1:
            pending_miss = None
        elif msg == 2:
            pending_miss = {"team": team, "event_num": event}
        elif msg == 3:
            active = None
            pending_miss = {"team": team, "event_num": event} if "MISS" in desc.upper() else None
        elif msg == 4:
            active = None
            if pending_miss and team == pending_miss["team"] and p1_id == ADAMS_ID:
                active = {
                    "team": team,
                    "secs_game": sec,
                    "rebound_event_num": event,
                    "miss_event_num": pending_miss["event_num"],
                }
            pending_miss = None
        elif msg in (5, 10, 12, 13):
            active = None
            pending_miss = None

    c = pd.DataFrame(candidates)
    if c.empty:
        raise SystemExit("No Adams -> Durant made kickout threes within 3.0s found")
    c = c.sort_values(["credited_adams_assist","elapsed_seconds","game_id","shot_event_num"],
                      ascending=[False,True,True,True]).reset_index(drop=True)
    c.to_csv(a.out / "all_adams_durant_kickout3_candidates.csv", index=False)

    # First proof: prefer official assist credit, then shortest elapsed time, and
    # force two different games for genuinely different examples.
    selected = []
    used_games = set()
    for rec in c.to_dict("records"):
        if rec["game_id"] in used_games:
            continue
        selected.append(rec)
        used_games.add(rec["game_id"])
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise SystemExit(f"Expected two distinct-game candidates; got {len(selected)}")

    manifest = {
        "tool_id": "SCREEN_TRACKER_KICKOUT_3",
        "version": "0.1.0-poc",
        "base_screen_tracker": {
            "version": "1.1.0",
            "protected_branch": "screen-tracker",
            "protected_source_sha": "15cde6374333c1956299c2355183549298a466db",
            "mutation_allowed": False,
        },
        "season": "2025-26",
        "scope": "Regular Season",
        "definition": {
            "rebounder": {"player_id": ADAMS_ID, "name": "Steven Adams"},
            "shooter": {"player_id": DURANT_ID, "name": "Kevin Durant"},
            "max_elapsed_seconds_inclusive": MAX_SECONDS,
            "made_three_required": True,
            "first_subsequent_team_fga_required": True,
            "direct_individual_oreb_required": True,
            "ranking_for_initial_two": "credited Adams assist first; then shortest elapsed time; distinct games",
        },
        "candidate_count": int(len(c)),
        "selected_events": selected,
    }
    (a.out / "kickout3_event_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
