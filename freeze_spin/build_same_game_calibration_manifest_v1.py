from __future__ import annotations

"""Build a deterministic same-game event manifest for camera-registry evidence.

The target event is always retained. Additional made field goals are sampled across
periods so the same official camera labels can be observed under different pan/tilt/
zoom states. This script does not calibrate or promote any camera.
"""

import argparse
import json
from pathlib import Path

import requests


def fetch_actions(game_id: str) -> list[dict]:
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    payload = r.json()
    return list(payload.get("game", {}).get("actions", []))


def is_made_fg(a: dict) -> bool:
    if str(a.get("shotResult", "")).lower() != "made":
        return False
    action = str(a.get("actionType", "")).lower()
    return action in {"2pt", "3pt"}


def evenly_pick(rows: list[dict], n: int) -> list[dict]:
    if n <= 0 or not rows:
        return []
    if len(rows) <= n:
        return rows[:]
    if n == 1:
        return [rows[len(rows) // 2]]
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    out = []
    seen = set()
    for i in idx:
        event = int(rows[i].get("actionNumber", -1))
        if event not in seen:
            seen.add(event)
            out.append(rows[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--game-date", required=True)
    ap.add_argument("--target-event", type=int, required=True)
    ap.add_argument("--per-period", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    actions = fetch_actions(args.game_id)
    by_period: dict[int, list[dict]] = {}
    for a in actions:
        if not is_made_fg(a):
            continue
        p = int(a.get("period", 0) or 0)
        if p <= 0:
            continue
        by_period.setdefault(p, []).append(a)

    selected: list[dict] = []
    for period in sorted(by_period):
        selected.extend(evenly_pick(by_period[period], args.per_period))

    target = next((a for a in actions if int(a.get("actionNumber", -1)) == args.target_event), None)
    if target is None:
        raise RuntimeError(f"Target event {args.target_event} not found in {args.game_id}")

    rows_by_event = {int(a.get("actionNumber", -1)): a for a in selected}
    rows_by_event[args.target_event] = target
    selected = sorted(rows_by_event.values(), key=lambda a: int(a.get("actionNumber", -1)))

    events = []
    for rank, a in enumerate(selected, 1):
        event_id = int(a.get("actionNumber", -1))
        person_id = int(a.get("personId", 0) or 0)
        desc = str(a.get("description") or a.get("actionType") or "same-game calibration sample")
        events.append({
            "rank": rank,
            "game_date": args.game_date,
            "game_id": args.game_id,
            "event_id": event_id,
            "period": int(a.get("period", 0) or 0),
            "clock": str(a.get("clock", "")),
            "player_id": person_id,
            "player": str(a.get("playerName", "")),
            "description": desc,
            "video_anchor": "made field goal event" if is_made_fg(a) else "locked target event",
            "camera_registry_role": "locked_target" if event_id == args.target_event else "same_game_calibration_evidence",
        })

    payload = {
        "selection": "same-game multi-frame physical-camera evidence",
        "game_id": args.game_id,
        "game_date": args.game_date,
        "target_event": args.target_event,
        "sampling": {"made_field_goals_per_period": args.per_period},
        "expected_count": len(events),
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event_count": len(events), "event_ids": [e["event_id"] for e in events]}, indent=2))


if __name__ == "__main__":
    main()
