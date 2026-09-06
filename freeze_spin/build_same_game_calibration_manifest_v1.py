from __future__ import annotations

"""Build a deterministic same-game event manifest for camera-registry evidence.

This calibration stage intentionally does not depend on play-by-play. GitHub-hosted
runners may be blocked by the NBA live-data CDN, while the official clips.nba.com
system is the durable video source already used by the project. We therefore probe
candidate event numbers directly through the proven clips inventory endpoint, retain
only real events with substantial multi-angle coverage, and sample them across the
game timeline. The immutable target event is always retained.
"""

import argparse
import json
import sys
from pathlib import Path

# The sampler lives under freeze_spin/ but reuses the proven root-level clip inventory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_kd_top10_all_angles import inventory


def evenly_pick(rows: list[dict], n: int) -> list[dict]:
    if n <= 0 or not rows:
        return []
    rows = sorted(rows, key=lambda r: int(r["event_id"]))
    if len(rows) <= n:
        return rows[:]
    if n == 1:
        return [rows[len(rows) // 2]]
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    out, seen = [], set()
    for i in idx:
        event = int(rows[i]["event_id"])
        if event not in seen:
            seen.add(event)
            out.append(rows[i])
    return out


def probe_event(game_id: str, event_id: int, min_angles: int) -> dict | None:
    try:
        page, title, opts = inventory(game_id, event_id)
    except Exception:
        return None
    if len(opts) < min_angles:
        return None
    return {
        "event_id": int(event_id),
        "angle_count": len(opts),
        "angle_labels": [o["label"] for o in opts],
        "clips_page": page,
        "clips_page_title": title,
    }


def discover_video_events(game_id: str, target_event: int, *, min_angles: int, max_events: int) -> tuple[list[dict], dict]:
    probes = sorted(set([target_event] + list(range(20, 761, 20))))
    found = []
    tested = []
    for eid in probes:
        tested.append(eid)
        row = probe_event(game_id, eid, min_angles)
        if row is not None:
            found.append(row)

    if len(found) < max_events:
        tested_set = set(tested)
        extra = [eid for eid in range(10, 771, 10) if eid not in tested_set]
        for eid in extra:
            tested.append(eid)
            row = probe_event(game_id, eid, min_angles)
            if row is not None:
                found.append(row)
            if len(found) >= max_events * 2:
                break

    if not any(int(r["event_id"]) == target_event for r in found):
        target = probe_event(game_id, target_event, 1)
        if target is None:
            raise RuntimeError(f"Target event {target_event} has no official clips inventory")
        found.append(target)

    dedup = {int(r["event_id"]): r for r in found}
    found = sorted(dedup.values(), key=lambda r: int(r["event_id"]))
    target = dedup[target_event]
    non_target = [r for r in found if int(r["event_id"]) != target_event]
    selected_non_target = evenly_pick(non_target, max(0, max_events - 1))
    selected = sorted(selected_non_target + [target], key=lambda r: int(r["event_id"]))
    audit = {
        "probe_count": len(tested),
        "probe_event_ids": tested,
        "qualified_event_count": len(found),
        "qualified_event_ids": [int(r["event_id"]) for r in found],
        "minimum_angles": min_angles,
    }
    return selected, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--game-date", required=True)
    ap.add_argument("--target-event", type=int, required=True)
    ap.add_argument("--event-count", type=int, default=9)
    ap.add_argument("--min-angles", type=int, default=8)
    ap.add_argument("--per-period", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    selected, discovery = discover_video_events(
        args.game_id, args.target_event, min_angles=args.min_angles, max_events=args.event_count
    )

    events = []
    for rank, row in enumerate(selected, 1):
        event_id = int(row["event_id"])
        events.append({
            "rank": rank,
            "game_date": args.game_date,
            "game_id": args.game_id,
            "event_id": event_id,
            "description": f"same-game camera calibration video event {event_id}",
            "video_anchor": "official clips.nba.com event inventory",
            "camera_registry_role": "locked_target" if event_id == args.target_event else "same_game_calibration_evidence",
            "discovered_angle_count": int(row["angle_count"]),
            "discovered_angle_labels": row["angle_labels"],
        })

    payload = {
        "selection": "video-driven same-game multi-frame physical-camera evidence",
        "game_id": args.game_id,
        "game_date": args.game_date,
        "target_event": args.target_event,
        "sampling": {
            "source": "official clips.nba.com event inventory",
            "requested_event_count": args.event_count,
            "minimum_angles_for_calibration_samples": args.min_angles,
            "discovery": discovery,
        },
        "expected_count": len(events),
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event_count": len(events), "event_ids": [e["event_id"] for e in events], "discovery": discovery}, indent=2))


if __name__ == "__main__":
    main()
