from __future__ import annotations

"""Prepare the immutable user-selected Frame C across current official clips.

The only authoritative timing is Right Slash 8.275733 s / decoded frame 248.
Audio correlation is used solely to map that physical instant into the other
freshly retrieved camera-local timelines.  This script fails rather than
selecting a different Right Slash frame.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2

LOCKED_TIME = 8.275733
LOCKED_FRAME = 248
GAME_ID = "0022500301"
EVENT_ID = 489
ORDERED = [
    "Broadcast", "Other Broadcast", "Mobile Broadcast", "Play by Play",
    "In Arena", "High Tight", "Left Slash", "Right Slash",
    "Left HandHeld", "Right HandHeld", "Left Above Rim", "Right Above Rim",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sync-script", type=Path, default=Path("freeze_spin/estimate_audio_sync_graph.py"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(args.clips.glob("*_489_*_SOURCE.mp4")):
        m = re.search(r"_489_(.+)_SOURCE\.mp4$", p.name)
        if m:
            rows.append({
                "label": m.group(1).replace("_", " "),
                "file": p.name,
                "freeze_time": LOCKED_TIME,
            })
    labels = {r["label"] for r in rows}
    missing = [x for x in ORDERED if x not in labels]
    if len(rows) != 12 or missing:
        raise RuntimeError(f"Expected exact 12-camera set; found {len(rows)}, missing={missing}")

    provisional = {
        "event": {"game_id": GAME_ID, "event_id": EVENT_ID},
        "source_fps": 29.97003,
        "reference_angle": "Left Above Rim",
        "angles": rows,
    }
    provisional_path = args.out / "provisional_sync_map.json"
    provisional_path.write_text(json.dumps(provisional, indent=2))
    graph_path = args.out / "audio_sync_graph.json"
    subprocess.run([
        sys.executable, str(args.sync_script),
        "--config", str(provisional_path),
        "--clips", str(args.clips),
        "--reference", "Left Above Rim",
        "--max-lag-seconds", "2.0",
        "--out", str(graph_path),
    ], check=True)

    sync = json.loads(graph_path.read_text())
    offsets = {r["label"]: float(r["offset_seconds_vs_reference"]) for r in sync["angles"]}
    files = {r["label"]: r["file"] for r in rows}
    if "Right Slash" not in offsets:
        raise RuntimeError("Right Slash synchronization offset missing")
    reference_time = LOCKED_TIME - offsets["Right Slash"]

    exported = []
    letters = list("ABCDEFGHIJKL")
    for letter, label in zip(letters, ORDERED):
        t = reference_time + offsets[label]
        p = args.clips / files[label]
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, image = cap.read()
        frame_index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot decode {label} at {t:.6f}s")
        if image.shape[:2] != (540, 960):
            raise RuntimeError(f"{label} unexpected source shape {image.shape}")
        fn = f"{letter}_{label.replace(' ', '_')}_{t:.6f}s_frame{frame_index:04d}.png"
        if not cv2.imwrite(str(frames_dir / fn), image):
            raise RuntimeError(f"Failed writing {fn}")
        exported.append({
            "option": letter,
            "camera": label,
            "requested_local_time": round(float(t), 6),
            "decoded_frame_index": frame_index,
            "file": fn,
        })

    rs = next(r for r in exported if r["camera"] == "Right Slash")
    if abs(float(rs["requested_local_time"]) - LOCKED_TIME) > 5e-7 or int(rs["decoded_frame_index"]) != LOCKED_FRAME:
        raise RuntimeError(f"FRAME C LOCK VIOLATED by current clip trim/decoder: {rs}")

    output = {
        "event": {"game_id": GAME_ID, "event_id": EVENT_ID, "date": "2025-11-30"},
        "chosen_timing": {
            "source_camera": "Right Slash",
            "option": "C",
            "right_slash_local_time": LOCKED_TIME,
            "decoded_frame_index_from_manual_chooser": LOCKED_FRAME,
        },
        "common_reference_time_left_above_rim": reference_time,
        "policy": "Frame C is immutable. Audio only maps this exact physical instant to other cameras.",
        "options": exported,
    }
    (args.out / "options.json").write_text(json.dumps(output, indent=2))
    print("FRAME_C_LOCK", rs, flush=True)
    print("COMMON_REFERENCE_TIME", reference_time, flush=True)
    for r in exported:
        print("SYNC_FRAME", r, flush=True)


if __name__ == "__main__":
    main()
