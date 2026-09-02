from __future__ import annotations

"""Export native frames from the exact target camera clip for intrinsics calibration.

The game-level physical camera centre is solved elsewhere.  This acquisition stage
captures the *same physical feed and same event clip* at many times around the
immutable freeze so event/clip-specific optical crop behaviour can be calibrated
without assuming one principal point for the whole game.

No scaling, interpolation, generated pixels, player/ball landmarks or timing changes.
The immutable freeze itself is excluded from the calibration sample bank.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--freeze-time", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-count", type=int, default=19)
    ap.add_argument("--exclude-radius-seconds", type=float, default=0.75)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.clip}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if (width, height) != (W, H) or fps <= 0 or count <= 0:
        raise RuntimeError(f"Unexpected native clip metadata {(width,height,fps,count)}")
    duration = count / fps

    # Broad temporal support. Keep clear of clip heads/tails and exact freeze state.
    candidates = np.linspace(0.08 * duration, 0.92 * duration, args.sample_count * 2 + 1)
    times = [float(t) for t in candidates if abs(float(t) - args.freeze_time) >= args.exclude_radius_seconds]
    if len(times) > args.sample_count:
        idx = np.linspace(0, len(times) - 1, args.sample_count).round().astype(int)
        times = [times[int(i)] for i in idx]

    rows = []
    for j, t in enumerate(times):
        frame_idx = int(np.clip(round(t * fps), 0, count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None or frame.shape[:2] != (H, W):
            raise RuntimeError(f"Decode failed at {t:.6f}s / frame {frame_idx}")
        actual_t = frame_idx / fps
        p = args.out / f"Left_Above_Rim_target_event__s{j:02d}__t{actual_t:.6f}s__f{frame_idx:04d}.png"
        cv2.imwrite(str(p), frame)
        rows.append({
            "sample_index": j,
            "requested_time_seconds": float(t),
            "decoded_frame_index": frame_idx,
            "decoded_time_seconds": float(actual_t),
            "relative_to_freeze_seconds": float(actual_t - args.freeze_time),
            "file": p.name,
        })
    cap.release()

    if len(rows) < 10:
        raise RuntimeError(f"Only {len(rows)} target-clip samples")
    if min(r["relative_to_freeze_seconds"] for r in rows) >= 0 or max(r["relative_to_freeze_seconds"] for r in rows) <= 0:
        raise RuntimeError("Samples do not span both sides of immutable freeze")
    if any(abs(r["relative_to_freeze_seconds"]) < args.exclude_radius_seconds - 1e-6 for r in rows):
        raise RuntimeError("Immutable freeze exclusion violated")

    payload = {
        "source_clip": str(args.clip),
        "source_resolution": [W, H],
        "fps": fps,
        "decoded_frame_count": count,
        "duration_seconds": duration,
        "immutable_freeze_time_seconds": float(args.freeze_time),
        "freeze_exclusion_radius_seconds": float(args.exclude_radius_seconds),
        "samples": rows,
        "guardrail": "acquisition only; native source pixels; exact freeze excluded; no camera promotion",
    }
    (args.out / "target_clip_intrinsics_sample_manifest_v31.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": len(rows),
        "time_min": min(r["decoded_time_seconds"] for r in rows),
        "time_max": max(r["decoded_time_seconds"] for r in rows),
        "relative_min": min(r["relative_to_freeze_seconds"] for r in rows),
        "relative_max": max(r["relative_to_freeze_seconds"] for r in rows),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
