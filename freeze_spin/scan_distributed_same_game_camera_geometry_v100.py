from __future__ import annotations

"""Distributed same-game camera geometry bank for fixed-centre R&D.

This is an acquisition/selection diagnostic only. It deliberately samples the
entire game instead of stopping after the first N discoverable clips. A shared
camera label is never treated as proof of one physical camera. No player/ball
landmarks and no camera promotion occur here.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.scan_same_game_camera_priors import (
    extract_frames,
    inventory_camera,
    make_sheet,
    safe_label,
)
import nba_video_worker as w


def frame_metrics(path: Path) -> dict:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (540, 960):
        raise RuntimeError(f"invalid native frame {path}")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 70, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=70, minLineLength=55, maxLineGap=8)
    nlines = 0 if lines is None else int(len(lines))
    long_lines = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            if math.hypot(float(x2 - x1), float(y2 - y1)) >= 110.0:
                long_lines += 1
    return {
        "sharpness_laplacian_var": sharp,
        "edge_fraction": float(np.mean(edges > 0)),
        "hough_line_count": nlines,
        "long_hough_line_count": int(long_lines),
    }


def discover_distributed(game_id: str, label: str, start: int, stop: int, step: int, segments: int, per_segment: int) -> list[dict]:
    bounds = np.linspace(start, stop + 1, segments + 1).astype(int)
    seen = set()
    found: list[dict] = []
    for seg in range(segments):
        lo = int(bounds[seg])
        hi = int(bounds[seg + 1] - 1)
        seg_found = 0
        first = lo + ((step - ((lo - start) % step)) % step)
        for event_id in range(first, hi + 1, step):
            try:
                url, title = inventory_camera(game_id, event_id, label)
            except Exception:
                continue
            key = (title, url.split("?")[0])
            if key in seen:
                continue
            seen.add(key)
            found.append({
                "segment": seg + 1,
                "segment_event_range": [lo, hi],
                "event_probe": int(event_id),
                "title": title,
                "url": url,
            })
            seg_found += 1
            if seg_found >= per_segment:
                break
    return found


def make_global_montage(rows: list[dict], root: Path, slug: str, out: Path) -> None:
    cells = []
    for rec in rows:
        if rec.get("status") != "ok":
            continue
        samples = rec.get("samples", [])
        if not samples:
            continue
        best = max(samples, key=lambda r: r["metrics"]["sharpness_laplacian_var"])
        p = root / f"event_{rec['event_probe']}_frames" / best["file"]
        im = cv2.imread(str(p))
        if im is None:
            continue
        thumb = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
        txt = f"seg{rec['segment']} e{rec['event_probe']} {best['file']}"
        cv2.rectangle(thumb, (0, 0), (480, 34), (0, 0, 0), -1)
        cv2.putText(thumb, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(thumb)
    if not cells:
        return
    cols = 3
    canvas_rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 255))
        canvas_rows.append(np.hstack(row))
    cv2.imwrite(str(out), np.vstack(canvas_rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--camera-label", required=True)
    ap.add_argument("--event-start", type=int, default=5)
    ap.add_argument("--event-stop", type=int, default=1405)
    ap.add_argument("--event-step", type=int, default=10)
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--per-segment", type=int, default=3)
    ap.add_argument("--frames-per-clip", type=int, default=7)
    ap.add_argument("--min-clips", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    slug = safe_label(args.camera_label)
    discovered = discover_distributed(
        args.game_id, args.camera_label, args.event_start, args.event_stop,
        args.event_step, args.segments, args.per_segment,
    )

    rows = []
    for d in discovered:
        eid = int(d["event_probe"])
        rec = {k: v for k, v in d.items() if k != "url"}
        rec["status"] = "failed"
        try:
            clip = args.out / f"event_{eid}_{slug}_SOURCE.mp4"
            w.download_hls_source(d["url"], clip)
            q = w.probe_video(clip)
            if not q.get("ok"):
                raise RuntimeError(q.get("reason"))
            if (int(q.get("width") or 0), int(q.get("height") or 0)) != (960, 540):
                raise RuntimeError(f"non-native raster {q.get('width')}x{q.get('height')}")
            froot = args.out / f"event_{eid}_frames"
            frames = extract_frames(clip, froot, n=args.frames_per_clip)
            samples = [{"file": p.name, "metrics": frame_metrics(p)} for p in frames]
            sheet = args.out / f"event_{eid}_{slug}_sheet.png"
            make_sheet(frames, eid, d["title"], args.camera_label, sheet)
            rec.update({"status": "ok", "probe": q, "sheet": sheet.name, "samples": samples})
            clip.unlink(missing_ok=True)
        except Exception as exc:
            rec["error"] = repr(exc)
        rows.append(rec)
        print("GEOMETRY_BANK", eid, rec["status"], flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    make_global_montage(ok, args.out, slug, args.out / "distributed_geometry_candidates_montage.png")
    payload = {
        "status": "DISCOVERY_ONLY_NO_PROMOTION",
        "game_id": args.game_id,
        "camera_label": args.camera_label,
        "purpose": "Full-game distributed native frame bank for visually selecting clean regulation basket/court geometry states before any fixed-centre metric solve.",
        "sampling": {
            "event_range": [args.event_start, args.event_stop],
            "event_step": args.event_step,
            "segments": args.segments,
            "per_segment_target": args.per_segment,
            "frames_per_clip": args.frames_per_clip,
        },
        "guardrails": [
            "camera label is not physical-camera proof",
            "no player or ball landmarks",
            "line/sharpness metrics are ranking diagnostics only and cannot promote a camera",
            "no scaling or generated pixels",
        ],
        "successful_clip_count": len(ok),
        "events": rows,
        "permissions": {
            "fixed_center_prior_allowed": False,
            "metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
    }
    (args.out / "distributed_same_game_geometry_bank_v100.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "successful_clip_count": len(ok), "event_probes": [r["event_probe"] for r in ok]}, indent=2), flush=True)
    if len(ok) < args.min_clips:
        raise SystemExit(f"Only {len(ok)} successful distributed clips; require {args.min_clips}")


if __name__ == "__main__":
    main()
