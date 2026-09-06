from __future__ import annotations

"""Adaptive Right Slash geometry-state recovery for the Adams/Jazz game.

This is a strict acquisition/selection stage.  It searches around evenly spaced
same-game event anchors instead of stopping at the first discoverable clips in a
large event-number segment.  Candidate frames are ranked only to surface clean
static NBA basket/court geometry for manual/metric calibration.  Ranking cannot
promote a camera and no player/ball landmarks are used.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.scan_same_game_camera_priors import extract_frames, inventory_camera, make_sheet, safe_label
import nba_video_worker as w

W, H = 960, 540


def _rectangular_evidence(gray: np.ndarray) -> dict:
    # Bright, locally rectangular structure is useful for surfacing board/target
    # candidates, but is explicitly not treated as a semantic detector.
    upper = gray[:390]
    _, bw = cv2.threshold(upper, 178, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        if peri < 30:
            continue
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, ww, hh = cv2.boundingRect(approx)
        if not (24 <= ww <= 230 and 14 <= hh <= 165):
            continue
        ar = ww / max(hh, 1)
        if not (1.05 <= ar <= 2.8):
            continue
        area = cv2.contourArea(c)
        fill = area / max(float(ww * hh), 1.0)
        if fill < 0.15:
            continue
        boxes.append([int(x), int(y), int(ww), int(hh), float(fill)])
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return {"count": len(boxes), "largest": boxes[:8]}


def frame_metrics(path: Path) -> dict:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (H, W):
        raise RuntimeError(f"invalid native frame {path}")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 65, 155)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=65, minLineLength=50, maxLineGap=9)
    lengths = []
    floor_lines = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            ln = math.hypot(float(x2 - x1), float(y2 - y1))
            lengths.append(ln)
            if max(y1, y2) >= 285 and ln >= 70:
                floor_lines += 1
    rect = _rectangular_evidence(gray)
    edge_fraction = float(np.mean(edges > 0))
    # Diagnostic ranking only.  Favour sharp frames with long static lines and
    # some rectangular structure, while retaining floor-line support.
    long_count = sum(ln >= 110.0 for ln in lengths)
    score = (
        2.2 * math.log1p(max(sharp, 0.0))
        + 0.09 * min(long_count, 40)
        + 0.07 * min(floor_lines, 40)
        + 0.22 * min(rect["count"], 10)
        + 5.0 * min(edge_fraction, 0.16)
    )
    return {
        "sharpness_laplacian_var": sharp,
        "edge_fraction": edge_fraction,
        "hough_line_count": 0 if lines is None else int(len(lines)),
        "long_hough_line_count": int(long_count),
        "floor_long_line_count": int(floor_lines),
        "rectangular_structure": rect,
        "diagnostic_geometry_score": float(score),
    }


def anchor_probe_order(anchor: int, lo: int, hi: int, radius: int) -> list[int]:
    out = []
    for d in range(radius + 1):
        vals = [anchor] if d == 0 else [anchor - d, anchor + d]
        for v in vals:
            if lo <= v <= hi and v not in out:
                out.append(v)
    return out


def discover_around_anchors(game_id: str, label: str, lo: int, hi: int, anchors: int, radius: int) -> list[dict]:
    anchor_vals = np.linspace(lo, hi, anchors).round().astype(int).tolist()
    seen = set()
    found = []
    for ai, anchor in enumerate(anchor_vals):
        winner = None
        for eid in anchor_probe_order(anchor, lo, hi, radius):
            try:
                url, title = inventory_camera(game_id, eid, label)
            except Exception:
                continue
            key = (title, url.split("?")[0])
            if key in seen:
                continue
            seen.add(key)
            winner = {
                "anchor_index": ai,
                "anchor_event": int(anchor),
                "event_probe": int(eid),
                "title": title,
                "url": url,
            }
            break
        if winner is not None:
            found.append(winner)
    return found


def make_montage(rows: list[dict], root: Path, out: Path) -> None:
    cells = []
    for rec in rows:
        if rec.get("status") != "ok" or not rec.get("selected_frames"):
            continue
        sf = rec["selected_frames"][0]
        p = root / f"event_{rec['event_probe']}_selected" / sf["file"]
        im = cv2.imread(str(p))
        if im is None:
            continue
        thumb = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
        txt = f"a{rec['anchor_index']:02d} e{rec['event_probe']} {sf['file']} score={sf['metrics']['diagnostic_geometry_score']:.2f}"
        cv2.rectangle(thumb, (0, 0), (480, 34), (0, 0, 0), -1)
        cv2.putText(thumb, txt, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255,255,255), 1, cv2.LINE_AA)
        cells.append(thumb)
    if not cells:
        return
    cols = 3
    rows_im = []
    for i in range(0, len(cells), cols):
        rr = cells[i:i+cols]
        while len(rr) < cols:
            rr.append(np.full_like(cells[0], 255))
        rows_im.append(np.hstack(rr))
    cv2.imwrite(str(out), np.vstack(rows_im))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--camera-label", default="Right Slash")
    ap.add_argument("--event-start", type=int, default=5)
    ap.add_argument("--event-stop", type=int, default=704)
    ap.add_argument("--anchors", type=int, default=18)
    ap.add_argument("--search-radius", type=int, default=16)
    ap.add_argument("--frames-per-clip", type=int, default=9)
    ap.add_argument("--retain-per-clip", type=int, default=2)
    ap.add_argument("--min-successful-anchors", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    slug = safe_label(args.camera_label)

    discovered = discover_around_anchors(
        args.game_id, args.camera_label, args.event_start, args.event_stop,
        args.anchors, args.search_radius,
    )
    results = []
    for d in discovered:
        eid = int(d["event_probe"])
        rec = {k:v for k,v in d.items() if k != "url"}
        rec["status"] = "failed"
        try:
            clip = args.out / f"event_{eid}_{slug}_SOURCE.mp4"
            w.download_hls_source(d["url"], clip)
            q = w.probe_video(clip)
            if not q.get("ok"):
                raise RuntimeError(q.get("reason"))
            if (int(q.get("width") or 0), int(q.get("height") or 0)) != (W, H):
                raise RuntimeError(f"non-native raster {q.get('width')}x{q.get('height')}")
            tmp = args.out / f"event_{eid}_all_frames"
            frames = extract_frames(clip, tmp, n=args.frames_per_clip)
            scored = [{"file":p.name, "metrics":frame_metrics(p)} for p in frames]
            scored.sort(key=lambda x:x["metrics"]["diagnostic_geometry_score"], reverse=True)
            selected = scored[:args.retain_per_clip]
            sheet = args.out / f"event_{eid}_{slug}_sheet.png"
            make_sheet(frames, eid, d["title"], args.camera_label, sheet)
            keepdir = args.out / f"event_{eid}_selected"
            keepdir.mkdir(exist_ok=True)
            for row in selected:
                (tmp / row["file"]).replace(keepdir / row["file"])
            for p in tmp.glob('*.png'):
                p.unlink()
            tmp.rmdir()
            clip.unlink(missing_ok=True)
            rec.update({"status":"ok", "probe":q, "sheet":sheet.name, "selected_frames":selected})
        except Exception as exc:
            rec["error"] = repr(exc)
        results.append(rec)
        print("ADAPTIVE_GEOMETRY", rec["anchor_index"], eid, rec["status"], flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    make_montage(ok, args.out, args.out / "right_slash_adaptive_geometry_montage_v101.png")
    payload = {
        "status": "GEOMETRY_SELECTION_BANK_V101" if len(ok) >= args.min_successful_anchors else "INSUFFICIENT_GEOMETRY_SELECTION_BANK_V101",
        "game_id": args.game_id,
        "camera_label": args.camera_label,
        "purpose": "Recover temporally distributed native Right Slash states likely to expose static regulation basket/court geometry for the shared physical-centre proof.",
        "sampling": {
            "event_range": [args.event_start, args.event_stop],
            "anchors": args.anchors,
            "search_radius": args.search_radius,
            "frames_per_clip": args.frames_per_clip,
            "retain_per_clip": args.retain_per_clip,
        },
        "guardrails": [
            "diagnostic ranking is not semantic landmark detection",
            "camera label is not physical-camera proof",
            "no player or ball landmarks",
            "no scaling or generated pixels",
            "no metric-camera or replay promotion in this stage",
        ],
        "successful_anchor_count": len(ok),
        "events": results,
        "permissions": {
            "shared_center_metric_attempt_allowed": len(ok) >= args.min_successful_anchors,
            "metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
    }
    (args.out / "right_slash_adaptive_geometry_bank_v101.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status":payload["status"], "successful_anchor_count":len(ok), "event_probes":[r["event_probe"] for r in ok]}, indent=2), flush=True)
    if len(ok) < args.min_successful_anchors:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
