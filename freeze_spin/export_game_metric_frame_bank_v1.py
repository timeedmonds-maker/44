from __future__ import annotations

"""Export native same-game frames for metric calibration of fixed camera centres.

This is a deliberately narrow acquisition tool.  It does not estimate any camera
matrix and it never uses player/ball points.  It retrieves only camera labels that
have already passed the game-level fixed-centre preflight, then samples several
native frames from several independent events so regulation court/basket geometry
can be calibrated jointly across pan/tilt/zoom states.
"""

import argparse
import html as htmlmod
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nba_video_worker as w


def inventory_label(game_id: str, event_id: int, wanted: str) -> tuple[str, str]:
    page = f"https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs"
    txt = w.http_bytes(page, w.H).decode("utf-8", "replace")
    title_m = re.search(r"<title>(.*?)</title>", txt, re.I | re.S)
    title = htmlmod.unescape(title_m.group(1).strip()) if title_m else ""
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', txt, re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        label = re.sub(r"<[^>]+>", "", htmlmod.unescape(m.group(3))).strip()
        if label == wanted and ".m3u8" in url.lower() and "lrmedia.nba.com" in url.lower():
            return url, title
    raise RuntimeError(f"No {wanted!r} HLS for {game_id}/{event_id}")


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def extract_frames(video: Path, outdir: Path, n: int) -> list[dict]:
    q = w.probe_video(video)
    if not q.get("ok"):
        raise RuntimeError(f"Bad video: {q}")
    width = int(q.get("width") or 0)
    height = int(q.get("height") or 0)
    if (width, height) != (960, 540):
        raise RuntimeError(f"Unexpected native raster {(width, height)} for {video.name}; no scaling is permitted")
    dur = float(q["duration"])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, frac in enumerate(np.linspace(0.20, 0.80, n)):
        t = max(0.05, min(dur - 0.05, dur * float(frac)))
        p = outdir / f"f{i:02d}_{t:.5f}s.png"
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-ss", f"{t:.5f}", "-i", str(video), "-frames:v", "1", str(p)
        ], check=True)
        image = cv2.imread(str(p))
        if image is None or image.shape[:2] != (540, 960):
            raise RuntimeError(f"Decoded frame validation failed for {p}")
        rows.append({"index": i, "fraction": float(frac), "local_time_seconds": round(t, 6), "file": p.name})
    return rows


def make_sheet(frame_paths: list[Path], title: str, out: Path) -> None:
    cells = []
    for i, p in enumerate(frame_paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        thumb = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"sample {i}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(thumb, f"sample {i}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(thumb)
    if not cells:
        return
    cols = 3
    rows = []
    for start in range(0, len(cells), cols):
        row = cells[start:start + cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 255))
        rows.append(np.hstack(row))
    body = np.vstack(rows)
    header = np.full((70, body.shape[1], 3), 255, np.uint8)
    cv2.putText(header, title[:120], (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([header, body]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--events", required=True, help="comma-separated official event ids")
    ap.add_argument("--labels", required=True, help="comma-separated exact camera labels")
    ap.add_argument("--frames-per-clip", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    events = [int(x.strip()) for x in args.events.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if len(events) < 3:
        raise SystemExit("At least three independent same-game events are required")
    if not labels:
        raise SystemExit("At least one camera label is required")
    if args.frames_per_clip < 3:
        raise SystemExit("frames-per-clip must be at least 3")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for label in labels:
        lroot = args.out / safe(label)
        lroot.mkdir(exist_ok=True)
        for event_id in events:
            rec = {"label": label, "event_id": event_id, "status": "failed"}
            try:
                url, title = inventory_label(args.game_id, event_id, label)
                clip = lroot / f"event_{event_id}_{safe(label)}_SOURCE.mp4"
                w.download_hls_source(url, clip)
                probe = w.probe_video(clip)
                if not probe.get("ok"):
                    raise RuntimeError(probe.get("reason"))
                froot = lroot / f"event_{event_id}_frames"
                frames = extract_frames(clip, froot, args.frames_per_clip)
                frame_paths = [froot / r["file"] for r in frames]
                sheet = lroot / f"event_{event_id}_{safe(label)}_sheet.jpg"
                make_sheet(frame_paths, f"{label} | event {event_id} | {title}", sheet)
                rec.update({"status": "ok", "title": title, "probe": probe, "frames": frames, "sheet": sheet.name})
                # Source clip is only a transport intermediate; the artifact is a native PNG frame bank.
                clip.unlink(missing_ok=True)
            except Exception as exc:
                rec["error"] = repr(exc)
            results.append(rec)
            print(label, event_id, rec["status"], flush=True)

    good = [r for r in results if r["status"] == "ok"]
    payload = {
        "game_id": args.game_id,
        "purpose": "native fixed-camera multi-event frame bank for shared-centre metric calibration",
        "guardrail": "acquisition only; no camera promotion; no player/ball landmarks; no scaling or generated pixels",
        "source_resolution": [960, 540],
        "events": events,
        "labels": labels,
        "frames_per_clip": args.frames_per_clip,
        "successful_clip_count": len(good),
        "expected_clip_count": len(events) * len(labels),
        "clips": results,
    }
    (args.out / "metric_frame_bank_manifest_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if len(good) != len(events) * len(labels):
        raise SystemExit(f"Only {len(good)}/{len(events) * len(labels)} fixed-camera clips exported")


if __name__ == "__main__":
    main()
