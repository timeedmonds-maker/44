from __future__ import annotations

"""Generic official same-game camera-state scanner for metric-camera calibration.

This is an acquisition/diagnostic stage only. A shared NBA camera label never
promotes a physical camera. Each returned frame remains untrusted until static
scene transfer and fixed-center geometry gates pass.
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


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def inventory_camera(game_id: str, event_id: int, camera_label: str) -> tuple[str, str]:
    page = f"https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs"
    txt = w.http_bytes(page, w.H).decode("utf-8", "replace")
    title_m = re.search(r"<title>(.*?)</title>", txt, re.I | re.S)
    title = htmlmod.unescape(title_m.group(1).strip()) if title_m else ""
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', txt, re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        label = re.sub(r"<[^>]+>", "", htmlmod.unescape(m.group(3))).strip()
        if label == camera_label and ".m3u8" in url.lower() and "lrmedia.nba.com" in url.lower():
            return url, title
    raise RuntimeError(f"No {camera_label!r} HLS for {game_id}/{event_id}")


def discover_events(game_id: str, camera_label: str, count: int, start: int, stop: int, step: int) -> list[dict]:
    found = []
    seen = set()
    probes = 0
    for event_id in range(start, stop + 1, step):
        probes += 1
        try:
            url, title = inventory_camera(game_id, event_id, camera_label)
        except Exception:
            continue
        key = (title, url.split("?")[0])
        if key in seen:
            continue
        seen.add(key)
        found.append({"event_id": event_id, "title": title, "url": url})
        if len(found) >= count:
            break
    if len(found) < count:
        raise RuntimeError(
            f"Found only {len(found)}/{count} distinct {camera_label} clips after {probes} official event probes"
        )
    return found


def extract_frames(video: Path, outdir: Path, n: int = 9) -> list[Path]:
    q = w.probe_video(video)
    if not q.get("ok"):
        raise RuntimeError(f"Bad video: {q}")
    dur = float(q["duration"])
    outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, frac in enumerate(np.linspace(0.18, 0.82, n)):
        t = max(0.05, min(dur - 0.05, dur * float(frac)))
        p = outdir / f"f{i:02d}.png"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{t:.5f}", "-i", str(video), "-frames:v", "1", str(p)],
            check=True,
        )
        frames.append(p)
    return frames


def make_sheet(frames: list[Path], event_id: int, title: str, camera_label: str, out: Path) -> None:
    cells = []
    for i, p in enumerate(frames):
        im = cv2.imread(str(p))
        if im is None:
            continue
        cv2.putText(im, f"sample {i+1}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(im, f"sample {i+1}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
        cells.append(im)
    if not cells:
        return
    rows = []
    for r in range(3):
        row = cells[r*3:(r+1)*3]
        while len(row) < 3:
            row.append(np.full_like(cells[0], 255))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    header = np.full((90, sheet.shape[1], 3), 255, np.uint8)
    desc = f"{camera_label} | event probe {event_id} | {title}"
    cv2.putText(header, desc[:180], (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([header, sheet]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--camera-label", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--event-start", type=int, default=5)
    ap.add_argument("--event-stop", type=int, default=1205)
    ap.add_argument("--event-step", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    slug = safe_label(args.camera_label)

    discovered = discover_events(
        args.game_id, args.camera_label, args.count,
        args.event_start, args.event_stop, args.event_step,
    )
    results = []
    for rank, d in enumerate(discovered, 1):
        eid = int(d["event_id"])
        rec = {"rank": rank, "event_probe": eid, "title": d["title"]}
        try:
            clip = args.out / f"event_{eid}_{slug}_SOURCE.mp4"
            w.download_hls_source(d["url"], clip)
            q = w.probe_video(clip)
            if not q.get("ok"):
                raise RuntimeError(q.get("reason"))
            frames = extract_frames(clip, args.out / f"event_{eid}_frames")
            sheet = args.out / f"event_{eid}_{slug}_sheet.png"
            make_sheet(frames, eid, d["title"], args.camera_label, sheet)
            rec.update({"status": "ok", "probe": q, "sheet": sheet.name})
        except Exception as e:
            rec.update({"status": "failed", "error": repr(e)})
        results.append(rec)

    payload = {
        "game_id": args.game_id,
        "camera_label": args.camera_label,
        "purpose": "Acquire diverse same-game optical states for a fixed-physical-camera test. Camera label is not promotion evidence.",
        "discovery_method": "official clips.nba.com event probing; deduplicate resolved HLS clips",
        "events": results,
        "promotion_allowed": False,
    }
    (args.out / "same_game_camera_prior_scan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ok = sum(r["status"] == "ok" for r in results)
    print(json.dumps(payload, indent=2))
    if ok < max(3, args.count // 2):
        raise SystemExit(f"Only {ok}/{args.count} {args.camera_label} clips downloaded")


if __name__ == "__main__":
    main()
