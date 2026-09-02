from __future__ import annotations

"""Acquire a small native same-game sample set for one exact camera label.

This is acquisition only. It intentionally avoids PBP and retrieves official
clips.nba.com media directly. Samples are native 960x540 source pixels at fixed
fractions of independent event clips; no scaling, interpolation or generated pixels.
"""

import argparse
import html as htmlmod
import json
import re
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import nba_video_worker as w

W, H = 960, 540
FRACTIONS = (0.25, 0.50, 0.75)


def inventory_label(game_id: str, event_id: int, wanted: str) -> tuple[str, str]:
    page = f"https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs"
    text = w.http_bytes(page, w.H).decode("utf-8", "replace")
    tm = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
    title = htmlmod.unescape(tm.group(1).strip()) if tm else ""
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', text, re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        label = re.sub(r"<[^>]+>", "", htmlmod.unescape(m.group(3))).strip()
        if label == wanted and ".m3u8" in url.lower() and "lrmedia.nba.com" in url.lower():
            return url, title
    raise RuntimeError(f"No {wanted!r} official HLS for {game_id}/{event_id}")


def safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    events = [int(x.strip()) for x in args.events.split(",") if x.strip()]
    if len(events) < 3:
        raise SystemExit("At least three independent events are required")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for event_id in events:
        url, title = inventory_label(args.game_id, event_id, args.label)
        clip = args.out / f"event_{event_id}_{safe(args.label)}_SOURCE.mp4"
        w.download_hls_source(url, clip)
        q = w.probe_video(clip)
        if not q.get("ok"):
            raise RuntimeError(f"Invalid media {event_id}: {q}")
        if (int(q.get("width") or 0), int(q.get("height") or 0)) != (W, H):
            raise RuntimeError(f"Non-native raster for event {event_id}: {(q.get('width'), q.get('height'))}")
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {clip}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            raise RuntimeError(f"No decoded frames for {clip}")
        samples = []
        for j, frac in enumerate(FRACTIONS):
            idx = int(round(frac * (count - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None or frame.shape[:2] != (H, W):
                raise RuntimeError(f"Decode failed {event_id} sample {j}")
            dst = args.out / f"{safe(args.label)}__event{event_id:04d}__s{j}.png"
            cv2.imwrite(str(dst), frame)
            samples.append({"sample": j, "fraction": frac, "decoded_frame_index": idx, "file": dst.name})
        cap.release()
        clip.unlink(missing_ok=True)
        rows.append({"event_id": event_id, "title": title, "probe": q, "samples": samples})
        print(args.label, event_id, "samples", len(samples), flush=True)
    payload = {
        "game_id": args.game_id,
        "camera_label": args.label,
        "events": rows,
        "fractions": list(FRACTIONS),
        "source_resolution": [W, H],
        "guardrail": "acquisition only; fixed source pixels; no camera promotion",
    }
    (args.out / "sample_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
