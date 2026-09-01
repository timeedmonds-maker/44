from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# The scanner lives under freeze_spin/, while nba_video_worker.py is repo-root.
# Add repo root explicitly so GitHub Actions and local invocations behave identically.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nba_video_worker as w


def fetch_pbp(game_id: str) -> list[dict]:
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    req = urllib.request.Request(url, headers={"User-Agent": w.UA, "Referer": "https://www.nba.com/"})
    with urllib.request.urlopen(req, timeout=45) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return payload["game"]["actions"]


def select_events(actions: list[dict], count: int) -> list[dict]:
    shots = []
    for a in actions:
        if str(a.get("actionType", "")).lower() not in {"2pt", "3pt"}:
            continue
        n = a.get("actionNumber")
        if n is None:
            continue
        shots.append(a)
    if len(shots) < count:
        raise RuntimeError(f"Only {len(shots)} field-goal actions available")
    idx = np.linspace(0, len(shots) - 1, count).round().astype(int)
    return [shots[int(i)] for i in idx]


def inventory_right_slash(game_id: str, event_id: int) -> tuple[str, str]:
    page = f"https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs"
    txt = w.http_bytes(page, w.H).decode("utf-8", "replace")
    title_m = re.search(r"<title>(.*?)</title>", txt, re.I | re.S)
    title = htmlmod.unescape(title_m.group(1).strip()) if title_m else ""
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', txt, re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        label = re.sub(r"<[^>]+>", "", htmlmod.unescape(m.group(3))).strip()
        if label == "Right Slash" and ".m3u8" in url.lower() and "lrmedia.nba.com" in url.lower():
            return url, title
    raise RuntimeError(f"No Right Slash HLS for {game_id}/{event_id}")


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
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{t:.5f}", "-i", str(video), "-frames:v", "1", str(p)], check=True)
        frames.append(p)
    return frames


def make_sheet(frames: list[Path], event: dict, out: Path) -> None:
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
    rows=[]
    for r in range(3):
        row=cells[r*3:(r+1)*3]
        while len(row)<3:
            row.append(np.full_like(cells[0],255))
        rows.append(np.hstack(row))
    sheet=np.vstack(rows)
    header=np.full((90, sheet.shape[1],3),255,np.uint8)
    desc=f"event {event.get('actionNumber')} | P{event.get('period')} {event.get('clock')} | {event.get('description','')}"
    cv2.putText(header, desc[:180], (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([header,sheet]))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    actions=fetch_pbp(args.game_id)
    events=select_events(actions,args.count)
    results=[]
    for rank,a in enumerate(events,1):
        eid=int(a["actionNumber"])
        rec={"rank":rank,"event_id":eid,"period":a.get("period"),"clock":a.get("clock"),"description":a.get("description")}
        try:
            url,title=inventory_right_slash(args.game_id,eid)
            clip=args.out/f"event_{eid}_Right_Slash_SOURCE.mp4"
            w.download_hls_source(url,clip)
            q=w.probe_video(clip)
            if not q.get("ok"):
                raise RuntimeError(q.get("reason"))
            frames=extract_frames(clip,args.out/f"event_{eid}_frames")
            sheet=args.out/f"event_{eid}_Right_Slash_sheet.png"
            make_sheet(frames,a,sheet)
            rec.update({"status":"ok","title":title,"probe":q,"sheet":sheet.name})
        except Exception as e:
            rec.update({"status":"failed","error":repr(e)})
        results.append(rec)
    payload={"game_id":args.game_id,"purpose":"Find clean same-game Right Slash fixed-geometry frames for a reusable per-game camera prior; no player points used.","events":results}
    (args.out/"same_game_right_slash_prior_scan.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    ok=sum(r["status"]=="ok" for r in results)
    print(json.dumps(payload,indent=2))
    if ok < max(3,args.count//2):
        raise SystemExit(f"Only {ok}/{args.count} Right Slash events retrieved")

if __name__=="__main__":
    main()
