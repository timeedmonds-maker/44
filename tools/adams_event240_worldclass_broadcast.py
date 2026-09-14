#!/usr/bin/env python3
"""Deterministic broadcast overlay for the validated event-240 Steven Adams screen.

The shot metric is resolved directly from the already-proven official NBA
shotqualityvideologs endpoint by exact (game_id, player_id, event_num).
No cross-repository CSV handoff is required.

The source is rendered at native resolution first.  Only after the native
master is complete is the repository's single approved deterministic
presentation renderer used to make the UHD delivery master.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

from render_deterministic_master import render as render_presentation

GAME_ID = "0022500001"
EVENT_NUM = 240
SHOOTER_ID = 1642263
XFG_URL = "https://stats.gleague.nba.com/stats/shotqualityvideologs"

# Validated event-240 track mapping from the accepted engineering pass.
PLAYERS = {
    29: {"label": "SHEPPARD", "team": "HOU", "role": "ballhandler"},
    34: {"label": "ADAMS", "team": "HOU", "role": "screener"},
    35: {"label": "MITCHELL", "team": "OKC", "role": "screened defender"},
    13: {"label": "HARTENSTEIN", "team": "OKC", "role": "screener defender"},
}

# BGR broadcast palette: strong enough to read, restrained enough to sit on TV footage.
COLORS = {"HOU": (45, 62, 218), "OKC": (218, 119, 28)}
DARK = {"HOU": (18, 25, 92), "OKC": (75, 43, 10)}
LABEL_TIER = {29: 0, 35: 0, 34: -5, 13: -24}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def run(cmd: list[str], timeout: int = 900) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def fetch_exact_xfg() -> dict:
    """Resolve one official shot-quality row for Reed Sheppard event 240."""
    s = requests.Session()
    s.headers.update(HEADERS)
    last = None
    payload = None
    for attempt in range(1, 6):
        try:
            r = s.get(XFG_URL, params={"GameID": GAME_ID, "PlayerID": SHOOTER_ID}, timeout=(8, 35))
            if r.status_code == 200:
                payload = r.json()
                if str(payload.get("gameId") or "").zfill(10) == GAME_ID and int(payload.get("playerId") or 0) == SHOOTER_ID:
                    break
                last = f"unexpected payload: {r.text[:240]}"
                payload = None
            else:
                last = f"HTTP {r.status_code}: {r.text[:240]}"
        except Exception as exc:
            last = repr(exc)
        if attempt < 5:
            time.sleep(min(6.0, 0.7 * (2 ** (attempt - 1))))
    if payload is None:
        raise RuntimeError(f"official xFG fetch failed: {last}")

    hits = []
    for sh in payload.get("shotList") or []:
        try:
            ev = int(sh.get("eventNum"))
        except Exception:
            continue
        if ev == EVENT_NUM:
            hits.append(sh)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one official xFG shot for event {EVENT_NUM}, got {len(hits)}")

    sh = hits[0]
    q = float(sh.get("shotQuality"))
    pct = q * 100.0 if q <= 1.5 else q
    rec = {
        "game_id": GAME_ID,
        "event_num": EVENT_NUM,
        "player_id": SHOOTER_ID,
        "xfg": q,
        "xfg_pct": pct,
        "period": sh.get("period"),
        "game_clock": sh.get("gameClock"),
        "action_type": sh.get("actionType"),
        "shot_type": sh.get("shotType"),
        "made": int(sh.get("success") or 0),
        "loc_x": sh.get("locX"),
        "loc_y": sh.get("locY"),
        "guid": sh.get("guid"),
        "source": XFG_URL,
        "join": "exact game_id + player_id + event_num",
    }
    return rec


def rolling_stable(v: np.ndarray, med: int, mean: int) -> np.ndarray:
    s = pd.Series(v, dtype="float64")
    s = s.rolling(med, center=True, min_periods=1).median()
    s = s.rolling(mean, center=True, min_periods=1).mean()
    return s.to_numpy(float)


def prepare_track(s: pd.DataFrame) -> dict:
    s = s.sort_values("time_s").copy()
    s = s[(s["y2"] - s["y1"]) >= 70].copy()
    if s.empty:
        raise ValueError("empty stable track")
    s = s.groupby("time_s", as_index=False)[["x1", "y1", "x2", "y2"]].median()
    t = s["time_s"].to_numpy(float)
    cx = ((s["x1"] + s["x2"]) / 2.0).to_numpy(float)
    head = s["y1"].to_numpy(float)
    foot = s["y2"].to_numpy(float)
    width = (s["x2"] - s["x1"]).to_numpy(float)
    rx = int(np.clip(np.median(width) * 1.00, 30, 46))
    ry = int(np.clip(rx * 0.27, 8, 12))
    return {
        "t": t,
        "cx_ring": rolling_stable(cx, 5, 5),
        "foot_ring": rolling_stable(foot, 5, 5),
        "cx_label": rolling_stable(cx, 7, 9),
        "head_label": rolling_stable(head, 7, 9),
        "rx": rx,
        "ry": ry,
    }


def interp_track(p: dict, t: float):
    ts = p["t"]
    if t < ts[0] - 0.08 or t > ts[-1] + 0.08:
        return None
    j = int(np.searchsorted(ts, t))
    if 0 < j < len(ts) and (ts[j] - ts[j - 1]) > 0.65:
        return None
    return {
        "ring_x": float(np.interp(t, ts, p["cx_ring"])),
        "ring_y": float(np.interp(t, ts, p["foot_ring"])),
        "label_x": float(np.interp(t, ts, p["cx_label"])),
        "head_y": float(np.interp(t, ts, p["head_label"])),
        "rx": p["rx"],
        "ry": p["ry"],
    }


def rounded_rect(img, p1, p2, color, radius=5):
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    for x, y in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)


def alpha_blend(frame, layer, alpha: float) -> None:
    cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_floor_ring(frame, anchor, color, dark) -> None:
    """Wide flat open horseshoe around the feet, broadcast-analysis style."""
    cx = int(round(anchor["ring_x"]))
    cy = int(round(anchor["ring_y"] + 2))
    rx, ry = int(anchor["rx"]), int(anchor["ry"])
    # Leave a broad gap behind the player; draw only the near/front horseshoe.
    cv2.ellipse(frame, (cx, cy + 1), (rx + 2, ry + 2), 0, 12, 168, dark, 7, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 12, 168, color, 4, cv2.LINE_AA)


def draw_label(frame, anchor, text: str, color, dark, tier: int = 0) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.44
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    px, py = 7, 4
    W, H = tw + 2 * px, th + base + 2 * py
    cx = int(round(anchor["label_x"]))
    x = int(round(cx - W / 2))
    y = int(round(anchor["head_y"] - H - 10 + tier))
    x = max(4, min(w - W - 4, x))
    y = max(4, min(h - H - 4, y))

    shadow = frame.copy()
    rounded_rect(shadow, (x + 1, y + 2), (x + W + 1, y + H + 2), (10, 10, 10), 5)
    alpha_blend(frame, shadow, 0.44)
    plate = frame.copy()
    rounded_rect(plate, (x, y), (x + W, y + H), dark, 5)
    rounded_rect(plate, (x + 1, y + 1), (x + W - 1, y + H - 1), color, 4)
    alpha_blend(frame, plate, 0.91)
    cv2.putText(frame, text, (x + px, y + py + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return x, y, W, H


def smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    q = min(1.0, max(0.0, (x - a) / (b - a)))
    return q * q * (3.0 - 2.0 * q)


def metric_alpha(t: float, start: float, end: float) -> float:
    if t < start - 0.16 or t > end + 0.16:
        return 0.0
    if t < start + 0.16:
        return smoothstep(start - 0.16, start + 0.16, t)
    if t > end - 0.16:
        return 1.0 - smoothstep(end - 0.16, end + 0.16, t)
    return 1.0


def draw_xfg_chip(frame, anchor, xfg_pct: float, alpha: float) -> None:
    if alpha <= 0.001:
        return
    h, w = frame.shape[:2]
    text = f"xFG {xfg_pct:.1f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.46
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    px, py = 8, 5
    W, H = tw + 2 * px, th + base + 2 * py
    cx = int(round(anchor["label_x"]))
    x = int(round(cx - W / 2))
    y = int(round(anchor["head_y"] + 8))
    x = max(4, min(w - W - 4, x))
    y = max(4, min(h - H - 4, y))

    plate = frame.copy()
    rounded_rect(plate, (x, y), (x + W, y + H), (24, 24, 24), 6)
    cv2.rectangle(plate, (x + 1, y + 1), (x + W - 1, y + H - 1), (190, 190, 190), 1, cv2.LINE_AA)
    alpha_blend(frame, plate, 0.90 * alpha)
    text_layer = frame.copy()
    cv2.putText(text_layer, text, (x + px, y + py + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    alpha_blend(frame, text_layer, alpha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=9.0)
    ap.add_argument("--end-s", type=float, default=13.45)
    ap.add_argument("--xfg-start-s", type=float, default=11.15)
    ap.add_argument("--xfg-end-s", type=float, default=12.45)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    xfg = fetch_exact_xfg()
    (args.out / "xfg_event240.json").write_text(json.dumps(xfg, indent=2), encoding="utf-8")
    print("EXACT_XFG=" + json.dumps(xfg), flush=True)

    tracks = pd.read_csv(args.tracks)
    prepared = {}
    for tid in PLAYERS:
        rows = tracks[tracks["track_id"] == tid]
        if rows.empty:
            raise RuntimeError(f"required validated participant track T{tid} missing")
        prepared[tid] = prepare_track(rows)

    cap = cv2.VideoCapture(str(args.source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start_s * 1000.0)

    silent = args.out / "event240_overlay_native_silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_count = 0
    preview_targets = [9.75, 10.55, 11.45, 12.15]
    saved = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t < args.start_s - 0.04:
            continue
        if t > args.end_s:
            break

        anchors = {}
        # Draw rings first so labels always sit cleanly on top.
        for tid, meta in PLAYERS.items():
            a = interp_track(prepared[tid], t)
            if a is None:
                continue
            anchors[tid] = a
            draw_floor_ring(frame, a, COLORS[meta["team"]], DARK[meta["team"]])

        for tid, meta in PLAYERS.items():
            a = anchors.get(tid)
            if a is None:
                continue
            draw_label(frame, a, meta["label"], COLORS[meta["team"]], DARK[meta["team"]], LABEL_TIER[tid])

        shooter = anchors.get(29)
        if shooter is not None:
            draw_xfg_chip(frame, shooter, float(xfg["xfg_pct"]), metric_alpha(t, args.xfg_start_s, args.xfg_end_s))

        writer.write(frame)
        frame_count += 1
        for target in preview_targets:
            if target not in saved and t >= target:
                cv2.imwrite(str(args.out / f"preview_{target:.2f}s.jpg"), frame)
                saved.add(target)

    writer.release()
    cap.release()
    if frame_count < 60:
        raise RuntimeError(f"render produced too few frames: {frame_count}")

    duration = max(0.1, args.end_s - args.start_s)
    native = args.out / "event240_broadcast_native.mp4"
    run([
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-i", str(silent),
        "-ss", f"{args.start_s:.3f}", "-t", f"{duration:.3f}", "-i", str(args.source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(native),
    ])

    uhd = args.out / "event240_broadcast_UHD.mp4"
    uhd_qa = render_presentation(native, uhd, "uhd", preset="veryfast")
    (args.out / "uhd_render_qa.json").write_text(json.dumps(uhd_qa, indent=2), encoding="utf-8")

    manifest = {
        "game_id": GAME_ID,
        "event_num": EVENT_NUM,
        "source": str(args.source),
        "tracks": str(args.tracks),
        "render_window_s": [args.start_s, args.end_s],
        "xfg_window_s": [args.xfg_start_s, args.xfg_end_s],
        "xfg": xfg,
        "players": PLAYERS,
        "native_output": str(native),
        "uhd_output": str(uhd),
        "native_resolution": [w, h],
        "frame_count": frame_count,
        "presentation_note": "UHD is deterministic presentation upscale from official native source; not native 4K.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
