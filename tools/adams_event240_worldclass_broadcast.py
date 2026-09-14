#!/usr/bin/env python3
"""World-class deterministic broadcast overlay for the validated event-240 Adams screen.

STRICTLY NON-GENERATIVE:
- official NBA Broadcast HLS pixels remain untouched except for deterministic graphics;
- graphics are rendered with OpenCV only;
- final HD/UHD treatment is deterministic FFmpeg only.

Only the four players directly involved in the screen action are annotated.
Validated participant mapping from the event-240 V3 interaction window:
  T29 Reed Sheppard   ballhandler
  T34 Steven Adams    screener
  T35 Ajay Mitchell   screened defender
  T13 Isaiah Hartenstein screener defender

Presentation changes versus the earlier QA render:
- surname only; no jersey number;
- smaller labels centered above the player's head;
- stronger offline temporal smoothing for label and ring anchors;
- single front/floor U-arc only (no rear arc visible behind the player);
- constant per-player ring size over the short clip to eliminate breathing/jitter;
- exact official shot-level xFG joined by (game_id,event_num,player_id).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GAME_ID_NUMERIC = 22500001
EVENT_NUM = 240
SHOOTER_ID = 1642263

PLAYERS = {
    29: {"label": "SHEPPARD", "team": "HOU", "role": "ballhandler", "name": "Reed Sheppard"},
    34: {"label": "ADAMS", "team": "HOU", "role": "screener", "name": "Steven Adams"},
    35: {"label": "MITCHELL", "team": "OKC", "role": "screened defender", "name": "Ajay Mitchell"},
    13: {"label": "HARTENSTEIN", "team": "OKC", "role": "screener defender", "name": "Isaiah Hartenstein"},
}

# BGR; restrained broadcast palette rather than neon tracker colors.
COLORS = {
    "HOU": (45, 62, 218),
    "OKC": (218, 119, 28),
}
DARK = {
    "HOU": (18, 25, 92),
    "OKC": (75, 43, 10),
}

# Fixed vertical tiers prevent label collisions while keeping every label horizontally
# centered over its player's head. These do not change frame-to-frame.
LABEL_TIER = {
    29: 0,
    35: 0,
    34: -4,
    13: -24,
}


def rolling_stable(v: np.ndarray, med: int, mean: int) -> np.ndarray:
    """Centered robust smoothing for offline broadcast graphics, with no phase lag."""
    s = pd.Series(v, dtype="float64")
    s = s.rolling(med, center=True, min_periods=1).median()
    s = s.rolling(mean, center=True, min_periods=1).mean()
    return s.to_numpy(float)


def prepare_track(s: pd.DataFrame) -> dict:
    s = s.sort_values("time_s").copy()
    s = s[(s["y2"] - s["y1"]) >= 70].copy()
    if s.empty:
        raise ValueError("empty stable track")

    # Collapse duplicate timestamps deterministically.
    s = s.groupby("time_s", as_index=False)[["x1", "y1", "x2", "y2"]].median()
    t = s["time_s"].to_numpy(float)
    cx = ((s["x1"] + s["x2"]) / 2.0).to_numpy(float)
    head = s["y1"].to_numpy(float)
    foot = s["y2"].to_numpy(float)
    width = (s["x2"] - s["x1"]).to_numpy(float)

    # Rings follow the feet with moderate smoothing; labels get stronger smoothing.
    cx_ring = rolling_stable(cx, 5, 5)
    foot_ring = rolling_stable(foot, 5, 5)
    cx_label = rolling_stable(cx, 7, 9)
    head_label = rolling_stable(head, 7, 9)

    # Constant ring geometry over this short broadcast clip removes size breathing.
    rx = int(np.clip(np.median(width) * 0.92, 28, 43))
    ry = int(np.clip(rx * 0.24, 7, 11))

    return {
        "t": t,
        "cx_ring": cx_ring,
        "foot_ring": foot_ring,
        "cx_label": cx_label,
        "head_label": head_label,
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


def rounded_rect(layer, p1, p2, color, radius=5):
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(layer, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(layer, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    for x, y in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(layer, (x, y), r, color, -1, cv2.LINE_AA)


def alpha_blend(frame, layer, alpha):
    cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_floor_ring(frame, anchor, color, dark):
    """Draw only the near/front floor half of the ellipse.

    The rear half is intentionally absent, so there is never a line visually
    passing behind or through the player's legs/body.
    """
    cx = int(round(anchor["ring_x"]))
    cy = int(round(anchor["ring_y"] + 2))
    rx, ry = int(anchor["rx"]), int(anchor["ry"])

    # Soft keyline first, then team stroke. Only 18..162 degrees = front U-arc.
    cv2.ellipse(frame, (cx, cy + 1), (rx + 1, ry + 1), 0, 18, 162, dark, 5, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 18, 162, color, 3, cv2.LINE_AA)


def label_geometry(frame, anchor, text, tier):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.43
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    px, py = 7, 4
    W, H = tw + 2 * px, th + base + 2 * py
    cx = int(round(anchor["label_x"]))
    x = int(round(cx - W / 2))
    y = int(round(anchor["head_y"] - H - 10 + tier))
    x = max(4, min(w - W - 4, x))
    y = max(4, min(h - H - 4, y))
    return font, scale, thick, px, py, tw, th, base, W, H, x, y


def draw_label(frame, anchor, text, color, dark, tier=0):
    font, scale, thick, px, py, tw, th, base, W, H, x, y = label_geometry(frame, anchor, text, tier)

    # Stable, restrained drop shadow.
    shadow = frame.copy()
    rounded_rect(shadow, (x + 1, y + 2), (x + W + 1, y + H + 2), (10, 10, 10), 5)
    alpha_blend(frame, shadow, 0.46)

    # Broadcast plate, slightly translucent to preserve the live picture underneath.
    plate = frame.copy()
    rounded_rect(plate, (x, y), (x + W, y + H), dark, 5)
    rounded_rect(plate, (x + 1, y + 1), (x + W - 1, y + H - 1), color, 4)
    alpha_blend(frame, plate, 0.90)

    # Small centered pointer; fixed geometry avoids wobble.
    cx = int(round(anchor["label_x"]))
    p0 = (max(x + 7, min(x + W - 7, cx)), y + H)
    tri = np.array([[p0[0] - 4, p0[1] - 1], [p0[0] + 4, p0[1] - 1], [p0[0], p0[1] + 5]], np.int32)
    cv2.fillConvexPoly(frame, tri, color, cv2.LINE_AA)

    cv2.putText(frame, text, (x + px, y + py + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return (x, y, W, H)


def load_exact_xfg(path: Path) -> dict:
    use = [
        "game_id", "event_num", "player_id", "description", "xfg_xfg", "xfg_xfg_pct",
        "xfg_expected_points", "xfg_shot_value", "xfg_available", "xfg_source_resource"
    ]
    hits = []
    for chunk in pd.read_csv(path, usecols=use, chunksize=120000):
        g = pd.to_numeric(chunk["game_id"], errors="coerce")
        e = pd.to_numeric(chunk["event_num"], errors="coerce")
        p = pd.to_numeric(chunk["player_id"], errors="coerce")
        q = chunk[(g == GAME_ID_NUMERIC) & (e == EVENT_NUM) & (p == SHOOTER_ID)]
        if not q.empty:
            hits.append(q)
    if not hits:
        raise SystemExit("Exact event-240 Reed Sheppard xFG row not found; refusing to estimate.")
    out = pd.concat(hits, ignore_index=True)
    if len(out) != 1:
        raise SystemExit(f"Expected exactly one xFG row, found {len(out)}; refusing ambiguity.")
    r = out.iloc[0]
    pct = float(r["xfg_xfg_pct"])
    xfg = float(r["xfg_xfg"])
    ep = float(r["xfg_expected_points"])
    return {
        "game_id": GAME_ID_NUMERIC,
        "event_num": EVENT_NUM,
        "player_id": SHOOTER_ID,
        "description": str(r["description"]),
        "xfg": xfg,
        "xfg_pct": pct,
        "expected_points": ep,
        "shot_value": float(r["xfg_shot_value"]),
        "source_resource": str(r["xfg_source_resource"]),
    }


def draw_xfg_chip(frame, anchor, xfg_pct: float, alpha: float):
    """Small shooter-attached stat chip; shown only around the shot phase."""
    if alpha <= 0:
        return
    h, w = frame.shape[:2]
    text = f"xFG {xfg_pct:.1f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.39
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    px, py = 7, 4
    W, H = tw + 2 * px, th + base + 2 * py
    cx = int(round(anchor["label_x"]))
    x = int(round(cx - W / 2))
    # The metric sits below the name but still above the player's head.
    y = int(round(anchor["head_y"] + 3))
    x = max(4, min(w - W - 4, x)); y = max(4, min(h - H - 4, y))

    layer = frame.copy()
    rounded_rect(layer, (x, y), (x + W, y + H), (28, 28, 28), 5)
    cv2.rectangle(layer, (x + 1, y + 1), (x + W - 1, y + H - 1), (130, 130, 130), 1, cv2.LINE_AA)
    alpha_blend(frame, layer, 0.88 * alpha)
    # Text is faded by drawing on its own layer.
    txt = frame.copy()
    cv2.putText(txt, text, (x + px, y + py + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    alpha_blend(frame, txt, alpha)


def smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    q = min(1.0, max(0.0, (x - a) / (b - a)))
    return q * q * (3.0 - 2.0 * q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--xfg-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=9.0)
    ap.add_argument("--end-s", type=float, default=13.45)
    ap.add_argument("--xfg-start-s", type=float, default=11.15)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    xfg = load_exact_xfg(a.xfg_csv)
    df = pd.read_csv(a.tracks)
    prepared = {}
    for tid in PLAYERS:
        s = df[df["track_id"] == tid]
        if s.empty:
            raise SystemExit(f"Required validated participant track T{tid} missing")
        prepared[tid] = prepare_track(s)

    cap = cv2.VideoCapture(str(a.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start_s * 1000)

    silent = a.out / "event240_worldclass_silent.mp4"
    wr = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames = 0
    preview_times = [9.7, 10.35, 11.35, 12.1]
    preview_done = set()

    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t < a.start_s - 0.05:
            continue
        if t > a.end_s:
            break

        anchors = {}
        for tid, meta in PLAYERS.items():
            anchor = interp_track(prepared[tid], t)
            if anchor is None:
                continue
            anchors[tid] = anchor
            draw_floor_ring(fr, anchor, COLORS[meta["team"]], DARK[meta["team"]])

        # Labels rendered after rings so typography stays crisp.
        for tid, meta in PLAYERS.items():
            anchor = anchors.get(tid)
            if anchor is None:
                continue
            draw_label(fr, anchor, meta["label"], COLORS[meta["team"]], DARK[meta["team"]], LABEL_TIER[tid])

        # Official xFG appears only around the terminal shot, attached to Sheppard.
        sh = anchors.get(29)
        if sh is not None and t >= a.xfg_start_s:
            fade_in = smoothstep(a.xfg_start_s, a.xfg_start_s + 0.18, t)
            fade_out = 1.0 - smoothstep(a.end_s - 0.28, a.end_s, t)
            draw_xfg_chip(fr, sh, xfg["xfg_pct"], min(fade_in, fade_out))

        wr.write(fr)
        frames += 1
        for pt in preview_times:
            if pt not in preview_done and abs(t - pt) <= max(0.018, 0.55 / fps):
                cv2.imwrite(str(a.out / f"preview_{pt:.2f}s.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 97])
                preview_done.add(pt)

    wr.release(); cap.release()

    native = a.out / "event240_worldclass_native.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(silent),
        "-ss", str(a.start_s), "-to", str(a.end_s), "-i", str(a.source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-profile:v", "high", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(native)
    ], check=True)

    uhd = a.out / "event240_worldclass_UHD.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(native),
        "-vf", "hqdn3d=0.6:0.6:2.0:2.0,scale=3840:2160:flags=lanczos,cas=0.22,fps=30",
        "-c:v", "libx264", "-profile:v", "high", "-crf", "16", "-maxrate", "36M", "-bufsize", "72M", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(uhd)
    ], check=True)

    manifest = {
        "game_id": "0022500001",
        "event_num": EVENT_NUM,
        "gold_positive_screen": True,
        "source": "official NBA Broadcast HLS, native 960x540",
        "clip_window_s": [a.start_s, a.end_s],
        "annotated_players": [
            {"name": m["name"], "label": m["label"], "team": m["team"], "track_id": tid, "role": m["role"]}
            for tid, m in PLAYERS.items()
        ],
        "xfg": xfg,
        "graphics": {
            "method": "deterministic OpenCV only",
            "ring": "front-floor U-arc only; no rear/behind-player arc; robust centered temporal smoothing; constant short-clip geometry",
            "labels": "surname only, smaller, horizontally centered above head, stronger centered temporal smoothing",
            "numbers_removed": True,
        },
        "video_pixels": "no generated/replaced basketball frames",
        "uhd": "repo-standard deterministic presentation upscale; not native 4K",
        "frames_rendered": frames,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (a.out / "xfg_event240.json").write_text(json.dumps(xfg, indent=2))

    preview_files = [str(a.out / f"preview_{pt:.2f}s.jpg") for pt in preview_times if (a.out / f"preview_{pt:.2f}s.jpg").exists()]
    zip_cmd = ["zip", "-j", "-q", str(a.out / "event240_worldclass_package.zip"), str(uhd), str(native), str(a.out / "manifest.json"), str(a.out / "xfg_event240.json")] + preview_files
    subprocess.run(zip_cmd, check=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
