from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

from select_jazz_predunk_ball_apex_v4 import (
    FPS,
    label_from_name,
    read_frames,
    detect_ball_batches,
    add_rim_and_hybrid_candidates,
    best_track,
)

BALL_CLASS = 37


def select_primary_apex(track: list[dict], old_local: float) -> tuple[dict | None, dict]:
    if len(track) < 10:
        return None, {"reason": f"primary track too short: {len(track)}"}
    sem = [q for q in track if q["source"] == "maskrcnn"]
    if len(sem) < 5:
        return None, {"reason": f"primary track has only {len(sem)} semantic observations"}

    track = sorted(track, key=lambda q: q["time"])
    t = np.array([float(q["time"]) for q in track], dtype=float)
    h = np.array([float(q["height"]) for q in track], dtype=float)

    # Hard user-QA boundary: v15's 9.635413 reference state was already post-dunk/hanging.
    keep = t < old_local - 0.08
    t = t[keep]; h = h[keep]
    if len(t) < 9:
        return None, {"reason": "too few primary observations safely before rejected hanging state"}

    grid = np.arange(t.min(), t.max() + 0.25 / FPS, 1.0 / FPS)
    hg = np.interp(grid, t, h)
    win = min(9, len(grid) if len(grid) % 2 == 1 else len(grid) - 1)
    if win < 5:
        return None, {"reason": "insufficient regularized trajectory for smoothing"}
    sm = savgol_filter(hg, win, 2, mode="interp")

    candidates = []
    for i in range(5, len(grid) - 4):
        # Local top only; never accept an endpoint or still-rising observation.
        if not (sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]):
            continue
        before = sm[max(0, i - 7):i]
        after = sm[i + 1:min(len(sm), i + 8)]
        rise = float(sm[i] - np.percentile(before, 20))
        fall = float(sm[i] - np.percentile(after, 20))
        if sm[i] < 8.0 or rise < 3.0 or fall < 4.0:
            continue
        apex_t = float(grid[i])
        sem_near = sum(1 for q in sem if abs(float(q["time"]) - apex_t) <= 0.105)
        if sem_near < 1:
            continue
        # Require actual tracked observations on both sides, not interpolation-only evidence.
        pre_obs = sum(1 for q in track if apex_t - 0.22 <= float(q["time"]) < apex_t)
        post_obs = sum(1 for q in track if apex_t < float(q["time"]) <= apex_t + 0.22)
        if pre_obs < 3 or post_obs < 3:
            continue
        # First descending approach to the rim plane after the top.
        crossing = None
        for j in range(i + 1, len(grid)):
            if sm[j] <= 4.0:
                crossing = float(grid[j]); break
        if crossing is None:
            continue
        lead = crossing - apex_t
        if not (0.025 <= lead <= 0.50):
            continue
        candidates.append({
            "apex_local_time": apex_t,
            "apex_height_px": float(sm[i]),
            "rise_px": rise,
            "fall_px": fall,
            "rim_crossing_local_time": crossing,
            "apex_to_crossing_s": lead,
            "semantic_near_apex": sem_near,
            "pre_observations": pre_obs,
            "post_observations": post_obs,
        })

    if not candidates:
        return None, {
            "reason": "no expanded-window interior rise-top-fall apex before rim crossing",
            "track_start": float(t.min()),
            "track_end": float(t.max()),
            "height_min_px": float(np.min(sm)),
            "height_max_px": float(np.max(sm)),
        }

    # A dunk should have one dominant top; prefer strongest combined rise+fall, then height.
    candidates.sort(key=lambda r: (r["rise_px"] + r["fall_px"], r["apex_height_px"]), reverse=True)
    return candidates[0], {"candidate_count": len(candidates)}


def primary_scan(model, path: Path, offset: float, old_ref: float, out: Path) -> dict:
    old_local = old_ref + offset
    # v16 showed the top was already at the left edge of its 0.90 s window (1 px rise, 9.25 px fall).
    # Expand substantially earlier instead of weakening the rise/top/descent gate.
    times = list(np.arange(old_local - 1.75, old_local - 0.035, 1.0 / FPS))
    rows = read_frames(path, times)
    detect_ball_batches(model, rows)
    add_rim_and_hybrid_candidates(rows)
    track = best_track(rows)
    apex, extra = select_primary_apex(track, old_local)
    diag = {
        "label": label_from_name(path),
        "offset_seconds_vs_reference": float(offset),
        "old_rejected_local_time": float(old_local),
        "sampled_frames": len(rows),
        "semantic_detection_frames": int(sum(r["semantic_count"] > 0 for r in rows)),
        "track_observations": len(track),
        "track_semantic_observations": int(sum(q["source"] == "maskrcnn" for q in track)),
        "track": [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in q.items()} for q in track],
    }
    if apex is None:
        return {**diag, "passed": False, "failure": extra}

    apex_ref = float(apex["apex_local_time"] - offset)
    crossing_ref = float(apex["rim_crossing_local_time"] - offset)

    chosen = []
    for dt in (-4/FPS, -3/FPS, -2/FPS, -1/FPS, 0, 1/FPS, 2/FPS, 3/FPS, 4/FPS):
        tt = apex["apex_local_time"] + dt
        row = min(rows, key=lambda r: abs(r["time"] - tt))
        im = row["image"].copy()
        cv2.putText(im, f"Right Slash  t={row['time']:.3f}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)
        chosen.append(im)
    cv2.imwrite(str(out / "Right_Slash_primary_apex_9frame_strip.png"), np.hstack(chosen))
    return {**diag, "passed": True, **extra, **apex,
            "apex_reference_time": apex_ref, "rim_crossing_reference_time": crossing_ref}


def semantic_confirm(model, path: Path, offset: float, apex_ref: float, out: Path) -> dict:
    local = float(apex_ref + offset)
    times = [local + k/FPS for k in (-2,-1,0,1,2)]
    rows = read_frames(path, times)
    detect_ball_batches(model, rows, batch=5)
    hits = []
    strip = []
    for r in rows:
        compact, rims = __import__("select_jazz_ball_apex_multiview_v3").orange_components(r["image"])
        rim = __import__("select_jazz_ball_apex_multiview_v3").nearest_rim(rims, r["image"].shape[1], r["image"].shape[0])
        im = r["image"].copy()
        local_hits = []
        if rim is not None:
            for b in r.get("semantic_balls", []):
                dx = abs(float(b["cx"]) - float(rim["cx"])); dy = float(b["cy"]) - float(rim["cy"])
                if dx <= 260 and -270 <= dy <= 150:
                    local_hits.append({"score": float(b["score"]), "cx": float(b["cx"]), "cy": float(b["cy"]), "dx_rim": dx, "dy_rim": dy})
                    cv2.circle(im, (int(round(b["cx"])), int(round(b["cy"]))), 14, (255,255,255), 2)
        if local_hits:
            hits.append({"time": float(r["time"]), "balls": local_hits})
        cv2.putText(im, f"{label_from_name(path)} t={r['time']:.3f}", (14,28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)
        strip.append(im)
    if strip:
        cv2.imwrite(str(out / f"{label_from_name(path).replace(' ','_')}_apex_confirmation_strip.png"), np.hstack(strip))
    return {"label": label_from_name(path), "mapped_local_time": local,
            "semantic_hit_frames": len(hits), "hits": hits,
            "passed": bool(hits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old-rejected-apex-ref", type=float, default=9.635413333166447)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    sync = json.load(open(args.sync))
    offsets = {r["label"]: float(r["offset_seconds_vs_reference"]) for r in sync["angles"]}
    files = {label_from_name(p): p for p in args.clips.glob("*_489_*_SOURCE.mp4")}
    primary = "Right Slash"
    confirms = ["Right HandHeld", "Left Slash", "High Tight", "Right Above Rim", "Left Above Rim"]
    missing = [x for x in [primary] + confirms if x not in files]
    if missing:
        raise RuntimeError(f"Missing apex views: {missing}")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, progress=True).eval()

    p = primary_scan(model, files[primary], offsets[primary], float(args.old_rejected_apex_ref), args.out)
    print(json.dumps({k:v for k,v in p.items() if k != "track"}, indent=2), flush=True)
    if not p.get("passed"):
        result = {"passed": False, "reason": "primary Right Slash semantic trajectory failed", "primary": p}
    else:
        apex_ref = float(p["apex_reference_time"])
        confirmations = [semantic_confirm(model, files[x], offsets[x], apex_ref, args.out) for x in confirms]
        passed_conf = [r for r in confirmations if r["passed"]]
        safe_before = float(args.old_rejected_apex_ref) - apex_ref
        passed = len(passed_conf) >= 2 and safe_before >= 0.08
        result = {
            "passed": bool(passed),
            "method": "expanded-window Right Slash semantic basketball trajectory + local rise/top/fall + first rim-plane crossing + synchronized multi-camera semantic confirmation",
            "apex_reference_time": apex_ref,
            "rim_crossing_reference_time": float(p["rim_crossing_reference_time"]),
            "seconds_before_user_rejected_hanging_state": safe_before,
            "primary": p,
            "confirmations": confirmations,
            "confirmation_count": len(passed_conf),
            "confirmation_labels": [r["label"] for r in passed_conf],
            "gate": {"primary_min_track_observations":10, "primary_min_semantic_observations":5,
                     "min_confirmation_views":2, "must_precede_rejected_hanging_state_s":0.08},
            "policy": "v15 freeze is visually falsified. No post-rim/hanging state may be reused; apex must be supported by a real basketball trajectory before its first observed descent to rim plane.",
        }
    (args.out / "predunk_ball_apex_v5.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("primary","confirmations")}, indent=2), flush=True)
    if not result.get("passed"):
        raise SystemExit("Expanded-window pre-contact apex gate failed")


if __name__ == "__main__":
    main()
