from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

SPORTS_BALL_CLASS = 37


def label_from_name(path: Path) -> str:
    m = re.search(r"_489_(.+)_SOURCE\.mp4$", path.name)
    if not m:
        return path.stem
    return m.group(1).replace("_", " ")


def orange_rim_candidates(frame: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Basketball rim orange spans a fairly compact hue range even under arena lighting.
    lo = np.array([2, 95, 65], np.uint8)
    hi = np.array([28, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    H, W = frame.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 35 or w < 18 or h < 2:
            continue
        aspect = w / max(h, 1)
        # Rim/brace fragments are elongated; reject huge floor/crowd orange regions.
        if aspect < 1.45 or w > W * 0.38 or h > H * 0.16:
            continue
        if y < H * 0.10 or y > H * 0.82:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        center_penalty = abs(cx - W / 2.0) / W
        score = float(area * min(aspect, 8.0) / (1.0 + 2.0 * center_penalty))
        out.append({"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy, "score": score})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:8]


def detect_ball(model, frame: np.ndarray) -> list[dict]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        pred = model([to_tensor(rgb)])[0]
    rows = []
    for score, label, box in zip(pred["scores"].cpu().numpy(), pred["labels"].cpu().numpy(), pred["boxes"].cpu().numpy()):
        if int(label) != SPORTS_BALL_CLASS or float(score) < 0.12:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        rows.append({
            "score": float(score),
            "box": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "radius": max(x2 - x1, y2 - y1) / 2.0,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-hz", type=float, default=5.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clips = sorted(args.clips.glob("*_R01_0022500301_489_*_SOURCE.mp4"))
    if len(clips) < 4:
        raise RuntimeError(f"Expected >=4 event-489 clips, found {len(clips)}")
    by_label = {label_from_name(p): p for p in clips}
    reference_label = "Left Above Rim" if "Left Above Rim" in by_label else ("Right Above Rim" if "Right Above Rim" in by_label else "Broadcast")
    reference_path = by_label[reference_label]

    cap = cv2.VideoCapture(str(reference_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = nframes / max(fps, 1e-6)
    center = duration / 2.0
    start = max(0.0, center - 2.2)
    end = min(duration - 0.05, center + 1.2)

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True).eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    candidates = []
    step = 1.0 / max(args.sample_hz, 1.0)
    t = start
    while t <= end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            t += step
            continue
        balls = detect_ball(model, frame)
        rims = orange_rim_candidates(frame)
        H, W = frame.shape[:2]
        for b in balls:
            for r in rims:
                dx = abs(b["cx"] - r["cx"])
                # Pre-dunk preference: ball bottom at/just above rim center and horizontally close.
                vertical = r["cy"] - (b["cy"] + b["radius"])
                if dx > max(160.0, r["w"] * 2.2):
                    continue
                if vertical < -28.0 or vertical > 120.0:
                    continue
                # Prefer a ball 0-35 px above the rim, high ball confidence, and a strong rim contour.
                vertical_cost = abs(vertical - 12.0)
                time_cost = abs(t - center) * 4.0
                score = 4.0 * b["score"] + math.log1p(r["score"]) - 0.035 * dx - 0.055 * vertical_cost - 0.12 * time_cost
                candidates.append({
                    "time": float(t),
                    "score": float(score),
                    "ball": b,
                    "rim": r,
                    "horizontal_distance_px": float(dx),
                    "ball_bottom_above_rim_center_px": float(vertical),
                })
        t += step
    cap.release()

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if candidates:
        chosen = candidates[0]
        confidence = "high" if chosen["ball"]["score"] >= 0.45 and abs(chosen["ball_bottom_above_rim_center_px"] - 12.0) <= 30 else "moderate"
        chosen_time = float(chosen["time"])
        method = "Mask R-CNN sports-ball detection + orange rim contour; highest pre-dunk ball-above-rim score"
    else:
        # Conservative fallback: event action is normally centered in NBA event clips; choose 0.35 s before center.
        chosen_time = max(0.0, center - 0.35)
        confidence = "low"
        chosen = None
        method = "fallback center-minus-0.35s; automated ball/rim pair unavailable"

    # Export the exact chosen reference frame for downstream QA/calibration.
    cap = cv2.VideoCapture(str(reference_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, chosen_time * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not export selected pre-dunk reference frame")
    cv2.imwrite(str(args.out / "predunk_reference_native.png"), frame)

    angles = []
    for label, p in sorted(by_label.items()):
        angles.append({"label": label, "file": p.name, "freeze_time": round(chosen_time, 5)})
    sync_map = {
        "event": {"game_id": "0022500301", "event_id": 489, "description": "S. Adams DUNK vs UTA immediately after block"},
        "source_fps": fps,
        "synchronization": "automatic reference pre-dunk state + transient-audio graph",
        "reference_angle": reference_label,
        "angles": angles,
    }
    (args.out / "jazz_predunk_sync_map.json").write_text(json.dumps(sync_map, indent=2), encoding="utf-8")

    report = {
        "reference_label": reference_label,
        "reference_file": reference_path.name,
        "fps": fps,
        "duration_seconds": duration,
        "search_window_seconds": [start, end],
        "selected_time_seconds": chosen_time,
        "selection_method": method,
        "confidence": confidence,
        "chosen": chosen,
        "top_candidates": candidates[:10],
        "native_resolution": [frame.shape[1], frame.shape[0]],
        "policy": "select the last clean pre-dunk geometry state, not rim-contact/completed dunk",
    }
    (args.out / "predunk_selection_v1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
