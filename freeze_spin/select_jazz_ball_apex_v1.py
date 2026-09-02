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
    return m.group(1).replace("_", " ") if m else path.stem


def orange_components(frame: np.ndarray) -> tuple[list[dict], list[dict]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broad enough for the NBA ball/rim under warm arena lighting, while retaining saturation.
    mask = cv2.inRange(hsv, np.array([1, 65, 45], np.uint8), np.array([32, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compact, elongated = [], []
    H, W = frame.shape[:2]
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 12:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        peri = float(cv2.arcLength(c, True))
        circ = float(4.0 * math.pi * area / max(peri * peri, 1e-6))
        fill = float(area / max(w * h, 1))
        roi = hsv[y:y+h, x:x+w]
        sat = float(np.mean(roi[..., 1])) if roi.size else 0.0
        row = {
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": float(x + w / 2.0), "cy": float(y + h / 2.0),
            "area": area, "circularity": circ, "fill": fill, "mean_saturation": sat,
        }
        aspect = w / max(h, 1)
        # Rim candidates: elongated orange geometry, not giant floor/crowd blobs.
        if area >= 28 and w >= 18 and aspect >= 1.45 and w <= W * 0.42 and h <= H * 0.18:
            row["rim_score"] = float(area * min(aspect, 8.0) * (0.5 + sat / 255.0))
            elongated.append(row)
        # Ball candidates: compact, roughly round, plausible native 540p size.
        if 4 <= w <= 64 and 4 <= h <= 64 and 0.50 <= aspect <= 1.90 and circ >= 0.12 and fill >= 0.16:
            size_pref = math.exp(-abs(math.log(max(math.sqrt(area), 1.0) / 17.0)) / 1.35)
            row["ball_color_score"] = float(1.4 * circ + 0.8 * fill + 0.9 * (sat / 255.0) + 0.6 * size_pref)
            compact.append(row)
    elongated.sort(key=lambda r: r["rim_score"], reverse=True)
    compact.sort(key=lambda r: r["ball_color_score"], reverse=True)
    return compact[:30], elongated[:12]


def maskrcnn_balls(model, frame: np.ndarray) -> list[dict]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        p = model([to_tensor(rgb)])[0]
    rows = []
    for score, label, box in zip(p["scores"].cpu().numpy(), p["labels"].cpu().numpy(), p["boxes"].cpu().numpy()):
        if int(label) != SPORTS_BALL_CLASS or float(score) < 0.025:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        rows.append({
            "score": float(score), "box": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "w": x2 - x1, "h": y2 - y1,
        })
    return rows[:12]


def nearest_rim(rims: list[dict], W: int, H: int) -> dict | None:
    if not rims:
        return None
    # Dunk clips are basket-centered. Penalize edge/crowd orange while keeping angle flexibility.
    def s(r: dict) -> float:
        center_pen = abs(r["cx"] - W / 2.0) / W + 0.35 * abs(r["cy"] - H * 0.43) / H
        return r["rim_score"] / (1.0 + 3.0 * center_pen)
    return max(rims, key=s)


def dedupe_candidates(color_rows: list[dict], net_rows: list[dict]) -> list[dict]:
    out = []
    for r in color_rows:
        row = dict(r)
        row["source"] = "orange"
        row["detector_score"] = float(r["ball_color_score"])
        out.append(row)
    for n in net_rows:
        merged = False
        for r in out:
            d = math.hypot(n["cx"] - r["cx"], n["cy"] - r["cy"])
            if d <= max(10.0, 0.55 * max(n["w"], n["h"], r["w"], r["h"])):
                r["source"] = "orange+maskrcnn"
                r["maskrcnn_score"] = float(n["score"])
                r["detector_score"] += 2.4 * float(n["score"])
                merged = True
                break
        if not merged:
            out.append({
                "x": int(n["box"][0]), "y": int(n["box"][1]),
                "w": float(n["w"]), "h": float(n["h"]),
                "cx": float(n["cx"]), "cy": float(n["cy"]),
                "source": "maskrcnn", "maskrcnn_score": float(n["score"]),
                "detector_score": float(2.4 * n["score"]),
            })
    return out


def build_track(frames: list[dict]) -> list[dict]:
    # Dynamic programming over the compact candidates. The ball must stay near the basket and move smoothly.
    prev_scores, prev_paths = {}, {}
    for i, fr in enumerate(frames):
        rim = fr["rim"]
        candidates = fr["candidates"]
        cur_scores, cur_paths = {}, {}
        for j, c in enumerate(candidates):
            dx_rim = abs(c["cx"] - rim["cx"])
            dy_rim = c["cy"] - rim["cy"]
            if dx_rim > 245 or dy_rim < -230 or dy_rim > 125:
                continue
            # Prefer compact evidence close enough to the basket to plausibly be the dunking ball.
            local = float(c["detector_score"] - 0.0028 * dx_rim - 0.0010 * abs(dy_rim + 45.0))
            if i == 0 or not prev_scores:
                cur_scores[j] = local
                cur_paths[j] = [(i, j)]
                continue
            best = None
            for pj, ps in prev_scores.items():
                pc = frames[i - 1]["candidates"][pj]
                dist = math.hypot(c["cx"] - pc["cx"], c["cy"] - pc["cy"])
                if dist > 48:
                    continue
                # Smooth motion, but do not reward a static orange object.
                step_score = ps + local - 0.018 * dist + min(dist, 18.0) * 0.012
                if best is None or step_score > best[0]:
                    best = (step_score, pj)
            if best is not None:
                cur_scores[j] = best[0]
                cur_paths[j] = prev_paths[best[1]] + [(i, j)]
        prev_scores, prev_paths = cur_scores, cur_paths
    if not prev_scores:
        return []
    best_j = max(prev_scores, key=prev_scores.get)
    idx_path = prev_paths[best_j]
    track = []
    for fi, cj in idx_path:
        fr = frames[fi]; c = fr["candidates"][cj]; rim = fr["rim"]
        track.append({
            "frame_index": int(fr["frame_index"]), "time": float(fr["time"]),
            "ball": c, "rim": rim,
            "height_proxy_px": float(rim["cy"] - c["cy"]),
        })
    return track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--impact-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="Left Above Rim")
    ap.add_argument("--search-before", type=float, default=0.95)
    ap.add_argument("--search-after", type=float, default=0.03)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clips = sorted(args.clips.glob("*_R01_0022500301_489_*_SOURCE.mp4"))
    by_label = {label_from_name(p): p for p in clips}
    if args.reference not in by_label:
        raise RuntimeError(f"Reference {args.reference!r} unavailable; labels={sorted(by_label)}")
    reference_path = by_label[args.reference]
    impact = json.loads(args.impact_json.read_text())
    impact_ref = float(impact["estimated_dunk_impact_reference_time"])
    if impact.get("reference_angle") != args.reference:
        raise RuntimeError(f"Impact reference {impact.get('reference_angle')} does not match {args.reference}")

    cap = cv2.VideoCapture(str(reference_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = max(0.0, impact_ref - args.search_before)
    hi = min((nframes - 1) / fps, impact_ref + args.search_after)
    f0, f1 = int(round(lo * fps)), int(round(hi * fps))

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True).eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    raw = []
    for fi in range(f0, f1 + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        compact, rims = orange_components(frame)
        rim = nearest_rim(rims, frame.shape[1], frame.shape[0])
        if rim is None:
            continue
        net = maskrcnn_balls(model, frame)
        candidates = dedupe_candidates(compact, net)
        raw.append({"frame_index": fi, "time": fi / fps, "rim": rim, "candidates": candidates, "image": frame})
    cap.release()
    if len(raw) < 10:
        raise RuntimeError(f"Only {len(raw)} frames with a usable rim in apex window")

    # Stabilize rim center against orange noise; relative ball-to-rim height is the key apex proxy.
    rim_cx = float(np.median([r["rim"]["cx"] for r in raw]))
    rim_cy = float(np.median([r["rim"]["cy"] for r in raw]))
    for r in raw:
        r["rim"] = dict(r["rim"], cx=rim_cx, cy=rim_cy)

    track = build_track(raw)
    if len(track) < 8:
        raise RuntimeError(f"Ball track too short: {len(track)} frames")

    times = np.array([r["time"] for r in track], np.float64)
    heights = np.array([r["height_proxy_px"] for r in track], np.float64)
    # Median then short moving-average smoothing. Apex is a visual height maximum, not an audio offset.
    if len(heights) >= 5:
        hmed = cv2.medianBlur(heights.astype(np.float32).reshape(-1, 1), 5).reshape(-1).astype(np.float64)
    else:
        hmed = heights.copy()
    hs = np.convolve(hmed, np.ones(3) / 3.0, mode="same")
    # Avoid edge maxima; require evidence on both sides of the apex.
    valid = np.arange(len(track))
    valid = valid[(valid >= 2) & (valid <= len(track) - 3)]
    if not len(valid):
        raise RuntimeError("No interior apex candidates")
    k = int(valid[np.argmax(hs[valid])])

    # Rising before and falling after in the image-space rim-relative height proxy.
    pre = hs[max(0, k - 3):k + 1]
    post = hs[k:min(len(hs), k + 4)]
    rise = float(hs[k] - np.min(pre)) if len(pre) else 0.0
    fall = float(hs[k] - np.min(post)) if len(post) else 0.0
    span = float(np.max(hs) - np.min(hs))
    apex = track[k]
    apex_time = float(apex["time"])
    confidence = "high" if len(track) >= 14 and rise >= 4.0 and fall >= 4.0 and span >= 15.0 else ("moderate" if span >= 10.0 else "low")
    if confidence == "low":
        raise RuntimeError(f"Apex visual trajectory gate failed: span={span:.2f}px rise={rise:.2f}px fall={fall:.2f}px")

    # Export apex and a compact +/-3-frame visual QA strip.
    apex_frame = next(r["image"] for r in raw if r["frame_index"] == apex["frame_index"])
    annotated = apex_frame.copy()
    bx, by = int(round(apex["ball"]["cx"])), int(round(apex["ball"]["cy"]))
    rr = int(max(7, round(max(float(apex["ball"].get("w", 14)), float(apex["ball"].get("h", 14))) / 2.0)))
    cv2.circle(annotated, (bx, by), rr, (0, 255, 0), 2)
    cv2.circle(annotated, (int(round(rim_cx)), int(round(rim_cy))), 7, (255, 0, 255), 2)
    cv2.putText(annotated, f"BALL APEX {apex_time:.3f}s", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "ball_apex_reference_native.png"), apex_frame)
    cv2.imwrite(str(args.out / "ball_apex_reference_annotated.png"), annotated)

    strip = []
    for off in range(-3, 4):
        fi = apex["frame_index"] + off
        rrw = next((r for r in raw if r["frame_index"] == fi), None)
        if rrw is None:
            continue
        im = rrw["image"].copy()
        tr = next((q for q in track if q["frame_index"] == fi), None)
        if tr:
            cv2.circle(im, (int(round(tr["ball"]["cx"])), int(round(tr["ball"]["cy"]))), 10, (0, 255, 0), 2)
            cv2.putText(im, f"{off:+d}  h={tr['height_proxy_px']:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        strip.append(im)
    if strip:
        cv2.imwrite(str(args.out / "ball_apex_qa_strip.png"), np.hstack(strip))

    sync_map = {
        "event": {"game_id": "0022500301", "event_id": 489, "description": "S. Adams DUNK vs UTA immediately after block"},
        "source_fps": fps,
        "synchronization": "visual ball-apex reference state + transient-audio offset graph",
        "reference_angle": args.reference,
        "angles": [{"label": label, "file": p.name, "freeze_time": round(apex_time, 5)} for label, p in sorted(by_label.items())],
    }
    (args.out / "jazz_ball_apex_sync_map.json").write_text(json.dumps(sync_map, indent=2))

    report = {
        "method": "frame-by-frame basketball tracking against stabilized rim + trajectory-reversal apex gate",
        "reference_label": args.reference,
        "reference_file": reference_path.name,
        "fps": fps,
        "audio_impact_reference_time": impact_ref,
        "search_window_reference_seconds": [lo, hi],
        "selected_apex_time_seconds": apex_time,
        "selected_apex_frame_index": int(apex["frame_index"]),
        "apex_lead_before_audio_impact_seconds": float(impact_ref - apex_time),
        "track_frames": len(track),
        "height_proxy_span_px": span,
        "rise_into_apex_px": rise,
        "fall_after_apex_px": fall,
        "apex_height_above_rim_proxy_px": float(apex["height_proxy_px"]),
        "ball_detector_source_at_apex": apex["ball"].get("source"),
        "ball_detector_score_at_apex": float(apex["ball"].get("detector_score", 0.0)),
        "confidence": confidence,
        "track": [{k: v for k, v in r.items()} for r in track],
        "policy": "Freeze at the visually measured highest point of the ball in the dunk motion. Audio is used only to bound/synchronize the event, not to define the apex.",
    }
    (args.out / "ball_apex_selection_v1.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "track"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
