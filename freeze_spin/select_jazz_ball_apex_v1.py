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
    mask = cv2.inRange(hsv, np.array([1, 58, 42], np.uint8), np.array([34, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compact, elongated = [], []
    H, W = frame.shape[:2]
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 9:
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
        if area >= 25 and w >= 18 and aspect >= 1.40 and w <= W * 0.42 and h <= H * 0.18:
            row["rim_score"] = float(area * min(aspect, 8.0) * (0.45 + sat / 255.0))
            elongated.append(row)
        if 4 <= w <= 68 and 4 <= h <= 68 and 0.45 <= aspect <= 2.05 and circ >= 0.09 and fill >= 0.12:
            size_pref = math.exp(-abs(math.log(max(math.sqrt(area), 1.0) / 17.0)) / 1.45)
            row["ball_color_score"] = float(1.35 * circ + 0.72 * fill + 0.92 * (sat / 255.0) + 0.58 * size_pref)
            compact.append(row)
    elongated.sort(key=lambda r: r["rim_score"], reverse=True)
    compact.sort(key=lambda r: r["ball_color_score"], reverse=True)
    return compact[:36], elongated[:14]


def maskrcnn_balls(model, frame: np.ndarray) -> list[dict]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        p = model([to_tensor(rgb)])[0]
    rows = []
    for score, label, box in zip(p["scores"].cpu().numpy(), p["labels"].cpu().numpy(), p["boxes"].cpu().numpy()):
        if int(label) != SPORTS_BALL_CLASS or float(score) < 0.018:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        rows.append({
            "score": float(score), "box": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "w": x2 - x1, "h": y2 - y1,
        })
    return rows[:14]


def nearest_rim(rims: list[dict], W: int, H: int) -> dict | None:
    if not rims:
        return None
    def score(r: dict) -> float:
        center_pen = abs(r["cx"] - W / 2.0) / W + 0.35 * abs(r["cy"] - H * 0.43) / H
        return r["rim_score"] / (1.0 + 3.0 * center_pen)
    return max(rims, key=score)


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
            if d <= max(11.0, 0.62 * max(n["w"], n["h"], r["w"], r["h"])):
                r["source"] = "orange+maskrcnn"
                r["maskrcnn_score"] = float(n["score"])
                r["detector_score"] += 3.0 * float(n["score"])
                merged = True
                break
        if not merged:
            out.append({
                "x": int(n["box"][0]), "y": int(n["box"][1]),
                "w": float(n["w"]), "h": float(n["h"]),
                "cx": float(n["cx"]), "cy": float(n["cy"]),
                "source": "maskrcnn", "maskrcnn_score": float(n["score"]),
                "detector_score": float(3.0 * n["score"]),
            })
    return out


def prepare_nodes(frames: list[dict]) -> list[dict]:
    nodes = []
    for fi, fr in enumerate(frames):
        rim = fr["rim"]
        for cj, c in enumerate(fr["candidates"]):
            dx = abs(float(c["cx"]) - float(rim["cx"]))
            dy = float(c["cy"]) - float(rim["cy"])
            if dx > 255 or dy < -240 or dy > 135:
                continue
            nodes.append({
                "raw_i": fi, "cand_i": cj,
                "frame_index": int(fr["frame_index"]), "time": float(fr["time"]),
                "cx": float(c["cx"]), "cy": float(c["cy"]),
                "height": float(rim["cy"] - c["cy"]),
                "detector_score": float(c["detector_score"]),
                "source": c.get("source", "unknown"),
                "dx_rim": dx, "dy_rim": dy,
            })
    # Static orange arena/rim lettering tends to recur at the same coordinates in most frames.
    for n in nodes:
        same_frames = set()
        for q in nodes:
            if abs(q["raw_i"] - n["raw_i"]) < 2:
                continue
            if math.hypot(q["cx"] - n["cx"], q["cy"] - n["cy"]) <= 5.5:
                same_frames.add(q["raw_i"])
        n["static_persistence"] = len(same_frames) / max(len(frames) - 1, 1)
        n["local_score"] = (
            n["detector_score"]
            - 0.0025 * n["dx_rim"]
            - 0.0010 * abs(n["dy_rim"] + 45.0)
            - 2.8 * max(0.0, n["static_persistence"] - 0.20)
        )
    nodes.sort(key=lambda n: (n["raw_i"], n["cand_i"]))
    return nodes


def build_gap_tolerant_track(frames: list[dict], max_gap: int = 4) -> tuple[list[dict], dict]:
    nodes = prepare_nodes(frames)
    if not nodes:
        return [], {"candidate_nodes": 0}
    best_score = np.full(len(nodes), -1e9, np.float64)
    best_len = np.ones(len(nodes), np.int32)
    prev = np.full(len(nodes), -1, np.int32)

    for i, n in enumerate(nodes):
        # Starting a new path is allowed, but longer coherent paths win through a coverage bonus.
        best_score[i] = float(n["local_score"] + 0.25)
        for j in range(i - 1, -1, -1):
            p = nodes[j]
            gap = n["raw_i"] - p["raw_i"]
            if gap <= 0:
                continue
            if gap > max_gap:
                if p["raw_i"] < n["raw_i"] - max_gap:
                    break
                continue
            dist = math.hypot(n["cx"] - p["cx"], n["cy"] - p["cy"])
            max_dist = 34.0 + 30.0 * gap
            if dist > max_dist:
                continue
            speed = dist / gap
            gap_pen = 0.24 * (gap - 1)
            speed_pen = 0.0065 * max(0.0, speed - 28.0) ** 1.35
            # Reward observation coverage. Do not require every frame to contain a detection.
            score = best_score[j] + n["local_score"] + 0.48 - gap_pen - speed_pen
            if score > best_score[i]:
                best_score[i] = score
                best_len[i] = best_len[j] + 1
                prev[i] = j

    # Prefer long paths with real spatial motion over isolated high-score blobs.
    end_scores = best_score + 0.48 * best_len
    order = np.argsort(end_scores)[::-1]
    chosen = None
    chosen_diag = None
    for end in order[:80]:
        idx = []
        cur = int(end)
        while cur >= 0:
            idx.append(cur)
            cur = int(prev[cur])
        idx.reverse()
        path = [nodes[k] for k in idx]
        if len(path) < 5:
            continue
        span_frames = path[-1]["raw_i"] - path[0]["raw_i"]
        displacement = math.hypot(path[-1]["cx"] - path[0]["cx"], path[-1]["cy"] - path[0]["cy"])
        bbox_span = math.hypot(
            max(x["cx"] for x in path) - min(x["cx"] for x in path),
            max(x["cy"] for x in path) - min(x["cy"] for x in path),
        )
        if span_frames < 8 or bbox_span < 18.0:
            continue
        chosen = path
        chosen_diag = {
            "candidate_nodes": len(nodes),
            "observations": len(path),
            "span_frames": int(span_frames),
            "endpoint_displacement_px": float(displacement),
            "track_bbox_span_px": float(bbox_span),
            "path_score": float(end_scores[end]),
            "max_gap_frames": max_gap,
        }
        break
    if chosen is None:
        return [], {"candidate_nodes": len(nodes), "reason": "no long moving path"}

    track = []
    for n in chosen:
        fr = frames[n["raw_i"]]
        c = fr["candidates"][n["cand_i"]]
        track.append({
            "frame_index": int(fr["frame_index"]), "time": float(fr["time"]),
            "ball": c, "rim": fr["rim"],
            "height_proxy_px": float(n["height"]),
            "static_persistence": float(n["static_persistence"]),
        })
    return track, chosen_diag or {}


def robust_quadratic_apex(track: list[dict], fps: float) -> tuple[dict, list[bool]]:
    times = np.array([r["time"] for r in track], np.float64)
    heights = np.array([r["height_proxy_px"] for r in track], np.float64)
    t0 = float(np.median(times))
    x = times - t0
    inliers = np.ones(len(track), dtype=bool)
    coeff = None
    for _ in range(5):
        if int(inliers.sum()) < 5:
            break
        coeff = np.polyfit(x[inliers], heights[inliers], 2)
        pred = np.polyval(coeff, x)
        resid = np.abs(heights - pred)
        med = float(np.median(resid[inliers]))
        mad = float(np.median(np.abs(resid[inliers] - med)))
        threshold = max(5.0, med + 3.2 * max(mad, 1.0))
        new_inliers = resid <= threshold
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    if coeff is None or int(inliers.sum()) < 5:
        raise RuntimeError("Quadratic apex fit has insufficient inliers")

    a, b, c = [float(v) for v in coeff]
    if a >= -0.5:
        raise RuntimeError(f"Apex trajectory is not concave-down enough: quadratic_a={a:.4f}")
    apex_x = -b / (2.0 * a)
    apex_time_fit = float(t0 + apex_x)
    inlier_times = times[inliers]
    margin = 1.25 / fps
    if apex_time_fit < float(inlier_times.min() + margin) or apex_time_fit > float(inlier_times.max() - margin):
        raise RuntimeError(
            f"Fitted apex {apex_time_fit:.5f}s is not interior to observed ball trajectory "
            f"[{inlier_times.min():.5f}, {inlier_times.max():.5f}]"
        )
    pred = np.polyval(coeff, x)
    rmse = float(np.sqrt(np.mean((heights[inliers] - pred[inliers]) ** 2)))
    if rmse > 11.0:
        raise RuntimeError(f"Apex trajectory quadratic fit too noisy: rmse={rmse:.2f}px")

    snap = int(np.argmin(np.abs(times - apex_time_fit)))
    apex = track[snap]
    before = heights[(times < apex_time_fit) & inliers]
    after = heights[(times > apex_time_fit) & inliers]
    if len(before) < 2 or len(after) < 2:
        raise RuntimeError("Apex trajectory lacks observations on both sides")
    fit_apex_height = float(np.polyval(coeff, apex_x))
    rise = float(fit_apex_height - np.percentile(before, 30))
    fall = float(fit_apex_height - np.percentile(after, 30))
    span = float(np.max(heights[inliers]) - np.min(heights[inliers]))
    if rise < 3.0 or fall < 3.0 or span < 9.0:
        raise RuntimeError(f"Rise/fall apex gate failed: rise={rise:.2f}px fall={fall:.2f}px span={span:.2f}px")
    return {
        "apex": apex,
        "apex_time_fit": apex_time_fit,
        "fit_coefficients_centered": [a, b, c],
        "fit_time_center": t0,
        "fit_rmse_px": rmse,
        "fit_apex_height_px": fit_apex_height,
        "rise_into_apex_px": rise,
        "fall_after_apex_px": fall,
        "height_proxy_span_px": span,
        "fit_inliers": int(inliers.sum()),
    }, inliers.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--impact-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="Left Above Rim")
    ap.add_argument("--search-before", type=float, default=1.05)
    ap.add_argument("--search-after", type=float, default=0.08)
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
    if len(raw) < 12:
        raise RuntimeError(f"Only {len(raw)} frames with a usable rim in apex window")

    rim_cx = float(np.median([r["rim"]["cx"] for r in raw]))
    rim_cy = float(np.median([r["rim"]["cy"] for r in raw]))
    for r in raw:
        r["rim"] = dict(r["rim"], cx=rim_cx, cy=rim_cy)

    track, track_diag = build_gap_tolerant_track(raw, max_gap=4)
    if len(track) < 6:
        raise RuntimeError(f"Ball track still too short after gap tolerance: {len(track)} frames; {track_diag}")
    fit, fit_inliers = robust_quadratic_apex(track, fps)
    apex = fit["apex"]
    apex_time = float(apex["time"])
    fit_apex_time = float(fit["apex_time_fit"])
    lead = float(impact_ref - apex_time)
    if lead < -0.08 or lead > 1.10:
        raise RuntimeError(f"Selected apex is implausibly placed relative to audio impact: lead={lead:.3f}s")

    confidence = "high" if (
        len(track) >= 10 and fit["fit_inliers"] >= 8 and fit["fit_rmse_px"] <= 7.5
        and fit["rise_into_apex_px"] >= 4.0 and fit["fall_after_apex_px"] >= 4.0
    ) else "moderate"

    apex_frame = next(r["image"] for r in raw if r["frame_index"] == apex["frame_index"])
    annotated = apex_frame.copy()
    bx, by = int(round(apex["ball"]["cx"])), int(round(apex["ball"]["cy"]))
    rr = int(max(7, round(max(float(apex["ball"].get("w", 14)), float(apex["ball"].get("h", 14))) / 2.0)))
    cv2.circle(annotated, (bx, by), rr, (0, 255, 0), 2)
    cv2.circle(annotated, (int(round(rim_cx)), int(round(rim_cy))), 7, (255, 0, 255), 2)
    cv2.putText(annotated, f"BALL APEX {apex_time:.3f}s", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"fit {fit_apex_time:.3f}s  rmse {fit['fit_rmse_px']:.1f}px", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "ball_apex_reference_native.png"), apex_frame)
    cv2.imwrite(str(args.out / "ball_apex_reference_annotated.png"), annotated)

    by_frame = {r["frame_index"]: r for r in raw}
    track_by_frame = {r["frame_index"]: r for r in track}
    strip = []
    for off in range(-4, 5):
        fi = apex["frame_index"] + off
        rrw = by_frame.get(fi)
        if rrw is None:
            continue
        im = rrw["image"].copy()
        tr = track_by_frame.get(fi)
        if tr:
            cv2.circle(im, (int(round(tr["ball"]["cx"])), int(round(tr["ball"]["cy"]))), 10, (0, 255, 0), 2)
            cv2.putText(im, f"{off:+d} h={tr['height_proxy_px']:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(im, f"{off:+d} gap", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
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
        "method": "gap-tolerant basketball candidate graph + robust concave-down rim-relative trajectory fit",
        "reference_label": args.reference,
        "reference_file": reference_path.name,
        "fps": fps,
        "audio_impact_reference_time": impact_ref,
        "search_window_reference_seconds": [lo, hi],
        "selected_apex_time_seconds": apex_time,
        "fitted_apex_time_seconds": fit_apex_time,
        "selected_apex_frame_index": int(apex["frame_index"]),
        "apex_lead_before_audio_impact_seconds": lead,
        "track_frames": len(track),
        "track_diagnostics": track_diag,
        "fit_inliers": fit["fit_inliers"],
        "fit_rmse_px": fit["fit_rmse_px"],
        "height_proxy_span_px": fit["height_proxy_span_px"],
        "rise_into_apex_px": fit["rise_into_apex_px"],
        "fall_after_apex_px": fit["fall_after_apex_px"],
        "apex_height_above_rim_proxy_px": float(apex["height_proxy_px"]),
        "ball_detector_source_at_apex": apex["ball"].get("source"),
        "ball_detector_score_at_apex": float(apex["ball"].get("detector_score", 0.0)),
        "confidence": confidence,
        "fit_inlier_mask": fit_inliers,
        "track": track,
        "policy": "Freeze at the visually measured highest point of the ball in the dunk motion. Audio bounds and synchronizes the event but does not define the apex. Missing detections may be bridged for tracking; the apex still requires a coherent concave-down rise/top/fall trajectory.",
    }
    (args.out / "ball_apex_selection_v1.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("track", "fit_inlier_mask")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
