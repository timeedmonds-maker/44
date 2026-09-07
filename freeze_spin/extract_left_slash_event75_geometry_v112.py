from __future__ import annotations

"""v112: source-pixel Left Slash event-75 basket-geometry extraction.

This is deliberately a diagnostic gate, not a metric-camera promotion.
Frame C supplies only a local search prior. The event-75 target opening and rim
are re-observed from the immutable event source pixels, then checked across
multiple adjacent source frames. A later shared-centre metric solve may consume
only a v112 PASS artifact.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

TARGET_ORDER = (
    "target_inner_top_left",
    "target_inner_top_right",
    "target_inner_bottom_right",
    "target_inner_bottom_left",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perspective(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(a, np.asarray(H, dtype=np.float64))[:, 0, :].astype(np.float64)


def canonical_ellipse(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(p) < 5:
        raise ValueError("need >=5 ellipse points")
    (cx, cy), (a, b), ang = cv2.fitEllipse(p)
    if a >= b:
        major, minor, angle = float(a), float(b), float(ang)
    else:
        major, minor, angle = float(b), float(a), float((ang + 90.0) % 180.0)
    angle %= 180.0
    return np.array([float(cx), float(cy), major, minor, angle], dtype=np.float64)


def ellipse_points(desc: np.ndarray, n: int = 256) -> np.ndarray:
    cx, cy, major, minor, angle = map(float, desc)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 0.5 * major * np.cos(t)
    y = 0.5 * minor * np.sin(t)
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    return np.column_stack([cx + ca * x - sa * y, cy + sa * x + ca * y])


def angle_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 1e9
    return float(abs(np.cross(v, p - a)) / n)


def line_angle(a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    return float(math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0)


def line_intersection(seg1: tuple[np.ndarray, np.ndarray], seg2: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    p, p2 = seg1
    q, q2 = seg2
    r = p2 - p
    s = q2 - q
    den = float(np.cross(r, s))
    if abs(den) < 1e-8:
        return None
    t = float(np.cross(q - p, s) / den)
    return p + t * r


def choose_target_lines(gray: np.ndarray, predicted: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]] | None:
    # LSD is deterministic and local-search constrained. Predicted geometry is a
    # prior only; accepted lines must be supported by source pixels.
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return None
    all_segs = []
    for item in lines.reshape(-1, 4):
        a = np.array(item[:2], dtype=np.float64)
        b = np.array(item[2:], dtype=np.float64)
        L = float(np.linalg.norm(b - a))
        if L >= 8.0:
            all_segs.append((a, b, L, line_angle(a, b)))

    chosen: list[tuple[np.ndarray, np.ndarray]] = []
    diagnostics: list[dict] = []
    for i in range(4):
        pa = predicted[i]
        pb = predicted[(i + 1) % 4]
        pm = 0.5 * (pa + pb)
        plen = float(np.linalg.norm(pb - pa))
        pang = line_angle(pa, pb)
        candidates = []
        for a, b, L, ang in all_segs:
            mid = 0.5 * (a + b)
            ad = angle_diff_deg(ang, pang)
            md = point_line_distance(mid, pa, pb)
            # midpoint must also be near the finite predicted edge span
            along = float(np.dot(mid - pa, pb - pa) / max(plen * plen, 1e-9))
            if ad > 12.0 or md > 10.0 or along < -0.30 or along > 1.30:
                continue
            if L < max(8.0, 0.22 * plen):
                continue
            # Prefer the inner stripe edge nearest the predicted inner opening.
            score = md + 0.28 * ad + 0.020 * abs(L - plen) + 1.5 * max(0.0, -along, along - 1.0)
            candidates.append((score, a, b, L, ad, md, along))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        score, a, b, L, ad, md, along = candidates[0]
        chosen.append((a, b))
        diagnostics.append({
            "edge_index": i,
            "score": float(score),
            "segment_length_px": float(L),
            "predicted_length_px": float(plen),
            "angle_error_deg": float(ad),
            "midline_distance_px": float(md),
            "along_fraction": float(along),
            "source_segment": [a.tolist(), b.tolist()],
        })

    # Corners are intersections of adjacent source-supported lines.
    corners = []
    for i in range(4):
        prev = chosen[(i - 1) % 4]
        cur = chosen[i]
        x = line_intersection(prev, cur)
        if x is None or not np.all(np.isfinite(x)):
            return None
        corners.append(x)
    refined = np.asarray(corners, dtype=np.float64)
    if not cv2.isContourConvex(np.round(refined).astype(np.int32).reshape(-1, 1, 2)):
        return None
    return chosen, diagnostics, refined


def nearest_curve_distance(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    a = np.asarray(points, dtype=np.float64)
    b = np.asarray(curve, dtype=np.float64)
    if not len(a) or not len(b):
        return np.empty((0,), dtype=np.float64)
    out = []
    for i in range(0, len(a), 2000):
        q = a[i:i + 2000]
        d2 = ((q[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        out.append(np.sqrt(d2.min(axis=1)))
    return np.concatenate(out)


def extract_rim(image: np.ndarray, predicted_samples: np.ndarray) -> dict | None:
    pred_desc = canonical_ellipse(predicted_samples)
    pred_curve = ellipse_points(pred_desc, 360)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 55, 145, L2gradient=True)
    ys, xs = np.nonzero(edges)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if not len(pts):
        return None

    xmin, ymin = pred_curve.min(axis=0) - 10.0
    xmax, ymax = pred_curve.max(axis=0) + 10.0
    roi = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    pts = pts[roi]
    if len(pts) < 12:
        return None
    d_pred = nearest_curve_distance(pts, pred_curve)
    support = pts[d_pred <= 5.0]
    if len(support) < 14:
        return None

    # Robustly fit/re-fit using the source edge support, then score held-out
    # alternating pixels. This rejects a fit supported only by one accidental edge.
    try:
        desc = canonical_ellipse(support)
    except cv2.error:
        return None
    curve = ellipse_points(desc, 480)
    d_fit = nearest_curve_distance(support, curve)
    keep = d_fit <= 2.75
    support2 = support[keep]
    if len(support2) >= 12:
        try:
            desc = canonical_ellipse(support2)
        except cv2.error:
            return None
        support = support2
        curve = ellipse_points(desc, 480)
        d_fit = nearest_curve_distance(support, curve)

    train = support[::2]
    held = support[1::2]
    if len(train) < 5 or len(held) < 5:
        return None
    try:
        train_desc = canonical_ellipse(train)
    except cv2.error:
        return None
    train_curve = ellipse_points(train_desc, 480)
    held_d = nearest_curve_distance(held, train_curve)

    center_shift = float(np.linalg.norm(desc[:2] - pred_desc[:2]))
    major_shift = abs(float(desc[2] - pred_desc[2]))
    minor_shift = abs(float(desc[3] - pred_desc[3]))
    angle_shift = angle_diff_deg(float(desc[4]), float(pred_desc[4]))
    return {
        "predicted_ellipse": pred_desc.tolist(),
        "source_ellipse": desc.tolist(),
        "source_support_count": int(len(support)),
        "source_fit_median_px": float(np.median(d_fit)),
        "source_fit_p95_px": float(np.percentile(d_fit, 95)),
        "heldout_count": int(len(held_d)),
        "heldout_median_px": float(np.median(held_d)),
        "heldout_p95_px": float(np.percentile(held_d, 95)),
        "center_shift_from_transfer_prior_px": center_shift,
        "major_axis_shift_from_transfer_prior_px": major_shift,
        "minor_axis_shift_from_transfer_prior_px": minor_shift,
        "angle_shift_from_transfer_prior_deg": angle_shift,
        "source_curve_samples_px": ellipse_points(desc, 24).tolist(),
        "source_edge_support_px": support.tolist(),
    }


def draw_overlay(image: np.ndarray, pred_target: np.ndarray, refined_target: np.ndarray, rim: dict, out: Path) -> None:
    im = image.copy()
    cv2.polylines(im, [np.round(pred_target).astype(np.int32)], True, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.polylines(im, [np.round(refined_target).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    pred_curve = ellipse_points(np.asarray(rim["predicted_ellipse"], dtype=np.float64), 180)
    src_curve = np.asarray(rim["source_curve_samples_px"], dtype=np.float64)
    cv2.polylines(im, [np.round(pred_curve).astype(np.int32)], True, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.polylines(im, [np.round(src_curve).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    for p in np.asarray(rim["source_edge_support_px"], dtype=np.float64):
        cv2.circle(im, tuple(np.round(p).astype(int)), 1, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.imwrite(str(out), im)


def frame_result(frame_c_target: np.ndarray, frame_c_rim: np.ndarray, burst_root: Path, row: dict, out: Path) -> dict:
    frame = burst_root / row["file"]
    image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (540, 960):
        return {"status": "bad_frame", "index": row.get("index"), "file": str(frame)}
    H_s2c = np.asarray(row["H_source_to_frame_c"], dtype=np.float64)
    if abs(float(np.linalg.det(H_s2c))) < 1e-10:
        return {"status": "singular_h", "index": row.get("index"), "file": str(frame)}
    H_c2s = np.linalg.inv(H_s2c)
    H_c2s /= H_c2s[2, 2]
    pred_target = perspective(H_c2s, frame_c_target)
    pred_rim = perspective(H_c2s, frame_c_rim)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target_fit = choose_target_lines(gray, pred_target)
    if target_fit is None:
        return {"status": "target_source_fit_failed", "index": row.get("index"), "time_s": row.get("time_s"), "file": row["file"]}
    _lines, line_diag, refined_target = target_fit
    target_shifts = np.linalg.norm(refined_target - pred_target, axis=1)
    rim = extract_rim(image, pred_rim)
    if rim is None:
        return {"status": "rim_source_fit_failed", "index": row.get("index"), "time_s": row.get("time_s"), "file": row["file"]}

    line_angle_max = max(float(d["angle_error_deg"]) for d in line_diag)
    line_mid_max = max(float(d["midline_distance_px"]) for d in line_diag)
    gates = {
        "target_corner_shift_max_at_most_8px": float(target_shifts.max()) <= 8.0,
        "target_line_angle_error_max_at_most_12deg": line_angle_max <= 12.0,
        "target_line_mid_distance_max_at_most_10px": line_mid_max <= 10.0,
        "rim_source_support_at_least_14": int(rim["source_support_count"]) >= 14,
        "rim_source_fit_p95_at_most_2_5px": float(rim["source_fit_p95_px"]) <= 2.5,
        "rim_heldout_p95_at_most_3px": float(rim["heldout_p95_px"]) <= 3.0,
        "rim_center_shift_from_transfer_at_most_7px": float(rim["center_shift_from_transfer_prior_px"]) <= 7.0,
        "rim_major_shift_from_transfer_at_most_10px": float(rim["major_axis_shift_from_transfer_prior_px"]) <= 10.0,
        "rim_minor_shift_from_transfer_at_most_6px": float(rim["minor_axis_shift_from_transfer_prior_px"]) <= 6.0,
        "rim_angle_shift_from_transfer_at_most_12deg": float(rim["angle_shift_from_transfer_prior_deg"]) <= 12.0,
    }
    passed = all(gates.values())
    overlay = out / f"event75_b{int(row['index']):02d}_source_geometry.png"
    draw_overlay(image, pred_target, refined_target, rim, overlay)
    score = (
        float(target_shifts.max())
        + 0.45 * float(rim["heldout_p95_px"])
        + 0.15 * float(rim["center_shift_from_transfer_prior_px"])
        + 0.05 * line_mid_max
        - 0.01 * int(rim["source_support_count"])
    )
    return {
        "status": "pass" if passed else "gate_fail",
        "index": int(row["index"]),
        "time_s": float(row["time_s"]),
        "file": row["file"],
        "image_sha256": sha256(frame),
        "v91_discovery_score": float(row.get("score", 1e9)),
        "v91_training_inliers": int(row.get("training_inliers", 0)),
        "v91_basket_good_within_3px": int(row.get("basket_good_within_3px", 0)),
        "H_source_to_frame_c": H_s2c.tolist(),
        "target_transfer_prior_px": pred_target.tolist(),
        "source_observed_target_inner_corners_px": refined_target.tolist(),
        "target_corner_shift_from_transfer_px": target_shifts.tolist(),
        "target_corner_shift_max_px": float(target_shifts.max()),
        "target_line_diagnostics": line_diag,
        "rim": rim,
        "gates": gates,
        "score": float(score),
        "overlay": overlay.name,
    }


def cross_frame_consistency(a: dict, b: dict) -> dict:
    Ha = np.asarray(a["H_source_to_frame_c"], dtype=np.float64)
    Hb = np.asarray(b["H_source_to_frame_c"], dtype=np.float64)
    ta = perspective(Ha, np.asarray(a["source_observed_target_inner_corners_px"], dtype=np.float64))
    tb = perspective(Hb, np.asarray(b["source_observed_target_inner_corners_px"], dtype=np.float64))
    target_delta = np.linalg.norm(ta - tb, axis=1)
    ra = perspective(Ha, np.asarray(a["rim"]["source_curve_samples_px"], dtype=np.float64))
    rb = perspective(Hb, np.asarray(b["rim"]["source_curve_samples_px"], dtype=np.float64))
    da = canonical_ellipse(ra)
    db = canonical_ellipse(rb)
    rim_center = float(np.linalg.norm(da[:2] - db[:2]))
    return {
        "frame_indices": [a["index"], b["index"]],
        "time_delta_s": abs(float(a["time_s"]) - float(b["time_s"])),
        "target_corner_delta_frame_c_px": target_delta.tolist(),
        "target_corner_max_delta_frame_c_px": float(target_delta.max()),
        "rim_center_delta_frame_c_px": rim_center,
        "rim_major_axis_delta_frame_c_px": abs(float(da[2] - db[2])),
        "rim_minor_axis_delta_frame_c_px": abs(float(da[3] - db[3])),
        "rim_angle_delta_frame_c_deg": angle_diff_deg(float(da[4]), float(db[4])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-c", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--burst-root", type=Path, required=True)
    ap.add_argument("--burst-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidate-count", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    obs = json.loads(args.observations.read_text())
    view = obs["views"][0]
    target = np.asarray([view["landmarks"][k] for k in TARGET_ORDER], dtype=np.float64)
    rim = np.asarray(view["rim_curve_samples_px"], dtype=np.float64)
    burst = json.loads(args.burst_json.read_text())
    rows = [r for r in burst["rows"] if r.get("status") == "ok" and "H_source_to_frame_c" in r]
    rows.sort(key=lambda r: float(r.get("score", 1e9)))

    results = [frame_result(target, rim, args.burst_root, r, args.out) for r in rows[: args.candidate_count]]
    passed = [r for r in results if r.get("status") == "pass"]
    passed.sort(key=lambda r: r["score"])

    consistency = []
    for i in range(len(passed)):
        for j in range(i + 1, len(passed)):
            if abs(float(passed[i]["time_s"]) - float(passed[j]["time_s"])) <= 0.16:
                consistency.append(cross_frame_consistency(passed[i], passed[j]))
    stable_pairs = [c for c in consistency if (
        c["target_corner_max_delta_frame_c_px"] <= 3.0
        and c["rim_center_delta_frame_c_px"] <= 3.0
        and c["rim_major_axis_delta_frame_c_px"] <= 6.0
        and c["rim_minor_axis_delta_frame_c_px"] <= 4.0
        and c["rim_angle_delta_frame_c_deg"] <= 8.0
    )]

    selected = None
    if stable_pairs:
        pair = min(stable_pairs, key=lambda c: c["target_corner_max_delta_frame_c_px"] + c["rim_center_delta_frame_c_px"])
        ids = set(pair["frame_indices"])
        pair_rows = [r for r in passed if r["index"] in ids]
        pair_rows.sort(key=lambda r: r["score"])
        selected = pair_rows[0]

    gates = {
        "at_least_two_source_geometry_frames_pass": len(passed) >= 2,
        "at_least_one_adjacent_cross_frame_stable_pair": len(stable_pairs) >= 1,
        "selected_frame_exists": selected is not None,
    }
    status = "PASS_LEFT_SLASH_EVENT75_SOURCE_GEOMETRY_V112" if all(gates.values()) else "FAIL_LEFT_SLASH_EVENT75_SOURCE_GEOMETRY_V112"
    payload = {
        "status": status,
        "purpose": "Source-pixel event75 target/rim re-observation for later two-state shared-centre Left Slash basin discrimination",
        "frame_c": args.frame_c.name,
        "frame_c_sha256": sha256(args.frame_c),
        "candidate_count": min(args.candidate_count, len(rows)),
        "passed_frame_count": len(passed),
        "stable_pair_count": len(stable_pairs),
        "gates": gates,
        "selected": selected,
        "stable_pairs": stable_pairs,
        "all_results": results,
        "permissions": {
            "left_slash_event75_source_geometry_allowed": all(gates.values()),
            "left_slash_metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "guardrail": "v112 is source-geometry evidence only. It cannot promote Left Slash or authorize a replay.",
    }
    (args.out / "left_slash_event75_source_geometry_v112.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "status": status,
        "passed_frame_count": len(passed),
        "stable_pair_count": len(stable_pairs),
        "selected": None if selected is None else {
            "index": selected["index"],
            "time_s": selected["time_s"],
            "score": selected["score"],
            "target_corner_shift_max_px": selected["target_corner_shift_max_px"],
            "rim_support": selected["rim"]["source_support_count"],
            "rim_heldout_p95_px": selected["rim"]["heldout_p95_px"],
        },
        "gates": gates,
    }, indent=2))
    raise SystemExit(0 if all(gates.values()) else 2)


if __name__ == "__main__":
    main()
