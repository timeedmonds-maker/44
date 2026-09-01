from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH = 960
HEIGHT = 540
COURT_LENGTH_CM = 2800.0
COURT_WIDTH_CM = 1500.0
BASKET_X_SHIFT_CM = 157.5
BASKET_Z_CM = -305.0
BASKETS = np.array([
    [BASKET_X_SHIFT_CM, COURT_WIDTH_CM / 2.0, BASKET_Z_CM],
    [COURT_LENGTH_CM - BASKET_X_SHIFT_CM, COURT_WIDTH_CM / 2.0, BASKET_Z_CM],
], dtype=np.float64)


def project(P: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    q = (P @ h.T).T
    valid = q[:, 2] > 1e-6
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    uv[valid] = q[valid, :2] / q[valid, 2:3]
    return uv, valid


def decompose(P: np.ndarray):
    K, R, C_h, *_ = cv2.decomposeProjectionMatrix(P.astype(np.float64))
    K /= K[2, 2]
    C = (C_h[:3] / C_h[3]).reshape(3)
    return K, R, C


def angle_wrap_deg(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def bilinear_remap(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def normalized_features(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # Per-camera normalization makes the cost less sensitive to broadcast color/exposure differences.
    channels = []
    for c in range(3):
        ch = lab[:, :, c]
        med = float(np.median(ch))
        mad = float(np.median(np.abs(ch - med))) + 1e-4
        channels.append(np.clip((ch - med) / (4.0 * mad), -2.0, 2.0))
    grad_scale = float(np.percentile(grad, 90)) + 1e-4
    channels.append(np.clip(grad / grad_scale, 0.0, 2.0))
    return np.dstack(channels).astype(np.float32)


def valid_camera_rows(predictions, mapping, images_dir: Path):
    rows = []
    for m in mapping:
        idx = int(m["index"])
        pred = predictions[idx] if idx < len(predictions) else {}
        pvals = pred.get("P")
        if not pvals or len(pvals) != 12:
            continue
        P = np.asarray(pvals, dtype=np.float64).reshape(3, 4)
        try:
            K, R, C = decompose(P)
        except cv2.error:
            continue
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(C)):
            continue
        if abs(K[0, 0]) < 100 or abs(K[1, 1]) < 100 or abs(K[0, 0]) > 20000 or abs(K[1, 1]) > 20000:
            continue
        img = cv2.imread(str(images_dir / m["image"]))
        if img is None:
            continue
        rows.append({
            "index": idx,
            "label": m["label"],
            "image_name": m["image"],
            "P": P,
            "K": K,
            "R": R,
            "C": C,
            "image": img,
            "features": normalized_features(img),
        })
    return rows


def choose_reference_and_sources(rows):
    if len(rows) < 2:
        raise RuntimeError("Need at least two calibrated camera views")

    by_label = {r["label"]: r for r in rows}
    ref = by_label.get("Broadcast") or by_label.get("Mobile Broadcast") or rows[0]

    hoop_uv, hoop_valid = project(ref["P"], BASKETS)
    scores = []
    for i in range(2):
        if hoop_valid[i]:
            u, v = hoop_uv[i]
            if -100 < u < WIDTH + 100 and -100 < v < HEIGHT + 100:
                scores.append((np.linalg.norm(np.array([u - WIDTH / 2, v - HEIGHT / 2])), i))
    if not scores:
        raise RuntimeError("Reference calibration does not place either basket near the image")
    basket_index = min(scores)[1]
    target = BASKETS[basket_index].copy()
    # Aim through the middle of the airborne action rather than the rim plane itself.
    action_target = target.copy()
    action_target[2] = -185.0

    ref_vec = ref["C"][:2] - target[:2]
    ref_az = math.degrees(math.atan2(ref_vec[1], ref_vec[0]))
    candidates = []
    for r in rows:
        if r is ref:
            continue
        uv, valid = project(r["P"], target[None, :])
        if not valid[0]:
            continue
        u, v = uv[0]
        if not (-160 <= u < WIDTH + 160 and -160 <= v < HEIGHT + 160):
            continue
        vec = r["C"][:2] - target[:2]
        az = math.degrees(math.atan2(vec[1], vec[0]))
        delta = angle_wrap_deg(az - ref_az)
        if abs(delta) < 1.0 or abs(delta) > 70.0:
            continue
        # Prefer 12-35 degree baselines: enough stereo parallax, not a radically different view.
        baseline_score = abs(abs(delta) - 24.0)
        label_penalty = 8.0 if "Above Rim" in r["label"] else 0.0
        candidates.append((baseline_score + label_penalty, abs(delta), delta, r))
    if not candidates:
        raise RuntimeError("No calibrated source camera with usable basket overlap")
    candidates.sort(key=lambda x: x[0])
    primary_delta = candidates[0][2]
    sign = 1.0 if primary_delta >= 0 else -1.0
    same_side = sorted(
        [x for x in candidates if x[2] * sign > 0],
        key=lambda x: (abs(abs(x[2]) - 20.0), x[0]),
    )
    selected = [x[3] for x in same_side[:3]]
    if len(selected) < 2:
        selected = [x[3] for x in candidates[: min(3, len(candidates))]]
    return ref, selected, basket_index, target, action_target, primary_delta


def plane_sweep(ref, sources, target, stride=2, depth_steps=88):
    target_uv, ok = project(ref["P"], target[None, :])
    if not ok[0]:
        raise RuntimeError("Target basket does not project in reference")
    hu, hv = target_uv[0]
    # Large enough to hold Adams/Cissoko and the ball while avoiding most irrelevant crowd pixels.
    x0 = max(0, int(round(hu - 330)))
    x1 = min(WIDTH - 1, int(round(hu + 330)))
    y0 = max(0, int(round(hv - 300)))
    y1 = min(HEIGHT - 1, int(round(hv + 245)))
    if x1 - x0 < 100 or y1 - y0 < 100:
        raise RuntimeError(f"Implausibly small action ROI {(x0, y0, x1, y1)}")

    xs = np.arange(x0, x1 + 1, stride, dtype=np.float64)
    ys = np.arange(y0, y1 + 1, stride, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)
    H, W = uu.shape

    pix = np.stack([uu.reshape(-1), vv.reshape(-1), np.ones(H * W)], axis=0)
    rays_cam = np.linalg.inv(ref["K"]) @ pix
    rays_world = ref["R"].T @ rays_cam
    rays_world /= np.linalg.norm(rays_world, axis=0, keepdims=True) + 1e-12
    rays_world = rays_world.T.reshape(H, W, 3)

    center_t = float(np.linalg.norm(ref["C"] - target))
    depth_min = max(100.0, center_t - 480.0)
    depth_max = center_t + 480.0
    depths = np.linspace(depth_min, depth_max, depth_steps, dtype=np.float64)

    ref_feat = ref["features"][vv.astype(np.int32), uu.astype(np.int32)]
    best = np.full((H, W), np.inf, dtype=np.float32)
    second = np.full((H, W), np.inf, dtype=np.float32)
    best_t = np.zeros((H, W), dtype=np.float32)
    support_at_best = np.zeros((H, W), dtype=np.uint8)

    for ti, depth in enumerate(depths):
        X = ref["C"].reshape(1, 1, 3) + rays_world * depth
        # Court coordinates use negative Z above the floor.
        physical = (
            (X[:, :, 2] <= 40.0) & (X[:, :, 2] >= -430.0) &
            (np.abs(X[:, :, 0] - target[0]) <= 650.0) &
            (np.abs(X[:, :, 1] - target[1]) <= 650.0)
        )
        cost_sum = np.zeros((H, W), dtype=np.float32)
        support = np.zeros((H, W), dtype=np.uint8)
        flatX = X.reshape(-1, 3)
        for src in sources:
            uv, valid = project(src["P"], flatX)
            su = uv[:, 0].reshape(H, W)
            sv = uv[:, 1].reshape(H, W)
            inside = valid.reshape(H, W) & (su >= 1) & (su < WIDTH - 2) & (sv >= 1) & (sv < HEIGHT - 2) & physical
            sample = bilinear_remap(src["features"], su, sv)
            # Robust truncated color + edge cost.
            d = np.abs(sample[:, :, :3] - ref_feat[:, :, :3]).mean(axis=2)
            dg = np.abs(sample[:, :, 3] - ref_feat[:, :, 3])
            c = np.minimum(d, 0.75) * 0.72 + np.minimum(dg, 0.75) * 0.28
            cost_sum += np.where(inside, c, 0.0).astype(np.float32)
            support += inside.astype(np.uint8)
        enough = support >= min(2, len(sources))
        cost = np.where(enough, cost_sum / np.maximum(support, 1), np.inf)
        improved = cost < best
        second = np.where(improved, best, np.minimum(second, cost))
        best = np.where(improved, cost, best)
        best_t = np.where(improved, depth, best_t)
        support_at_best = np.where(improved, support, support_at_best)

    margin = second - best
    finite = np.isfinite(best)
    good = finite & (best < 0.42) & (margin > 0.006) & (support_at_best >= min(2, len(sources)))
    # Favor textured pixels because constant-color court areas are depth ambiguous.
    good &= ref_feat[:, :, 3] > 0.035

    Xbest = ref["C"].reshape(1, 1, 3) + rays_world * best_t[:, :, None]
    points = Xbest[good]
    colors = ref["image"][vv.astype(np.int32), uu.astype(np.int32)][good]
    metrics = {
        "roi": [x0, y0, x1, y1],
        "grid_shape": [H, W],
        "depth_range_cm": [round(depth_min, 2), round(depth_max, 2)],
        "depth_steps": depth_steps,
        "accepted_points": int(len(points)),
        "median_best_cost": None if not np.any(good) else round(float(np.median(best[good])), 5),
        "median_margin": None if not np.any(good) else round(float(np.median(margin[good])), 5),
    }
    return points.astype(np.float64), colors.astype(np.uint8), metrics


def write_ply(path: Path, points: np.ndarray, colors_bgr: np.ndarray):
    colors_rgb = colors_bgr[:, ::-1]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors_rgb):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def rotate_z(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def render_points(points, colors, ref, target, yaw_deg, scale=2):
    Q = rotate_z(yaw_deg)
    C0 = ref["C"]
    C = target + Q @ (C0 - target)
    R = ref["R"] @ Q.T
    t = -R @ C.reshape(3, 1)
    K = ref["K"].copy()
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    K2 = S @ K
    P = K2 @ np.hstack([R, t])
    uv, valid = project(P, points)
    cam = (R @ (points - C).T).T
    depth = cam[:, 2]
    valid &= depth > 1.0
    out_w, out_h = WIDTH * scale, HEIGHT * scale
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < out_w) & (uv[:, 1] >= 0) & (uv[:, 1] < out_h)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if valid.sum() == 0:
        return canvas, P
    idx = np.where(valid)[0]
    # Far-to-near splatting provides a simple z-buffer approximation.
    idx = idx[np.argsort(depth[idx])[::-1]]
    radius = max(2, int(round(2.2 * scale)))
    for i in idx:
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        cv2.circle(canvas, (x, y), radius, tuple(int(v) for v in colors[i]), -1, cv2.LINE_AA)
    return canvas, P


def make_montage(frames, labels, out_path: Path):
    thumb_w, thumb_h = 960, 540
    canvas = np.zeros((2160, 3840, 3), dtype=np.uint8)
    slots = [(0,0),(960,0),(1920,0),(2880,0),(480,700),(1440,700),(2400,700)]
    for frame, label, (x, y) in zip(frames, labels, slots):
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y:y+thumb_h, x:x+thumb_w] = thumb
        cv2.putText(canvas, label, (x+22, y+48), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "CALIBRATED STATIC RECONSTRUCTION TEST | REAL NBA PIXELS | NO IMAGE MORPH", (90, 1510), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "0 to 25 degree virtual-camera orbit from plane-sweep 3D point reconstruction", (90, 1580), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (190,190,190), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Black regions are deliberately left unresolved rather than hallucinated.", (90, 1640), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (190,190,190), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--depth-steps", type=int, default=88)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["views"]
    rows = valid_camera_rows(predictions, mapping, args.images)
    ref, sources, basket_index, basket_target, action_target, primary_delta = choose_reference_and_sources(rows)

    points, colors, sweep_qa = plane_sweep(ref, sources, action_target, stride=args.stride, depth_steps=args.depth_steps)
    if len(points) < 500:
        raise RuntimeError(f"Reconstruction too sparse for a meaningful test: {len(points)} points")

    write_ply(args.out / "action_plane_sweep_point_cloud.ply", points, colors)
    direction = 1.0 if primary_delta >= 0 else -1.0
    degrees = [0, 5, 10, 15, 20, 25]
    synthetic = []
    projection_matrices = {}
    for deg in degrees:
        frame, P = render_points(points, colors, ref, action_target, direction * deg, scale=2)
        synthetic.append(frame)
        projection_matrices[str(deg)] = P.reshape(-1).tolist()
        cv2.imwrite(str(args.out / f"virtual_{deg:02d}deg_uhd.png"), frame)

    # First panel is the actual reference image, enlarged deterministically only for layout comparison.
    actual = cv2.resize(ref["image"], (WIDTH * 2, HEIGHT * 2), interpolation=cv2.INTER_NEAREST)
    frames = [actual] + synthetic[1:]
    labels = [f"REAL {ref['label']} 0 deg"] + [f"VIRTUAL {d} deg" for d in degrees[1:]]
    make_montage(frames, labels, args.out / "static_reconstruction_0_to_25deg_uhd.png")

    qa = {
        "mode": "calibrated_multiview_plane_sweep_point_reconstruction",
        "source_policy": "official synchronized NBA impact frames only; no diffusion, generative fill, optical-flow morph or mesh image warp",
        "reference": ref["label"],
        "sources": [s["label"] for s in sources],
        "basket_index": basket_index,
        "basket_target_cm": basket_target.tolist(),
        "action_target_cm": action_target.tolist(),
        "primary_source_azimuth_delta_deg": round(float(primary_delta), 3),
        "orbit_direction": "positive" if direction > 0 else "negative",
        "virtual_degrees": degrees,
        "valid_calibrated_views": [r["label"] for r in rows],
        "plane_sweep": sweep_qa,
        "projection_matrices": projection_matrices,
        "quality_gate": {
            "purpose": "Expose geometry errors without hiding them behind whip blur or cuts.",
            "pass_condition": "Player/ball/basket geometry remains coherent across 0-25 degrees with enough reconstructed coverage to identify the action.",
            "note": "Black/unresolved pixels are considered missing data, not filled content.",
        },
    }
    (args.out / "static_reconstruction_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
