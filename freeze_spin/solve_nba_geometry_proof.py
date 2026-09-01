from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.nba_geometry import (
    BACKBOARD_INNER_RECT_HEIGHT_CM,
    BACKBOARD_INNER_RECT_WIDTH_CM,
    FOOT_CM,
    FREE_THROW_LINE_BOARD_DISTANCE_CM,
    PAINT_WIDTH_CM,
    RIM_CENTER_FROM_BOARD_CM,
    RIM_INSIDE_RADIUS_CM,
    RIM_TOP_HEIGHT_CM,
)

IMAGE_W = 960
IMAGE_H = 540


def world_landmarks() -> dict[str, np.ndarray]:
    """Basket-local metric landmarks.

    X is perpendicular to the backboard toward the court, Y is horizontal
    across the board, and Z is up from the floor. The baseline is four feet
    behind the board plane; the free-throw line is fifteen feet in front of it.
    """
    hw = BACKBOARD_INNER_RECT_WIDTH_CM / 2.0
    top = RIM_TOP_HEIGHT_CM + BACKBOARD_INNER_RECT_HEIGHT_CM
    z = RIM_TOP_HEIGHT_CM
    r = RIM_INSIDE_RADIUS_CM
    x = RIM_CENTER_FROM_BOARD_CM
    lane_half = PAINT_WIDTH_CM / 2.0
    return {
        "inner_rect_top_left": np.array([0.0, -hw, top], dtype=np.float64),
        "inner_rect_top_right": np.array([0.0, +hw, top], dtype=np.float64),
        "inner_rect_bottom_right": np.array([0.0, +hw, z], dtype=np.float64),
        "inner_rect_bottom_left": np.array([0.0, -hw, z], dtype=np.float64),
        "rim_left": np.array([x, -r, z], dtype=np.float64),
        "rim_right": np.array([x, +r, z], dtype=np.float64),
        "baseline_left_lane": np.array([-4.0 * FOOT_CM, -lane_half, 0.0], dtype=np.float64),
        "baseline_right_lane": np.array([-4.0 * FOOT_CM, +lane_half, 0.0], dtype=np.float64),
        "ft_left_lane": np.array([FREE_THROW_LINE_BOARD_DISTANCE_CM, -lane_half, 0.0], dtype=np.float64),
        "ft_right_lane": np.array([FREE_THROW_LINE_BOARD_DISTANCE_CM, +lane_half, 0.0], dtype=np.float64),
    }


def view_points(view: dict, world: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, np.ndarray]:
    if "landmarks" in view:
        names = list(view["landmarks"].keys())
        missing = [n for n in names if n not in world]
        if missing:
            raise KeyError(f"Unknown world landmarks for {view['label']}: {missing}")
        obs = np.asarray([view["landmarks"][n] for n in names], dtype=np.float64)
    else:
        names = [
            "inner_rect_top_left", "inner_rect_top_right",
            "inner_rect_bottom_right", "inner_rect_bottom_left",
            "rim_left", "rim_right",
        ]
        obs = np.asarray([view[n] for n in names], dtype=np.float64)
    obj = np.asarray([world[n] for n in names], dtype=np.float64)
    return names, obj, obs


def project(params: np.ndarray, obj: np.ndarray):
    rvec = params[:3]
    tvec = params[3:6]
    focal = float(np.exp(params[6]))
    cx, cy = float(params[7]), float(params[8])
    R, _ = cv2.Rodrigues(rvec)
    cam = (R @ obj.T).T + tvec
    uv = np.column_stack([
        focal * cam[:, 0] / cam[:, 2] + cx,
        focal * cam[:, 1] / cam[:, 2] + cy,
    ])
    return uv, cam, focal, R


def solve_one(obj: np.ndarray, obs: np.ndarray):
    candidates = []
    for f0 in (300.0, 500.0, 800.0, 1200.0, 1800.0):
        K = np.asarray([[f0, 0.0, IMAGE_W / 2.0], [0.0, f0, IMAGE_H / 2.0], [0.0, 0.0, 1.0]])
        ok, rvec, tvec = cv2.solvePnP(obj, obs, K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        p0 = np.r_[rvec.ravel(), tvec.ravel(), np.log(f0), IMAGE_W / 2.0, IMAGE_H / 2.0]

        def residual(p):
            uv, cam, focal, _ = project(p, obj)
            # Keep residual vector length constant. This matters when a bad
            # trial temporarily pushes a point behind the camera.
            depth_penalty = np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0
            return np.r_[
                (uv - obs).ravel(),
                (p[7] - IMAGE_W / 2.0) / 100.0,
                (p[8] - IMAGE_H / 2.0) / 100.0,
                (np.log(focal) - np.log(800.0)) / 1.2,
                depth_penalty,
            ]

        try:
            opt = least_squares(residual, p0, loss="soft_l1", max_nfev=15000)
        except Exception:
            continue
        uv, cam, focal, R = project(opt.x, obj)
        rmse = float(np.sqrt(np.mean(np.sum((uv - obs) ** 2, axis=1))))
        center = -R.T @ opt.x[3:6]
        plausible = (
            np.all(cam[:, 2] > 20.0)
            and 150.0 <= focal <= 4000.0
            and -300.0 <= opt.x[7] <= IMAGE_W + 300.0
            and -300.0 <= opt.x[8] <= IMAGE_H + 300.0
            and -2000.0 <= center[2] <= 3000.0
        )
        candidates.append((not plausible, rmse, abs(np.log(focal / 800.0)), opt.x, uv, cam, focal, center))

    if not candidates:
        raise RuntimeError("No camera solution converged")
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0]


def ring_circle() -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
    return np.column_stack([
        RIM_CENTER_FROM_BOARD_CM + RIM_INSIDE_RADIUS_CM * np.cos(theta),
        RIM_INSIDE_RADIUS_CM * np.sin(theta),
        np.full_like(theta, RIM_TOP_HEIGHT_CM),
    ])


def projected_rim_bbox(params: np.ndarray) -> list[float]:
    uv, _, _, _ = project(params, ring_circle())
    return [float(np.min(uv[:, 0])), float(np.min(uv[:, 1])), float(np.max(uv[:, 0])), float(np.max(uv[:, 1]))]


def half_pixel_sensitivity(obj: np.ndarray, obs: np.ndarray, base_center: np.ndarray) -> float:
    amp = 0.5
    patterns: list[np.ndarray] = []
    for sx, sy in ((amp, 0.0), (-amp, 0.0), (0.0, amp), (0.0, -amp)):
        p = np.zeros_like(obs)
        p[:, 0] = sx
        p[:, 1] = sy
        patterns.append(p)
    p = np.zeros_like(obs)
    p[::2, 0] = amp
    p[1::2, 0] = -amp
    patterns.append(p)
    p = np.zeros_like(obs)
    p[::2, 1] = amp
    p[1::2, 1] = -amp
    patterns.append(p)
    ctr = np.mean(obs, axis=0)
    directions = obs - ctr
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9
    patterns.append(amp * directions)
    patterns.append(-amp * directions)

    shifts = []
    for perturb in patterns:
        rejected, _, _, _, _, _, _, center = solve_one(obj, obs + perturb)
        if rejected:
            return float("inf")
        shifts.append(float(np.linalg.norm(center - base_center)))
    return max(shifts) if shifts else float("inf")


def draw_overlay(image: np.ndarray, params: np.ndarray, names: list[str], obj: np.ndarray, obs: np.ndarray, out: Path):
    uv, _, _, _ = project(params, obj)
    overlay = image.copy()
    index = {name: i for i, name in enumerate(names)}
    rect_names = ["inner_rect_top_left", "inner_rect_top_right", "inner_rect_bottom_right", "inner_rect_bottom_left"]
    if all(n in index for n in rect_names):
        rect = np.round(np.asarray([uv[index[n]] for n in rect_names])).astype(int)
        for a, b in zip(rect, np.roll(rect, -1, axis=0)):
            cv2.line(overlay, tuple(a), tuple(b), (0, 255, 0), 2, cv2.LINE_AA)
    circle_uv, _, _, _ = project(params, ring_circle())
    cv2.polylines(overlay, [np.round(circle_uv).astype(int)], True, (255, 0, 255), 2, cv2.LINE_AA)
    for p in np.round(obs).astype(int):
        cv2.circle(overlay, tuple(p), 3, (0, 255, 255), -1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.landmarks.read_text(encoding="utf-8"))
    world = world_landmarks()
    args.out.mkdir(parents=True, exist_ok=True)

    max_rmse = float(payload.get("max_landmark_rmse_px", 3.0))
    max_rim_bbox_error = float(payload.get("max_rim_bbox_edge_error_px", 5.0))
    max_sensitivity = float(payload.get("max_half_pixel_camera_center_shift_cm", 75.0))
    required_views = int(payload.get("minimum_required_passed_views", 2))
    min_baseline_gate = float(payload.get("min_distinct_camera_baseline_cm", 50.0))

    rows = []
    for view in payload["views"]:
        names, obj, obs = view_points(view, world)
        rejected, rmse, _, params, uv, cam, focal, center = solve_one(obj, obs)
        image = cv2.imread(str(args.images / view["image"]))
        if image is None:
            raise RuntimeError(f"Missing image {view['image']}")
        draw_overlay(image, params, names, obj, obs, args.out / f"{view['index']:02d}_{view['label'].replace(' ', '_')}_overlay.png")

        rim_bbox = projected_rim_bbox(params)
        expected_bbox = view.get("rim_bbox_px")
        rim_bbox_edge_error = None
        if expected_bbox is not None:
            rim_bbox_edge_error = float(np.max(np.abs(np.asarray(rim_bbox) - np.asarray(expected_bbox, dtype=float))))

        sensitivity = half_pixel_sensitivity(obj, obs, center)
        status = (
            not rejected
            and rmse <= max_rmse
            and sensitivity <= max_sensitivity
            and (rim_bbox_edge_error is None or rim_bbox_edge_error <= max_rim_bbox_error)
        )
        rows.append({
            "index": int(view["index"]),
            "label": view["label"],
            "landmark_names": names,
            "landmark_rmse_px": round(float(rmse), 4),
            "focal_px": round(float(focal), 3),
            "principal_point_px": [round(float(params[7]), 3), round(float(params[8]), 3)],
            "camera_center_basket_local_cm": [round(float(x), 3) for x in center],
            "min_landmark_depth_cm": round(float(np.min(cam[:, 2])), 3),
            "projected_rim_bbox_px": [round(float(x), 3) for x in rim_bbox],
            "expected_rim_bbox_px": expected_bbox,
            "rim_bbox_max_edge_error_px": None if rim_bbox_edge_error is None else round(rim_bbox_edge_error, 3),
            "half_pixel_max_camera_center_shift_cm": round(float(sensitivity), 3),
            "reprojection_px": [[round(float(x), 3), round(float(y), 3)] for x, y in uv],
            "status": "pass" if status else "reject",
        })

    passed_rows = [r for r in rows if r["status"] == "pass"]
    centers = [np.asarray(r["camera_center_basket_local_cm"], dtype=float) for r in passed_rows]
    baselines = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            baselines.append(float(np.linalg.norm(centers[i] - centers[j])))
    min_baseline = min(baselines) if baselines else 0.0
    passed = len(centers) >= required_views and min_baseline >= min_baseline_gate

    report = {
        "method": "known NBA basket/court metric geometry + deterministic nonlinear camera solve",
        "nonplanar_constraints": [
            "rim centre plane is 15 inches in front of backboard target plane when rim landmarks are used",
            "backboard target plane plus regulation court floor anchors when floor landmarks are used",
        ],
        "view_count": len(rows),
        "passed_view_count": len(centers),
        "minimum_pairwise_camera_baseline_cm": round(min_baseline, 3),
        "gate": {
            "max_landmark_rmse_px": max_rmse,
            "max_rim_bbox_edge_error_px": max_rim_bbox_error,
            "max_half_pixel_camera_center_shift_cm": max_sensitivity,
            "minimum_required_passed_views": required_views,
            "min_distinct_camera_baseline_cm": min_baseline_gate,
            "pass": bool(passed),
        },
        "views": rows,
    }
    (args.out / "nba_geometry_proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
