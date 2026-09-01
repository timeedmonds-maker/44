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
    RIM_CENTER_FROM_BOARD_CM,
    RIM_INSIDE_RADIUS_CM,
    RIM_TOP_HEIGHT_CM,
)

IMAGE_W = 960
IMAGE_H = 540


def world_points() -> np.ndarray:
    """Six non-coplanar basket landmarks in a basket-local metric frame.

    X is perpendicular to the backboard toward the court, Y is horizontal
    across the board, and Z is up from the floor. The four target corners are
    on X=0; the rim endpoints are RIM_CENTER_FROM_BOARD_CM in front of that
    plane, which is the key constraint that breaks planar homography ambiguity.
    """
    hw = BACKBOARD_INNER_RECT_WIDTH_CM / 2.0
    top = RIM_TOP_HEIGHT_CM + BACKBOARD_INNER_RECT_HEIGHT_CM
    z = RIM_TOP_HEIGHT_CM
    r = RIM_INSIDE_RADIUS_CM
    x = RIM_CENTER_FROM_BOARD_CM
    return np.asarray([
        [0.0, -hw, top],
        [0.0, +hw, top],
        [0.0, +hw, z],
        [0.0, -hw, z],
        [x, -r, z],
        [x, +r, z],
    ], dtype=np.float64)


def image_points(view: dict) -> np.ndarray:
    names = [
        "inner_rect_top_left", "inner_rect_top_right",
        "inner_rect_bottom_right", "inner_rect_bottom_left",
        "rim_left", "rim_right",
    ]
    return np.asarray([view[n] for n in names], dtype=np.float64)


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
            res = (uv - obs).ravel()
            # Broadcast crops may move the principal point, so these are soft
            # regularizers rather than hard assumptions.
            res = np.r_[
                res,
                (p[7] - IMAGE_W / 2.0) / 100.0,
                (p[8] - IMAGE_H / 2.0) / 100.0,
                (np.log(focal) - np.log(800.0)) / 1.2,
            ]
            if np.any(cam[:, 2] <= 20.0):
                res = np.r_[res, np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0]
            return res

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
    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    return np.column_stack([
        RIM_CENTER_FROM_BOARD_CM + RIM_INSIDE_RADIUS_CM * np.cos(theta),
        RIM_INSIDE_RADIUS_CM * np.sin(theta),
        np.full_like(theta, RIM_TOP_HEIGHT_CM),
    ])


def draw_overlay(image: np.ndarray, params: np.ndarray, obj: np.ndarray, obs: np.ndarray, out: Path):
    uv, _, _, _ = project(params, obj)
    overlay = image.copy()
    rect = np.round(uv[:4]).astype(int)
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
    obj = world_points()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for view in payload["views"]:
        obs = image_points(view)
        rejected, rmse, _, params, uv, cam, focal, center = solve_one(obj, obs)
        image = cv2.imread(str(args.images / view["image"]))
        if image is None:
            raise RuntimeError(f"Missing image {view['image']}")
        draw_overlay(image, params, obj, obs, args.out / f"{view['index']:02d}_{view['label'].replace(' ', '_')}_overlay.png")
        rows.append({
            "index": int(view["index"]),
            "label": view["label"],
            "landmark_rmse_px": round(float(rmse), 4),
            "focal_px": round(float(focal), 3),
            "principal_point_px": [round(float(params[7]), 3), round(float(params[8]), 3)],
            "camera_center_basket_local_cm": [round(float(x), 3) for x in center],
            "min_landmark_depth_cm": round(float(np.min(cam[:, 2])), 3),
            "reprojection_px": [[round(float(x), 3), round(float(y), 3)] for x, y in uv],
            "status": "pass" if (not rejected and rmse <= 3.0) else "reject",
        })

    centers = [np.asarray(r["camera_center_basket_local_cm"], dtype=float) for r in rows if r["status"] == "pass"]
    baselines = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            baselines.append(float(np.linalg.norm(centers[i] - centers[j])))
    min_baseline = min(baselines) if baselines else 0.0
    passed = len(centers) >= 2 and min_baseline >= 50.0

    report = {
        "method": "known NBA basket metric geometry + deterministic nonlinear camera solve",
        "nonplanar_constraint": "rim centre plane is 15 inches in front of backboard target plane",
        "view_count": len(rows),
        "passed_view_count": len(centers),
        "minimum_pairwise_camera_baseline_cm": round(min_baseline, 3),
        "gate": {
            "max_landmark_rmse_px": 3.0,
            "min_distinct_camera_baseline_cm": 50.0,
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
