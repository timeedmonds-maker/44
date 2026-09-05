from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

CAMERA_INDEX = {
    "In Arena": 4,
    "Left Slash": 6,
    "Left HandHeld": 8,
    "Left Above Rim": 10,
}
STATIC_ROI = {
    "In Arena": (250, 30, 600, 220),
    "Left Slash": (250, 20, 700, 210),
    "Left HandHeld": (300, 20, 850, 260),
    "Left Above Rim": (250, 20, 720, 190),
}


def load_camera_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["label"]: row for row in payload["cameras"]}


def estimate_selected_to_anchor_homography(anchor_bgr: np.ndarray, selected_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, dict]:
    anchor = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)
    selected = cv2.cvtColor(selected_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = roi
    mask = np.zeros_like(anchor)
    mask[y1:y2, x1:x2] = 255
    pts_anchor = cv2.goodFeaturesToTrack(anchor, maxCorners=500, qualityLevel=0.01, minDistance=7, mask=mask, blockSize=7)
    if pts_anchor is None or len(pts_anchor) < 20:
        raise RuntimeError("Insufficient static features for camera-motion compensation")
    pts_selected, status, err = cv2.calcOpticalFlowPyrLK(
        anchor, selected, pts_anchor, None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
    )
    good = (status.ravel() == 1) & (err.ravel() < 15.0)
    pa = pts_anchor[good].reshape(-1, 2)
    ps = pts_selected[good].reshape(-1, 2)
    H, inliers = cv2.findHomography(ps, pa, cv2.RANSAC, 1.0)
    if H is None or inliers is None:
        raise RuntimeError("Static homography estimation failed")
    keep = inliers.ravel().astype(bool)
    pred = cv2.perspectiveTransform(ps[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
    residual = np.linalg.norm(pred - pa[keep], axis=1)
    qa = {
        "tracked_features": int(len(pa)),
        "inlier_features": int(np.sum(keep)),
        "median_static_residual_px": float(np.median(residual)),
        "p95_static_residual_px": float(np.percentile(residual, 95)),
    }
    if qa["inlier_features"] < 30 or qa["median_static_residual_px"] > 0.75 or qa["p95_static_residual_px"] > 1.5:
        raise RuntimeError(f"Camera-motion compensation QA failed: {qa}")
    return H, qa


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    q = P @ np.r_[X, 1.0]
    return q[:2] / q[2]


def dlt_initial(cameras: dict[str, dict], observations: dict[str, np.ndarray], labels: list[str]) -> np.ndarray:
    A = []
    for label in labels:
        P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
        u, v = observations[label]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    xh = vt[-1]
    return xh[:3] / xh[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--foreground", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--anchor-images", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cameras = load_camera_map(args.cameras)
    foreground = json.loads(args.foreground.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    ball_views = foreground["ball"]["views"]

    obs_anchor: dict[str, np.ndarray] = {}
    homography_qa = {}
    homographies = {}
    for label in CAMERA_INDEX:
        frame_index = int(state["selected_frames"][label])
        anchor_path = args.anchor_images / f"{CAMERA_INDEX[label]}.png"
        selected_path = args.locked_images / f"{label.replace(' ', '_')}_F{frame_index}.png"
        anchor = cv2.imread(str(anchor_path))
        selected = cv2.imread(str(selected_path))
        if anchor is None or selected is None:
            raise FileNotFoundError(f"Missing anchor/selected image for {label}")
        H, qa = estimate_selected_to_anchor_homography(anchor, selected, STATIC_ROI[label])
        p = np.asarray(ball_views[label]["pixel_xy_selected_frame"], dtype=np.float64)
        q = cv2.perspectiveTransform(p.reshape(1, 1, 2), H)[0, 0]
        obs_anchor[label] = q
        homography_qa[label] = qa
        homographies[label] = H

    core = ["In Arena", "Left Slash", "Left HandHeld"]
    x0 = dlt_initial(cameras, obs_anchor, core)

    def residual(X: np.ndarray) -> np.ndarray:
        rows = []
        for label in CAMERA_INDEX:
            P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
            uv = project(P, X)
            sigma = float(ball_views[label]["uncertainty_px"])
            rows.extend((uv - obs_anchor[label]) / sigma)
        return np.asarray(rows)

    fit = least_squares(residual, x0, loss="huber", f_scale=1.0)
    X = fit.x

    views = []
    max_normalized = 0.0
    for label in CAMERA_INDEX:
        P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
        pred_anchor = project(P, X)
        observed_anchor = obs_anchor[label]
        error_anchor = float(np.linalg.norm(pred_anchor - observed_anchor))
        sigma = float(ball_views[label]["uncertainty_px"])
        normalized = error_anchor / sigma
        max_normalized = max(max_normalized, normalized)
        Hinv = np.linalg.inv(homographies[label])
        pred_selected = cv2.perspectiveTransform(pred_anchor.reshape(1, 1, 2), Hinv)[0, 0]
        observed_selected = np.asarray(ball_views[label]["pixel_xy_selected_frame"], dtype=np.float64)
        views.append({
            "label": label,
            "selected_frame": int(state["selected_frames"][label]),
            "observed_ball_selected_px": [round(float(x), 4) for x in observed_selected],
            "observed_ball_anchor_px": [round(float(x), 4) for x in observed_anchor],
            "predicted_ball_anchor_px": [round(float(x), 4) for x in pred_anchor],
            "predicted_ball_selected_px": [round(float(x), 4) for x in pred_selected],
            "reprojection_error_px": round(error_anchor, 4),
            "uncertainty_px": sigma,
            "normalized_error_sigma": round(normalized, 4),
            "camera_motion_homography_selected_to_anchor": [[round(float(v), 10) for v in row] for row in homographies[label]],
            "static_motion_compensation_qa": homography_qa[label],
        })

    # Basket-local coordinates: rim centre is [38.1, 0, 304.8] cm in this solver.
    rim_center = np.asarray([38.1, 0.0, 304.8], dtype=np.float64)
    distance_from_rim_center = float(np.linalg.norm(X - rim_center))
    gate = {
        "ball_height_near_rim": bool(270.0 <= X[2] <= 325.0),
        "max_normalized_reprojection_error_le_1_5_sigma": bool(max_normalized <= 1.5),
        "all_static_motion_compensation_passed": True,
    }
    gate["pass"] = bool(all(gate.values()))

    payload = {
        "method": "locked-state real-pixel ball centres + static camera-motion compensation + robust four-camera metric triangulation",
        "coordinate_system": "basket-local centimetres (+X board toward court, +Y across board, +Z upward)",
        "ball_center_world_cm": [round(float(x), 6) for x in X],
        "ball_height_cm": round(float(X[2]), 6),
        "rim_top_height_cm": 304.8,
        "ball_height_below_rim_top_cm": round(float(304.8 - X[2]), 6),
        "distance_from_rim_center_cm": round(distance_from_rim_center, 6),
        "optimizer_cost": round(float(fit.cost), 8),
        "gate": gate,
        "views": views,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
