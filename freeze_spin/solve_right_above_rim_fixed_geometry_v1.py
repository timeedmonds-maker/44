from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

IMAGE_W = 960
IMAGE_H = 540
INCH_CM = 2.54
FOOT_CM = 12.0 * INCH_CM
RIM_X_CM = 15.0 * INCH_CM
RIM_Z_CM = 10.0 * FOOT_CM
RIM_RADIUS_CM = 9.0 * INCH_CM
RESTRICTED_RADIUS_CM = 4.0 * FOOT_CM


def circle(cx: float, z: float, radius: float, samples: int = 180) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return np.column_stack([
        cx + radius * np.cos(t),
        radius * np.sin(t),
        np.full_like(t, z),
    ]).astype(np.float64)


def look_at_rvec(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - center
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def project(params: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(params[:3])
    center = params[3:6]
    focal = float(np.exp(params[6]))
    cx, cy = float(params[7]), float(params[8])
    cam = (R @ (obj - center).T).T
    uv = np.column_stack([
        focal * cam[:, 0] / cam[:, 2] + cx,
        focal * cam[:, 1] / cam[:, 2] + cy,
    ])
    return uv, cam, R


def nearest_curve_distances(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    d2 = np.sum((obs[:, None, :] - pred[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def solve(spec: dict) -> tuple[np.ndarray, dict]:
    rim3 = circle(RIM_X_CM, RIM_Z_CM, RIM_RADIUS_CM)
    restricted3 = circle(RIM_X_CM, 0.0, RESTRICTED_RADIUS_CM)
    rim_obs = np.asarray(spec["rim_inner_edge_samples_px"], dtype=np.float64)
    restricted_obs = np.asarray(spec["restricted_area_centerline_samples_px"], dtype=np.float64)

    def residual(p: np.ndarray) -> np.ndarray:
        rim_uv, rim_cam, _ = project(p, rim3)
        restricted_uv, restricted_cam, _ = project(p, restricted3)
        rim_d = nearest_curve_distances(rim_obs, rim_uv)
        restricted_d = nearest_curve_distances(restricted_obs, restricted_uv)
        depth_min = min(float(np.min(rim_cam[:, 2])), float(np.min(restricted_cam[:, 2])))
        depth_penalty = max(0.0, 20.0 - depth_min) / 2.0
        priors = np.asarray([
            (p[7] - IMAGE_W / 2.0) / 100.0,
            (p[8] - IMAGE_H / 2.0) / 100.0,
            (p[6] - math.log(1000.0)) / 1.5,
            depth_penalty,
        ])
        return np.concatenate([rim_d, restricted_d / 1.5, priors])

    starts = [
        (20.0, -50.0, 550.0, 1100.0),
        (20.0, 50.0, 550.0, 1100.0),
        (0.0, 0.0, 650.0, 1300.0),
    ]
    lower = np.r_[[-np.inf] * 3, [-1000.0, -1000.0, 250.0], math.log(150.0), 300.0, 100.0]
    upper = np.r_[[np.inf] * 3, [1000.0, 1000.0, 3000.0], math.log(4000.0), 660.0, 440.0]
    candidates = []
    for cx, cy, cz, focal in starts:
        center = np.asarray([cx, cy, cz], dtype=np.float64)
        p0 = np.r_[
            look_at_rvec(center, np.asarray([RIM_X_CM, 0.0, 150.0])),
            center,
            math.log(focal),
            IMAGE_W / 2.0,
            IMAGE_H / 2.0,
        ]
        fit = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=1400,
        )
        p = fit.x
        rim_uv, rim_cam, R = project(p, rim3)
        restricted_uv, restricted_cam, _ = project(p, restricted3)
        rim_d = nearest_curve_distances(rim_obs, rim_uv)
        restricted_d = nearest_curve_distances(restricted_obs, restricted_uv)
        all_d = np.r_[rim_d, restricted_d]
        plausible = (
            float(np.min(rim_cam[:, 2])) > 20.0
            and float(np.min(restricted_cam[:, 2])) > 20.0
            and 150.0 <= math.exp(float(p[6])) <= 4000.0
        )
        candidates.append((not plausible, float(np.sqrt(np.mean(all_d ** 2))), fit.cost, p, rim_d, restricted_d, R))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    bad, rms, cost, p, rim_d, restricted_d, R = candidates[0]
    center = p[3:6]
    qa = {
        "plausible": not bad,
        "combined_curve_rms_px": float(rms),
        "combined_curve_p95_px": float(np.percentile(np.r_[rim_d, restricted_d], 95)),
        "rim_curve_rms_px": float(np.sqrt(np.mean(rim_d ** 2))),
        "restricted_curve_rms_px": float(np.sqrt(np.mean(restricted_d ** 2))),
        "optimizer_cost": float(cost),
        "camera_center_world_cm": [float(v) for v in center],
        "focal_px": float(math.exp(float(p[6]))),
        "principal_point_px": [float(p[7]), float(p[8])],
        "R_world_to_camera": R.tolist(),
        "t_world_to_camera_cm": (-R @ center).tolist(),
    }
    K = np.asarray([[qa["focal_px"], 0.0, p[7]], [0.0, qa["focal_px"], p[8]], [0.0, 0.0, 1.0]])
    qa["projection_matrix_KRt"] = (K @ np.column_stack([R, -R @ center])).tolist()
    return p, qa


def draw_overlay(image: np.ndarray, p: np.ndarray, spec: dict, out: Path) -> None:
    rim3 = circle(RIM_X_CM, RIM_Z_CM, RIM_RADIUS_CM, 360)
    restricted3 = circle(RIM_X_CM, 0.0, RESTRICTED_RADIUS_CM, 360)
    rim_uv = project(p, rim3)[0]
    restricted_uv = project(p, restricted3)[0]
    overlay = image.copy()
    cv2.polylines(overlay, [np.round(rim_uv).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(overlay, [np.round(restricted_uv).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    for q in np.asarray(spec["rim_inner_edge_samples_px"], dtype=int):
        cv2.circle(overlay, tuple(q), 3, (0, 0, 255), -1, cv2.LINE_AA)
    for q in np.asarray(spec["restricted_area_centerline_samples_px"], dtype=int):
        cv2.circle(overlay, tuple(q), 3, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(out), overlay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    image = cv2.imread(str(args.image))
    if image is None or image.shape[:2] != (IMAGE_H, IMAGE_W):
        raise RuntimeError("Expected native 960x540 Right Above Rim reference frame")
    args.out.mkdir(parents=True, exist_ok=True)
    p, qa = solve(spec)
    curve_gate = (
        qa["plausible"]
        and qa["combined_curve_rms_px"] <= float(spec["strict_max_combined_curve_rms_px"])
        and qa["combined_curve_p95_px"] <= float(spec["strict_max_combined_curve_p95_px"])
    )
    qa["half_pixel_sensitivity_status"] = "not_evaluated_until_curve_gate_passes"
    gate = {
        "source_geometry_independent_of_players": True,
        "strict_curve_fit": bool(curve_gate),
        "half_pixel_camera_center_stability": False,
        "pass": False,
    }
    payload = {
        "status": "strict_metric_camera_not_accepted" if not gate["pass"] else "strict_metric_camera_accepted",
        "method": "known 3D rim circle + known 4ft restricted-area floor circle; no player or ball points",
        "qa": qa,
        "gate": gate,
        "next_action": "Refine/automate source-curve extraction until <=3px RMS, then run half-pixel camera-center perturbation before promoting this camera.",
    }
    (args.out / "right_above_rim_fixed_geometry_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_overlay(image, p, spec, args.out / "right_above_rim_fixed_geometry_overlay_v1.png")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
