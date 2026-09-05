from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.solve_nba_geometry_proof_v3 import solve_camera, world_landmarks


def matrix(a: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 9) for v in row] for row in np.asarray(a)]


def vector(a: np.ndarray) -> list[float]:
    return [round(float(v), 9) for v in np.asarray(a).ravel()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.landmarks.read_text(encoding="utf-8"))
    world = world_landmarks()
    cameras = []

    for view in spec["views"]:
        names, obj, obs, rim_samples, board_obs, solved = solve_camera(view, world)
        rejected, score, params, rmse, center, focal, rim_metrics, board_rmse = solved
        if rejected:
            raise RuntimeError(f"Rejected camera while exporting: {view['label']}")

        rvec = np.asarray(params[:3], dtype=np.float64)
        tvec = np.asarray(params[3:6], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        cx, cy = float(params[7]), float(params[8])
        K = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        P = K @ np.column_stack([R, tvec])

        row = {
            "index": int(view["index"]),
            "label": view["label"],
            "calibration_anchor_frame": int(spec.get("calibration_anchor_frame", 28)),
            "image_size_px": [960, 540],
            "focal_px": round(float(focal), 9),
            "principal_point_px": [round(cx, 9), round(cy, 9)],
            "K": matrix(K),
            "rvec_world_to_camera": vector(rvec),
            "R_world_to_camera": matrix(R),
            "t_world_to_camera_cm": vector(tvec),
            "camera_center_world_cm": vector(center),
            "projection_matrix_KRt": matrix(P),
            "landmark_rmse_px": round(float(rmse), 9),
            "solver_cost": round(float(score), 9),
            "estimated_backboard_bottom_z_cm": round(float(params[9]), 9) if len(params) > 9 else None,
        }
        cameras.append(row)

    payload = {
        "coordinate_system": "basket-local metric coordinates in centimetres: +X from board toward court, +Y across board, +Z upward",
        "transform_convention": "X_camera = R_world_to_camera @ X_world + t_world_to_camera_cm; pixel homogeneous ~ K @ X_camera",
        "source_landmarks": str(args.landmarks),
        "camera_count": len(cameras),
        "cameras": cameras,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
