from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

WIDTH = 960
HEIGHT = 540
COURT_LENGTH_CM = 2800.0
COURT_WIDTH_CM = 1500.0
BASKET_X_SHIFT_CM = 157.5
BASKET_HEIGHT_CM = -305.0


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def field_points() -> np.ndarray:
    points = []
    u = 175.0
    s = 0.0
    for _ in range(7):
        for i in range(13):
            points.append([i * COURT_LENGTH_CM / 12.0, COURT_WIDTH_CM - s, 0.0])
        s += u
        u += 30.0
    points.append([BASKET_X_SHIFT_CM, COURT_WIDTH_CM / 2.0, BASKET_HEIGHT_CM])
    points.append([COURT_LENGTH_CM - BASKET_X_SHIFT_CM, COURT_WIDTH_CM / 2.0, BASKET_HEIGHT_CM])
    return np.asarray(points, dtype=np.float64)


def project(P: np.ndarray, points: np.ndarray):
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    q = (P @ homogeneous.T).T
    valid = np.abs(q[:, 2]) > 1e-9
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid] = q[valid, :2] / q[valid, 2:3]
    return uv, valid


def camera_parameters(P: np.ndarray):
    K, R, camera_h, *_ = cv2.decomposeProjectionMatrix(P.astype(np.float64))
    if abs(K[2, 2]) > 1e-12:
        K = K / K[2, 2]
    center = (camera_h[:3] / camera_h[3]).reshape(-1)
    return K, R, center


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["views"]
    args.out.mkdir(parents=True, exist_ok=True)
    points = field_points()
    rows = []

    for row in mapping:
        index = int(row["index"])
        label = row["label"]
        prediction = predictions[index] if index < len(predictions) else {}
        image = cv2.imread(str(args.images / row["image"]))
        if image is None:
            raise RuntimeError(f"Missing impact image: {row['image']}")

        p_values = prediction.get("P")
        result = {"index": index, "label": label, "has_projection_matrix": bool(p_values)}
        if not p_values or len(p_values) != 12:
            result["status"] = "no_valid_projection_matrix"
            rows.append(result)
            cv2.putText(image, "NO CALIBRATION", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.out / f"{index:02d}_{safe_name(label)}.png"), image)
            continue

        P = np.asarray(p_values, dtype=np.float64).reshape(3, 4)
        uv, valid = project(P, points)
        K, R, center = camera_parameters(P)
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT)
        inside_court = int(inside[:-2].sum())

        hoop_uv = uv[-2:]
        image_center = np.array([WIDTH / 2.0, HEIGHT / 2.0])
        hoop_dist = np.linalg.norm(hoop_uv - image_center, axis=1)
        finite_hoops = np.isfinite(hoop_dist)
        visible_basket_index = int(np.argmin(np.where(finite_hoops, hoop_dist, np.inf))) if finite_hoops.any() else None
        visible_basket_uv = hoop_uv[visible_basket_index].tolist() if visible_basket_index is not None else None

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        camera_distance_cm = float(np.linalg.norm(center - np.array([COURT_LENGTH_CM / 2.0, COURT_WIDTH_CM / 2.0, 0.0])))
        plausible_intrinsics = 100.0 <= abs(fx) <= 20000.0 and 100.0 <= abs(fy) <= 20000.0
        plausible_center = bool(np.all(np.isfinite(center)) and camera_distance_cm <= 30000.0)
        enough_floor_support = inside_court >= 4
        status = "candidate" if plausible_intrinsics and plausible_center and enough_floor_support else "reject"

        result.update({
            "status": status,
            "camera_center_cm": [round(float(v), 3) for v in center],
            "camera_distance_from_court_center_cm": round(camera_distance_cm, 3),
            "intrinsics": {"fx": round(fx, 3), "fy": round(fy, 3), "cx": round(cx, 3), "cy": round(cy, 3)},
            "inside_projected_court_grid_points": inside_court,
            "visible_basket_index": visible_basket_index,
            "visible_basket_uv": None if visible_basket_uv is None else [round(float(v), 3) for v in visible_basket_uv],
        })
        rows.append(result)

        for x, y in uv[:-2]:
            if np.isfinite(x) and np.isfinite(y) and -100 <= x < WIDTH + 100 and -100 <= y < HEIGHT + 100:
                cv2.circle(image, (int(round(x)), int(round(y))), 3, (0, 255, 255), -1, cv2.LINE_AA)
        for basket_index, (x, y) in enumerate(hoop_uv):
            if np.isfinite(x) and np.isfinite(y) and -250 <= x < WIDTH + 250 and -250 <= y < HEIGHT + 250:
                cv2.circle(image, (int(round(x)), int(round(y))), 8, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"B{basket_index}", (int(x) + 10, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"{label} | {status} | grid={inside_court}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if status == "candidate" else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.out / f"{index:02d}_{safe_name(label)}.png"), image)

    payload = {
        "model": "CEA-LIST/KaliCalib model_challenge.pth",
        "image_size": [WIDTH, HEIGHT],
        "criteria": {
            "min_inside_projected_court_grid_points": 4,
            "intrinsic_abs_focal_px_range": [100, 20000],
            "max_camera_distance_from_court_center_cm": 30000,
            "note": "Candidate is necessary but not sufficient; overlays still require visual court/basket agreement.",
        },
        "candidate_count": sum(row.get("status") == "candidate" for row in rows),
        "views": rows,
    }
    (args.out / "calibration_qa.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
