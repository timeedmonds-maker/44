from __future__ import annotations

"""v12b plumbing fixes for the three-camera high-standard hypothesis test.

The first v12 run stopped before geometry because RF-DETR's shared helper returns
(xy, confidence, detection_confidence, boxes, covariance) as a tuple while v12
expected a dict.  The next run reached geometry and built three player meshes,
but OpenCV 5 rejected a very tall remap buffer used by the far-background
source warp (the destination exceeded SHRT_MAX rows).  This wrapper fixes only
those execution adapters and then executes the unchanged v12 three-camera
experiment.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_rfdetr_v10 as v10


def attach_rfdetr_poses_fixed(pose_model, image, instances):
    v10.POSE_MODEL = pose_model
    xy, conf, det_conf, boxes, cov = v10._pose_predict(image)
    pose = {
        "xy": np.asarray(xy, np.float64),
        "conf": np.asarray(conf, np.float64),
        "det_conf": np.asarray(det_conf, np.float64),
        "boxes": np.asarray(boxes, np.float64),
        "cov": None if cov is None else np.asarray(cov, np.float64),
    }
    mat = np.zeros((len(instances), len(pose["boxes"])), np.float64)
    for i, inst in enumerate(instances):
        for j, box in enumerate(pose["boxes"]):
            mat[i, j] = v12.iou(inst["box"], box)
    matched = []
    if mat.size:
        rr, cc = linear_sum_assignment(-mat)
        for i, j in zip(rr, cc):
            i, j = int(i), int(j)
            score = float(mat[i, j])
            if score < 0.08:
                continue
            row = {
                "pose_index": j,
                "xy": pose["xy"][j].tolist(),
                "confidence": pose["conf"][j].tolist(),
                "detection_confidence": float(pose["det_conf"][j]),
                "bbox": pose["boxes"][j].tolist(),
                "mask_pose_iou": score,
            }
            if pose["cov"] is not None and j < len(pose["cov"]):
                row["covariance"] = pose["cov"][j].tolist()
            instances[i]["rfdetr_pose"] = row
            matched.append({"mask_instance": i, "pose_index": j, "iou": score})
    return pose, matched


def bilinear_sample_chunked(image: np.ndarray, uv: np.ndarray, chunk: int = 30000):
    """Equivalent to v8.bilinear_sample without OpenCV's SHRT_MAX remap limit."""
    uv = np.asarray(uv)
    out = np.zeros((len(uv), 3), dtype=np.uint8)
    for start in range(0, len(uv), chunk):
        end = min(len(uv), start + chunk)
        mapx = uv[start:end, 0].astype(np.float32).reshape(-1, 1)
        mapy = uv[start:end, 1].astype(np.float32).reshape(-1, 1)
        out[start:end] = cv2.remap(
            image, mapx, mapy, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).reshape(-1, 3)
    return out


def main():
    v12.attach_rfdetr_poses = attach_rfdetr_poses_fixed
    v12.v8.bilinear_sample = bilinear_sample_chunked
    v12.main()


if __name__ == "__main__":
    main()
