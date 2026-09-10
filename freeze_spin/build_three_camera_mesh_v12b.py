from __future__ import annotations

"""v12b plumbing fix for the three-camera high-standard hypothesis test.

v12's first run stopped before geometry because RF-DETR's shared helper returns
(xy, confidence, detection_confidence, boxes, covariance) as a tuple.  v12
incorrectly treated that tuple as a dict.  This wrapper fixes only that adapter
and then executes the unchanged v12 three-camera experiment.
"""

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


def main():
    v12.attach_rfdetr_poses = attach_rfdetr_poses_fixed
    v12.main()


if __name__ == "__main__":
    main()
