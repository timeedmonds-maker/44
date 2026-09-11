from __future__ import annotations

"""v16: remove residual player contamination from static reprojection.

v15 proved the long white court streak is not a reconstructed player mesh: it
survived identity-mask-safe mesh rendering and lies well outside every projected
mesh bbox. The remaining failure mode is a real player pixel that one source
segmentation did not exclude before that source was sampled as floor/background.

v16 keeps all v15 geometry/rendering choices and adds an independent semantic
safety net: every RF-DETR pose whose inferred support lies on/near the regulation
court expands the source dynamic-exclusion mask in-place before static floor and
background visibility are computed. This never creates appearance; it only
prevents known human pixels from being mislabelled as static court/arena pixels.

Native 960x540 only. No generated texture, interpolation fill, upscale or UHD.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v12b as v12b
from freeze_spin import build_three_camera_mesh_v15 as v15


_ORIGINAL_BROAD_DETECT = v15.detect_near_play_with_broad_static_mask
_ORIGINAL_ATTACH = v12b.attach_rfdetr_poses_fixed
_DYNAMIC_BY_IMAGE = {}
_AUGMENT_QA = {}


def detect_store_dynamic(model, image, K, R, C):
    dyn, inst, balls = _ORIGINAL_BROAD_DETECT(model, image, K, R, C)
    _DYNAMIC_BY_IMAGE[id(image)] = {
        "mask": dyn,
        "K": np.asarray(K, np.float64),
        "R": np.asarray(R, np.float64),
        "C": np.asarray(C, np.float64),
    }
    return dyn, inst, balls


def _pose_support_pixel(pose, idx, box):
    xy = np.asarray(pose["xy"][idx], np.float64)
    cf = np.asarray(pose["conf"][idx], np.float64)
    ankle_ids = [j for j in (15, 16) if j < len(cf) and cf[j] >= 0.14 and np.isfinite(xy[j]).all()]
    if ankle_ids:
        p = np.median(xy[ankle_ids], axis=0)
        return float(p[0]), float(p[1]), "ankle"
    x1, y1, x2, y2 = [float(v) for v in box]
    return 0.5 * (x1 + x2), y2, "bbox_bottom"


def attach_pose_and_expand_static_exclusion(pose_model, image, instances):
    pose, matched = _ORIGINAL_ATTACH(pose_model, image, instances)
    state = _DYNAMIC_BY_IMAGE.get(id(image))
    if state is None:
        return pose, matched

    dyn = state["mask"]
    K, R, C = state["K"], state["R"], state["C"]
    rows = []
    boxes = np.asarray(pose.get("boxes", []), np.float64)
    det_conf = np.asarray(pose.get("det_conf", np.ones(len(boxes))), np.float64)
    for i, box in enumerate(boxes):
        if i >= len(det_conf) or float(det_conf[i]) < 0.14 or not np.isfinite(box).all():
            continue
        sx, sy, support_method = _pose_support_pixel(pose, i, box)
        P, _ = v12.v5.pixel_floor_point(sx, sy, K, R, C)
        if P is None:
            continue
        xw, yw = float(P[0]), float(P[1])
        oncourt = (-260.0 <= xw <= 1650.0 and abs(yw) <= 880.0)
        if not oncourt:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        pad = 5
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(v12.W - 1, x2 + pad); y2 = min(v12.H - 1, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        before = int(dyn[y1:y2 + 1, x1:x2 + 1].sum())
        dyn[y1:y2 + 1, x1:x2 + 1] = True
        added = int((y2 - y1 + 1) * (x2 - x1 + 1) - before)
        rows.append({
            "pose_index": int(i),
            "detection_confidence": float(det_conf[i]),
            "support_method": support_method,
            "support_px": [float(sx), float(sy)],
            "floor_world_cm": [float(P[0]), float(P[1]), float(P[2])],
            "bbox_with_pad": [int(x1), int(y1), int(x2), int(y2)],
            "newly_excluded_pixels_bbox": int(max(0, added)),
        })
    _AUGMENT_QA[str(id(image))] = rows
    return pose, matched


def main():
    _DYNAMIC_BY_IMAGE.clear(); _AUGMENT_QA.clear()
    v15.detect_near_play_with_broad_static_mask = detect_store_dynamic
    v12b.attach_rfdetr_poses_fixed = attach_pose_and_expand_static_exclusion

    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v15.main()

    try:
        oi = sys.argv.index("--out")
        out = Path(sys.argv[oi + 1])
        qp = out / "three_camera_mesh_v12_qa.json"
        q = json.loads(qp.read_text())
        q["v16_renderer"] = {
            "resolution": [960, 540],
            "semantic_static_exclusion": "v15 broad Mask R-CNN on-court mask plus RF-DETR on-court pose bboxes padded 5px before static visibility sampling",
            "rfdetr_excluded_pose_count": int(sum(len(v) for v in _AUGMENT_QA.values())),
            "rfdetr_exclusion_audit": list(_AUGMENT_QA.values()),
            "generated_texture": False,
            "upscale": False,
            "uhd": False,
        }
        qp.write_text(json.dumps(q, indent=2))
    except Exception as exc:
        print("V16_QA_APPEND_WARNING", repr(exc), flush=True)


if __name__ == "__main__":
    main()
