from __future__ import annotations

"""v12b execution fixes for the native three-camera high-standard test.

The core v12 experiment remains source-grounded and native 960x540 only.  This
wrapper adapts current-runner compatibility issues and one brittle identity handoff:
RF-DETR detection indices are not stable identifiers across separate inference
runs, so the strict v11 three-camera track is rematched to current masks by
projecting its saved 3-D joints back into each solved camera.

No UHD/upscale path exists here.
"""

import copy

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_rfdetr_v10 as v10


_ORIGINAL_NP_CROSS = np.cross
_ORIGINAL_ATTACH_RAR = v12.attach_right_above_rim


def cross_compat(a, b, *args, **kwargs):
    """Restore the historical scalar 2-D cross product used by v12's triangle-area test."""
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.ndim >= 1 and bb.ndim >= 1 and aa.shape[-1] == 2 and bb.shape[-1] == 2:
        return aa[..., 0] * bb[..., 1] - aa[..., 1] * bb[..., 0]
    return _ORIGINAL_NP_CROSS(a, b, *args, **kwargs)


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


def _strict_track_mask_match(cams, label, instances, strict_track, conf_min: float = 0.12):
    """Match a saved v11 strict 3-D track to one current RF-DETR/mask instance.

    v11 pose ids were array positions, not durable identities.  The physically
    meaningful object is the saved world-space skeleton, so project that skeleton
    into this solved camera and choose the current pose that best explains it.
    """
    targets = {}
    for row in strict_track.get("joints", []):
        j = int(row.get("joint_index", -1))
        world = row.get("world_cm")
        if j < 0 or world is None:
            continue
        X = np.asarray(world, np.float64).reshape(1, 3)
        uv, _, valid = v12.v8.project_metric(cams[label], X)
        if bool(valid[0]):
            targets[j] = np.asarray(uv[0], np.float64)

    candidates = []
    for mi, inst in enumerate(instances):
        p = inst.get("rfdetr_pose")
        if not p:
            continue
        xy = np.asarray(p["xy"], np.float64)
        cf = np.asarray(p["confidence"], np.float64)
        errs = []
        for j, uv0 in targets.items():
            if j >= len(xy) or j >= len(cf) or cf[j] < conf_min:
                continue
            errs.append(float(np.linalg.norm(xy[j] - uv0)))
        if len(errs) < 4:
            continue
        e = np.asarray(errs, np.float64)
        med = float(np.median(e))
        p75 = float(np.percentile(e, 75))
        good35 = int(np.sum(e <= 35.0))
        cost = med + 0.20 * p75 - 1.5 * good35 - 0.25 * len(e)
        candidates.append({
            "mask_instance": int(mi),
            "pose_index": int(p.get("pose_index", -1)),
            "joint_count": int(len(e)),
            "median_px": med,
            "p75_px": p75,
            "good35": good35,
            "cost": float(cost),
        })

    candidates.sort(key=lambda x: (x["cost"], x["median_px"], -x["joint_count"]))
    if not candidates:
        return None, {"status": "NO_CANDIDATE", "candidates": []}
    best = candidates[0]
    accepted = bool(best["joint_count"] >= 6 and best["median_px"] <= 75.0 and best["p75_px"] <= 110.0)
    return (best["mask_instance"] if accepted else None), {
        "status": "MATCHED" if accepted else "REJECTED",
        "best": best,
        "candidates": candidates[:5],
    }


def attach_right_above_rim_fixed(cams, instances, mb, exact_qa):
    """Preserve v12 association logic but make the v11 strict handoff identity-stable."""
    qa = copy.deepcopy(exact_qa)
    remap_audit = []
    for ti, track in enumerate(qa.get("selected", {}).get("assigned_tracks", [])):
        if not track.get("strict"):
            continue
        old_ids = dict(track.get("ids", {}))
        new_ids = dict(old_ids)
        per_camera = {}
        complete = True
        for label in v12.CAMERAS:
            mi, mq = _strict_track_mask_match(cams, label, instances[label], track)
            per_camera[label] = mq
            if mi is None:
                complete = False
                continue
            pose = instances[label][mi].get("rfdetr_pose")
            if not pose:
                complete = False
                continue
            new_ids[label] = int(pose["pose_index"])
            per_camera[label]["selected_mask_instance"] = int(mi)
            per_camera[label]["selected_current_pose_index"] = int(pose["pose_index"])
        if complete:
            track["ids"] = new_ids
        remap_audit.append({
            "strict_track_index": int(ti),
            "complete": bool(complete),
            "old_pose_ids": old_ids,
            "current_pose_ids": new_ids if complete else None,
            "per_camera": per_camera,
        })

    tracks, audit = _ORIGINAL_ATTACH_RAR(cams, instances, mb, qa)
    audit["v12b_strict_projection_remap"] = remap_audit
    return tracks, audit


def main():
    v12.attach_rfdetr_poses = attach_rfdetr_poses_fixed
    v12.attach_right_above_rim = attach_right_above_rim_fixed
    v12.v8.bilinear_sample = bilinear_sample_chunked
    v12.np.cross = cross_compat

    # Hard policy guard for this project: native source resolution only, never UHD/upscale.
    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    assert float(v12.np.cross(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))) == 1.0
    v12.main()


if __name__ == "__main__":
    main()
