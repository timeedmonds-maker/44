from __future__ import annotations

"""v12b execution fixes for the native three-camera high-standard test.

The core v12 experiment remains source-grounded and native 960x540 only.  This
wrapper adapts current-runner compatibility issues and hardens the v11 -> v12
identity handoff.  The v11 strict three-camera track is defined by RF-DETR pose,
not by Mask R-CNN instance indices.  For the exact selected frames the strict
pose is present in all three cameras, but the highly overlapped jumping player is
not assigned an individual Mask R-CNN instance in Left Above Rim or Right Above
Rim.  When that happens, build an identity-specific deterministic segmentation
from the validated RF-DETR pose using pose-seeded GrabCut.  The resulting mask is
only a geometry/ownership constraint; rendered appearance still comes from real
source RGB pixels.

No UHD or upscale path exists here.  The hard output policy is native 960x540.
"""

import copy

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_rfdetr_v10 as v10


_ORIGINAL_NP_CROSS = np.cross
_ORIGINAL_ATTACH_RAR = v12.attach_right_above_rim
POSE_CACHE = {}


def cross_compat(a, b, *args, **kwargs):
    """Restore the historical scalar 2-D cross product used by v12's triangle-area test."""
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.ndim >= 1 and bb.ndim >= 1 and aa.shape[-1] == 2 and bb.shape[-1] == 2:
        return aa[..., 0] * bb[..., 1] - aa[..., 1] * bb[..., 0]
    return _ORIGINAL_NP_CROSS(a, b, *args, **kwargs)


def _camera_label_for_image(image):
    for label, src in getattr(v12, "IMAGES", {}).items():
        if image is src:
            return label
    return None


def _pose_row(pose, j, mask_iou=None, segmentation_source=None):
    row = {
        "pose_index": int(j),
        "xy": pose["xy"][j].tolist(),
        "confidence": pose["conf"][j].tolist(),
        "detection_confidence": float(pose["det_conf"][j]),
        "bbox": pose["boxes"][j].tolist(),
    }
    if mask_iou is not None:
        row["mask_pose_iou"] = float(mask_iou)
    if segmentation_source is not None:
        row["segmentation_source"] = str(segmentation_source)
    if pose.get("cov") is not None and j < len(pose["cov"]):
        row["covariance"] = pose["cov"][j].tolist()
    return row


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
    label = _camera_label_for_image(image)
    if label is not None:
        POSE_CACHE[label] = pose

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
            instances[i]["rfdetr_pose"] = _pose_row(pose, j, mask_iou=score, segmentation_source="mask_rcnn")
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


def _find_mask_for_pose(instances, pose_index):
    for i, inst in enumerate(instances):
        p = inst.get("rfdetr_pose")
        if p and int(p.get("pose_index", -1)) == int(pose_index):
            return int(i)
    return None


def _projection_validation(cams, label, pose, pose_index, strict_track, conf_min=0.12):
    if pose_index < 0 or pose_index >= len(pose["xy"]):
        return {"status": "POSE_INDEX_MISSING", "pose_index": int(pose_index)}
    xy = np.asarray(pose["xy"][pose_index], np.float64)
    cf = np.asarray(pose["conf"][pose_index], np.float64)
    errs = []
    for row in strict_track.get("joints", []):
        j = int(row.get("joint_index", -1))
        world = row.get("world_cm")
        if j < 0 or world is None or j >= len(xy) or j >= len(cf) or cf[j] < conf_min:
            continue
        X = np.asarray(world, np.float64).reshape(1, 3)
        uv, _, valid = v12.v8.project_metric(cams[label], X)
        if bool(valid[0]):
            errs.append(float(np.linalg.norm(xy[j] - uv[0])))
    if not errs:
        return {"status": "NO_COMMON_JOINTS", "pose_index": int(pose_index), "joint_count": 0}
    e = np.asarray(errs, np.float64)
    med = float(np.median(e)); p75 = float(np.percentile(e, 75)); good35 = int(np.sum(e <= 35.0))
    accepted = bool(len(e) >= 6 and med <= 45.0 and p75 <= 75.0)
    return {
        "status": "VALIDATED" if accepted else "REJECTED",
        "pose_index": int(pose_index),
        "joint_count": int(len(e)),
        "median_px": med,
        "p75_px": p75,
        "good35": good35,
    }


def _draw_pose_seed(shape, xy, cf, bbox):
    seed = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(8.0, x2 - x1); bh = max(12.0, y2 - y1)
    line_t = max(4, int(round(0.055 * min(bw, bh))))
    joint_r = max(3, int(round(0.032 * min(bw, bh))))
    for a, b in v10.DRAW_EDGES:
        if a < len(cf) and b < len(cf) and cf[a] >= 0.16 and cf[b] >= 0.16:
            pa = tuple(np.rint(xy[a]).astype(int)); pb = tuple(np.rint(xy[b]).astype(int))
            cv2.line(seed, pa, pb, 255, line_t, cv2.LINE_AA)
    for j in range(min(17, len(cf))):
        if cf[j] >= 0.16:
            cv2.circle(seed, tuple(np.rint(xy[j]).astype(int)), joint_r, 255, -1, cv2.LINE_AA)
    torso = [5, 6, 12, 11]
    if all(j < len(cf) and cf[j] >= 0.12 for j in torso):
        poly = np.rint(xy[torso]).astype(np.int32)
        cv2.fillConvexPoly(seed, poly, 255, cv2.LINE_AA)
    if len(cf) > 0 and cf[0] >= 0.12:
        cv2.circle(seed, tuple(np.rint(xy[0]).astype(int)), max(joint_r + 2, int(round(0.07 * bh))), 255, -1, cv2.LINE_AA)
    return seed


def _pose_prompt_mask(image, pose, target_idx):
    """Identity-specific source segmentation using target pose as hard foreground seed.

    Other overlapping RF-DETR poses supply negative seeds.  GrabCut never creates
    texture; it only decides which real source pixels belong to the target body.
    """
    h, w = image.shape[:2]
    box = np.asarray(pose["boxes"][target_idx], np.float64)
    xy = np.asarray(pose["xy"][target_idx], np.float64)
    cf = np.asarray(pose["conf"][target_idx], np.float64)
    x1, y1, x2, y2 = box
    pad_x = max(6, int(round(0.08 * max(1.0, x2 - x1))))
    pad_y = max(6, int(round(0.06 * max(1.0, y2 - y1))))
    X1 = max(0, int(np.floor(x1)) - pad_x); Y1 = max(0, int(np.floor(y1)) - pad_y)
    X2 = min(w - 1, int(np.ceil(x2)) + pad_x); Y2 = min(h - 1, int(np.ceil(y2)) + pad_y)
    if X2 <= X1 or Y2 <= Y1:
        return None, {"status": "BAD_BBOX"}

    positive = _draw_pose_seed((h, w), xy, cf, box)
    positive[:Y1, :] = 0; positive[Y2 + 1:, :] = 0; positive[:, :X1] = 0; positive[:, X2 + 1:] = 0
    if int((positive > 0).sum()) < 40:
        return None, {"status": "INSUFFICIENT_POSE_SEED"}

    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[Y1:Y2 + 1, X1:X2 + 1] = cv2.GC_PR_BGD
    gc[positive > 0] = cv2.GC_FGD

    # Negative identity prompts from other overlapping pose detections.
    neg_pixels = 0
    for oi in range(len(pose["xy"])):
        if oi == target_idx or oi >= len(pose["boxes"]):
            continue
        ob = pose["boxes"][oi]
        if v12.iou(box, ob) < 0.025:
            continue
        oxy = np.asarray(pose["xy"][oi], np.float64); ocf = np.asarray(pose["conf"][oi], np.float64)
        for j in range(min(17, len(ocf))):
            if ocf[j] < 0.22:
                continue
            px, py = np.rint(oxy[j]).astype(int)
            if not (X1 <= px <= X2 and Y1 <= py <= Y2):
                continue
            yy0, yy1 = max(0, py - 5), min(h, py + 6); xx0, xx1 = max(0, px - 5), min(w, px + 6)
            if np.any(positive[yy0:yy1, xx0:xx1] > 0):
                continue
            cv2.circle(gc, (int(px), int(py)), 4, int(cv2.GC_BGD), -1)
            neg_pixels += 1

    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
        mask = np.logical_or(gc == cv2.GC_FGD, gc == cv2.GC_PR_FGD)
    except cv2.error:
        mask = positive > 0

    roi = np.zeros((h, w), bool); roi[Y1:Y2 + 1, X1:X2 + 1] = True
    mask &= roi
    # Keep only connected components touched by the target skeleton.
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask)
    touched = []
    for cc in range(1, n):
        cm = lab == cc
        ov = int(np.sum(cm & (positive > 0)))
        if ov > 0:
            touched.append((ov, int(cm.sum()), cc))
    if touched:
        touched.sort(reverse=True)
        for ov, area, cc in touched:
            if ov >= max(4, int(0.025 * int((positive > 0).sum()))) or cc == touched[0][2]:
                keep |= lab == cc
        mask = keep

    bbox_area = float((X2 - X1 + 1) * (Y2 - Y1 + 1))
    area = int(mask.sum())
    fallback = False
    if area < max(100, int(0.045 * bbox_area)) or area > int(0.92 * bbox_area):
        # Conservative identity envelope if colour segmentation becomes unstable.
        k = max(3, int(round(0.035 * min(X2 - X1 + 1, Y2 - Y1 + 1))))
        if k % 2 == 0: k += 1
        mask = cv2.dilate(positive, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1) > 0
        mask &= roi
        area = int(mask.sum()); fallback = True
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask, {
        "status": "POSE_PROMPT_MASK_OK",
        "target_pose_index": int(target_idx),
        "bbox": [float(v) for v in box],
        "mask_pixels": int(mask.sum()),
        "bbox_pixels": int(bbox_area),
        "negative_pose_seeds": int(neg_pixels),
        "fallback_to_pose_envelope": bool(fallback),
    }


def _append_pose_prompt_instance(cams, label, instances, pose, pose_index):
    image = v12.IMAGES[label]
    mask, mq = _pose_prompt_mask(image, pose, pose_index)
    if mask is None or int(mask.sum()) < 80:
        mq["status"] = "POSE_PROMPT_MASK_REJECTED"
        return None, mq
    ys, xs = np.where(mask)
    ycut = max(int(np.percentile(ys, 96)), int(ys.max()) - 4)
    near_bottom = xs[ys >= ycut]
    foot_x = int(round(float(np.median(near_bottom)))) if len(near_bottom) else int(round(float(np.median(xs))))
    foot_y = int(ys.max())
    Cc, Rc, Kc = cams[label]
    P, t = v12.v5.pixel_floor_point(foot_x, foot_y, Kc, Rc, Cc)
    inst = {
        "score": float(pose["det_conf"][pose_index]),
        "mask": mask,
        "box": [float(v) for v in pose["boxes"][pose_index]],
        "foot_px": [foot_x, foot_y],
        "segmentation_source": "rfdetr_pose_seeded_grabcut",
        "rfdetr_pose": _pose_row(pose, pose_index, segmentation_source="rfdetr_pose_seeded_grabcut"),
    }
    if P is not None:
        inst["foot_world_cm"] = [float(v) for v in P]
        inst["floor_camera_depth_cm"] = float(t)
    instances.append(inst)
    mi = len(instances) - 1
    mq["mask_instance"] = int(mi)
    mq["appearance"] = v12.appearance_descriptor(image, inst)
    return int(mi), mq


def _ensure_strict_pose_masks(cams, instances, exact_qa):
    audit = []
    for ti, track in enumerate(exact_qa.get("selected", {}).get("assigned_tracks", [])):
        if not track.get("strict"):
            continue
        row = {"strict_track_index": int(ti), "per_camera": {}, "complete": True}
        for label in v12.CAMERAS:
            pose = POSE_CACHE.get(label)
            pose_index = int(track.get("ids", {}).get(label, -1))
            if pose is None:
                row["per_camera"][label] = {"status": "POSE_CACHE_MISSING"}; row["complete"] = False; continue
            val = _projection_validation(cams, label, pose, pose_index, track)
            info = {"validation": val}
            if val.get("status") != "VALIDATED":
                info["status"] = "STRICT_POSE_VALIDATION_FAILED"
                row["per_camera"][label] = info; row["complete"] = False; continue
            existing = _find_mask_for_pose(instances[label], pose_index)
            if existing is not None:
                info.update({"status": "EXISTING_MASK", "mask_instance": int(existing)})
                row["per_camera"][label] = info; continue
            mi, mq = _append_pose_prompt_instance(cams, label, instances[label], pose, pose_index)
            info["pose_prompt_mask"] = mq
            if mi is None:
                info["status"] = "POSE_PROMPT_MASK_FAILED"; row["complete"] = False
            else:
                info.update({"status": "POSE_PROMPT_MASK_ADDED", "mask_instance": int(mi)})
            row["per_camera"][label] = info
        audit.append(row)
    return audit


def _strict_track_mask_match(cams, label, instances, strict_track, conf_min: float = 0.12):
    """Fallback rematch when a validated saved pose id cannot be used directly."""
    targets = {}
    for row in strict_track.get("joints", []):
        j = int(row.get("joint_index", -1)); world = row.get("world_cm")
        if j < 0 or world is None: continue
        uv, _, valid = v12.v8.project_metric(cams[label], np.asarray(world, np.float64).reshape(1, 3))
        if bool(valid[0]): targets[j] = np.asarray(uv[0], np.float64)
    candidates = []
    for mi, inst in enumerate(instances):
        p = inst.get("rfdetr_pose")
        if not p: continue
        xy = np.asarray(p["xy"], np.float64); cf = np.asarray(p["confidence"], np.float64); errs = []
        for j, uv0 in targets.items():
            if j >= len(xy) or j >= len(cf) or cf[j] < conf_min: continue
            errs.append(float(np.linalg.norm(xy[j] - uv0)))
        if len(errs) < 4: continue
        e = np.asarray(errs, np.float64); med = float(np.median(e)); p75 = float(np.percentile(e, 75)); good35 = int(np.sum(e <= 35.0))
        cost = med + 0.20 * p75 - 1.5 * good35 - 0.25 * len(e)
        candidates.append({"mask_instance": int(mi), "pose_index": int(p.get("pose_index", -1)), "joint_count": int(len(e)), "median_px": med, "p75_px": p75, "good35": good35, "cost": float(cost)})
    candidates.sort(key=lambda x: (x["cost"], x["median_px"], -x["joint_count"]))
    if not candidates: return None, {"status": "NO_CANDIDATE", "candidates": []}
    best = candidates[0]; accepted = bool(best["joint_count"] >= 6 and best["median_px"] <= 75.0 and best["p75_px"] <= 110.0)
    return (best["mask_instance"] if accepted else None), {"status": "MATCHED" if accepted else "REJECTED", "best": best, "candidates": candidates[:5]}


def attach_right_above_rim_fixed(cams, instances, mb, exact_qa):
    """Recover missing strict-pose masks, then preserve the original v12 association logic."""
    qa = copy.deepcopy(exact_qa)
    mask_audit = _ensure_strict_pose_masks(cams, instances, qa)

    # Only remap if a strict saved pose still has no current mask after recovery.
    remap_audit = []
    for ti, track in enumerate(qa.get("selected", {}).get("assigned_tracks", [])):
        if not track.get("strict"): continue
        old_ids = dict(track.get("ids", {})); new_ids = dict(old_ids); per_camera = {}; complete = True
        for label in v12.CAMERAS:
            pi = int(old_ids.get(label, -999))
            existing = _find_mask_for_pose(instances[label], pi)
            if existing is not None:
                per_camera[label] = {"status": "DIRECT_STRICT_POSE_MASK", "selected_mask_instance": int(existing), "selected_current_pose_index": int(pi)}
                continue
            mi, mq = _strict_track_mask_match(cams, label, instances[label], track); per_camera[label] = mq
            if mi is None:
                complete = False; continue
            pose = instances[label][mi].get("rfdetr_pose")
            if not pose:
                complete = False; continue
            new_ids[label] = int(pose["pose_index"])
            per_camera[label]["selected_mask_instance"] = int(mi); per_camera[label]["selected_current_pose_index"] = int(pose["pose_index"])
        if complete: track["ids"] = new_ids
        remap_audit.append({"strict_track_index": int(ti), "complete": bool(complete), "old_pose_ids": old_ids, "current_pose_ids": new_ids if complete else None, "per_camera": per_camera})

    tracks, audit = _ORIGINAL_ATTACH_RAR(cams, instances, mb, qa)
    audit["v12b_pose_prompt_mask_recovery"] = mask_audit
    audit["v12b_strict_projection_remap"] = remap_audit
    return tracks, audit


def main():
    POSE_CACHE.clear()
    v12.attach_rfdetr_poses = attach_rfdetr_poses_fixed
    v12.attach_right_above_rim = attach_right_above_rim_fixed
    v12.v8.bilinear_sample = bilinear_sample_chunked
    v12.np.cross = cross_compat

    # Hard project policy: native source resolution only, never UHD/upscale.
    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    assert float(v12.np.cross(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))) == 1.0
    v12.main()


if __name__ == "__main__":
    main()
