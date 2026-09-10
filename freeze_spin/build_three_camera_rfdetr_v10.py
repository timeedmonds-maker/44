from __future__ import annotations

"""RF-DETR-Keypoint constrained three-camera diagnostic v10.

This is an integration layer over semantic v9. It keeps v9's exact metric
court/backboard/rim, individual Mask R-CNN silhouettes, deterministic ball
triangulation and native 960x540 render, then adds RF-DETR Keypoint Preview as
semantic geometry:

* infer COCO-17 human pose independently in each accepted source frame;
* attach RF-DETR poses to the corresponding v9 person masks by one-to-one IoU;
* use two-view triangulated joint/bone plausibility as an additional cross-camera
  identity-association term;
* triangulate per-track 3-D joints from all available accepted cameras;
* when the skeleton is sufficiently supported and physically plausible, reject
  visual-hull surface points far from the articulated body skeleton.

No generated pixels, synthetic player texture, optical-flow morph or upscale.
RF-DETR is used only to constrain geometry/identity; source RGB remains the
appearance source.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v6 as v6
from freeze_spin import build_three_camera_semantic_v9 as v9
from freeze_spin import build_three_camera_volumetric_v8 as v8

try:
    from rfdetr import RFDETRKeypointPreview
except Exception as e:
    raise RuntimeError("RF-DETR Keypoint Preview unavailable; install rfdetr>=1.8.1") from e

CAMERA_ORDER = v9.CAMERA_ORDER
RIM = v9.RIM
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
BONES = [
    (5, 6, 30.0, 20.0, 70.0),
    (5, 7, 19.0, 15.0, 55.0), (7, 9, 17.0, 12.0, 55.0),
    (6, 8, 19.0, 15.0, 55.0), (8, 10, 17.0, 12.0, 55.0),
    (5, 11, 27.0, 25.0, 95.0), (6, 12, 27.0, 25.0, 95.0),
    (11, 12, 28.0, 12.0, 65.0),
    (11, 13, 23.0, 25.0, 75.0), (13, 15, 20.0, 20.0, 75.0),
    (12, 14, 23.0, 25.0, 75.0), (14, 16, 20.0, 20.0, 75.0),
]
DRAW_EDGES = [(a, b) for a, b, *_ in BONES] + [(0, 5), (0, 6)]

POSE_MODEL = None
GLOBAL_CAMS = None
POSE_AUDIT = {}
ORIG_LOAD_CAMERAS = base.load_cameras
ORIG_DETECT = v6.detect_near_play
ORIG_BUILD_HULL = v9.build_track_hull


def _out_dir() -> Path:
    try:
        i = sys.argv.index("--out")
        return Path(sys.argv[i + 1])
    except Exception:
        return Path("three_camera_v10")


def _camera_label(C) -> str:
    if GLOBAL_CAMS:
        return min(CAMERA_ORDER, key=lambda k: float(np.linalg.norm(np.asarray(GLOBAL_CAMS[k][0]) - np.asarray(C))))
    return f"camera_{len(POSE_AUDIT)}"


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a); bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + bb - inter)


def _load_cameras(*args, **kwargs):
    global GLOBAL_CAMS
    GLOBAL_CAMS = ORIG_LOAD_CAMERAS(*args, **kwargs)
    return GLOBAL_CAMS


def _pose_predict(image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kp = POSE_MODEL.predict(rgb, threshold=0.18)
    try:
        kp = kp.with_nms()
    except Exception:
        pass
    xy = np.asarray(kp.xy, dtype=np.float64)
    conf = getattr(kp, "keypoint_confidence", None)
    if conf is None:
        conf = np.ones(xy.shape[:2], np.float64)
    else:
        conf = np.asarray(conf, dtype=np.float64)
    det_conf = getattr(kp, "detection_confidence", None)
    if det_conf is None:
        det_conf = np.ones(len(xy), np.float64)
    else:
        det_conf = np.asarray(det_conf, dtype=np.float64)
    boxes = np.asarray(kp.data.get("xyxy", np.empty((0, 4))), dtype=np.float64)
    cov = kp.data.get("covariance", None)
    cov = np.asarray(cov, dtype=np.float64) if cov is not None else None
    return xy, conf, det_conf, boxes, cov


def _draw_pose(image, xy, conf, boxes, matches, label):
    out = image.copy()
    for j in range(len(xy)):
        if j < len(boxes):
            x1, y1, x2, y2 = np.rint(boxes[j]).astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
        for a, b in DRAW_EDGES:
            if a < xy.shape[1] and b < xy.shape[1] and conf[j, a] >= 0.20 and conf[j, b] >= 0.20:
                pa = tuple(np.rint(xy[j, a]).astype(int)); pb = tuple(np.rint(xy[j, b]).astype(int))
                cv2.line(out, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)
        for k in range(min(17, xy.shape[1])):
            if conf[j, k] >= 0.20:
                p = tuple(np.rint(xy[j, k]).astype(int))
                cv2.circle(out, p, 2, (255, 255, 255), -1, cv2.LINE_AA)
    for mi, pj, iou in matches:
        box = boxes[pj]
        x1, y1 = np.rint(box[:2]).astype(int)
        cv2.putText(out, f"mask{mi}/pose{pj} iou={iou:.2f}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"RF-DETR POSE | {label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _detect_with_pose(model, image, K, R, C):
    dyn, instances, balls = ORIG_DETECT(model, image, K, R, C)
    label = _camera_label(C)
    xy, conf, det_conf, boxes, cov = _pose_predict(image)
    mat = np.zeros((len(instances), len(boxes)), np.float64)
    for i, inst in enumerate(instances):
        for j, box in enumerate(boxes):
            mat[i, j] = _iou(inst["box"], box)
    matches = []
    if mat.size:
        rr, cc = linear_sum_assignment(-mat)
        for i, j in zip(rr, cc):
            score = float(mat[int(i), int(j)])
            if score < 0.08:
                continue
            pose = {
                "xy": xy[int(j)].tolist(),
                "confidence": conf[int(j)].tolist(),
                "detection_confidence": float(det_conf[int(j)]),
                "bbox": boxes[int(j)].tolist(),
                "mask_pose_iou": score,
            }
            if cov is not None and int(j) < len(cov):
                pose["covariance"] = np.asarray(cov[int(j)]).tolist()
            instances[int(i)]["rfdetr_pose"] = pose
            matches.append((int(i), int(j), score))
    summary = {
        "rfdetr_detections": int(len(xy)),
        "mask_instances": int(len(instances)),
        "matched_mask_pose_pairs": int(len(matches)),
        "matches": [{"mask_instance": a, "pose_instance": b, "iou": s} for a, b, s in matches],
        "mean_confident_joints_per_detection": float(np.mean(np.sum(conf >= 0.20, axis=1))) if len(conf) else 0.0,
    }
    POSE_AUDIT[label] = summary
    out = _out_dir(); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"rfdetr_pose_{label.replace(' ', '_')}.png"), _draw_pose(image, xy, conf, boxes, matches, label))
    return dyn, instances, balls


def _triangulate_pair_joints(label_a, label_b, inst_a, inst_b, threshold=0.22):
    pa, pb = inst_a.get("rfdetr_pose"), inst_b.get("rfdetr_pose")
    if not pa or not pb or GLOBAL_CAMS is None:
        return {}, {"pose_available": False, "common_confident_joints": 0}
    xa = np.asarray(pa["xy"], np.float64); xb = np.asarray(pb["xy"], np.float64)
    ca = np.asarray(pa["confidence"], np.float64); cb = np.asarray(pb["confidence"], np.float64)
    joints = {}
    for j in range(min(17, len(xa), len(xb))):
        if ca[j] < threshold or cb[j] < threshold:
            continue
        X = v8.dlt_point(GLOBAL_CAMS, {label_a: xa[j], label_b: xb[j]})
        if X is None or not np.isfinite(X).all():
            continue
        if not (-320 <= X[0] <= 1250 and -720 <= X[1] <= 720 and -40 <= X[2] <= 390):
            continue
        joints[j] = X
    plausible = 0; measured = 0; lengths = {}
    for a, b, _rad, lo, hi in BONES:
        if a in joints and b in joints:
            L = float(np.linalg.norm(joints[a] - joints[b]))
            lengths[f"{COCO_NAMES[a]}-{COCO_NAMES[b]}"] = L
            measured += 1
            plausible += int(lo <= L <= hi)
    frac = float(plausible / measured) if measured else None
    return joints, {
        "pose_available": True,
        "common_confident_joints": int(len(joints)),
        "measured_major_bones": int(measured),
        "plausible_major_bones": int(plausible),
        "plausible_bone_fraction": frac,
        "bone_lengths_cm": lengths,
    }


def _one_to_one_pose(primary_label, other_label, instances, supports):
    A, B = supports[primary_label], supports[other_label]
    if not A or not B:
        return {}, []
    mat = np.full((len(A), len(B)), -1e6, np.float64)
    details = {}
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            base_score, q = v9.pair_score(a, b, instances[primary_label][i], instances[other_label][j])
            if base_score <= -1e5:
                details[(i, j)] = {**q, "base_score": float(base_score), "pose_bonus": 0.0, "pose": {"pose_available": False}}
                continue
            _, pq = _triangulate_pair_joints(primary_label, other_label, instances[primary_label][i], instances[other_label][j])
            bonus = 0.0
            if pq.get("pose_available"):
                n = pq["common_confident_joints"]
                frac = pq.get("plausible_bone_fraction")
                bonus += min(1.25, 0.12 * n)
                if frac is not None:
                    bonus += 2.0 * (frac - 0.50)
                if n < 4:
                    bonus -= 0.75
            score = float(base_score + bonus)
            mat[i, j] = score
            details[(i, j)] = {**q, "base_score": float(base_score), "pose_bonus": float(bonus), "pose": pq}
    rr, cc = linear_sum_assignment(-mat)
    out = {}; qa = []
    for i, j in zip(rr, cc):
        i, j = int(i), int(j)
        s = float(mat[i, j]); q = details[(i, j)]
        geom_ok = q.get("intersection_voxels", 0) >= 18 and q.get("cosine", 0.0) >= 0.004
        pose = q.get("pose", {})
        pose_ok = True
        if pose.get("pose_available") and pose.get("measured_major_bones", 0) >= 3:
            pose_ok = (pose.get("plausible_bone_fraction") or 0.0) >= 0.34
        accepted = bool(s > 4.2 and geom_ok and pose_ok)
        qa.append({"primary_instance": i, "other_instance": j, "score": s, "accepted": accepted, **q})
        if accepted:
            out[i] = j
    return out, qa


def _triangulate_track_joints(cams, track, instances, threshold=0.20):
    joints = {}; joint_qa = {}
    for j in range(17):
        obs = {}; weights = {}
        for label in CAMERA_ORDER:
            idx = track.get(label)
            if idx is None:
                continue
            pose = instances[label][idx].get("rfdetr_pose")
            if not pose:
                continue
            conf = np.asarray(pose["confidence"], np.float64)
            xy = np.asarray(pose["xy"], np.float64)
            if j >= len(conf) or conf[j] < threshold:
                continue
            obs[label] = xy[j]; weights[label] = float(conf[j])
        if len(obs) < 2:
            continue
        X = v8.dlt_point(cams, obs)
        if X is None or not np.isfinite(X).all():
            continue
        if not (-320 <= X[0] <= 1250 and -720 <= X[1] <= 720 and -40 <= X[2] <= 390):
            continue
        errs = {}
        for label, uv0 in obs.items():
            uv, _, valid = v8.project_metric(cams[label], np.asarray(X).reshape(1, 3))
            if valid[0]:
                errs[label] = float(np.linalg.norm(uv[0] - uv0))
        rms = float(np.sqrt(np.mean(np.square(list(errs.values()))))) if errs else 999.0
        if len(obs) >= 3 and rms > 18.0:
            continue
        joints[j] = np.asarray(X, np.float64)
        joint_qa[COCO_NAMES[j]] = {
            "world_cm": [float(v) for v in X], "views": list(obs),
            "mean_confidence": float(np.mean(list(weights.values()))), "reprojection_rms_px": rms,
        }
    return joints, joint_qa


def _point_segment_distance(pts, a, b):
    ab = b - a; den = float(np.dot(ab, ab))
    if den < 1e-8:
        return np.linalg.norm(pts - a, axis=1)
    t = ((pts - a) @ ab) / den
    t = np.clip(t, 0.0, 1.0)
    q = a.reshape(1, 3) + t[:, None] * ab.reshape(1, 3)
    return np.linalg.norm(pts - q, axis=1)


def _build_pose_constrained_hull(cams, track, instances, voxel_cm=4.0):
    surface, qa = ORIG_BUILD_HULL(cams, track, instances, voxel_cm)
    if len(surface) == 0:
        qa["rfdetr_pose_constraint"] = {"applied": False, "reason": "no_base_surface"}
        return surface, qa
    joints, joint_qa = _triangulate_track_joints(cams, track, instances)
    plausible = 0; measured = 0; bone_qa = {}; distances = []
    for a, b, radius, lo, hi in BONES:
        if a not in joints or b not in joints:
            continue
        L = float(np.linalg.norm(joints[a] - joints[b])); ok = lo <= L <= hi
        bone_qa[f"{COCO_NAMES[a]}-{COCO_NAMES[b]}"] = {"length_cm": L, "plausible": bool(ok), "radius_cm": radius}
        measured += 1; plausible += int(ok)
        if ok:
            distances.append(_point_segment_distance(surface.astype(np.float64), joints[a], joints[b]) / radius)
    frac = float(plausible / measured) if measured else 0.0
    applied = len(joints) >= 6 and plausible >= 3 and frac >= 0.42 and bool(distances)
    info = {
        "applied": bool(applied), "triangulated_joints": int(len(joints)),
        "measured_major_bones": int(measured), "plausible_major_bones": int(plausible),
        "plausible_bone_fraction": frac, "joints": joint_qa, "bones": bone_qa,
        "surface_before": int(len(surface)),
    }
    if applied:
        d = np.min(np.vstack(distances), axis=0); keep = d <= 1.0
        for X in joints.values():
            keep |= np.linalg.norm(surface.astype(np.float64) - X.reshape(1, 3), axis=1) <= 22.0
        candidate = surface[keep]
        info["surface_after"] = int(len(candidate)); info["kept_fraction"] = float(len(candidate) / max(1, len(surface)))
        if len(candidate) >= 180 and len(candidate) >= 0.14 * len(surface):
            surface = candidate; qa["status"] = "TRACK_HULL_RFDETR_POSE_CONSTRAINED"
        else:
            info["applied"] = False; info["reason"] = "pose_silhouette_disagreement_gate"
    else:
        info["surface_after"] = int(len(surface)); info["reason"] = "insufficient_or_implausible_pose_geometry"
    qa["rfdetr_pose_constraint"] = info; qa["surface_voxels"] = int(len(surface))
    return surface, qa


def main():
    global POSE_MODEL
    print("Loading RF-DETR Keypoint Preview...", flush=True)
    POSE_MODEL = RFDETRKeypointPreview()
    print("RFDETR_KEYPOINT_MODEL_READY", flush=True)
    base.load_cameras = _load_cameras
    v6.detect_near_play = _detect_with_pose
    v9.one_to_one_match = _one_to_one_pose
    v9.build_track_hull = _build_pose_constrained_hull
    v9.main()
    out = _out_dir(); src = out / "three_camera_semantic_v9_qa.json"
    if src.exists():
        qa = json.loads(src.read_text())
        qa["schema_version"] = 10
        qa["status"] = "DIAGNOSTIC_THREE_CAMERA_RFDETR_KEYPOINT_V10_RENDERED"
        qa["rfdetr_keypoint"] = {
            "model": "RFDETRKeypointPreview COCO-17",
            "role": "semantic cross-camera identity + articulated 3-D hull constraint only",
            "source_appearance_policy": "unchanged: source RGB only",
            "camera_pose_policy": "unchanged accepted metric cameras; homography is not substituted for 3-D calibration",
            "source_pose_audit": POSE_AUDIT,
        }
        qa["render_resolution_policy"] = "native 960x540 only; no upscale"
        qa["representation_change_from_v9"] = (
            "RF-DETR COCO-17 poses are attached one-to-one to source masks, used to score cross-camera "
            "identity association and, when multi-view bone geometry passes physical gates, constrain each "
            "per-player visual-hull surface around triangulated articulated joints."
        )
        (out / "three_camera_rfdetr_v10_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
        print(json.dumps({
            "status": qa["status"], "pose_audit": POSE_AUDIT,
            "accepted_track_count": qa.get("accepted_track_count"), "ball": qa.get("ball", {}),
            "tracks": [t.get("rfdetr_pose_constraint", {}) for t in qa.get("tracks", [])],
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
