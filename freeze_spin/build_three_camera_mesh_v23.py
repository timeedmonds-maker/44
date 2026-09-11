from __future__ import annotations

"""v23: focal #12 recovery when the Left-Above-Rim detector is occlusion-confused.

v22 proved that a dedicated focal mesh could be forced through the renderer, but
visual QA showed the Left Above Rim identity prompt had locked onto the white
leaper in front of the focal black-uniform #12 player.  That is not an acceptable
identity-preservation pass.

v23 removes the requirement for an independent LAR pose detection for the focal
player.  The focal black-uniform player is cleanly detected in Broadcast and Right
Above Rim.  Those two solved cameras triangulate a 3-D skeleton; that skeleton is
then projected into Left Above Rim and used only to recover the *visible* source
pixels of the same player behind the occluder.  The anatomical mesh uses the
Broadcast/RAR triangulated joints and silhouettes from all three solved cameras.

No appearance is generated.  All rendered RGB comes from the three native source
frames.  Output remains hard-locked to native 960x540; no UHD or upscale path.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v12b as v12b
from freeze_spin import build_three_camera_mesh_v14 as v14
from freeze_spin import build_three_camera_mesh_v22 as v22


_ORIG_TRIANGULATE = v12.triangulate_track_joints
_ORIG_POST = v22._post_focal_visibility
_V23_AUDIT = {}


def _pose_arrays(pose, idx):
    return (
        np.asarray(pose["xy"][idx], np.float64),
        np.asarray(pose["conf"][idx], np.float64),
    )


def _triangulate_bc_focal(cams, pose_b, bi, pose_c, ci, conf_min=0.12):
    xb, cb = _pose_arrays(pose_b, bi)
    xc, cc = _pose_arrays(pose_c, ci)
    joints = {}
    conf = {}
    jq = {}
    for j in range(min(17, len(xb), len(xc))):
        if cb[j] < conf_min or cc[j] < conf_min:
            continue
        X = v12.v8.dlt_point(cams, {v12.B: xb[j], v12.C: xc[j]})
        if X is None or not np.isfinite(X).all():
            continue
        X = np.asarray(X, np.float64)
        if not (-320 <= X[0] <= 1250 and -720 <= X[1] <= 720 and -40 <= X[2] <= 390):
            continue
        errs = {}
        for label, uv0 in ((v12.B, xb[j]), (v12.C, xc[j])):
            uv, _, ok = v12.v8.project_metric(cams[label], X.reshape(1, 3))
            if bool(ok[0]):
                errs[label] = float(np.linalg.norm(uv[0] - uv0))
        if len(errs) != 2 or max(errs.values()) > 28.0:
            continue
        joints[j] = X
        conf[j] = float(min(cb[j], cc[j]))
        jq[str(j)] = {
            "world_cm": [float(x) for x in X],
            "broadcast_confidence": float(cb[j]),
            "right_above_rim_confidence": float(cc[j]),
            "per_view_error_px": errs,
        }

    measured = 0
    plausible = 0
    bone_rows = []
    # Broad human-segment gate: enough to reject impossible cross-player pairings
    # without imposing a body-size prior specific to one player.
    for a, b in v12.v10.DRAW_EDGES:
        if a not in joints or b not in joints:
            continue
        d = float(np.linalg.norm(joints[a] - joints[b]))
        measured += 1
        ok = 7.0 <= d <= 105.0
        plausible += int(ok)
        bone_rows.append({"a": int(a), "b": int(b), "length_cm": d, "plausible": bool(ok)})
    frac = float(plausible / measured) if measured else 0.0
    qa = {
        "joint_count": int(len(joints)),
        "measured_bones": int(measured),
        "plausible_bones": int(plausible),
        "plausible_bone_fraction": frac,
        "joints": jq,
        "bones": bone_rows,
    }
    if len(joints) < 8 or measured < 5 or frac < 0.60:
        raise RuntimeError(f"v23 Broadcast/RAR focal skeleton failed plausibility: {qa}")
    return joints, conf, qa


def _project_skeleton(cams, label, joints, joint_conf):
    xy = np.full((17, 2), np.nan, np.float64)
    cf = np.zeros(17, np.float64)
    for j, X in joints.items():
        uv, _, ok = v12.v8.project_metric(cams[label], np.asarray(X, np.float64).reshape(1, 3))
        if bool(ok[0]) and np.isfinite(uv[0]).all():
            xy[j] = uv[0]
            cf[j] = max(0.13, float(joint_conf.get(j, 0.13)))
    good = np.where(cf >= 0.12)[0]
    if len(good) < 6:
        raise RuntimeError(f"v23 too few focal skeleton joints project into {label}")
    pts = xy[good]
    x1, y1 = np.min(pts, axis=0); x2, y2 = np.max(pts, axis=0)
    bw, bh = max(12.0, x2 - x1), max(20.0, y2 - y1)
    box = np.asarray([
        max(0.0, x1 - 0.30 * bw - 8.0), max(0.0, y1 - 0.16 * bh - 8.0),
        min(v12.W - 1.0, x2 + 0.30 * bw + 8.0), min(v12.H - 1.0, y2 + 0.12 * bh + 8.0),
    ], np.float64)
    return xy, cf, box


def _not_white_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Reject bright, low-saturation uniform/court pixels while retaining black
    # uniform plus exposed skin/arms.  This is source-pixel classification only.
    white = (hsv[:, :, 2] >= 180) & (hsv[:, :, 1] <= 88)
    return ~white


def _clean_instance_mask(image, mask, min_fraction=0.28):
    m = mask.astype(bool)
    clean = m & _not_white_mask(image)
    if int(clean.sum()) >= max(120, int(min_fraction * max(1, int(m.sum())))):
        clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1) > 0
        return clean, True
    return m, False


def _append_bc_pose_instance(cams, label, instances, pose, pose_index):
    mi, mq = v12b._append_pose_prompt_instance(cams, label, instances, pose, pose_index)
    if mi is None:
        mi = v12b._find_mask_for_pose(instances, pose_index)
        if mi is None:
            return None, {**mq, "status": "V23_BC_MASK_FAILED"}
    inst = instances[mi]
    clean, used = _clean_instance_mask(v12.IMAGES[label], inst["mask"])
    inst["mask"] = clean
    mq = {**mq, "v23_white_pixel_rejection_applied": bool(used), "v23_mask_pixels": int(clean.sum())}
    return int(mi), mq


def _append_projected_lar_instance(cams, instances, joints, joint_conf):
    xy, cf, box = _project_skeleton(cams, v12.A, joints, joint_conf)
    actual = v12b.POSE_CACHE.get(v12.A)
    if actual is None:
        raise RuntimeError("v23 LAR RF-DETR cache missing")

    # Target projected skeleton at row 0; current LAR detections are included as
    # negative identity prompts, which is particularly useful for the white leaper
    # occluding the focal player.
    xys = [xy] + [np.asarray(x, np.float64) for x in actual["xy"]]
    cfs = [cf] + [np.asarray(x, np.float64) for x in actual["conf"]]
    boxes = [box] + [np.asarray(x, np.float64) for x in actual["boxes"]]
    dets = [1.0] + [float(x) for x in actual["det_conf"]]
    fake = {
        "xy": np.asarray(xys, np.float64),
        "conf": np.asarray(cfs, np.float64),
        "boxes": np.asarray(boxes, np.float64),
        "det_conf": np.asarray(dets, np.float64),
        "cov": None,
    }
    mask, mq = v12b._pose_prompt_mask(v12.IMAGES[v12.A], fake, 0)
    if mask is None:
        raise RuntimeError(f"v23 projected LAR focal mask failed: {mq}")
    mask, cleaned = _clean_instance_mask(v12.IMAGES[v12.A], mask, min_fraction=0.18)
    if int(mask.sum()) < 90:
        raise RuntimeError(f"v23 projected LAR focal mask too small: {int(mask.sum())}")

    ys, xs = np.where(mask)
    inst = {
        "score": 1.0,
        "mask": mask,
        "box": [float(x) for x in box],
        "foot_px": [int(round(float(np.median(xs[ys >= np.percentile(ys, 92)])))) if len(xs) else int(round(float(np.mean(box[[0,2]])))), int(ys.max())],
        "segmentation_source": "v23_bc_triangulated_projection_pose_prompt",
        "rfdetr_pose": {
            "pose_index": -230,
            "xy": xy.tolist(),
            "confidence": cf.tolist(),
            "detection_confidence": 1.0,
            "bbox": [float(x) for x in box],
            "segmentation_source": "v23_bc_triangulated_projection_pose_prompt",
        },
    }
    instances.append(inst)
    mi = len(instances) - 1
    qa = {
        **mq,
        "status": "V23_PROJECTED_LAR_MASK_OK",
        "mask_instance": int(mi),
        "projected_bbox": [float(x) for x in box],
        "projected_joint_count": int(np.sum(cf >= 0.12)),
        "mask_pixels": int(mask.sum()),
        "white_pixel_rejection_applied": bool(cleaned),
    }
    return int(mi), qa


def _dedupe_bc(tracks, instances, bi, ci):
    kept, dropped = [], []
    for ti, tr in enumerate(tracks):
        hits = []
        for label, pi in ((v12.B, bi), (v12.C, ci)):
            mi = tr.get(label)
            if mi is None or mi < 0 or mi >= len(instances[label]):
                continue
            p = instances[label][mi].get("rfdetr_pose")
            if p and int(p.get("pose_index", -999)) == int(pi):
                hits.append(label)
        if hits:
            dropped.append({"track_index": int(ti), "matching_focal_pose_views": hits})
        else:
            kept.append(tr)
    return kept, dropped


def attach_right_above_rim_v23(cams, instances, mb, exact_qa):
    global _V23_AUDIT
    tracks, audit = v22._ORIGINAL_ATTACH_FIXED(cams, instances, mb, exact_qa)

    pb = v12b.POSE_CACHE.get(v12.B)
    pc = v12b.POSE_CACHE.get(v12.C)
    if pb is None or pc is None:
        raise RuntimeError("v23 Broadcast/RAR pose cache missing")
    bi, bq = v22._select_focal_pose(v12.B, pb)
    ci, cq = v22._select_focal_pose(v12.C, pc)
    joints, jconf, jq = _triangulate_bc_focal(cams, pb, bi, pc, ci)

    bmi, bmq = _append_bc_pose_instance(cams, v12.B, instances[v12.B], pb, bi)
    cmi, cmq = _append_bc_pose_instance(cams, v12.C, instances[v12.C], pc, ci)
    if bmi is None or cmi is None:
        raise RuntimeError(f"v23 failed focal B/C masks: B={bmq} C={cmq}")
    ami, amq = _append_projected_lar_instance(cams, instances[v12.A], joints, jconf)

    tracks, dropped = _dedupe_bc(tracks, instances, bi, ci)
    focal_track = {
        v12.A: int(ami), v12.B: int(bmi), v12.C: int(cmi),
        "two_view_joints": {int(k): np.asarray(v, np.float64) for k, v in joints.items()},
        "appearance": {"class": "dark"},
        "focal_subject_lock": "black_uniform_12_bc_geometry_lar_projection",
        "v23_focal_subject": True,
        "v23_bc_joints": {int(k): np.asarray(v, np.float64) for k, v in joints.items()},
        "v23_bc_joint_qa": jq,
        "rar_attachment": {"method": "v23_broadcast_rar_triangulated_focal_identity"},
    }
    tracks.append(focal_track)

    _V23_AUDIT = {
        "status": "V23_FOCAL_BC_GEOMETRY_LAR_PROJECTED_MASK",
        "identity_basis": "black-uniform focal player selected independently in Broadcast and Right Above Rim; LAR recovered by projecting their triangulated 3-D skeleton",
        "broadcast_pose_selection": bq,
        "right_above_rim_pose_selection": cq,
        "broadcast_pose_index": int(bi),
        "right_above_rim_pose_index": int(ci),
        "bc_skeleton": jq,
        "mask_instances": {v12.A: int(ami), v12.B: int(bmi), v12.C: int(cmi)},
        "mask_qa": {v12.A: amq, v12.B: bmq, v12.C: cmq},
        "dropped_duplicate_tracks": dropped,
    }
    # v22's downstream diagnostics read this global as well.
    v22._FOCAL_AUDIT = _V23_AUDIT
    audit["v23_focal_subject_lock"] = _V23_AUDIT
    return tracks, audit


def triangulate_track_joints_v23(cams, track, instances, conf_min=0.20):
    if track.get("v23_focal_subject") and track.get("v23_bc_joints"):
        joints = {int(k): np.asarray(v, np.float64) for k, v in track["v23_bc_joints"].items()}
        qa = {}
        for j, X in joints.items():
            qa[v12.v10.COCO_NAMES[j]] = {
                "world_cm": [float(x) for x in X],
                "method": "broadcast_right_above_rim_two_view_focal",
                "views_observed": [v12.B, v12.C],
            }
        return joints, qa
    return _ORIG_TRIANGULATE(cams, track, instances, conf_min=conf_min)


def _focal_visible_appearance(cams, angle):
    C0, R0, K0 = cams[v12.A]
    Rt, Ct = v12.base.orbit_pose(C0, R0, v12.RIM, float(angle))
    neutral = np.full((v12.H, v12.W, 3), 112, np.uint8)
    all_img, _ = v14.render_triangles_mask_safe(neutral, cams, K0, Rt, Ct, v22._ALL_MESHES)
    others = [m for m in v22._ALL_MESHES if m is not v22._FOCAL_MESH]
    wo_img, _ = v14.render_triangles_mask_safe(neutral, cams, K0, Rt, Ct, others)
    delta = np.max(cv2.absdiff(all_img, wo_img), axis=2) > 5
    pix = all_img[delta]
    if len(pix) == 0:
        return {"visible_pixels": 0, "dark_fraction_v_lt_120": 0.0, "white_fraction": 1.0, "median_v": 255.0}, all_img, delta
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    vv = hsv[:, 2].astype(np.float32); ss = hsv[:, 1].astype(np.float32)
    q = {
        "visible_pixels": int(len(pix)),
        "dark_fraction_v_lt_120": float(np.mean(vv < 120.0)),
        "white_fraction": float(np.mean((vv >= 180.0) & (ss <= 88.0))),
        "median_v": float(np.median(vv)),
        "p75_v": float(np.percentile(vv, 75)),
    }
    return q, all_img, delta


def post_focal_visibility_v23(out: Path, cams):
    rows = _ORIG_POST(out, cams)
    app = {}
    for ang in (5, 10, 15, 20, 25):
        q, im, delta = _focal_visible_appearance(cams, ang)
        app[str(ang)] = q
        vis = im.copy()
        vis[~delta] = 112
        cv2.imwrite(str(out / f"v23_focal_visible_rgb_{ang:02d}deg.png"), vis)
    for row in rows:
        a = int(row["angle_deg"])
        if a > 0:
            row["v23_visible_appearance"] = app[str(a)]
    return rows


def _arg_path(name):
    i = sys.argv.index(name)
    return Path(sys.argv[i + 1])


def main():
    global _V23_AUDIT
    _V23_AUDIT = {}
    v12.triangulate_track_joints = triangulate_track_joints_v23
    v22.attach_right_above_rim_focal_locked = attach_right_above_rim_v23
    v22._post_focal_visibility = post_focal_visibility_v23

    assert (int(v12.W), int(v12.H)) == (960, 540)
    v22.main()

    out = _arg_path("--out")
    qp = out / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    rows = q.get("v22_focal_subject", {}).get("visibility_by_angle", [])
    q["v23_focal_subject"] = {
        "resolution": [960, 540],
        "audit": _V23_AUDIT,
        "accepted_focal_mesh_count": q.get("v22_focal_subject", {}).get("accepted_focal_mesh_count", 0),
        "accepted_focal_mesh_views": q.get("v22_focal_subject", {}).get("accepted_focal_mesh_views", []),
        "visibility_by_angle": rows,
        "identity_correction": "LAR focal pose detector is not trusted under rim occlusion; focal 3-D joints come from independently selected dark Broadcast/RAR poses and recover only visible LAR source pixels by projection",
        "generated_texture": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q["v23_focal_subject"], indent=2), flush=True)


if __name__ == "__main__":
    main()
