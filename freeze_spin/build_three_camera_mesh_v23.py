from __future__ import annotations

"""v23: focal #12 recovery under severe LAR/RAR pose occlusion.

v22 demonstrated why a generic three-view human-track gate is insufficient: the
Left Above Rim RF-DETR pose selected the white leaper in front of the focal dark
player.  The first v23 attempt then showed that the same focal player is cleanly
identified in Broadcast and Right Above Rim, but the overhead keypoint detector
only has five mutually confident joints because the bodies overlap at the rim.

This revision therefore uses the information that is actually reliable:

* Broadcast supplies the focal player's detailed 2-D pose.
* Right Above Rim supplies a clean focal identity mask and a handful of direct
  keypoint correspondences.
* Reliable Broadcast/RAR joints establish metric depth.
* Missing Broadcast joints are back-projected along their metric camera rays and
  their depth is selected by intersection with the focal RAR silhouette, with a
  conservative human-segment continuity prior.
* That recovered 3-D skeleton is projected into Left Above Rim to recover only
  the visible dark-player source pixels behind the white occluder.
* The anatomical volume then uses the recovered skeleton plus focal silhouettes
  from all three solved cameras.

No appearance is generated.  All rendered RGB remains sampled from official
native source frames.  Resolution is hard-locked to 960x540; no UHD/upscale path.
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
    return np.asarray(pose["xy"][idx], np.float64), np.asarray(pose["conf"][idx], np.float64)


def _not_white_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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


def _camera_depth(cam, X):
    C, R, _K = cam
    s = float(v12.v3.forward_sign(R, C))
    Xc = R @ (np.asarray(X, np.float64) - C)
    return float(s * Xc[2])


def _world_on_ray(cam, uv, depth):
    C, R, K = cam
    s = float(v12.v3.forward_sign(R, C))
    u, v = map(float, uv)
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    Xc = s * np.asarray([xn * depth, yn * depth, depth], np.float64)
    return R.T @ Xc + C


def _bone_neighbours(j):
    out = []
    for a, b in v12.v10.DRAW_EDGES:
        if a == j:
            out.append(int(b))
        elif b == j:
            out.append(int(a))
    return out


def _direct_bc_joints(cams, pb, bi, pc, ci):
    xb, cb = _pose_arrays(pb, bi)
    xc, cc = _pose_arrays(pc, ci)
    joints, conf, qa = {}, {}, {}
    for j in range(min(17, len(xb), len(xc))):
        if cb[j] < 0.10 or cc[j] < 0.055:
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
        if len(errs) != 2 or max(errs.values()) > 38.0:
            continue
        joints[j] = X
        conf[j] = float(min(cb[j], max(cc[j], 0.08)))
        qa[j] = {
            "method": "direct_broadcast_rar_keypoints",
            "world_cm": [float(x) for x in X],
            "broadcast_confidence": float(cb[j]),
            "right_above_rim_confidence": float(cc[j]),
            "per_view_error_px": errs,
        }
    return joints, conf, qa


def _recover_broadcast_rays_with_rar_mask(cams, pb, bi, rar_mask, joints, conf, qa):
    xb, cb = _pose_arrays(pb, bi)
    if len(joints) < 3:
        raise RuntimeError(f"v23 needs at least three direct B/RAR depth anchors, got {len(joints)}")

    depths = np.asarray([_camera_depth(cams[v12.B], X) for X in joints.values()], np.float64)
    depths = depths[np.isfinite(depths) & (depths > 50)]
    if not len(depths):
        raise RuntimeError("v23 no valid Broadcast-camera depth anchors")
    z_med = float(np.median(depths))
    z_span = max(150.0, min(280.0, 2.8 * float(np.std(depths)) + 150.0))

    m = rar_mask.astype(bool)
    inside_dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    outside_dt = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)

    # Torso/hips first, then limbs/head so recovered neighbours can constrain the
    # next joint's depth.  Direct joints remain untouched.
    order = [11, 12, 5, 6, 13, 14, 7, 8, 15, 16, 9, 10, 0, 1, 2, 3, 4]
    for j in order:
        if j in joints or j >= len(cb) or cb[j] < 0.085 or not np.isfinite(xb[j]).all():
            continue
        best = None
        for z in np.linspace(max(80.0, z_med - z_span), z_med + z_span, 181):
            X = _world_on_ray(cams[v12.B], xb[j], float(z))
            if not (-320 <= X[0] <= 1250 and -720 <= X[1] <= 720 and -40 <= X[2] <= 390):
                continue
            uv, _, valid = v12.v8.project_metric(cams[v12.C], X.reshape(1, 3))
            if not bool(valid[0]) or not np.isfinite(uv[0]).all():
                continue
            u, v = np.rint(uv[0]).astype(int)
            if not (0 <= u < v12.W and 0 <= v < v12.H):
                continue
            signed = float(inside_dt[v, u]) if m[v, u] else -float(outside_dt[v, u])
            score = signed - 0.010 * abs(float(z) - z_med)

            # Broad connected-bone continuity; it nudges the ray solution toward
            # anatomy but does not manufacture an unseen pose.
            nrows = []
            for n in _bone_neighbours(j):
                if n not in joints:
                    continue
                d = float(np.linalg.norm(X - joints[n]))
                plausible = 7.0 <= d <= 110.0
                score += 1.2 if plausible else -0.035 * min(180.0, abs(d - 58.0))
                nrows.append((n, d, plausible))
            candidate = (score, signed, -abs(float(z) - z_med), X, uv[0], nrows, float(z))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or best[1] < -5.0:
            continue
        score, signed, _prior, X, uvc, nrows, zsel = best
        joints[j] = np.asarray(X, np.float64)
        conf[j] = float(cb[j])
        qa[j] = {
            "method": "broadcast_ray_depth_from_rar_focal_silhouette",
            "world_cm": [float(x) for x in X],
            "broadcast_confidence": float(cb[j]),
            "rar_signed_mask_distance_px": float(signed),
            "rar_projected_px": [float(x) for x in uvc],
            "broadcast_camera_depth_cm": float(zsel),
            "neighbour_bones": [
                {"joint": int(n), "length_cm": float(d), "plausible": bool(ok)} for n, d, ok in nrows
            ],
        }
    return z_med, z_span


def _triangulate_bc_focal(cams, pose_b, bi, pose_c, ci, rar_mask):
    joints, conf, jq = _direct_bc_joints(cams, pose_b, bi, pose_c, ci)
    direct_count = int(len(joints))
    z_med, z_span = _recover_broadcast_rays_with_rar_mask(cams, pose_b, bi, rar_mask, joints, conf, jq)

    measured = 0
    plausible = 0
    bone_rows = []
    for a, b in v12.v10.DRAW_EDGES:
        if a not in joints or b not in joints:
            continue
        d = float(np.linalg.norm(joints[a] - joints[b]))
        measured += 1
        ok = 7.0 <= d <= 115.0
        plausible += int(ok)
        bone_rows.append({"a": int(a), "b": int(b), "length_cm": d, "plausible": bool(ok)})
    frac = float(plausible / measured) if measured else 0.0
    qaj = {str(k): v for k, v in jq.items()}
    qa = {
        "joint_count": int(len(joints)),
        "direct_keypoint_joint_count": direct_count,
        "silhouette_ray_recovered_joint_count": int(len(joints) - direct_count),
        "broadcast_anchor_depth_median_cm": z_med,
        "broadcast_anchor_depth_search_halfspan_cm": z_span,
        "measured_bones": int(measured),
        "plausible_bones": int(plausible),
        "plausible_bone_fraction": frac,
        "joints": qaj,
        "bones": bone_rows,
    }
    if len(joints) < 9 or measured < 7 or frac < 0.52:
        raise RuntimeError(f"v23 Broadcast/RAR silhouette-resolved focal skeleton failed plausibility: {qa}")
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
    if len(good) < 7:
        raise RuntimeError(f"v23 too few recovered focal joints project into {label}")
    pts = xy[good]
    x1, y1 = np.min(pts, axis=0); x2, y2 = np.max(pts, axis=0)
    bw, bh = max(12.0, x2 - x1), max(20.0, y2 - y1)
    box = np.asarray([
        max(0.0, x1 - 0.30 * bw - 8.0), max(0.0, y1 - 0.16 * bh - 8.0),
        min(v12.W - 1.0, x2 + 0.30 * bw + 8.0), min(v12.H - 1.0, y2 + 0.12 * bh + 8.0),
    ], np.float64)
    return xy, cf, box


def _append_projected_lar_instance(cams, instances, joints, joint_conf):
    xy, cf, box = _project_skeleton(cams, v12.A, joints, joint_conf)
    actual = v12b.POSE_CACHE.get(v12.A)
    if actual is None:
        raise RuntimeError("v23 LAR RF-DETR cache missing")

    xys = [xy] + [np.asarray(x, np.float64) for x in actual["xy"]]
    cfs = [cf] + [np.asarray(x, np.float64) for x in actual["conf"]]
    boxes = [box] + [np.asarray(x, np.float64) for x in actual["boxes"]]
    dets = [1.0] + [float(x) for x in actual["det_conf"]]
    fake = {
        "xy": np.asarray(xys, np.float64), "conf": np.asarray(cfs, np.float64),
        "boxes": np.asarray(boxes, np.float64), "det_conf": np.asarray(dets, np.float64), "cov": None,
    }
    mask, mq = v12b._pose_prompt_mask(v12.IMAGES[v12.A], fake, 0)
    if mask is None:
        raise RuntimeError(f"v23 projected LAR focal mask failed: {mq}")
    mask, cleaned = _clean_instance_mask(v12.IMAGES[v12.A], mask, min_fraction=0.14)
    if int(mask.sum()) < 75:
        raise RuntimeError(f"v23 projected LAR focal mask too small: {int(mask.sum())}")

    ys, xs = np.where(mask)
    ycut = float(np.percentile(ys, 92))
    near = xs[ys >= ycut]
    foot_x = int(round(float(np.median(near)))) if len(near) else int(round(float(np.mean(box[[0, 2]]))))
    inst = {
        "score": 1.0,
        "mask": mask,
        "box": [float(x) for x in box],
        "foot_px": [foot_x, int(ys.max())],
        "segmentation_source": "v23_bc_silhouette_resolved_projection_pose_prompt",
        "rfdetr_pose": {
            "pose_index": -230, "xy": xy.tolist(), "confidence": cf.tolist(),
            "detection_confidence": 1.0, "bbox": [float(x) for x in box],
            "segmentation_source": "v23_bc_silhouette_resolved_projection_pose_prompt",
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

    bmi, bmq = _append_bc_pose_instance(cams, v12.B, instances[v12.B], pb, bi)
    cmi, cmq = _append_bc_pose_instance(cams, v12.C, instances[v12.C], pc, ci)
    if bmi is None or cmi is None:
        raise RuntimeError(f"v23 failed focal B/C masks: B={bmq} C={cmq}")

    joints, jconf, jq = _triangulate_bc_focal(cams, pb, bi, pc, ci, instances[v12.C][cmi]["mask"])
    ami, amq = _append_projected_lar_instance(cams, instances[v12.A], joints, jconf)

    tracks, dropped = _dedupe_bc(tracks, instances, bi, ci)
    focal_track = {
        v12.A: int(ami), v12.B: int(bmi), v12.C: int(cmi),
        "two_view_joints": {int(k): np.asarray(v, np.float64) for k, v in joints.items()},
        "appearance": {"class": "dark"},
        "focal_subject_lock": "black_uniform_12_broadcast_pose_rar_silhouette_lar_projection",
        "v23_focal_subject": True,
        "v23_bc_joints": {int(k): np.asarray(v, np.float64) for k, v in joints.items()},
        "v23_bc_joint_qa": jq,
        "rar_attachment": {"method": "v23_broadcast_pose_rar_silhouette_metric_depth"},
    }
    tracks.append(focal_track)

    _V23_AUDIT = {
        "status": "V23_FOCAL_BROADCAST_RAY_RAR_SILHOUETTE_LAR_PROJECTED_MASK",
        "identity_basis": "dark focal player selected independently in Broadcast and RAR; Broadcast pose rays receive metric depth from direct RAR correspondences plus the same player's RAR silhouette; recovered skeleton is then projected into LAR",
        "broadcast_pose_selection": bq,
        "right_above_rim_pose_selection": cq,
        "broadcast_pose_index": int(bi),
        "right_above_rim_pose_index": int(ci),
        "bc_skeleton": jq,
        "mask_instances": {v12.A: int(ami), v12.B: int(bmi), v12.C: int(cmi)},
        "mask_qa": {v12.A: amq, v12.B: bmq, v12.C: cmq},
        "dropped_duplicate_tracks": dropped,
    }
    v22._FOCAL_AUDIT = _V23_AUDIT
    audit["v23_focal_subject_lock"] = _V23_AUDIT
    return tracks, audit


def triangulate_track_joints_v23(cams, track, instances, conf_min=0.20):
    if track.get("v23_focal_subject") and track.get("v23_bc_joints"):
        joints = {int(k): np.asarray(v, np.float64) for k, v in track["v23_bc_joints"].items()}
        qa = {}
        for j, X in joints.items():
            method = track.get("v23_bc_joint_qa", {}).get("joints", {}).get(str(j), {}).get("method", "broadcast_rar_focal")
            qa[v12.v10.COCO_NAMES[j]] = {
                "world_cm": [float(x) for x in X],
                "method": method,
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
        "identity_correction": "LAR pose detector is not trusted under rim occlusion; focal metric skeleton comes from Broadcast pose plus RAR direct correspondences/silhouette and recovers only visible LAR source pixels by projection",
        "generated_texture": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q["v23_focal_subject"], indent=2), flush=True)


if __name__ == "__main__":
    main()
