from __future__ import annotations

"""Three-camera high-standard hypothesis test v12.

This experiment explicitly tests whether the three already-solved physical cameras
(Left Above Rim, Broadcast, Right Above Rim) can support a public-quality static
0..25 degree free-view arc at the exact v11 visual state.

Key changes from v10/v11:
- v11 exact-state frames are treated as fixed inputs;
- Left/Broadcast identity tracks are attached to Right-Above-Rim by projecting
  triangulated two-view joints into the overhead camera, rather than requiring
  the old foot/silhouette-cone association to work from the overhead view;
- the strict v11 three-view RF-DETR correspondence is used as a hard identity
  anchor when it maps cleanly to segmentation masks;
- each player is reconstructed as a smooth anatomy-constrained silhouette volume
  and converted to a triangle mesh with marching cubes;
- mesh colour is sampled only from the assigned real source views;
- the distant arena is source-grounded with an infinity-depth rotational warp,
  while court/backboard/rim remain metric geometry.

No AI upscaling, generated texture, invented player pixels, optical-flow morph or
camera cross-fade is used. Output is six native 960x540 stills only; animation is
intentionally deferred until the static hypothesis passes visual QA.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.measure import marching_cubes
from moge.model.v2 import MoGeModel
from rfdetr import RFDETRKeypointPreview
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v4 as v4
from freeze_spin import build_three_camera_diagnostic_v5 as v5
from freeze_spin import build_three_camera_diagnostic_v6 as v6
from freeze_spin import build_three_camera_semantic_v9 as v9
from freeze_spin import build_three_camera_rfdetr_v10 as v10
from freeze_spin import build_three_camera_volumetric_v8 as v8
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W, H = base.W, base.H
RIM = base.RIM.astype(np.float64)
CAMERAS = ("Left Above Rim", "Broadcast", "Right Above Rim")
A, B, C = CAMERAS

CAPSULES = [
    (5, 6, 13.0),
    (5, 7, 10.0), (7, 9, 8.5),
    (6, 8, 10.0), (8, 10, 8.5),
    (5, 11, 17.0), (6, 12, 17.0),
    (11, 12, 15.0),
    (11, 13, 12.0), (13, 15, 9.5),
    (12, 14, 12.0), (14, 16, 9.5),
    (0, 5, 12.0), (0, 6, 12.0),
]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + bb - inter)


def find_frame(root: Path, label: str) -> Path:
    token = label.replace(" ", "_")
    xs = sorted(root.rglob(f"*{token}*.png"))
    if label == "Broadcast":
        xs = [p for p in xs if "Mobile" not in p.name and "Other" not in p.name]
    xs = [p for p in xs if "selected_pose" not in p.name and "rfdetr_pose" not in p.name and "montage" not in p.name]
    if len(xs) != 1:
        raise RuntimeError(f"expected exactly one selected {label} frame; found {xs}")
    return xs[0]


def attach_rfdetr_poses(pose_model, image, instances):
    v10.POSE_MODEL = pose_model
    pose = v10._pose_predict(image)
    mat = np.zeros((len(instances), len(pose["boxes"])), np.float64)
    for i, inst in enumerate(instances):
        for j, box in enumerate(pose["boxes"]):
            mat[i, j] = iou(inst["box"], box)
    matched = []
    if mat.size:
        rr, cc = linear_sum_assignment(-mat)
        for i, j in zip(rr, cc):
            i, j = int(i), int(j)
            s = float(mat[i, j])
            if s < 0.08:
                continue
            instances[i]["rfdetr_pose"] = {
                "pose_index": j,
                "xy": pose["xy"][j].tolist(),
                "confidence": pose["conf"][j].tolist(),
                "detection_confidence": float(pose["det_conf"][j]),
                "bbox": pose["boxes"][j].tolist(),
                "mask_pose_iou": s,
            }
            matched.append({"mask_instance": i, "pose_index": j, "iou": s})
    return pose, matched


def appearance_descriptor(image, inst):
    mask = inst["mask"].astype(bool)
    x1, y1, x2, y2 = [int(round(x)) for x in inst["box"]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return {"class": "unknown", "median_v": 0.0, "median_s": 0.0, "dark_fraction": 0.0, "light_fraction": 0.0}
    yy2 = min(y2, int(round(y1 + 0.68 * (y2 - y1 + 1))))
    roi_mask = mask[y1:yy2 + 1, x1:x2 + 1]
    pix = image[y1:yy2 + 1, x1:x2 + 1][roi_mask]
    if len(pix) < 20:
        pix = image[mask]
    if len(pix) < 20:
        return {"class": "unknown", "median_v": 0.0, "median_s": 0.0, "dark_fraction": 0.0, "light_fraction": 0.0}
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    v = hsv[:, 2].astype(np.float32); s = hsv[:, 1].astype(np.float32)
    medv, meds = float(np.median(v)), float(np.median(s))
    dark = float(np.mean(v < 105)); light = float(np.mean(v > 175))
    if dark >= 0.48 or medv < 100:
        cls = "dark"
    elif light >= 0.46 or medv > 165:
        cls = "light"
    else:
        cls = "mid"
    return {"class": cls, "median_v": medv, "median_s": meds, "dark_fraction": dark, "light_fraction": light}


def hard_left_broadcast_tracks(cams, instances):
    v10.GLOBAL_CAMS = cams
    _, supports, coarse_bounds = v9.coarse_instance_support(cams, instances, step=10.0)
    _mapping, rows = v10._one_to_one_pose(A, B, instances, supports)
    mapping = {}
    audit = []
    for row in rows:
        i, j = int(row["primary_instance"]), int(row["other_instance"])
        base_score = float(row.get("base_score", -1e6))
        foot = row.get("foot_distance_cm")
        geom_ok = row.get("intersection_voxels", 0) >= 18 and row.get("cosine", 0.0) >= 0.004
        base_ok = base_score > 4.2
        floor_ok = foot is None or float(foot) <= 360.0
        pose = row.get("pose", {})
        pose_ok = True
        if pose.get("pose_available") and pose.get("measured_major_bones", 0) >= 3:
            pose_ok = (pose.get("plausible_bone_fraction") or 0.0) >= 0.34
        accepted = bool(base_ok and geom_ok and floor_ok and pose_ok)
        r = dict(row)
        r["v12_left_broadcast_gate"] = {
            "base_score_gt_4_2": base_ok,
            "silhouette_geometry_ok": bool(geom_ok),
            "floor_anchor_le_360cm": bool(floor_ok),
            "pose_plausibility_ok": bool(pose_ok),
        }
        r["accepted"] = accepted
        audit.append(r)
        if accepted:
            mapping[i] = j
    return mapping, audit, coarse_bounds


def two_view_joints(cams, inst_a, inst_b, conf_min=0.20):
    pa, pb = inst_a.get("rfdetr_pose"), inst_b.get("rfdetr_pose")
    if not pa or not pb:
        return {}
    xa, xb = np.asarray(pa["xy"], np.float64), np.asarray(pb["xy"], np.float64)
    ca, cb = np.asarray(pa["confidence"], np.float64), np.asarray(pb["confidence"], np.float64)
    out = {}
    for j in range(min(17, len(xa), len(xb))):
        if ca[j] < conf_min or cb[j] < conf_min:
            continue
        X = v8.dlt_point(cams, {A: xa[j], B: xb[j]})
        if X is None or not np.isfinite(X).all():
            continue
        if not (-320 <= X[0] <= 1250 and -720 <= X[1] <= 720 and -40 <= X[2] <= 390):
            continue
        out[j] = np.asarray(X, np.float64)
    return out


def projected_rar_cost(cams, joints, rar_inst, desc_ab, desc_rar, conf_min=0.16):
    p = rar_inst.get("rfdetr_pose")
    if not p or not joints:
        return 1e6, {"joint_count": 0, "median_px": 999.0, "p75_px": 999.0, "good45": 0, "appearance_mismatch": False}
    xy = np.asarray(p["xy"], np.float64)
    cf = np.asarray(p["confidence"], np.float64)
    errs = []
    for j, X in joints.items():
        if j >= len(cf) or cf[j] < conf_min:
            continue
        uv, _, valid = v8.project_metric(cams[C], X.reshape(1, 3))
        if not valid[0]:
            continue
        errs.append(float(np.linalg.norm(uv[0] - xy[j])))
    if not errs:
        med, p75, good = 999.0, 999.0, 0
    else:
        e = np.asarray(errs, np.float64)
        med, p75, good = float(np.median(e)), float(np.percentile(e, 75)), int(np.sum(e <= 45.0))
    ca, cr = desc_ab.get("class", "unknown"), desc_rar.get("class", "unknown")
    mismatch = ca in ("dark", "light") and cr in ("dark", "light") and ca != cr
    cost = med + 0.25 * p75 - 2.5 * good - 1.0 * len(errs) + (55.0 if mismatch else 0.0)
    return float(cost), {
        "joint_count": int(len(errs)), "median_px": med, "p75_px": p75,
        "good45": good, "appearance_mismatch": bool(mismatch),
        "left_broadcast_class": ca, "rar_class": cr,
    }


def pose_to_mask(instances, pose_index):
    for i, inst in enumerate(instances):
        p = inst.get("rfdetr_pose")
        if p and int(p.get("pose_index", -1)) == int(pose_index):
            return i
    return None


def attach_right_above_rim(cams, instances, mb, exact_qa):
    base_tracks = []
    for li, bi in mb.items():
        ja = two_view_joints(cams, instances[A][li], instances[B][bi])
        da = appearance_descriptor(IMAGES[A], instances[A][li])
        db = appearance_descriptor(IMAGES[B], instances[B][bi])
        if da["class"] == db["class"]:
            cls = da["class"]
        elif da["class"] in ("dark", "light") and db["class"] == "mid":
            cls = da["class"]
        elif db["class"] in ("dark", "light") and da["class"] == "mid":
            cls = db["class"]
        else:
            cls = "mid"
        desc = {"class": cls, "median_v": 0.5 * (da["median_v"] + db["median_v"])}
        base_tracks.append({"Left Above Rim": li, "Broadcast": bi, "Right Above Rim": None, "two_view_joints": ja, "appearance": desc})

    forced = {}
    for e in exact_qa.get("selected", {}).get("assigned_tracks", []):
        if not e.get("strict"):
            continue
        ids = e.get("ids", {})
        lm = pose_to_mask(instances[A], ids.get(A, -999))
        bm = pose_to_mask(instances[B], ids.get(B, -999))
        rm = pose_to_mask(instances[C], ids.get(C, -999))
        if lm is None or bm is None or rm is None:
            continue
        for ti, tr in enumerate(base_tracks):
            if tr[A] == lm and tr[B] == bm:
                forced[ti] = rm
                break
        else:
            base_tracks.append({A: lm, B: bm, C: rm, "two_view_joints": two_view_joints(cams, instances[A][lm], instances[B][bm]), "appearance": {"class": "unknown"}, "forced_exact_sync": True})
            forced[len(base_tracks) - 1] = rm

    desc_r = [appearance_descriptor(IMAGES[C], x) for x in instances[C]]
    nT, nR = len(base_tracks), len(instances[C])
    cost = np.full((nT, nR), 1e6, np.float64)
    details = {}
    for ti, tr in enumerate(base_tracks):
        for ri in range(nR):
            cst, q = projected_rar_cost(cams, tr["two_view_joints"], instances[C][ri], tr["appearance"], desc_r[ri])
            cost[ti, ri] = cst
            details[(ti, ri)] = q

    used_r = set(forced.values())
    for ti, ri in forced.items():
        base_tracks[ti][C] = ri
        base_tracks[ti]["rar_attachment"] = {"method": "v11_strict_three_view_identity", **details.get((ti, ri), {})}

    free_t = [i for i in range(nT) if i not in forced]
    free_r = [i for i in range(nR) if i not in used_r]
    if free_t and free_r:
        sub = cost[np.ix_(free_t, free_r)]
        rr, cc = linear_sum_assignment(sub)
        for a, b in zip(rr, cc):
            ti, ri = free_t[int(a)], free_r[int(b)]
            q = details[(ti, ri)]
            accept = bool(q["joint_count"] >= 3 and q["median_px"] <= 70.0 and q["p75_px"] <= 105.0 and not q["appearance_mismatch"])
            if accept:
                base_tracks[ti][C] = ri
                base_tracks[ti]["rar_attachment"] = {"method": "projected_two_view_skeleton_plus_appearance", **q, "cost": float(cost[ti, ri])}
            else:
                base_tracks[ti]["rar_attachment"] = {"method": "rejected", **q, "cost": float(cost[ti, ri])}

    audit = {
        "forced_from_v11_strict": {str(k): int(v) for k, v in forced.items()},
        "rar_instance_appearance": desc_r,
        "tracks": [
            {
                "track_id": i,
                "instances": {k: (int(tr[k]) if tr.get(k) is not None else None) for k in CAMERAS},
                "rar_attachment": tr.get("rar_attachment", {"method": "none"}),
                "two_view_joint_count": int(len(tr.get("two_view_joints", {}))),
                "appearance": tr.get("appearance", {}),
            }
            for i, tr in enumerate(base_tracks)
        ],
    }
    return base_tracks, audit


def triangulate_track_joints(cams, track, instances, conf_min=0.20):
    joints, qa = {}, {}
    for j in range(17):
        obs = {}
        for label in CAMERAS:
            idx = track.get(label)
            if idx is None:
                continue
            p = instances[label][idx].get("rfdetr_pose")
            if not p:
                continue
            xy = np.asarray(p["xy"], np.float64); cf = np.asarray(p["confidence"], np.float64)
            if j < len(cf) and cf[j] >= conf_min:
                obs[label] = xy[j]
        if len(obs) < 2:
            continue
        candidates = []
        if A in obs and B in obs:
            X = v8.dlt_point(cams, {A: obs[A], B: obs[B]})
            if X is not None and np.isfinite(X).all():
                candidates.append(("left_broadcast", X))
        if len(obs) >= 3:
            X = v8.dlt_point(cams, obs)
            if X is not None and np.isfinite(X).all():
                candidates.append(("all_three", X))
        if not candidates:
            labels = list(obs)[:2]
            X = v8.dlt_point(cams, {labels[0]: obs[labels[0]], labels[1]: obs[labels[1]]})
            if X is not None and np.isfinite(X).all():
                candidates.append(("fallback_pair", X))
        best = None
        for method, X in candidates:
            if not (-320 <= X[0] <= 1250 and -720 <= X[1] <= 720 and -40 <= X[2] <= 390):
                continue
            errs = {}
            for label, uv0 in obs.items():
                uv, _, valid = v8.project_metric(cams[label], np.asarray(X).reshape(1, 3))
                if valid[0]:
                    errs[label] = float(np.linalg.norm(uv[0] - uv0))
            if len(errs) < 2:
                continue
            e = np.asarray(list(errs.values()), np.float64)
            med = float(np.median(e)); p75 = float(np.percentile(e, 75))
            penalty = med + 0.25 * p75
            if method == "all_three" and len(errs) == 3 and med <= 24.0 and p75 <= 38.0:
                penalty -= 4.0
            if best is None or penalty < best[0]:
                best = (penalty, method, np.asarray(X, np.float64), errs, med, p75)
        if best is None:
            continue
        _, method, X, errs, med, p75 = best
        joints[j] = X
        qa[v10.COCO_NAMES[j]] = {
            "world_cm": [float(x) for x in X], "method": method,
            "views_observed": list(obs), "per_view_error_px": errs,
            "median_reprojection_px": med, "p75_reprojection_px": p75,
        }
    return joints, qa


def point_segment_ratio(pts, a, b, radius):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    ab = b - a; den = float(np.dot(ab, ab))
    if den < 1e-8:
        return np.linalg.norm(pts - a.reshape(1, 3), axis=1) / float(radius)
    t = ((pts - a.reshape(1, 3)) @ ab) / den
    t = np.clip(t, 0.0, 1.0)
    q = a.reshape(1, 3) + t[:, None] * ab.reshape(1, 3)
    return np.linalg.norm(pts - q, axis=1) / float(radius)


def build_anatomical_mesh(cams, track, instances, voxel_cm=2.5):
    joints, joint_qa = triangulate_track_joints(cams, track, instances)
    if len(joints) < 6:
        return None, {"status": "REJECTED_TOO_FEW_JOINTS", "joint_count": int(len(joints)), "joints": joint_qa}
    J = np.asarray(list(joints.values()), np.float64)
    x0, y0, z0 = np.min(J, axis=0) - np.asarray([34.0, 34.0, 34.0])
    x1, y1, z1 = np.max(J, axis=0) + np.asarray([34.0, 34.0, 34.0])
    x0, x1 = max(-300.0, x0), min(1180.0, x1)
    y0, y1 = max(-620.0, y0), min(620.0, y1)
    z0, z1 = max(-8.0, z0), min(360.0, z1)
    xs = np.arange(x0, x1 + 0.5 * voxel_cm, voxel_cm, np.float32)
    ys = np.arange(y0, y1 + 0.5 * voxel_cm, voxel_cm, np.float32)
    zs = np.arange(z0, z1 + 0.5 * voxel_cm, voxel_cm, np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    ratio = np.full(len(pts), np.inf, np.float32)
    used_segments = 0
    for a, b, r in CAPSULES:
        if a in joints and b in joints:
            ratio = np.minimum(ratio, point_segment_ratio(pts, joints[a], joints[b], r).astype(np.float32))
            used_segments += 1
    if 5 in joints and 6 in joints and 11 in joints and 12 in joints:
        sh = 0.5 * (joints[5] + joints[6]); hp = 0.5 * (joints[11] + joints[12])
        ratio = np.minimum(ratio, point_segment_ratio(pts, sh, hp, 19.0).astype(np.float32))
        used_segments += 1
    if 0 in joints:
        ratio = np.minimum(ratio, (np.linalg.norm(pts - joints[0].reshape(1, 3), axis=1) / 14.0).astype(np.float32))

    assigned = [k for k in CAMERAS if track.get(k) is not None]
    support = np.zeros(len(pts), np.uint8)
    per_view = {}
    for label in assigned:
        mask = instances[label][track[label]]["mask"].astype(bool)
        hit = v8.mask_membership(cams[label], pts, mask, chunk=250_000)
        support += hit.astype(np.uint8)
        per_view[label] = int(hit.sum())

    occ = ((support >= 2) & (ratio <= 1.30)) | ((support >= 3) & (ratio <= 1.58))
    occ |= (support >= 1) & (ratio <= 0.68)
    occ = occ.reshape(X.shape)
    st = ndimage.generate_binary_structure(3, 1)
    occ = ndimage.binary_closing(occ, structure=st, iterations=1)
    lab, n = ndimage.label(occ, structure=st)
    if n == 0:
        return None, {"status": "REJECTED_EMPTY_OCCUPANCY", "joint_count": int(len(joints)), "joints": joint_qa}
    sizes = np.bincount(lab.ravel())
    largest = int(np.argmax(sizes[1:]) + 1)
    occ = lab == largest
    vox = int(occ.sum())
    if vox < 90:
        return None, {"status": "REJECTED_SMALL_OCCUPANCY", "occupied_voxels": vox, "joints": joint_qa}
    field = ndimage.gaussian_filter(occ.astype(np.float32), sigma=0.72)
    level = min(0.48, max(0.24, 0.38 * float(field.max())))
    verts, faces, normals, values = marching_cubes(field, level=level, spacing=(voxel_cm, voxel_cm, voxel_cm))
    verts[:, 0] += float(xs[0]); verts[:, 1] += float(ys[0]); verts[:, 2] += float(zs[0])
    verts = verts.astype(np.float32); faces = faces.astype(np.int32)

    vis_by_source, col_by_source = {}, {}
    visible_counts = {}
    for label in CAMERAS:
        if track.get(label) is None:
            vis_by_source[label] = np.zeros(len(verts), bool)
            col_by_source[label] = np.zeros((len(verts), 3), np.uint8)
            visible_counts[label] = 0
            continue
        mask = instances[label][track[label]]["mask"].astype(bool)
        vis, col, q = v8.source_surface_data(label, cams[label], verts, mask, IMAGES[label], voxel_cm)
        vis_by_source[label] = vis; col_by_source[label] = col
        visible_counts[label] = int(q["source_visible_surface_voxels"])

    any_vis = np.logical_or.reduce([vis_by_source[k] for k in CAMERAS])
    return {
        "verts": verts, "faces": faces,
        "vis": vis_by_source, "cols": col_by_source,
        "joints": joints,
    }, {
        "status": "ANATOMICAL_MESH_OK",
        "views": assigned,
        "joint_count": int(len(joints)), "used_capsule_segments": int(used_segments),
        "joints": joint_qa,
        "bounds_cm": {"x": [float(xs[0]), float(xs[-1])], "y": [float(ys[0]), float(ys[-1])], "z": [float(zs[0]), float(zs[-1])]},
        "occupied_voxels": vox, "mesh_vertices": int(len(verts)), "mesh_faces": int(len(faces)),
        "per_view_inside_voxels": per_view, "visible_mesh_vertices": visible_counts,
        "vertices_visible_from_any_source": int(np.sum(any_vis)),
        "vertex_source_grounded_fraction": float(np.mean(any_vis)),
    }


def mesh_colours_for_target(mesh, cams, Ct, beta=7.0):
    pts = mesh["verts"].astype(np.float64)
    tv = Ct.reshape(1, 3) - pts
    tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-8)
    scores = []
    for label in CAMERAS:
        sv = cams[label][0].reshape(1, 3) - pts
        sv /= np.maximum(np.linalg.norm(sv, axis=1, keepdims=True), 1e-8)
        s = np.sum(sv * tv, axis=1)
        s[~mesh["vis"][label]] = -99.0
        scores.append(s)
    S = np.vstack(scores).T
    maxs = np.max(S, axis=1, keepdims=True)
    Wt = np.exp(beta * np.clip(S - maxs, -6.0, 0.0))
    Wt[S < -10] = 0.0
    denom = np.sum(Wt, axis=1, keepdims=True)
    out = np.zeros((len(pts), 3), np.float64)
    for si, label in enumerate(CAMERAS):
        out += Wt[:, si:si+1] * mesh["cols"][label].astype(np.float64)
    good = denom[:, 0] > 1e-8
    out[good] /= denom[good]
    return np.clip(out, 0, 255).astype(np.uint8), good


def make_uv_sphere(center, radius=12.0, nlat=10, nlon=20):
    verts = []
    for i in range(nlat + 1):
        phi = math.pi * i / nlat
        for j in range(nlon):
            th = 2.0 * math.pi * j / nlon
            verts.append(center + radius * np.asarray([math.sin(phi)*math.cos(th), math.sin(phi)*math.sin(th), math.cos(phi)]))
    faces = []
    for i in range(nlat):
        for j in range(nlon):
            a = i*nlon + j; b = i*nlon + ((j+1)%nlon)
            c = (i+1)*nlon + j; d = (i+1)*nlon + ((j+1)%nlon)
            faces.append([a,c,b]); faces.append([b,c,d])
    return np.asarray(verts,np.float32), np.asarray(faces,np.int32)


def source_ball_colour(images, candidates, used_labels):
    pix = []
    for label in used_labels:
        if not candidates.get(label):
            continue
        q = candidates[label][0]
        x1,y1,x2,y2 = [int(round(v)) for v in q["box"]]
        x1,y1=max(0,x1),max(0,y1);x2,y2=min(W,x2),min(H,y2)
        roi=images[label][y1:y2,x1:x2]
        if roi.size:
            hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            m=(hsv[:,:,0]>=2)&(hsv[:,:,0]<=24)&(hsv[:,:,1]>=70)
            if np.any(m): pix.append(roi[m])
    if pix:
        p=np.concatenate(pix,axis=0)
        return np.median(p,axis=0).astype(np.uint8)
    return np.asarray([45,105,190],np.uint8)


def enhance_ball_three_view(cams, candidates):
    X, qa = v8.triangulate_ball(cams, candidates)
    if X is None:
        return None, qa
    missing = [k for k in CAMERAS if k not in qa.get("views", []) and candidates.get(k)]
    for label in missing:
        uv,_,valid=v8.project_metric(cams[label],X.reshape(1,3))
        if not valid[0]: continue
        q=min(candidates[label],key=lambda b:(b["cx"]-uv[0,0])**2+(b["cy"]-uv[0,1])**2)
        d=float(math.hypot(q["cx"]-uv[0,0],q["cy"]-uv[0,1]))
        if d>42.0: continue
        obs={}
        for k in qa["views"]:
            b=qa["detections"][k]; obs[k]=np.asarray([b["cx"],b["cy"]],np.float64)
        obs[label]=np.asarray([q["cx"],q["cy"]],np.float64)
        X3=v8.dlt_point(cams,obs)
        if X3 is None or not np.isfinite(X3).all(): continue
        errs={}
        for k,u0 in obs.items():
            u,_,ok=v8.project_metric(cams[k],X3.reshape(1,3))
            if ok[0]: errs[k]=float(np.linalg.norm(u[0]-u0))
        if len(errs)==3 and max(errs.values())<=26.0 and np.linalg.norm(X3-RIM)<=90.0:
            return X3.astype(np.float64), {
                "status":"BALL_TRIANGULATED_THREE_VIEW_ENHANCED",
                "center_world_cm":[float(x) for x in X3],
                "views":list(obs),
                "detections":{**qa["detections"],label:{"cx":float(q["cx"]),"cy":float(q["cy"]),"score":float(q["score"]) }},
                "reprojection_errors_px":errs,
                "rms_reprojection_px":float(np.sqrt(np.mean(np.square(list(errs.values()))))),
                "distance_from_rim_center_cm":float(np.linalg.norm(X3-RIM)),
                "third_view_projection_distance_px":d,
            }
    return X, qa


def far_background(Kt, Rt, cams, images, dynamic_masks):
    yy, xx = np.indices((H, W), np.float64)
    hp = np.stack([xx.ravel(), yy.ravel(), np.ones(H*W)], axis=0)
    dcam = np.linalg.inv(Kt) @ hp
    dw = Rt.T @ dcam
    target_axis = (RIM - CURRENT_CT); target_axis /= max(1e-9,float(np.linalg.norm(target_axis)))
    pref=[]
    for label in CAMERAS:
        a=(RIM-cams[label][0]); a/=max(1e-9,float(np.linalg.norm(a)))
        pref.append((float(np.dot(a,target_axis)),label))
    pref.sort(reverse=True)
    out=np.zeros((H*W,3),np.uint8); filled=np.zeros(H*W,bool)
    for _,label in pref:
        Cc,Rc,Kc=cams[label]
        ds=Rc@dw
        q=Kc@ds
        with np.errstate(divide='ignore',invalid='ignore'):
            uv=(q[:2]/q[2:3]).T
        sign=float(v3.forward_sign(Rc,Cc))
        valid=np.isfinite(uv).all(axis=1)&(sign*ds[2]>1e-6)&(uv[:,0]>=0)&(uv[:,0]<W-1)&(uv[:,1]>=0)&(uv[:,1]<H-1)
        if not np.any(valid): continue
        cols=v8.bilinear_sample(images[label],uv)
        ui=np.rint(uv[:,0]).astype(np.int32); vi=np.rint(uv[:,1]).astype(np.int32)
        inside=valid.copy(); ids=np.where(valid)[0]
        if len(ids): inside[ids]&=~dynamic_masks[label][vi[ids],ui[ids]]
        take=inside&~filled
        out[take]=cols[take];filled[take]=True
    return out.reshape(H,W,3), filled.reshape(H,W)


def render_triangles(base_img, cams, Kt, Rt, Ct, meshes, ball_mesh=None, ball_color=None):
    tri_rows=[]
    for mesh in meshes:
        cols, grounded = mesh_colours_for_target(mesh, cams, Ct)
        uv,depth,valid=v8.project_metric((Ct,Rt,Kt),mesh["verts"])
        faces=mesh["faces"]
        ok=np.all(valid[faces],axis=1)&np.all(grounded[faces],axis=1)
        for fi in np.where(ok)[0]:
            f=faces[fi]; p=uv[f]
            if np.max(p[:,0])<0 or np.min(p[:,0])>=W or np.max(p[:,1])<0 or np.min(p[:,1])>=H: continue
            area=abs(float(np.cross(p[1]-p[0],p[2]-p[0])))
            if area<0.18 or area>6000: continue
            tri_rows.append((float(np.mean(depth[f])),p,np.mean(cols[f],axis=0)))
    if ball_mesh is not None:
        bv,bf=ball_mesh; uv,depth,valid=v8.project_metric((Ct,Rt,Kt),bv)
        for f in bf:
            if not np.all(valid[f]): continue
            p=uv[f]
            if np.max(p[:,0])<0 or np.min(p[:,0])>=W or np.max(p[:,1])<0 or np.min(p[:,1])>=H: continue
            tri_rows.append((float(np.mean(depth[f])),p,np.asarray(ball_color,np.float64)))
    tri_rows.sort(key=lambda x:x[0],reverse=True)
    img=base_img.copy()
    for _d,p,col in tri_rows:
        poly=np.rint(p).astype(np.int32)
        cv2.fillConvexPoly(img,poly,tuple(int(x) for x in np.clip(col,0,255)),lineType=cv2.LINE_AA)
    return img, len(tri_rows)


def sample_rim_colour(cams, images):
    pts=v8.rim_points()[::18]
    pix=[]
    for label in CAMERAS:
        uv,_,valid=v8.project_metric(cams[label],pts)
        good=valid&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        if np.any(good): pix.append(v8.bilinear_sample(images[label],uv[good]))
    if pix:
        p=np.concatenate(pix,axis=0)
        hsv=cv2.cvtColor(p.reshape(-1,1,3).astype(np.uint8),cv2.COLOR_BGR2HSV).reshape(-1,3)
        m=(hsv[:,0]>=2)&(hsv[:,0]<=30)&(hsv[:,1]>=80)
        if np.any(m): return np.median(p[m],axis=0).astype(np.uint8)
        return np.median(p,axis=0).astype(np.uint8)
    return np.asarray([40,90,200],np.uint8)


def draw_metric_rim(img, Kt, Rt, Ct, colour):
    r=9.0*2.54
    th=np.linspace(0,2*math.pi,181)
    pts=np.column_stack([RIM[0]+r*np.cos(th),RIM[1]+r*np.sin(th),np.full_like(th,RIM[2])]).astype(np.float32)
    uv,_,valid=v8.project_metric((Ct,Rt,Kt),pts)
    for i in range(len(pts)-1):
        if valid[i] and valid[i+1]:
            p0=tuple(np.rint(uv[i]).astype(int));p1=tuple(np.rint(uv[i+1]).astype(int))
            if -20<=p0[0]<W+20 and -20<=p0[1]<H+20 and -20<=p1[0]<W+20 and -20<=p1[1]<H+20:
                cv2.line(img,p0,p1,tuple(int(x) for x in colour),2,cv2.LINE_AA)


def montage(paths, out):
    ims=[cv2.imread(str(p)) for p in paths]
    c=np.zeros((H*2,W*3,3),np.uint8)
    for i,im in enumerate(ims): c[(i//3)*H:(i//3+1)*H,(i%3)*W:(i%3+1)*W]=im
    cv2.imwrite(str(out),c)


IMAGES = {}
CURRENT_CT = np.zeros(3,np.float64)


def main():
    global IMAGES, CURRENT_CT
    ap=argparse.ArgumentParser()
    ap.add_argument('--frames-dir',type=Path,required=True)
    ap.add_argument('--sync-qa',type=Path,required=True)
    ap.add_argument('--registry',type=Path,required=True)
    ap.add_argument('--rar-report',type=Path,required=True)
    ap.add_argument('--broadcast-event-frame',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tokens',type=int,default=900)
    ap.add_argument('--voxel-cm',type=float,default=2.5)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)

    paths={label:find_frame(args.frames_dir,label) for label in CAMERAS}
    IMAGES={k:cv2.imread(str(p)) for k,p in paths.items()}
    for k,im in IMAGES.items():
        if im is None or im.shape[:2]!=(H,W): raise RuntimeError(f'bad selected frame {k}: {paths[k]}')
    exact_qa=json.loads(args.sync_qa.read_text())
    cams=base.load_cameras(args.registry,args.rar_report,args.broadcast_event_frame)

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    print('Loading RF-DETR Keypoint Preview and Mask R-CNN...',flush=True)
    pose_model=RFDETRKeypointPreview()
    seg_model=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    print('PLAYER_MODELS_READY',flush=True)

    instances={};pose_audit={};dynamic_masks={}
    for label in CAMERAS:
        Cc,Rc,Kc=cams[label]
        dyn,inst,_=v6.detect_near_play(seg_model,IMAGES[label],Kc,Rc,Cc)
        pose,matches=attach_rfdetr_poses(pose_model,IMAGES[label],inst)
        instances[label]=inst;dynamic_masks[label]=dyn
        pose_audit[label]={
            'mask_instances':len(inst),'pose_detections':int(len(pose['xy'])),'matched_mask_pose_pairs':matches,
            'appearance':[appearance_descriptor(IMAGES[label],x) for x in inst],
        }
        overlay=IMAGES[label].copy()
        for m in matches:
            i=m['mask_instance'];box=inst[i]['box'];x1,y1,x2,y2=np.rint(box).astype(int)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),1)
            cv2.putText(overlay,f"m{i}/p{m['pose_index']}",(x1,max(15,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(args.out/f"v12_pose_mask_{label.replace(' ','_')}.png"),overlay)

    mb,ab_audit,coarse_bounds=hard_left_broadcast_tracks(cams,instances)
    if not mb: raise RuntimeError('v12 found no hardened Left/Broadcast tracks')
    tracks,rar_audit=attach_right_above_rim(cams,instances,mb,exact_qa)
    if not tracks: raise RuntimeError('v12 found no player tracks')

    meshes=[];mesh_qas=[];kept_tracks=[]
    for ti,tr in enumerate(tracks):
        mesh,mqa=build_anatomical_mesh(cams,tr,instances,voxel_cm=args.voxel_cm)
        mqa['track_id']=ti;mqa['instances']={k:(int(tr[k]) if tr.get(k) is not None else None) for k in CAMERAS}
        mqa['rar_attachment']=tr.get('rar_attachment',{'method':'none'})
        mesh_qas.append(mqa)
        if mesh is not None:
            meshes.append(mesh);kept_tracks.append(ti)
            np.savez_compressed(args.out/f'v12_track_{ti:02d}_mesh.npz',verts=mesh['verts'],faces=mesh['faces'])
    if len(meshes)<2: raise RuntimeError(f'only {len(meshes)} anatomical meshes survived')

    print('Loading MoGe for static-plane visibility...',flush=True)
    depth_model=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval()
    sources={};static_qa={};ball_candidates={}
    for label in CAMERAS:
        Cc,Rc,Kc=cams[label]
        depth,_,valid,_,_=moge_infer(depth_model,IMAGES[label],args.tokens)
        align,dqa=v3.robust_depth_align(depth,valid,Kc,Rc,Cc)
        fv,bv=v5.source_visibility_v5(IMAGES[label],depth,valid,dynamic_masks[label],Kc,Rc,Cc,align)
        sources[label]={'image':IMAGES[label],'C':Cc,'R':Rc,'K':Kc,'floor_vis':fv,'board_vis':bv,'dynamic':dynamic_masks[label]}
        static_qa[label]={'depth_alignment':dqa,'floor_visible_pixels':int(fv.sum()),'board_visible_pixels':int(bv.sum())}
        ball_candidates[label]=v9.roi_ball_candidates(IMAGES[label],cams[label],[])

    ball_center,ball_qa=enhance_ball_three_view(cams,ball_candidates)
    ball_mesh=None;ball_colour=None
    if ball_center is not None:
        ball_mesh=make_uv_sphere(ball_center)
        ball_colour=source_ball_colour(IMAGES,ball_candidates,ball_qa.get('views',[]))
    rim_colour=sample_rim_colour(cams,IMAGES)

    C0,R0,K0=cams[A]
    angles=[0.0,5.0,10.0,15.0,20.0,25.0]
    frame_qa=[];clean_paths=[]
    for ang in angles:
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang));CURRENT_CT=np.asarray(Ct,np.float64)
        bg,bgmask=far_background(K0,Rt,cams,IMAGES,dynamic_masks)
        Pf,sf=v4.virtual_plane_points(K0,Rt,Ct,'floor')
        floor_img,floor_mask=v4.sample_plane_from_sources(Pf,sf,sources,'floor')
        Pb,sb=v4.virtual_plane_points(K0,Rt,Ct,'board')
        board_img,board_mask=v4.sample_plane_from_sources(Pb,sb,sources,'board')
        img=bg.copy();img[floor_mask]=floor_img[floor_mask];img[board_mask]=board_img[board_mask]
        img,tri_count=render_triangles(img,cams,K0,Rt,Ct,meshes,ball_mesh,ball_colour)
        draw_metric_rim(img,K0,Rt,Ct,rim_colour)
        clean=args.out/f'v12_{int(ang):02d}deg.png';cv2.imwrite(str(clean),img);clean_paths.append(clean)
        nonblack=float(np.mean(np.any(img>3,axis=2)))
        frame_qa.append({'angle_deg':ang,'background_coverage':float(bgmask.mean()),'floor_pixels':int(floor_mask.sum()),'board_pixels':int(board_mask.sum()),'rendered_triangles':int(tri_count),'nonblack_fraction':nonblack})
    montage(clean_paths,args.out/'v12_key_arc_montage.png')

    qa={
        'schema_version':12,
        'status':'THREE_CAMERA_HIGH_STANDARD_HYPOTHESIS_STATIC_RENDERED',
        'hypothesis':'three solved physical cameras may be sufficient for a high-standard 0-25 degree static free-view arc; four cameras are not assumed necessary',
        'source_resolution':[W,H],
        'selected_exact_state':{'offsets':exact_qa.get('selected',{}).get('offsets'),'frames':exact_qa.get('selected',{}).get('frames'),'strict_three_view_tracks':exact_qa.get('selected',{}).get('strict_track_count')},
        'source_files':{k:paths[k].name for k in CAMERAS},
        'pose_mask_audit':pose_audit,
        'left_broadcast_association':ab_audit,
        'coarse_association_bounds_cm':[float(x) for x in coarse_bounds],
        'right_above_rim_attachment':rar_audit,
        'player_meshes':mesh_qas,
        'accepted_mesh_track_ids':kept_tracks,
        'accepted_mesh_count':len(meshes),
        'tracks_using_right_above_rim':int(sum(1 for q in mesh_qas if C in q.get('views',[]))),
        'ball':ball_qa,
        'static_plane_qa':static_qa,
        'frames':frame_qa,
        'minimum_nonblack_fraction':float(min(x['nonblack_fraction'] for x in frame_qa)),
        'minimum_background_coverage':float(min(x['background_coverage'] for x in frame_qa)),
        'appearance_policy':'real source RGB only for player mesh colour; distant arena uses source-image rotation warp; no generated texture or upscale',
        'geometry_policy':'metric court/backboard/rim; exact-state multi-view pose plus assigned silhouettes; anatomical capsules only regularize occupancy',
        'render_resolution_policy':'native 960x540 only; static six-frame proof, no UHD and no animation',
        'visual_pass_required':True,
    }
    (args.out/'three_camera_mesh_v12_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({
        'status':qa['status'],'accepted_mesh_count':qa['accepted_mesh_count'],'tracks_using_right_above_rim':qa['tracks_using_right_above_rim'],
        'ball':qa['ball'].get('status'),'minimum_nonblack_fraction':qa['minimum_nonblack_fraction'],
        'rar_attachment':[x['rar_attachment'] for x in rar_audit['tracks']],
    },indent=2),flush=True)


if __name__=='__main__':
    main()
