from __future__ import annotations

"""Three-camera volumetric free-view diagnostic v8.

This deliberately replaces the v7 source-facing flat person cards with a true
3-D shape-from-silhouette representation. The accepted metric Left Above Rim,
Right Above Rim and Broadcast cameras are held fixed. Court and backboard stay
on their exact NBA metric planes. Player geometry is the boundary of a 3-D
visual hull supported by at least two accepted camera silhouettes. The rim is
rendered from its known metric circle and the ball is independently triangulated
from real detector observations when a stable solution exists.

Every rendered colour is sampled from the three real NBA source frames. There
is no crossfade, optical-flow morph, generative fill or invented texture.
Unsupported pixels remain black.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from moge.model.v2 import MoGeModel
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v4 as v4
from freeze_spin import build_three_camera_diagnostic_v5 as v5
from freeze_spin import build_three_camera_diagnostic_v6 as v6
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W, H = base.W, base.H
RIM = base.RIM.astype(np.float64)
CAMERA_ORDER = ("Left Above Rim", "Broadcast", "Right Above Rim")
BALL_RADIUS_CM = 12.0


def project_metric(cam, pts: np.ndarray):
    C, R, K = cam
    Xc = (R @ (pts.astype(np.float64) - C).T).T
    q = (K @ Xc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    sign = float(v3.forward_sign(R, C))
    depth = sign * Xc[:, 2]
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 20.0)
    return uv, depth, valid


def bilinear_sample(image: np.ndarray, uv: np.ndarray):
    mapx = uv[:, 0].astype(np.float32).reshape(-1, 1)
    mapy = uv[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(
        image, mapx, mapy, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).reshape(-1, 3)


def mask_membership(cam, pts: np.ndarray, mask: np.ndarray, chunk=350_000):
    hit = np.zeros(len(pts), dtype=bool)
    for start in range(0, len(pts), chunk):
        end = min(len(pts), start + chunk)
        uv, _, valid = project_metric(cam, pts[start:end])
        u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
        vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
        inside = valid & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        ids = np.where(inside)[0]
        local = np.zeros(end - start, dtype=bool)
        if len(ids):
            local[ids] = mask[vv[ids], u[ids]]
        hit[start:end] = local
    return hit


def action_bounds(instances_by_camera):
    pts = []
    for inst in instances_by_camera.get("Left Above Rim", []):
        p = np.asarray(inst["foot_world_cm"], dtype=np.float64)
        if np.isfinite(p).all() and np.linalg.norm(p[:2] - RIM[:2]) <= 520.0:
            pts.append(p)
    if not pts:
        for rows in instances_by_camera.values():
            for inst in rows:
                p = np.asarray(inst["foot_world_cm"], dtype=np.float64)
                if np.isfinite(p).all() and np.linalg.norm(p[:2] - RIM[:2]) <= 520.0:
                    pts.append(p)
    if pts:
        p = np.asarray(pts)
        x0 = max(-240.0, float(np.min(p[:, 0])) - 115.0)
        x1 = min(760.0, float(np.max(p[:, 0])) + 135.0)
        y0 = max(-470.0, float(np.min(p[:, 1])) - 115.0)
        y1 = min(470.0, float(np.max(p[:, 1])) + 115.0)
    else:
        x0, x1, y0, y1 = -220.0, 700.0, -430.0, 430.0
    x0 = min(x0, float(RIM[0] - 110.0)); x1 = max(x1, float(RIM[0] + 220.0))
    y0 = min(y0, -180.0); y1 = max(y1, 180.0)
    return x0, x1, y0, y1, -8.0, 372.0


def build_visual_hull(cams, body_masks, instances_by_camera, voxel_cm: float):
    x0, x1, y0, y1, z0, z1 = action_bounds(instances_by_camera)
    xs = np.arange(x0, x1 + 0.5 * voxel_cm, voxel_cm, dtype=np.float32)
    ys = np.arange(y0, y1 + 0.5 * voxel_cm, voxel_cm, dtype=np.float32)
    zs = np.arange(z0, z1 + 0.5 * voxel_cm, voxel_cm, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    counts = np.zeros(len(pts), dtype=np.uint8)
    hits = {}
    for label in CAMERA_ORDER:
        h = mask_membership(cams[label], pts, body_masks[label])
        counts += h.astype(np.uint8)
        hits[label] = int(h.sum())

    shape = X.shape
    occ = counts.reshape(shape) >= 2
    st = ndimage.generate_binary_structure(3, 1)
    occ = ndimage.binary_opening(occ, structure=st, iterations=1)
    occ = ndimage.binary_closing(occ, structure=st, iterations=1)
    labels, n = ndimage.label(occ, structure=st)
    sizes = np.bincount(labels.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= 90]
    if not keep:
        raise RuntimeError(f"no substantial 3-D silhouette component; raw occupied={int(occ.sum())}")
    occ = np.isin(labels, keep)
    eroded = ndimage.binary_erosion(occ, structure=st, border_value=0)
    surface = occ & ~eroded
    surface_idx = np.flatnonzero(surface.ravel())
    surface_pts = pts[surface_idx]
    surface_support = counts[surface_idx]
    if len(surface_pts) < 1000:
        raise RuntimeError(f"visual hull surface too small: {len(surface_pts)}")
    qa = {
        "voxel_cm": float(voxel_cm),
        "bounds_cm": {"x": [float(xs[0]), float(xs[-1])], "y": [float(ys[0]), float(ys[-1])], "z": [float(zs[0]), float(zs[-1])]},
        "grid_shape": list(shape),
        "grid_voxels": int(len(pts)),
        "minimum_support_views": 2,
        "per_camera_inside_mask_voxels": hits,
        "occupied_voxels": int(occ.sum()),
        "surface_voxels": int(len(surface_pts)),
        "components_before_filter": int(n),
        "components_kept": int(len(keep)),
        "kept_component_sizes": [int(sizes[i]) for i in keep],
        "surface_support_histogram": {"2_views": int(np.sum(surface_support == 2)), "3_views": int(np.sum(surface_support == 3))},
    }
    return surface_pts, qa


def source_surface_data(label, cam, pts, mask, image, voxel_cm):
    uv, depth, valid = project_metric(cam, pts)
    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    inside = valid & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    ids = np.where(inside)[0]
    inmask = np.zeros(len(pts), dtype=bool)
    if len(ids):
        inmask[ids] = mask[vv[ids], u[ids]]
    inside &= inmask
    pix = vv * W + u
    zbuf = np.full(H * W, np.inf, dtype=np.float32)
    ids = np.where(inside)[0]
    if len(ids):
        np.minimum.at(zbuf, pix[ids], depth[ids].astype(np.float32))
    visible = np.zeros(len(pts), dtype=bool)
    if len(ids):
        visible[ids] = depth[ids] <= zbuf[pix[ids]] + max(6.0, 2.1 * voxel_cm)
    colours = bilinear_sample(image, uv)
    return visible, colours, {"source_visible_surface_voxels": int(visible.sum())}


def select_view_colours(pts, Ct, cams, vis_by_source, colours_by_source):
    n = len(pts)
    target_vec = Ct.reshape(1, 3) - pts.astype(np.float64)
    target_vec /= np.maximum(np.linalg.norm(target_vec, axis=1, keepdims=True), 1e-8)
    best_score = np.full(n, -2.0, dtype=np.float64)
    best_idx = np.full(n, -1, dtype=np.int16)
    for si, label in enumerate(CAMERA_ORDER):
        C = cams[label][0]
        sv = C.reshape(1, 3) - pts.astype(np.float64)
        sv /= np.maximum(np.linalg.norm(sv, axis=1, keepdims=True), 1e-8)
        score = np.sum(sv * target_vec, axis=1)
        valid = vis_by_source[label]
        take = valid & (score > best_score)
        best_score[take] = score[take]
        best_idx[take] = si
    cols = np.zeros((n, 3), dtype=np.uint8)
    for si, label in enumerate(CAMERA_ORDER):
        m = best_idx == si
        cols[m] = colours_by_source[label][m]
    return cols, best_idx


def dlt_point(cams, observations):
    A = []
    for label, uv in observations.items():
        C, R, K = cams[label]
        P = K @ np.hstack([R, (-R @ C.reshape(3, 1))])
        u, v = [float(x) for x in uv]
        A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    xh = vt[-1]
    if abs(float(xh[3])) < 1e-9:
        return None
    return xh[:3] / xh[3]


def triangulate_ball(cams, detections):
    available = [k for k in CAMERA_ORDER if detections.get(k)]
    candidates = []
    for nviews in (3, 2):
        if len(available) < nviews:
            continue
        for labels in itertools.combinations(available, nviews):
            choices = [detections[k][:3] for k in labels]
            for combo in itertools.product(*choices):
                obs = {k: np.asarray([b["cx"], b["cy"]], dtype=np.float64) for k, b in zip(labels, combo)}
                X = dlt_point(cams, obs)
                if X is None or not np.isfinite(X).all():
                    continue
                reproj = []
                for k, uv_obs in obs.items():
                    uv, _, valid = project_metric(cams[k], X.reshape(1, 3))
                    if not valid[0]:
                        reproj = []; break
                    reproj.append(float(np.linalg.norm(uv[0] - uv_obs)))
                if not reproj:
                    continue
                rim_dist = float(np.linalg.norm(X - RIM))
                physical = (-120.0 <= X[0] <= 360.0 and abs(X[1]) <= 220.0 and 170.0 <= X[2] <= 385.0)
                if not physical:
                    continue
                detector_bonus = float(sum(float(b["score"]) for b in combo))
                score = float(np.sqrt(np.mean(np.square(reproj))) + 0.012 * rim_dist - 0.8 * detector_bonus + (0.0 if nviews == 3 else 1.5))
                candidates.append((score, X, labels, combo, reproj, rim_dist))
    if not candidates:
        return None, {"status": "NO_STABLE_BALL_TRIANGULATION"}
    candidates.sort(key=lambda x: x[0])
    score, X, labels, combo, reproj, rim_dist = candidates[0]
    qa = {
        "status": "BALL_TRIANGULATED",
        "center_world_cm": [float(x) for x in X],
        "views": list(labels),
        "detections": {k: {"cx": float(b["cx"]), "cy": float(b["cy"]), "score": float(b["score"])} for k, b in zip(labels, combo)},
        "reprojection_errors_px": {k: float(e) for k, e in zip(labels, reproj)},
        "rms_reprojection_px": float(np.sqrt(np.mean(np.square(reproj)))),
        "distance_from_rim_center_cm": rim_dist,
        "selection_score": float(score),
    }
    return X.astype(np.float64), qa


def sphere_points(center, radius=BALL_RADIUS_CM, n_lat=16, n_lon=32):
    pts = []
    for i in range(1, n_lat):
        phi = math.pi * i / n_lat
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(n_lon):
            th = 2.0 * math.pi * j / n_lon
            pts.append(center + radius * np.asarray([sp * math.cos(th), sp * math.sin(th), cp]))
    pts.append(center + np.asarray([0.0, 0.0, radius])); pts.append(center - np.asarray([0.0, 0.0, radius]))
    return np.asarray(pts, dtype=np.float32)


def rim_points():
    r = 9.0 * 2.54
    pts = []
    for dz in (-0.9, 0.0, 0.9):
        for dr in (-0.9, 0.0, 0.9):
            rr = r + dr
            for th in np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False):
                pts.append([RIM[0] + rr * math.cos(th), RIM[1] + rr * math.sin(th), RIM[2] + dz])
    return np.asarray(pts, dtype=np.float32)


def source_generic_surface_data(cam, pts, image):
    uv, depth, valid = project_metric(cam, pts)
    u = np.rint(uv[:, 0]).astype(np.int32, copy=False); vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    inside = valid & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    pix = vv * W + u
    zbuf = np.full(H * W, np.inf, np.float32)
    ids = np.where(inside)[0]
    if len(ids):
        np.minimum.at(zbuf, pix[ids], depth[ids].astype(np.float32))
    visible = np.zeros(len(pts), bool)
    if len(ids): visible[ids] = depth[ids] <= zbuf[pix[ids]] + 4.0
    return visible, bilinear_sample(image, uv)


def subtract_metric_rim_from_plane_visibility(source):
    pts = rim_points()[::18]
    uv, _, valid = project_metric((source["C"], source["R"], source["K"]), pts)
    rm = np.zeros((H, W), np.uint8)
    poly = []
    for p, ok in zip(uv, valid):
        if ok and -20 <= p[0] < W + 20 and -20 <= p[1] < H + 20:
            poly.append((int(round(p[0])), int(round(p[1]))))
    if len(poly) >= 8:
        cv2.polylines(rm, [np.asarray(poly, np.int32)], True, 255, 7, cv2.LINE_AA)
        rm = cv2.dilate(rm, np.ones((5, 5), np.uint8), iterations=1)
        source["floor_vis"] &= rm == 0
        source["board_vis"] &= rm == 0
    return int(np.sum(rm > 0))


def annotate(img, angle, cov, body_cov, hull_qa, ball_status):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (760, 67), (0, 0, 0), -1)
    cv2.putText(out, "3-CAMERA DIAGNOSTIC v8 | VOLUMETRIC SILHOUETTE 3D", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .46, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, f"orbit {angle:04.1f} deg | grounded {cov*100:05.1f}% | 3D body {body_cov*100:04.1f}% | hull {hull_qa['surface_voxels']} vox | {ball_status}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, "court/board metric planes + 2-of-3 visual hull + metric rim + triangulated ball; real source pixels only", (12, 63), cv2.FONT_HERSHEY_SIMPLEX, .34, (210,210,210), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--rar-report", type=Path, required=True)
    ap.add_argument("--broadcast-event-frame", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=1200)
    ap.add_argument("--voxel-cm", type=float, default=4.0)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    lar = base.find_one(args.frames_dir, "Left_Above_Rim")
    rar = base.find_one(args.frames_dir, "Right_Above_Rim")
    br = [p for p in sorted(args.frames_dir.rglob("*Broadcast*.png")) if "Mobile" not in p.name and "Other" not in p.name]
    if len(br) != 1: raise RuntimeError(f"Broadcast ambiguity {br}")
    paths = {"Left Above Rim": lar, "Right Above Rim": rar, "Broadcast": br[0]}
    images = {k: cv2.imread(str(p)) for k, p in paths.items()}
    for k, im in images.items():
        if im is None or im.shape[:2] != (H, W): raise RuntimeError(f"bad source frame {k}: {paths[k]}")
    cams = base.load_cameras(args.registry, args.rar_report, args.broadcast_event_frame)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    seg_model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

    sources = {}; body_masks = {}; instances = {}; balls = {}; qa = {"sources": {}, "frames": []}
    for label in CAMERA_ORDER:
        C, R, K = cams[label]; image = images[label]
        depth, _, valid, _, _ = moge_infer(depth_model, image, args.tokens)
        align, dqa = v3.robust_depth_align(depth, valid, K, R, C)
        _dyn, inst, bdet = v6.detect_near_play(seg_model, image, K, R, C)
        body = np.zeros((H, W), dtype=bool)
        for row in inst: body |= row["mask"].astype(bool)
        body = cv2.dilate(body.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1) > 0
        fv, bv = v5.source_visibility_v5(image, depth, valid, _dyn, K, R, C, align)
        sources[label] = {"image": image, "C": C, "R": R, "K": K, "floor_vis": fv, "board_vis": bv, "dynamic": _dyn}
        body_masks[label] = body; instances[label] = inst; balls[label] = bdet[:3]
        cv2.imwrite(str(args.out / f"{label.replace(' ','_')}_body_mask.png"), body.astype(np.uint8) * 255)
        qa["sources"][label] = {
            "file": paths[label].name,
            "depth_alignment_for_static_plane_visibility": dqa,
            "near_play_person_instances": len(inst),
            "body_mask_pixels": int(body.sum()),
            "ball_detections": bdet[:3],
            "floor_visible_pixels": int(fv.sum()),
            "board_visible_pixels": int(bv.sum()),
            "camera_center_cm": [float(x) for x in C],
        }

    body_pts, hull_qa = build_visual_hull(cams, body_masks, instances, args.voxel_cm)
    body_vis = {}; body_cols = {}
    for label in CAMERA_ORDER:
        vis, cols, sqa = source_surface_data(label, cams[label], body_pts, body_masks[label], images[label], args.voxel_cm)
        body_vis[label] = vis; body_cols[label] = cols; qa["sources"][label].update(sqa)

    ball_center, ball_qa = triangulate_ball(cams, balls)
    ball_pts = sphere_points(ball_center) if ball_center is not None else np.empty((0, 3), np.float32)
    ball_vis = {}; ball_cols = {}
    if len(ball_pts):
        for label in CAMERA_ORDER:
            uv_c, _, ok_c = project_metric(cams[label], ball_center.reshape(1, 3))
            bm = np.zeros((H, W), dtype=bool)
            if ok_c[0] and balls.get(label):
                px = uv_c[0]
                nearest = min(balls[label], key=lambda b: (float(b["cx"])-px[0])**2 + (float(b["cy"])-px[1])**2)
                x1,y1,x2,y2 = [int(round(v)) for v in nearest["box"]]
                x1=max(0,x1-4); y1=max(0,y1-4); x2=min(W-1,x2+4); y2=min(H-1,y2+4)
                bm[y1:y2+1, x1:x2+1] = True
            vis, cols, _ = source_surface_data(label, cams[label], ball_pts, bm, images[label], 2.0)
            ball_vis[label] = vis; ball_cols[label] = cols

    rp = rim_points(); rim_vis = {}; rim_cols = {}
    for label in CAMERA_ORDER:
        vis, cols = source_generic_surface_data(cams[label], rp, images[label])
        rim_vis[label] = vis; rim_cols[label] = cols
        qa["sources"][label]["metric_rim_plane_exclusion_pixels"] = subtract_metric_rim_from_plane_visibility(sources[label])

    C0, R0, K0 = cams["Left Above Rim"]
    angles = np.r_[np.zeros(12), np.linspace(0.0, 25.0, 76), np.full(18, 25.0)]
    for i, ang in enumerate(angles):
        Rt, Ct = base.orbit_pose(C0, R0, RIM, float(ang))
        Pf, sf = v4.virtual_plane_points(K0, Rt, Ct, "floor")
        floor_img, floor_mask = v4.sample_plane_from_sources(Pf, sf, sources, "floor")
        Pb, sb = v4.virtual_plane_points(K0, Rt, Ct, "board")
        board_img, board_mask = v4.sample_plane_from_sources(Pb, sb, sources, "board")
        img = floor_img.copy(); grounded = floor_mask.copy(); img[board_mask] = board_img[board_mask]; grounded[board_mask] = True

        body_chosen, owners = select_view_colours(body_pts, Ct, cams, body_vis, body_cols)
        rim_chosen, rim_owners = select_view_colours(rp, Ct, cams, rim_vis, rim_cols)
        fg_pts = [body_pts[owners >= 0], rp[rim_owners >= 0]]
        fg_cols = [body_chosen[owners >= 0], rim_chosen[rim_owners >= 0]]
        ball_owners = np.empty(0, dtype=np.int16)
        if len(ball_pts):
            ball_chosen, ball_owners = select_view_colours(ball_pts, Ct, cams, ball_vis, ball_cols)
            if np.any(ball_owners >= 0):
                fg_pts.append(ball_pts[ball_owners >= 0]); fg_cols.append(ball_chosen[ball_owners >= 0])
        if fg_pts and sum(len(x) for x in fg_pts):
            fg_img, fg_m = v4.raster((np.concatenate(fg_pts), np.concatenate(fg_cols)), K0, Rt, Ct, radius=2)
        else:
            fg_img = np.zeros((H,W,3), np.uint8); fg_m = np.zeros((H,W), np.uint8)
        fm = fg_m > 0; img[fm] = fg_img[fm]

        body_img_qa, body_m_qa = v4.raster((body_pts[owners >= 0], body_chosen[owners >= 0]), K0, Rt, Ct, radius=2) if np.any(owners >= 0) else (np.zeros((H,W,3),np.uint8), np.zeros((H,W),np.uint8))
        rim_img_qa, rim_m_qa = v4.raster((rp[rim_owners >= 0], rim_chosen[rim_owners >= 0]), K0, Rt, Ct, radius=2) if np.any(rim_owners >= 0) else (np.zeros((H,W,3),np.uint8), np.zeros((H,W),np.uint8))
        bm = body_m_qa > 0; rm = rim_m_qa > 0
        ball_pixels = 0
        if len(ball_pts) and np.any(ball_owners >= 0):
            ball_img_qa, ball_m_qa = v4.raster((ball_pts[ball_owners >= 0], ball_chosen[ball_owners >= 0]), K0, Rt, Ct, radius=2)
            ball_pixels = int(np.sum(ball_m_qa > 0))
        total = grounded | fm
        cov = float(total.mean()); body_cov = float(bm.mean())
        frame_qa = {
            "frame": int(i), "angle_deg": float(ang), "source_grounded_fraction": cov,
            "body_fraction": body_cov, "foreground_pixels_common_zbuffer": int(fm.sum()), "body_pixels_qa": int(bm.sum()), "rim_pixels_qa": int(rm.sum()), "ball_pixels_qa": ball_pixels,
            "body_source_ownership_points": {CAMERA_ORDER[j]: int(np.sum(owners == j)) for j in range(len(CAMERA_ORDER))},
        }
        qa["frames"].append(frame_qa)
        cv2.imwrite(str(args.out / f"frame_{i:03d}.png"), annotate(img, float(ang), cov, body_cov, hull_qa, ball_qa["status"]))

    qa.update({
        "schema_version": 8,
        "status": "DIAGNOSTIC_THREE_CAMERA_VOLUMETRIC_RENDERED",
        "game_id": "0022500301",
        "event_id": 489,
        "source_resolution": [W, H],
        "geometry": {
            "court": "exact metric floor plane",
            "backboard": "exact metric board plane",
            "players": "3-D visual-hull boundary from >=2 of 3 accepted metric camera silhouettes",
            "rim": "known metric 18-inch inside circle represented in world 3-D",
            "ball": "detector-centre multi-camera DLT triangulation + 12 cm radius physical prior when stable",
        },
        "appearance_policy": "real source RGB samples only; nearest-view ownership selected per 3-D surface point; no crossfade, optical-flow morph, generative fill or invented texture",
        "visual_hull": hull_qa,
        "ball": ball_qa,
        "minimum_source_grounded_fraction": float(min(x["source_grounded_fraction"] for x in qa["frames"])),
        "certification_scope": "three-camera R&D diagnostic only; does not satisfy the project >=4-camera production gate",
        "perspective_test": "v8 removes v7's source-facing flat person cards; apparent player shape is produced by projecting a fixed 3-D silhouette volume through the virtual metric camera",
    })
    (args.out / "three_camera_volumetric_v8_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"status": qa["status"], "visual_hull": hull_qa, "ball": ball_qa, "minimum_source_grounded_fraction": qa["minimum_source_grounded_fraction"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
