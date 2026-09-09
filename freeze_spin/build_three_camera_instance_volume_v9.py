from __future__ import annotations

"""Jazz event 489: instance-separated metric 3-D diagnostic v9.

v8 proved the global silhouette hull is the wrong foreground representation: a
union-of-people silhouette in one view can intersect a different person in a
second view and create geometrically valid but physically false ghost volumes.

v9 makes the Left Above Rim frame the immutable reference view for the requested
0->25 degree arc. Each reference person is reconstructed independently:

1. detect on-court person instances independently in all three accepted views;
2. back-project every individual silhouette into the common metric NBA world;
3. match Left Above Rim <-> Broadcast instances using 3-D cone-intersection
   compactness plus source appearance, one-to-one;
4. select the physically compact connected 3-D component for each matched pair;
5. use Right Above Rim only when it supports that SAME selected component;
6. render dense occupied voxels through a real target-view z-buffer.

This guarantees the 0-degree body geometry is anchored to real Left Above Rim
silhouettes and prevents Broadcast+overhead-only ghost bodies from appearing in
the reference view. Court/backboard remain exact metric planes. The rim is an
explicit metric circle. Ball candidates combine COCO and deterministic orange
component detection, then require multi-view metric triangulation.

No generated pixels, crossfade, optical-flow morph, or synthetic texture.
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
from freeze_spin import build_three_camera_volumetric_v8 as v8
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W, H = base.W, base.H
RIM = base.RIM.astype(np.float64)
CAMERA_ORDER = ("Left Above Rim", "Broadcast", "Right Above Rim")
REF = "Left Above Rim"
MATCH = "Broadcast"
OVERHEAD = "Right Above Rim"
BALL_RADIUS_CM = 12.0


def descriptor(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = cv2.erode(mask.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
    if int(np.sum(m > 0)) < 60:
        m = mask.astype(np.uint8) * 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], m, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    n = float(np.linalg.norm(h))
    return (h / n).astype(np.float32) if n > 1e-8 else h.astype(np.float32)


def appearance_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def make_grid(voxel_cm: float):
    xs = np.arange(-260.0, 721.0, voxel_cm, dtype=np.float32)
    ys = np.arange(-560.0, 561.0, voxel_cm, dtype=np.float32)
    zs = np.arange(-8.0, 373.0, voxel_cm, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    shape = X.shape
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    del X, Y, Z
    return xs, ys, zs, shape, pts


def instance_metric_prior(depth, valid, align, K, R, C, mask):
    s = v3.forward_sign(R, C)
    yy, xx = np.indices((H, W))
    pick = mask & valid & ((xx % 3) == 0) & ((yy % 3) == 0)
    z = align[0] * depth.astype(np.float64) + align[1]
    pick &= np.isfinite(z) & (z > 20.0) & (z < 12000.0)
    ys, xs = np.where(pick)
    if len(xs) < 20:
        return None
    zz = z[ys, xs]
    xn = (xs.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (ys.astype(np.float64) - K[1, 2]) / K[1, 1]
    Xc = s * np.column_stack([xn * zz, yn * zz, zz])
    Xw = (R.T @ Xc.T).T + C
    plausible = (
        np.isfinite(Xw).all(axis=1)
        & (Xw[:, 0] >= -400.0) & (Xw[:, 0] <= 1000.0)
        & (np.abs(Xw[:, 1]) <= 750.0)
        & (Xw[:, 2] >= -80.0) & (Xw[:, 2] <= 500.0)
    )
    Xw = Xw[plausible]
    if len(Xw) < 15:
        return None
    return np.median(Xw, axis=0).astype(np.float64)


def support_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1) > 0


def pair_metrics(h1, h2, pts, prior, sim):
    ids = np.flatnonzero(h1 & h2)
    n = int(len(ids))
    if n < 120:
        return {"score": -1e6, "overlap_voxels": n, "appearance_similarity": float(sim)}
    step = max(1, n // 5000)
    p = pts[ids[::step]].astype(np.float64)
    med = np.median(p, axis=0)
    lo = np.percentile(p, 5.0, axis=0); hi = np.percentile(p, 95.0, axis=0)
    span = hi - lo
    norm_overlap = n / max(1.0, math.sqrt(float(np.sum(h1)) * float(np.sum(h2))))
    prior_dist = float(np.linalg.norm(med[:2] - prior[:2])) if prior is not None else 0.0
    compact_pen = (
        max(0.0, float(span[0]) - 180.0) / 110.0
        + max(0.0, float(span[1]) - 180.0) / 110.0
        + max(0.0, float(span[2]) - 340.0) / 170.0
    )
    score = 18.0 * norm_overlap + 1.35 * float(sim) - prior_dist / 260.0 - compact_pen
    return {
        "score": float(score),
        "overlap_voxels": n,
        "normalized_overlap": float(norm_overlap),
        "appearance_similarity": float(sim),
        "intersection_median_world_cm": med.tolist(),
        "intersection_p05_p95_span_cm": span.tolist(),
        "distance_to_reference_moge_prior_xy_cm": prior_dist,
    }


def component_from_pair(pair_occ, shape, pts, prior, overhead_hits, overhead_descs, ref_desc):
    flat = np.flatnonzero(pair_occ)
    if len(flat) < 120:
        return None, None
    ix, iy, iz = np.unravel_index(flat, shape)
    pad = 2
    x0, x1 = max(0, int(ix.min()) - pad), min(shape[0], int(ix.max()) + pad + 1)
    y0, y1 = max(0, int(iy.min()) - pad), min(shape[1], int(iy.max()) + pad + 1)
    z0, z1 = max(0, int(iz.min()) - pad), min(shape[2], int(iz.max()) + pad + 1)
    sub = pair_occ.reshape(shape)[x0:x1, y0:y1, z0:z1]
    st = ndimage.generate_binary_structure(3, 1)
    sub = ndimage.binary_closing(sub, structure=st, iterations=1)
    labels, nlab = ndimage.label(sub, structure=st)
    sizes = np.bincount(labels.ravel())
    rows = []
    for lab in range(1, nlab + 1):
        size = int(sizes[lab])
        if size < 60:
            continue
        c = np.argwhere(labels == lab)
        gx = c[:, 0] + x0; gy = c[:, 1] + y0; gz = c[:, 2] + z0
        gflat = np.ravel_multi_index((gx, gy, gz), shape)
        step = max(1, len(gflat) // 4000)
        wp = pts[gflat[::step]].astype(np.float64)
        med = np.median(wp, axis=0)
        lo = np.percentile(wp, 3.0, axis=0); hi = np.percentile(wp, 97.0, axis=0)
        span = hi - lo
        prior_dist = float(np.linalg.norm(med[:2] - prior[:2])) if prior is not None else 0.0
        best_r = None; best_r_ratio = 0.0; best_r_score = -1.0
        for ri, hr in enumerate(overhead_hits):
            ov = int(np.sum(hr[gflat]))
            ratio = ov / max(1, size)
            app = appearance_similarity(ref_desc, overhead_descs[ri])
            rscore = ratio + 0.08 * app
            if rscore > best_r_score:
                best_r_score = rscore; best_r_ratio = ratio; best_r = ri
        compact_pen = (
            max(0.0, float(span[0]) - 175.0) / 80.0
            + max(0.0, float(span[1]) - 175.0) / 80.0
            + max(0.0, float(span[2]) - 345.0) / 150.0
        )
        too_flat_pen = max(0.0, 70.0 - float(span[2])) / 100.0
        score = math.log1p(size) + 3.0 * best_r_ratio - prior_dist / 240.0 - compact_pen - too_flat_pen
        rows.append({
            "label": lab,
            "size": size,
            "global_flat_indices": gflat,
            "median_world_cm": med.tolist(),
            "span_cm": span.tolist(),
            "distance_to_reference_moge_prior_xy_cm": prior_dist,
            "best_overhead_instance": best_r,
            "best_overhead_overlap_ratio": float(best_r_ratio),
            "score": float(score),
        })
    if not rows:
        return None, {"components_found": int(nlab), "components_scored": 0}
    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    gflat = best["global_flat_indices"]

    # The overhead view is only allowed to carve a player when it sees a large
    # fraction of the already-selected SAME compact component. This prevents the
    # v8 cross-person failure from returning through an overhead union mask.
    refined = False
    ri = best["best_overhead_instance"]
    if ri is not None and best["best_overhead_overlap_ratio"] >= 0.35:
        keep = overhead_hits[ri][gflat]
        if int(np.sum(keep)) >= 0.35 * len(gflat):
            gflat = gflat[keep]
            refined = True

    qa_rows = []
    for r in rows[:8]:
        rr = {k: v for k, v in r.items() if k != "global_flat_indices"}
        qa_rows.append(rr)
    qa = {
        "components_found": int(nlab),
        "components_scored": len(rows),
        "top_components": qa_rows,
        "selected_component": qa_rows[0],
        "overhead_carve_applied": bool(refined),
        "final_voxels": int(len(gflat)),
    }
    return gflat, qa


def match_reference_to_broadcast(ref_instances, br_instances, ref_hits, br_hits, pts, ref_priors, ref_descs, br_descs):
    nr, nb = len(ref_instances), len(br_instances)
    scores = np.full((nr, nb), -1e6, dtype=np.float64)
    metrics = [[None for _ in range(nb)] for _ in range(nr)]
    for i in range(nr):
        for j in range(nb):
            sim = appearance_similarity(ref_descs[i], br_descs[j])
            m = pair_metrics(ref_hits[i], br_hits[j], pts, ref_priors[i], sim)
            metrics[i][j] = m; scores[i, j] = m["score"]
    row, col = linear_sum_assignment(-scores)
    assignment = {int(i): int(j) for i, j in zip(row, col) if metrics[int(i)][int(j)]["overlap_voxels"] >= 120}
    return assignment, scores, metrics


def dense_raster(pts, cols, K, R, C, radius=1):
    img = np.zeros((H * W, 3), np.uint8); mask = np.zeros(H * W, np.uint8)
    if len(pts) == 0:
        return img.reshape(H, W, 3), mask.reshape(H, W)
    Xc = (R @ (pts.astype(np.float64) - C).T).T
    z = Xc[:, 2]
    q = (K @ Xc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    u0 = np.rint(uv[:, 0]).astype(np.int32); v0 = np.rint(uv[:, 1]).astype(np.int32)
    good = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 20.0)
    ids0 = np.where(good)[0]
    pix_all = []; dep_all = []; src_all = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu = u0[ids0] + dx; vv = v0[ids0] + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not np.any(ok):
                continue
            ids = ids0[ok]
            pix_all.append((vv[ok] * W + uu[ok]).astype(np.int32))
            dep_all.append(z[ids].astype(np.float32))
            src_all.append(ids.astype(np.int32))
    if not pix_all:
        return img.reshape(H, W, 3), mask.reshape(H, W)
    pix = np.concatenate(pix_all); dep = np.concatenate(dep_all); src = np.concatenate(src_all)
    order = np.argsort(dep, kind="stable")
    p = pix[order]
    _, first = np.unique(p, return_index=True)
    winners = order[first]
    wpix = pix[winners]; wsrc = src[winners]
    img[wpix] = cols[wsrc]; mask[wpix] = 255
    return img.reshape(H, W, 3), mask.reshape(H, W)


def mask_iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = int(np.sum(a & b)); union = int(np.sum(a | b))
    return float(inter / union) if union else 1.0


def orange_ball_candidates(image, cam):
    C, R, K = cam
    uv, depth, ok = v8.project_metric(cam, RIM.reshape(1, 3))
    if not ok[0]:
        return [], np.zeros((H, W), bool)
    rc = uv[0]
    rim_depth = max(float(depth[0]), 50.0)
    expected_r = float(np.clip(K[0, 0] * BALL_RADIUS_CM / rim_depth, 4.0, 45.0))
    expected_area = math.pi * expected_r * expected_r
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # NBA orange basketball; intentionally broad enough for arena white balance.
    raw = ((hsv[:, :, 0] >= 3) & (hsv[:, :, 0] <= 28) & (hsv[:, :, 1] >= 85) & (hsv[:, :, 2] >= 55))
    yy, xx = np.indices((H, W))
    near = (xx - rc[0]) ** 2 + (yy - rc[1]) ** 2 <= 230.0 ** 2
    m = (raw & near).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    out = []
    chosen_mask = np.zeros((H, W), bool)
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < max(12.0, 0.10 * expected_area) or area > max(3800.0, 7.0 * expected_area):
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if min(w, h) < 3 or max(w, h) / max(1.0, min(w, h)) > 2.4:
            continue
        comp = (lab == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = max([cv2.arcLength(c, True) for c in contours], default=0.0)
        circ = float(4.0 * math.pi * area / max(per * per, 1e-6))
        cx, cy = [float(v) for v in cents[i]]
        dist = float(np.linalg.norm(np.asarray([cx, cy]) - rc))
        area_pen = abs(math.log(max(area, 1.0) / max(expected_area, 1.0)))
        raw_score = dist / 230.0 + 0.65 * area_pen + 1.2 * max(0.0, 0.65 - circ)
        conf = float(1.0 / (1.0 + raw_score))
        out.append({"score": conf, "cx": cx, "cy": cy, "box": [x, y, x + w - 1, y + h - 1], "area": area, "circularity": circ, "distance_to_rim_px": dist})
    out.sort(key=lambda r: r["score"], reverse=True)
    if out:
        b = out[0]; x1, y1, x2, y2 = [int(v) for v in b["box"]]
        chosen_mask[max(0, y1 - 3):min(H, y2 + 4), max(0, x1 - 3):min(W, x2 + 4)] = True
    return out[:5], chosen_mask


def annotate(img, angle, body_iou, persons, ball_status):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (850, 70), (0, 0, 0), -1)
    cv2.putText(out, "JAZZ EVENT 489 | 3-CAMERA v9 | INSTANCE-SEPARATED METRIC 3D", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, f"orbit {angle:04.1f} deg | retained people {persons} | reference silhouette IoU {body_iou:0.3f} | {ball_status}", (12, 45), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, "per-person LAR↔Broadcast 3D hull; overhead only same-person support; dense target z-buffer; source RGB only", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, .33, (210,210,210), 1, cv2.LINE_AA)
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
    ap.add_argument("--full-arc", action="store_true")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    lar = base.find_one(args.frames_dir, "Left_Above_Rim")
    rar = base.find_one(args.frames_dir, "Right_Above_Rim")
    br = [p for p in sorted(args.frames_dir.rglob("*Broadcast*.png")) if "Mobile" not in p.name and "Other" not in p.name]
    if len(br) != 1: raise RuntimeError(f"Broadcast ambiguity {br}")
    paths = {REF: lar, MATCH: br[0], OVERHEAD: rar}
    images = {k: cv2.imread(str(p)) for k, p in paths.items()}
    for k, im in images.items():
        if im is None or im.shape[:2] != (H, W): raise RuntimeError(f"bad source {k}: {paths[k]}")
    cams = base.load_cameras(args.registry, args.rar_report, args.broadcast_event_frame)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    seg_model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

    sources = {}; instances = {}; priors = {}; descs = {}; body_union = {}; ball_obs = {}; ball_masks = {}
    qa = {"schema_version": 9, "game_id": "0022500301", "event_id": 489, "sources": {}, "players": [], "frames": []}
    for label in CAMERA_ORDER:
        C, R, K = cams[label]; im = images[label]
        depth, _, valid, _, _ = moge_infer(depth_model, im, args.tokens)
        align, dqa = v3.robust_depth_align(depth, valid, K, R, C)
        dyn, inst, coco_balls = v6.detect_near_play(seg_model, im, K, R, C)
        instances[label] = inst
        um = np.zeros((H, W), bool)
        p = []; d = []
        for row in inst:
            row["support_mask"] = support_mask(row["mask"])
            um |= row["support_mask"]
            p.append(instance_metric_prior(depth, valid, align, K, R, C, row["mask"]))
            d.append(descriptor(im, row["mask"]))
        body_union[label] = um; priors[label] = p; descs[label] = d
        floor_vis, board_vis = v5.source_visibility_v5(im, depth, valid, dyn, K, R, C, align)
        sources[label] = {"image": im, "C": C, "R": R, "K": K, "floor_vis": floor_vis, "board_vis": board_vis, "dynamic": dyn}
        orange, orange_mask = orange_ball_candidates(im, cams[label])
        merged = list(coco_balls[:3]) + orange
        merged.sort(key=lambda r: float(r["score"]), reverse=True)
        ball_obs[label] = merged[:6]; ball_masks[label] = orange_mask
        qa["sources"][label] = {
            "file": paths[label].name,
            "person_instances": len(inst),
            "body_union_pixels": int(um.sum()),
            "depth_alignment": dqa,
            "moge_instance_priors_world_cm": [None if x is None else [float(y) for y in x] for x in p],
            "instance_foot_world_cm": [row["foot_world_cm"] for row in inst],
            "coco_ball_detections": coco_balls[:3],
            "orange_ball_candidates": orange,
            "camera_center_cm": [float(x) for x in C],
        }
        cv2.imwrite(str(args.out / f"{label.replace(' ','_')}_instance_union.png"), um.astype(np.uint8) * 255)

    xs, ys, zs, shape, pts = make_grid(args.voxel_cm)
    ref_hits = [v8.mask_membership(cams[REF], pts, r["support_mask"]) for r in instances[REF]]
    br_hits = [v8.mask_membership(cams[MATCH], pts, r["support_mask"]) for r in instances[MATCH]]
    rhits = [v8.mask_membership(cams[OVERHEAD], pts, r["support_mask"]) for r in instances[OVERHEAD]]

    assignment, score_matrix, pair_matrix = match_reference_to_broadcast(
        instances[REF], instances[MATCH], ref_hits, br_hits, pts, priors[REF], descs[REF], descs[MATCH]
    )
    qa["matching"] = {
        "reference": REF,
        "matched_view": MATCH,
        "score_matrix": score_matrix.tolist(),
        "assignment": {str(k): int(v) for k, v in assignment.items()},
        "pair_metrics": [[m for m in row] for row in pair_matrix],
    }

    body_pts = []; body_cols = []
    ref_image = images[REF]
    for i, inst in enumerate(instances[REF]):
        if i not in assignment:
            qa["players"].append({"reference_instance": i, "status": "UNMATCHED"})
            continue
        j = assignment[i]
        pair_occ = ref_hits[i] & br_hits[j]
        gflat, cqa = component_from_pair(pair_occ, shape, pts, priors[REF][i], rhits, descs[OVERHEAD], descs[REF][i])
        pqa = {
            "reference_instance": i,
            "broadcast_instance": j,
            "reference_score": float(inst["score"]),
            "reference_box": inst["box"],
            "reference_foot_world_cm": inst["foot_world_cm"],
            "reference_moge_prior_world_cm": None if priors[REF][i] is None else priors[REF][i].tolist(),
            "pair_metric": pair_matrix[i][j],
            "component_selection": cqa,
        }
        if gflat is None or len(gflat) < 80:
            pqa["status"] = "NO_COMPACT_COMPONENT"; qa["players"].append(pqa); continue
        pp = pts[gflat]
        uv, _, validp = v8.project_metric(cams[REF], pp)
        cc = v8.bilinear_sample(ref_image, uv)
        keep = validp & np.isfinite(uv).all(axis=1)
        pp = pp[keep]; cc = cc[keep]
        if len(pp) < 80:
            pqa["status"] = "NO_TEXTURED_COMPONENT"; qa["players"].append(pqa); continue
        body_pts.append(pp); body_cols.append(cc)
        pqa["status"] = "RETAINED"; pqa["render_voxels"] = int(len(pp)); qa["players"].append(pqa)

    if not body_pts:
        raise RuntimeError("No player volumes survived instance-separated reconstruction")
    body_pts = np.concatenate(body_pts).astype(np.float32); body_cols = np.concatenate(body_cols).astype(np.uint8)
    qa["retained_player_count"] = int(sum(1 for x in qa["players"] if x.get("status") == "RETAINED"))
    qa["body_voxels"] = int(len(body_pts))
    qa["grid"] = {"voxel_cm": float(args.voxel_cm), "shape": list(shape), "voxel_count": int(len(pts)), "bounds_cm": {"x": [float(xs[0]), float(xs[-1])], "y": [float(ys[0]), float(ys[-1])], "z": [float(zs[0]), float(zs[-1])]}}

    ball_center, ball_qa = v8.triangulate_ball(cams, ball_obs)
    qa["ball"] = ball_qa
    ball_pts = v8.sphere_points(ball_center) if ball_center is not None else np.empty((0, 3), np.float32)
    ball_cols = np.empty((0, 3), np.uint8)
    if len(ball_pts):
        best_label = max(CAMERA_ORDER, key=lambda k: float(ball_obs[k][0]["score"]) if ball_obs[k] else -1.0)
        cand = ball_obs[best_label][0]
        im = images[best_label]
        x1, y1, x2, y2 = [int(round(x)) for x in cand["box"]]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
        crop = im[y1:y2+1, x1:x2+1]
        med = np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8) if crop.size else np.asarray([50, 110, 190], np.uint8)
        ball_cols = np.repeat(med.reshape(1, 3), len(ball_pts), axis=0)

    rp = v8.rim_points()
    ruv, _, rv = v8.project_metric(cams[REF], rp)
    rcols = v8.bilinear_sample(images[REF], ruv)
    rp = rp[rv]; rcols = rcols[rv]

    C0, R0, K0 = cams[REF]
    if args.full_arc:
        angles = np.r_[np.zeros(12), np.linspace(0.0, 25.0, 76), np.full(18, 25.0)]
    else:
        angles = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], dtype=np.float64)

    ref_union = body_union[REF]
    ref_iou = None
    for fi, ang in enumerate(angles):
        Rt, Ct = base.orbit_pose(C0, R0, RIM, float(ang))
        Pf, sf = v4.virtual_plane_points(K0, Rt, Ct, "floor")
        floor_img, floor_mask = v4.sample_plane_from_sources(Pf, sf, sources, "floor")
        Pb, sb = v4.virtual_plane_points(K0, Rt, Ct, "board")
        board_img, board_mask = v4.sample_plane_from_sources(Pb, sb, sources, "board")
        img = floor_img.copy(); grounded = floor_mask.copy(); img[board_mask] = board_img[board_mask]; grounded[board_mask] = True

        bimg, bmask = dense_raster(body_pts, body_cols, K0, Rt, Ct, radius=1)
        fg_pts = [body_pts, rp]; fg_cols = [body_cols, rcols]
        if len(ball_pts): fg_pts.append(ball_pts); fg_cols.append(ball_cols)
        fimg, fmask = dense_raster(np.concatenate(fg_pts), np.concatenate(fg_cols), K0, Rt, Ct, radius=1)
        fm = fmask > 0; img[fm] = fimg[fm]
        biou = mask_iou(bmask > 0, ref_union) if abs(float(ang)) < 1e-8 else (ref_iou if ref_iou is not None else 0.0)
        if abs(float(ang)) < 1e-8: ref_iou = biou
        total = grounded | fm
        qa["frames"].append({
            "frame": int(fi), "angle_deg": float(ang), "source_grounded_fraction": float(total.mean()),
            "foreground_pixels": int(fm.sum()), "body_pixels": int(np.sum(bmask > 0)),
            "reference_body_silhouette_iou_at_zero_deg": float(biou) if abs(float(ang)) < 1e-8 else None,
        })
        name = f"frame_{fi:03d}.png" if args.full_arc else f"v9_{int(round(float(ang))):02d}deg.png"
        cv2.imwrite(str(args.out / name), annotate(img, float(ang), float(ref_iou or 0.0), qa["retained_player_count"], ball_qa["status"]))

    qa["reference_zero_degree_body_iou"] = float(ref_iou or 0.0)
    qa["status"] = "STATIC_SIX_RENDERED" if not args.full_arc else "FULL_ARC_RENDERED"
    qa["method"] = "instance-separated Left Above Rim to Broadcast metric silhouette-cone intersections; compact connected-component selection; conditional same-player overhead support; dense target-view z-buffer; exact court/backboard planes; source RGB only"
    qa["v8_failure_addressed"] = "eliminates union-of-people cross-view ghost volumes and disallows non-reference-supported bodies at the 0-degree anchor"
    (args.out / "three_camera_instance_volume_v9_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"status": qa["status"], "retained_player_count": qa["retained_player_count"], "body_voxels": qa["body_voxels"], "reference_zero_degree_body_iou": qa["reference_zero_degree_body_iou"], "ball": qa["ball"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
