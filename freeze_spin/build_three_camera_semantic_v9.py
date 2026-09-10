from __future__ import annotations

"""Three-camera semantic volumetric diagnostic v9.

v8 intersected one union-of-all-players silhouette per camera. In a crowded
paint that permits the silhouette cone of player A in one view to intersect
player B in another, producing phantom merged bodies. v9 preserves person
identity before 3-D reconstruction:

1. detect individual near-play person instances in each accepted camera;
2. associate Left Above Rim instances one-to-one with Broadcast and Right Above
   Rim using metric silhouette-cone overlap, with foot-world location only as a
   secondary cue;
3. build one spatially bounded visual hull per associated physical player;
4. colour each player's surface only from source cameras assigned to that track;
5. augment ball detection deterministically inside a projected metric-rim ROI
   and triangulate a single physical 3-D ball when >=2 views support it.

Court/backboard/rim remain exact metric geometry. There is no generative fill,
optical-flow morph, crossfade or invented player/ball texture. Unsupported
pixels remain black. This script renders only native 960x540 source geometry;
upscaling is intentionally outside this workflow.
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


def _mask_hit(cam, pts, mask):
    return v8.mask_membership(cam, pts, mask.astype(bool), chunk=300_000)


def _grid(bounds, step):
    x0, x1, y0, y1, z0, z1 = bounds
    xs = np.arange(x0, x1 + 0.5 * step, step, dtype=np.float32)
    ys = np.arange(y0, y1 + 0.5 * step, step, dtype=np.float32)
    zs = np.arange(z0, z1 + 0.5 * step, step, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    return pts, X.shape, (xs, ys, zs)


def coarse_instance_support(cams, instances, step=10.0):
    bounds = v8.action_bounds(instances)
    pts, _, _ = _grid(bounds, step)
    supports = {}
    for label in CAMERA_ORDER:
        rows = []
        for idx, inst in enumerate(instances[label]):
            hit = _mask_hit(cams[label], pts, inst["mask"])
            rows.append({"idx": idx, "hit": hit, "count": int(hit.sum())})
        supports[label] = rows
    return pts, supports, bounds


def pair_score(a, b, inst_a, inst_b):
    inter = int(np.count_nonzero(a["hit"] & b["hit"]))
    if inter < 12:
        return -1e6, {"intersection_voxels": inter, "cosine": 0.0, "foot_distance_cm": None}
    denom = math.sqrt(max(1, a["count"]) * max(1, b["count"]))
    cosine = inter / denom
    pa = np.asarray(inst_a["foot_world_cm"], np.float64)
    pb = np.asarray(inst_b["foot_world_cm"], np.float64)
    d = None
    foot_term = 0.0
    if np.isfinite(pa).all() and np.isfinite(pb).all():
        d = float(np.linalg.norm(pa[:2] - pb[:2]))
        foot_term = 0.85 * math.exp(-d / 140.0)
        if d > 360.0:
            foot_term -= min(2.0, (d - 360.0) / 180.0)
    score = math.log1p(inter) + 5.0 * cosine + foot_term
    return score, {"intersection_voxels": inter, "cosine": float(cosine), "foot_distance_cm": d}


def one_to_one_match(primary_label, other_label, instances, supports):
    A = supports[primary_label]
    B = supports[other_label]
    if not A or not B:
        return {}, []
    mat = np.full((len(A), len(B)), -1e6, np.float64)
    details = {}
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            s, q = pair_score(a, b, instances[primary_label][i], instances[other_label][j])
            mat[i, j] = s
            details[(i, j)] = q
    rr, cc = linear_sum_assignment(-mat)
    out = {}
    qa = []
    for i, j in zip(rr, cc):
        s = float(mat[i, j])
        q = details[(int(i), int(j))]
        accepted = s > 4.2 and q["intersection_voxels"] >= 18 and q["cosine"] >= 0.004
        qa.append({"primary_instance": int(i), "other_instance": int(j), "score": s, "accepted": bool(accepted), **q})
        if accepted:
            out[int(i)] = int(j)
    return out, qa


def track_anchor(track, instances):
    pts = []
    for label, idx in track.items():
        if idx is None:
            continue
        p = np.asarray(instances[label][idx]["foot_world_cm"], np.float64)
        if np.isfinite(p).all() and np.linalg.norm(p[:2] - RIM[:2]) < 700.0:
            pts.append(p)
    if pts:
        p = np.median(np.asarray(pts), axis=0)
        return float(p[0]), float(p[1])
    return float(RIM[0]), float(RIM[1])


def build_track_hull(cams, track, instances, voxel_cm=4.0):
    ax, ay = track_anchor(track, instances)
    x0, x1 = max(-260.0, ax - 100.0), min(1180.0, ax + 100.0)
    y0, y1 = max(-590.0, ay - 100.0), min(590.0, ay + 100.0)
    z0, z1 = -8.0, 330.0
    pts, shape, axes = _grid((x0, x1, y0, y1, z0, z1), voxel_cm)
    matched = [k for k in CAMERA_ORDER if track.get(k) is not None]
    if len(matched) < 2:
        return np.empty((0, 3), np.float32), {"status": "REJECTED_LT2_VIEWS", "views": matched}
    counts = np.zeros(len(pts), np.uint8)
    per_view = {}
    for label in matched:
        mask = instances[label][track[label]]["mask"].astype(bool)
        h = _mask_hit(cams[label], pts, mask)
        counts += h.astype(np.uint8)
        per_view[label] = int(h.sum())
    occ = counts.reshape(shape) >= 2
    st = ndimage.generate_binary_structure(3, 1)
    occ = ndimage.binary_closing(occ, structure=st, iterations=1)
    labels, n = ndimage.label(occ, structure=st)
    sizes = np.bincount(labels.ravel()) if n else np.asarray([0])
    if n == 0:
        return np.empty((0, 3), np.float32), {"status": "REJECTED_EMPTY", "views": matched, "per_view_inside": per_view}
    largest = int(np.argmax(sizes[1:]) + 1)
    occ = labels == largest
    if int(occ.sum()) < 70:
        return np.empty((0, 3), np.float32), {"status": "REJECTED_SMALL", "views": matched, "largest_component_voxels": int(occ.sum()), "per_view_inside": per_view}
    er = ndimage.binary_erosion(occ, structure=st, border_value=0)
    surf = occ & ~er
    idx = np.flatnonzero(surf.ravel())
    surface_pts = pts[idx]
    support = counts[idx]
    return surface_pts, {
        "status": "TRACK_HULL_OK",
        "views": matched,
        "anchor_xy_cm": [ax, ay],
        "bounds_cm": {"x": [float(axes[0][0]), float(axes[0][-1])], "y": [float(axes[1][0]), float(axes[1][-1])], "z": [float(axes[2][0]), float(axes[2][-1])]},
        "occupied_voxels": int(occ.sum()),
        "surface_voxels": int(len(surface_pts)),
        "surface_support_2": int(np.sum(support == 2)),
        "surface_support_3": int(np.sum(support == 3)),
        "per_view_inside": per_view,
    }


def roi_ball_candidates(image, cam, existing):
    out = [dict(x, source="maskrcnn") for x in existing[:3]]
    uv, _, valid = v8.project_metric(cam, RIM.reshape(1, 3))
    if not valid[0]:
        return out
    cx0, cy0 = [float(x) for x in uv[0]]
    rad = 135
    x0 = max(0, int(cx0 - rad)); x1 = min(W, int(cx0 + rad + 1))
    y0 = max(0, int(cy0 - rad)); y1 = min(H, int(cy0 + rad + 1))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return out
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([2, 75, 55], np.uint8), np.array([24, 255, 255], np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        xx, yy, ww, hh, area = [int(v) for v in stats[i]]
        if area < 12 or area > 2400 or ww < 3 or hh < 3:
            continue
        ar = ww / max(1.0, float(hh))
        if not (0.45 <= ar <= 2.2):
            continue
        comp = (lab[yy:yy+hh, xx:xx+ww] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        circ = float(4.0 * math.pi * max(1.0, cv2.contourArea(c)) / max(1e-6, per * per))
        gx = float(cents[i][0] + x0); gy = float(cents[i][1] + y0)
        dr = math.hypot(gx - cx0, gy - cy0)
        if dr > rad:
            continue
        score = 0.06 + 0.22 * max(0.0, min(1.0, circ)) + 0.08 * (1.0 - dr / rad)
        out.append({"score": float(score), "box": [float(xx+x0), float(yy+y0), float(xx+ww+x0), float(yy+hh+y0)], "cx": gx, "cy": gy, "source": "rim_roi_orange", "circularity": circ})
    out.sort(key=lambda q: float(q["score"]), reverse=True)
    kept = []
    for q in out:
        if any((q["cx"]-k["cx"])**2 + (q["cy"]-k["cy"])**2 < 9.0**2 for k in kept):
            continue
        kept.append(q)
        if len(kept) >= 8:
            break
    return kept


def source_track_surface(label, cams, pts, track, instances, images, voxel_cm):
    if track.get(label) is None:
        return np.zeros(len(pts), bool), np.zeros((len(pts), 3), np.uint8), {"source_visible_surface_voxels": 0}
    mask = instances[label][track[label]]["mask"].astype(bool)
    return v8.source_surface_data(label, cams[label], pts, mask, images[label], voxel_cm)


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
    if len(br) != 1:
        raise RuntimeError(f"Broadcast ambiguity {br}")
    paths = {"Left Above Rim": lar, "Broadcast": br[0], "Right Above Rim": rar}
    images = {k: cv2.imread(str(p)) for k, p in paths.items()}
    for k, im in images.items():
        if im is None or im.shape[:2] != (H, W):
            raise RuntimeError(f"bad source frame {k}: {paths[k]}")
    cams = base.load_cameras(args.registry, args.rar_report, args.broadcast_event_frame)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    seg_model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

    instances = {}; balls = {}; sources = {}; qa = {"schema_version": 9, "sources": {}, "frames": []}
    for label in CAMERA_ORDER:
        C, R, K = cams[label]; image = images[label]
        depth, _, valid, _, _ = moge_infer(depth_model, image, args.tokens)
        align, dqa = v3.robust_depth_align(depth, valid, K, R, C)
        dyn, inst, bdet = v6.detect_near_play(seg_model, image, K, R, C)
        fv, bv = v5.source_visibility_v5(image, depth, valid, dyn, K, R, C, align)
        sources[label] = {"image": image, "C": C, "R": R, "K": K, "floor_vis": fv, "board_vis": bv, "dynamic": dyn}
        instances[label] = inst
        balls[label] = roi_ball_candidates(image, cams[label], bdet)
        qa["sources"][label] = {
            "file": paths[label].name,
            "depth_alignment_for_static_plane_visibility": dqa,
            "near_play_person_instances": len(inst),
            "instance_support": [{"score": float(x["score"]), "foot_px": x["foot_px"], "foot_world_cm": x["foot_world_cm"], "box": x["box"]} for x in inst],
            "ball_candidates": balls[label],
            "floor_visible_pixels": int(fv.sum()), "board_visible_pixels": int(bv.sum()),
            "camera_center_cm": [float(x) for x in C],
        }
        for j, row in enumerate(inst):
            cv2.imwrite(str(args.out / f"{label.replace(' ','_')}_person_{j:02d}.png"), row["mask"].astype(np.uint8) * 255)

    _, coarse, coarse_bounds = coarse_instance_support(cams, instances, step=10.0)
    mb, qab = one_to_one_match("Left Above Rim", "Broadcast", instances, coarse)
    mr, qar = one_to_one_match("Left Above Rim", "Right Above Rim", instances, coarse)
    tracks = []
    for i in range(len(instances["Left Above Rim"])):
        tr = {"Left Above Rim": i, "Broadcast": mb.get(i), "Right Above Rim": mr.get(i)}
        if sum(v is not None for v in tr.values()) >= 2:
            tracks.append(tr)
    if not tracks:
        raise RuntimeError("identity association produced no >=2-view player tracks")

    track_points = []; track_vis = []; track_cols = []; track_qas = []
    for ti, tr in enumerate(tracks):
        pts, tqa = build_track_hull(cams, tr, instances, voxel_cm=args.voxel_cm)
        tqa["track_id"] = ti; tqa["instances"] = tr
        if len(pts) == 0:
            track_qas.append(tqa); continue
        vis = {}; cols = {}
        for label in CAMERA_ORDER:
            vv, cc, sqa = source_track_surface(label, cams, pts, tr, instances, images, args.voxel_cm)
            vis[label] = vv; cols[label] = cc
            tqa[f"{label}_visible_surface_voxels"] = int(sqa["source_visible_surface_voxels"])
        track_points.append(pts); track_vis.append(vis); track_cols.append(cols); track_qas.append(tqa)
    if not track_points:
        raise RuntimeError("all associated track hulls rejected")

    ball_center, ball_qa = v8.triangulate_ball(cams, balls)
    ball_pts = v8.sphere_points(ball_center) if ball_center is not None else np.empty((0, 3), np.float32)
    ball_vis = {}; ball_cols = {}
    if len(ball_pts):
        for label in CAMERA_ORDER:
            uv, _, ok = v8.project_metric(cams[label], ball_center.reshape(1, 3))
            bm = np.zeros((H, W), np.uint8)
            if ok[0] and balls[label]:
                q = min(balls[label], key=lambda b: (b["cx"]-uv[0,0])**2 + (b["cy"]-uv[0,1])**2)
                r = max(4, int(round(0.35 * max(q["box"][2]-q["box"][0], q["box"][3]-q["box"][1]))))
                cv2.circle(bm, (int(round(q["cx"])), int(round(q["cy"]))), r, 1, -1)
            vv, cc, _ = v8.source_surface_data(label, cams[label], ball_pts, bm > 0, images[label], 2.0)
            ball_vis[label] = vv; ball_cols[label] = cc

    rp = v8.rim_points(); rim_vis = {}; rim_cols = {}
    for label in CAMERA_ORDER:
        vv, cc = v8.source_generic_surface_data(cams[label], rp, images[label])
        rim_vis[label] = vv; rim_cols[label] = cc
        qa["sources"][label]["metric_rim_plane_exclusion_pixels"] = v8.subtract_metric_rim_from_plane_visibility(sources[label])

    C0, R0, K0 = cams["Left Above Rim"]
    angles = np.r_[np.zeros(12), np.linspace(0.0, 25.0, 76), np.full(18, 25.0)]
    for fi, ang in enumerate(angles):
        Rt, Ct = base.orbit_pose(C0, R0, RIM, float(ang))
        Pf, sf = v4.virtual_plane_points(K0, Rt, Ct, "floor")
        floor_img, floor_mask = v4.sample_plane_from_sources(Pf, sf, sources, "floor")
        Pb, sb = v4.virtual_plane_points(K0, Rt, Ct, "board")
        board_img, board_mask = v4.sample_plane_from_sources(Pb, sb, sources, "board")
        img = floor_img.copy(); grounded = floor_mask.copy(); img[board_mask] = board_img[board_mask]; grounded[board_mask] = True

        fg_pts = []; fg_cols = []; body_rendered_points = 0
        owner_qa = {}
        for ti, pts in enumerate(track_points):
            chosen, owners = v8.select_view_colours(pts, Ct, cams, track_vis[ti], track_cols[ti])
            ok = owners >= 0
            if np.any(ok):
                fg_pts.append(pts[ok]); fg_cols.append(chosen[ok]); body_rendered_points += int(ok.sum())
            owner_qa[str(ti)] = {CAMERA_ORDER[j]: int(np.sum(owners == j)) for j in range(len(CAMERA_ORDER))}
        rim_chosen, rim_owners = v8.select_view_colours(rp, Ct, cams, rim_vis, rim_cols)
        if np.any(rim_owners >= 0):
            fg_pts.append(rp[rim_owners >= 0]); fg_cols.append(rim_chosen[rim_owners >= 0])
        if len(ball_pts):
            bc, bo = v8.select_view_colours(ball_pts, Ct, cams, ball_vis, ball_cols)
            if np.any(bo >= 0):
                fg_pts.append(ball_pts[bo >= 0]); fg_cols.append(bc[bo >= 0])
        if fg_pts:
            fg_img, fg_m = v4.raster((np.concatenate(fg_pts), np.concatenate(fg_cols)), K0, Rt, Ct, radius=2)
            fm = fg_m > 0; img[fm] = fg_img[fm]
        else:
            fm = np.zeros((H, W), bool)
        total = grounded | fm
        cv2.rectangle(img, (0, 0), (820, 67), (0,0,0), -1)
        cv2.putText(img, "3-CAMERA DIAGNOSTIC v9 | IDENTITY-AWARE PER-PLAYER 3D", (12,21), cv2.FONT_HERSHEY_SIMPLEX, .46, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, f"orbit {float(ang):04.1f} deg | tracks {len(track_points)} | surface {sum(len(x) for x in track_points)} | ball {ball_qa['status']}", (12,46), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, "individual-mask association -> bounded per-player hulls; exact metric court/board/rim; source pixels only", (12,63), cv2.FONT_HERSHEY_SIMPLEX, .34, (210,210,210), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.out / f"frame_{fi:03d}.png"), img)
        qa["frames"].append({"frame": int(fi), "angle_deg": float(ang), "source_grounded_fraction": float(total.mean()), "foreground_pixels": int(fm.sum()), "body_rendered_surface_points": body_rendered_points, "track_source_ownership": owner_qa})

    qa.update({
        "status": "DIAGNOSTIC_THREE_CAMERA_SEMANTIC_V9_RENDERED",
        "game_id": "0022500301", "event_id": 489,
        "source_resolution": [W, H],
        "coarse_association_bounds_cm": [float(x) for x in coarse_bounds],
        "association": {"Left_to_Broadcast": qab, "Left_to_RightAboveRim": qar},
        "tracks": track_qas,
        "accepted_track_count": int(len(track_points)),
        "total_player_surface_voxels": int(sum(len(x) for x in track_points)),
        "ball": ball_qa,
        "minimum_source_grounded_fraction": float(min(x["source_grounded_fraction"] for x in qa["frames"])),
        "representation_change_from_v8": "person identity preserved before hull construction; no union-of-crowd silhouette intersection",
        "appearance_policy": "real source RGB samples only from cameras assigned to each physical player track; no generated fill/crossfade/morph",
        "render_resolution_policy": "native source geometry only (960x540); no UHD upscale in reconstruction workflow",
        "certification_scope": "three-camera R&D diagnostic only; does not satisfy >=4-camera production gate",
    })
    (args.out / "three_camera_semantic_v9_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"status": qa["status"], "tracks": qa["accepted_track_count"], "surface": qa["total_player_surface_voxels"], "ball": ball_qa, "minimum": qa["minimum_source_grounded_fraction"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
