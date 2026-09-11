from __future__ import annotations

"""v22: hard-lock the focal black-uniform #12 subject across all three solved views.

User visual QA on v20 identified a fundamental foreground failure: the focal
black-uniform #12 player visible at the rim in the native source frame is not
reliably preserved in the novel views.  The earlier pipeline proved that *a*
three-camera human track exists, but did not prove that this track is the focal
subject.  That invalidates any static free-view pass.

v22 therefore treats focal identity preservation as a hard gate.  It uses manual
source-frame seed boxes for the black-uniform #12 subject, selects the RF-DETR pose
inside each seed using spatial overlap plus dark-uniform evidence, builds a fresh
pose-prompted identity mask in every camera, and injects one explicit three-view
focal track into the existing reconstruction.  Any anonymous track reusing the
same selected pose is removed to prevent duplicates.

The production renderer remains source-grounded and native 960x540.  No generated
texture, inpainting, interpolation fill, upscale or UHD is introduced.  Additional
focal-only masks/overlays are diagnostics only and are used to assert that the
focal mesh has visible contribution at every novel angle.
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
from freeze_spin import build_three_camera_mesh_v20 as v20


# Manual visual anchors for the black-uniform #12 focal player in the exact v11
# frames.  These are identity prompts, not camera calibration constraints.
FOCAL_SEEDS = {
    v12.A: {"bbox": [438.0, 160.0, 512.0, 326.0], "point": [474.0, 246.0]},
    v12.B: {"bbox": [472.0, 100.0, 570.0, 312.0], "point": [518.0, 205.0]},
    v12.C: {"bbox": [514.0, 184.0, 662.0, 340.0], "point": [590.0, 254.0]},
}

_ORIGINAL_ATTACH_FIXED = v12b.attach_right_above_rim_fixed
_ORIGINAL_BUILD = v14._ORIGINAL_BUILD_MESH
_FOCAL_AUDIT = {}
_FOCAL_MESH = None
_ALL_MESHES = []


def _box_center(box):
    x1, y1, x2, y2 = map(float, box)
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], np.float64)


def _dark_torso_evidence(image, pose, idx):
    box = np.asarray(pose["boxes"][idx], np.float64)
    xy = np.asarray(pose["xy"][idx], np.float64)
    cf = np.asarray(pose["conf"][idx], np.float64)
    mask = np.zeros(image.shape[:2], np.uint8)
    torso = [5, 6, 12, 11]
    if all(j < len(cf) and cf[j] >= 0.12 and np.isfinite(xy[j]).all() for j in torso):
        cv2.fillConvexPoly(mask, np.rint(xy[torso]).astype(np.int32), 255, cv2.LINE_AA)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    else:
        x1, y1, x2, y2 = box
        xa = max(0, int(round(x1 + 0.22 * (x2 - x1))))
        xb = min(image.shape[1] - 1, int(round(x1 + 0.78 * (x2 - x1))))
        ya = max(0, int(round(y1 + 0.22 * (y2 - y1))))
        yb = min(image.shape[0] - 1, int(round(y1 + 0.70 * (y2 - y1))))
        if xb > xa and yb > ya:
            mask[ya:yb + 1, xa:xb + 1] = 255
    pix = image[mask > 0]
    if len(pix) < 20:
        return {"dark_fraction": 0.0, "median_v": 255.0, "sample_pixels": int(len(pix))}
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    vv = hsv[:, 2].astype(np.float32)
    return {
        "dark_fraction": float(np.mean(vv < 120.0)),
        "median_v": float(np.median(vv)),
        "sample_pixels": int(len(vv)),
    }


def _select_focal_pose(label, pose):
    seed = FOCAL_SEEDS[label]
    sb = np.asarray(seed["bbox"], np.float64)
    sp = np.asarray(seed["point"], np.float64)
    rows = []
    for i, box in enumerate(np.asarray(pose["boxes"], np.float64)):
        ov = float(v12.iou(sb, box))
        c = _box_center(box)
        dist = float(np.linalg.norm(c - sp))
        dark = _dark_torso_evidence(v12.IMAGES[label], pose, i)
        det = float(pose["det_conf"][i]) if i < len(pose["det_conf"]) else 0.0
        score = 3.2 * ov + 1.55 * math.exp(-dist / 95.0) + 1.45 * dark["dark_fraction"] + 0.20 * det
        rows.append({
            "pose_index": int(i), "score": float(score), "seed_iou": ov,
            "center_distance_px": dist, "detection_confidence": det, **dark,
            "pose_box": [float(x) for x in box],
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    if not rows:
        raise RuntimeError(f"v22 no RF-DETR poses for focal subject in {label}")
    best = rows[0]
    if best["seed_iou"] < 0.035 or best["center_distance_px"] > 125.0:
        raise RuntimeError(f"v22 focal pose seed failed for {label}: {best}")
    return int(best["pose_index"]), {"seed": seed, "selected": best, "top_candidates": rows[:5]}


def _dedupe_tracks(tracks, instances, target_pose):
    kept, dropped = [], []
    for ti, tr in enumerate(tracks):
        hits = []
        for label in v12.CAMERAS:
            mi = tr.get(label)
            if mi is None or mi < 0 or mi >= len(instances[label]):
                continue
            p = instances[label][mi].get("rfdetr_pose")
            if p and int(p.get("pose_index", -999)) == int(target_pose[label]):
                hits.append(label)
        if hits:
            dropped.append({"track_index": int(ti), "matching_focal_pose_views": hits})
        else:
            kept.append(tr)
    return kept, dropped


def attach_right_above_rim_focal_locked(cams, instances, mb, exact_qa):
    global _FOCAL_AUDIT
    tracks, audit = _ORIGINAL_ATTACH_FIXED(cams, instances, mb, exact_qa)

    target_pose = {}
    selection = {}
    prompt_masks = {}
    mask_instances = {}
    for label in v12.CAMERAS:
        pose = v12b.POSE_CACHE.get(label)
        if pose is None:
            raise RuntimeError(f"v22 pose cache missing {label}")
        pi, sq = _select_focal_pose(label, pose)
        target_pose[label] = int(pi)
        selection[label] = sq

        # Always request an identity-specific pose-prompt mask even if Mask R-CNN
        # already attached this pose.  The rim collision is exactly where generic
        # instance masks can merge the focal player with the defender.
        mi, mq = v12b._append_pose_prompt_instance(cams, label, instances[label], pose, pi)
        if mi is None:
            mi = v12b._find_mask_for_pose(instances[label], pi)
            if mi is None:
                raise RuntimeError(f"v22 could not build focal mask in {label}: {mq}")
            mq = {**mq, "fallback_existing_mask_instance": int(mi)}
        mask_instances[label] = int(mi)
        prompt_masks[label] = mq

    tracks, dropped = _dedupe_tracks(tracks, instances, target_pose)
    focal_track = {
        v12.A: mask_instances[v12.A],
        v12.B: mask_instances[v12.B],
        v12.C: mask_instances[v12.C],
        "two_view_joints": v12.two_view_joints(
            cams, instances[v12.A][mask_instances[v12.A]], instances[v12.B][mask_instances[v12.B]], conf_min=0.14
        ),
        "appearance": {"class": "dark"},
        "focal_subject_lock": "black_uniform_12_user_visual_anchor",
        "rar_attachment": {"method": "v22_manual_visual_anchor_three_view"},
    }
    tracks.append(focal_track)
    _FOCAL_AUDIT = {
        "status": "FOCAL_THREE_VIEW_TRACK_INJECTED",
        "subject_definition": "black-uniform #12 player identified by user visual QA in the exact v11 source frames",
        "pose_selection": selection,
        "selected_pose_indices": target_pose,
        "selected_mask_instances": mask_instances,
        "pose_prompt_masks": prompt_masks,
        "dropped_duplicate_tracks": dropped,
        "three_view_track": True,
    }
    audit["v22_focal_subject_lock"] = _FOCAL_AUDIT
    return tracks, audit


def build_mesh_capture(cams, track, instances, voxel_cm=2.5):
    global _FOCAL_MESH, _ALL_MESHES
    mesh, qa = _ORIGINAL_BUILD(cams, track, instances, voxel_cm=voxel_cm)
    focal = bool(track.get("focal_subject_lock"))
    qa["v22_focal_subject"] = focal
    if focal:
        qa["v22_focal_subject_definition"] = track.get("focal_subject_lock")
    if mesh is not None:
        mesh["v22_focal_subject"] = focal
        _ALL_MESHES.append(mesh)
        if focal:
            _FOCAL_MESH = mesh
    return mesh, qa


def _projected_silhouette(mesh, cam):
    C, R, K = cam
    uv, depth, valid = v12.v8.project_metric((C, R, K), mesh["verts"].astype(np.float64))
    mask = np.zeros((v12.H, v12.W), np.uint8)
    for f in mesh["faces"]:
        if not np.all(valid[f]):
            continue
        p = uv[f]
        if not np.isfinite(p).all():
            continue
        if np.max(p[:, 0]) < 0 or np.min(p[:, 0]) >= v12.W or np.max(p[:, 1]) < 0 or np.min(p[:, 1]) >= v12.H:
            continue
        area = abs(float(v12b.cross_compat(p[1] - p[0], p[2] - p[0])))
        if area < 0.12 or area > 6500:
            continue
        cv2.fillConvexPoly(mask, np.rint(p).astype(np.int32), 255, cv2.LINE_8)
    return mask


def _post_focal_visibility(out: Path, cams):
    if _FOCAL_MESH is None:
        raise RuntimeError("v22 focal mesh was not accepted")
    C0, R0, K0 = cams[v12.A]
    rows = []
    overlay_frames = []
    for ang in (0, 5, 10, 15, 20, 25):
        Rt, Ct = v12.base.orbit_pose(C0, R0, v12.RIM, float(ang))
        sil = _projected_silhouette(_FOCAL_MESH, (Ct, Rt, K0))
        ys, xs = np.where(sil > 0)
        area = int(len(xs))
        if len(xs):
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            bh = int(ys.max() - ys.min() + 1); bw = int(xs.max() - xs.min() + 1)
        else:
            bbox = None; bh = bw = 0

        visible_effect = None
        if ang > 0:
            neutral = np.full((v12.H, v12.W, 3), 112, np.uint8)
            all_img, _ = v14.render_triangles_mask_safe(neutral, cams, K0, Rt, Ct, _ALL_MESHES)
            other = [m for m in _ALL_MESHES if m is not _FOCAL_MESH]
            wo_img, _ = v14.render_triangles_mask_safe(neutral, cams, K0, Rt, Ct, other)
            delta = np.max(cv2.absdiff(all_img, wo_img), axis=2) > 5
            visible_effect = int(delta.sum())
            cv2.imwrite(str(out / f"v22_focal_visible_effect_{ang:02d}deg.png"), delta.astype(np.uint8) * 255)

        cv2.imwrite(str(out / f"v22_focal_projected_mask_{ang:02d}deg.png"), sil)
        base = cv2.imread(str(out / f"v12_{ang:02d}deg.png"), cv2.IMREAD_COLOR)
        if base is not None:
            ov = base.copy()
            contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ov, f"FOCAL #12 LOCK | {ang} deg", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"v22_focal_overlay_{ang:02d}deg.png"), ov)
            overlay_frames.append(ov)
        rows.append({
            "angle_deg": int(ang), "projected_silhouette_pixels": area,
            "projected_bbox_xyxy": bbox, "projected_bbox_width_px": bw,
            "projected_bbox_height_px": bh, "visible_render_effect_pixels": visible_effect,
        })

    if len(overlay_frames) == 6:
        top = np.hstack(overlay_frames[:3]); bot = np.hstack(overlay_frames[3:])
        cv2.imwrite(str(out / "v22_focal_visibility_montage.png"), np.vstack([top, bot]))
    return rows


def _arg_path(name):
    i = sys.argv.index(name)
    return Path(sys.argv[i + 1])


def main():
    global _FOCAL_MESH, _ALL_MESHES, _FOCAL_AUDIT
    _FOCAL_MESH = None; _ALL_MESHES = []; _FOCAL_AUDIT = {}

    # v12b.main installs this symbol into v12 at execution time.
    v12b.attach_right_above_rim_fixed = attach_right_above_rim_focal_locked
    # v14's wrapper calls this object to construct each mesh, then adds source masks.
    v14._ORIGINAL_BUILD_MESH = build_mesh_capture

    assert (int(v12.W), int(v12.H)) == (960, 540)
    v20.main()

    out = _arg_path("--out")
    cams = v12.base.load_cameras(_arg_path("--registry"), _arg_path("--rar-report"), _arg_path("--broadcast-event-frame"))
    visibility = _post_focal_visibility(out, cams)

    qp = out / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    focal_mesh_rows = [x for x in q.get("player_meshes", []) if x.get("v22_focal_subject")]
    q["v22_focal_subject"] = {
        "resolution": [960, 540],
        "identity_basis": "user-specified black-uniform #12 visual anchor in all three exact-state source frames",
        "lock_audit": _FOCAL_AUDIT,
        "accepted_focal_mesh_count": int(len(focal_mesh_rows)),
        "accepted_focal_mesh_views": focal_mesh_rows[0].get("views", []) if focal_mesh_rows else [],
        "visibility_by_angle": visibility,
        "hard_requirement": "focal subject must have a dedicated accepted three-view mesh and visible contribution in every novel 5..25 degree frame",
        "generated_texture": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))

    print(json.dumps({
        "v22_focal_meshes": q["v22_focal_subject"]["accepted_focal_mesh_count"],
        "v22_focal_views": q["v22_focal_subject"]["accepted_focal_mesh_views"],
        "v22_visibility": visibility,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
