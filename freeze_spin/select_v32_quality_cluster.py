from __future__ import annotations

"""Select only the people we can reasonably require at high quality in v32 Stage B.

Policy:
- Steven Adams is mandatory and is identified from the already accepted v23/v31
  focal-player visual anchors, not from loose court-foot clustering.
- Non-focal people are optional.  They enter the Stage-B quality cluster only if
  they are close to the action AND are independently observed by >=2 solved
  cameras with useful source area.
- Peripheral / single-view people are explicitly excluded from the quality
  contract.  The final virtual-camera framing must keep excluded people outside
  the expected visible region rather than rendering them badly.

This script does not synthesize pixels and does not alter the source bursts.
It produces a deterministic selection manifest used to focus Gaussian capacity,
held-out QA, and final framing.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

CAMS = ("Left Above Rim", "Broadcast", "Right Above Rim")


def iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a); bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    ub = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    return inter / max(1e-9, ua + ub - inter)


def focus_boxes_from_v31(q):
    lock = q.get("v23_focal_subject", {}).get("audit", {})
    if not lock:
        lock = q.get("right_above_rim_attachment", {}).get("v23_focal_subject_lock", {})
    out = {}
    b = lock.get("broadcast_pose_selection", {}).get("selected", {}).get("pose_box")
    r = lock.get("right_above_rim_pose_selection", {}).get("selected", {}).get("pose_box")
    if b: out["Broadcast"] = b
    if r: out["Right Above Rim"] = r
    return out


def sharpness(stage: Path, obs: dict):
    p = stage / obs["rgba_crop"]
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return 0.0
    a = im[:, :, 3] > 0
    if int(a.sum()) < 64:
        return 0.0
    g = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    vals = np.abs(lap[a])
    return float(np.percentile(vals, 75)) if len(vals) else 0.0


def best_anchor_match(per_camera, box):
    rows = []
    for o in per_camera:
        ov = iou(o["bbox_xyxy"], box)
        rows.append((ov, float(o.get("score", 0.0)), int(o.get("mask_pixels", 0)), o))
    rows.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return rows[0] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--v31-qa", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius-cm", type=float, default=600.0)
    ap.add_argument("--min-cameras", type=int, default=2)
    ap.add_argument("--min-median-mask-px", type=float, default=1400.0)
    args = ap.parse_args()

    m = json.loads((args.stage_a / "v32_scene_manifest.json").read_text())
    q = json.loads(args.v31_qa.read_text())
    fboxes = focus_boxes_from_v31(q)
    if "Broadcast" not in fboxes or "Right Above Rim" not in fboxes:
        raise RuntimeError("missing accepted v31 focal boxes")

    matches = {}
    focal_world = []
    contaminated_track_ids = set()
    for cam in ("Broadcast", "Right Above Rim"):
        best = best_anchor_match(m["all_on_court_people"]["per_camera"][cam], fboxes[cam])
        if best is None or best[0] < 0.20:
            raise RuntimeError(f"unable to lock focal player in {cam}: {best}")
        ov, det_score, mask_px, obs = best
        sh = sharpness(args.stage_a, obs)
        matches[cam] = {
            "camera_person_id": int(obs["camera_person_id"]),
            "bbox_xyxy": obs["bbox_xyxy"],
            "anchor_iou": float(ov),
            "detector_score": float(det_score),
            "mask_pixels": int(mask_px),
            "crop_edge_sharpness_p75": float(sh),
            "foot_world_cm": obs["foot_world_cm"],
            "mask_full": obs["mask_full"],
            "rgba_crop": obs["rgba_crop"],
        }
        focal_world.append(np.asarray(obs["foot_world_cm"][:2], np.float64))
        for tr in m["all_on_court_people"]["loose_world_union_tracks"]:
            if any(o["camera"] == cam and int(o["camera_person_id"]) == int(obs["camera_person_id"]) for o in tr["observations"]):
                contaminated_track_ids.add(int(tr["track_id"]))

    action_xy = np.median(np.stack(focal_world), axis=0)
    selected = []
    rejected = []
    for tr in m["all_on_court_people"]["loose_world_union_tracks"]:
        tid = int(tr["track_id"])
        xy = np.asarray(tr["centroid_xy_cm"], np.float64)
        dist = float(np.linalg.norm(xy - action_xy))
        masks = [int(o.get("mask_pixels", 0)) for o in tr.get("observations", [])]
        med_mask = float(np.median(masks)) if masks else 0.0
        shs = [sharpness(args.stage_a, o) for o in tr.get("observations", [])]
        med_sharp = float(np.median(shs)) if shs else 0.0
        reasons = []
        if tid in contaminated_track_ids:
            reasons.append("contains_focal_observation_or_bad_loose_pairing")
        if int(tr.get("camera_count", 0)) < args.min_cameras:
            reasons.append("single_view")
        if dist > args.radius_cm:
            reasons.append("peripheral_distance")
        if med_mask < args.min_median_mask_px:
            reasons.append("insufficient_source_area")
        row = {
            "track_id": tid,
            "centroid_xy_cm": tr["centroid_xy_cm"],
            "distance_from_focal_cm": dist,
            "camera_count": int(tr.get("camera_count", 0)),
            "cameras": tr.get("cameras", []),
            "median_mask_pixels": med_mask,
            "median_crop_edge_sharpness_p75": med_sharp,
            "observations": tr.get("observations", []),
        }
        if reasons:
            row["reject_reasons"] = reasons
            rejected.append(row)
        else:
            row["quality_role"] = "supporting_action_player"
            selected.append(row)

    # Focal player is one logical entity even if the earlier loose foot-position
    # join split/mispaired its Broadcast and RAR observations.
    focal = {
        "quality_role": "mandatory_focal_player",
        "identity": "Steven Adams / accepted dark-uniform #12 focal lock",
        "centroid_xy_cm": action_xy.tolist(),
        "observations": matches,
        "source_camera_count": len(matches),
        "note": "v31 accepted visual identity anchors supersede loose world-track association in the crowded paint",
    }

    # Tight final-camera contract: selected people may remain visible; rejected
    # people are not required and should be kept outside the final crop whenever
    # the virtual path permits.
    pts = [action_xy] + [np.asarray(x["centroid_xy_cm"], np.float64) for x in selected]
    p = np.stack(pts)
    world_bounds = {
        "xmin_cm": float(p[:,0].min() - 140.0),
        "xmax_cm": float(p[:,0].max() + 140.0),
        "ymin_cm": float(p[:,1].min() - 140.0),
        "ymax_cm": float(p[:,1].max() + 140.0),
    }

    out = {
        "version": "v32-quality-cluster-1",
        "policy": "quality over player count; Adams mandatory; only independently multi-view, near-action supporting people are quality-required",
        "focal_player": focal,
        "supporting_players": selected,
        "supporting_player_count": len(selected),
        "quality_required_person_count": 1 + len(selected),
        "excluded_people": rejected,
        "excluded_person_track_count": len(rejected),
        "selection_thresholds": {
            "radius_cm": args.radius_cm,
            "min_cameras": args.min_cameras,
            "min_median_mask_pixels": args.min_median_mask_px,
        },
        "final_camera_contract": {
            "world_action_bounds_cm": world_bounds,
            "peripheral_people_quality_required": False,
            "rule": "do not allow an excluded peripheral person to remain conspicuously inside the final 0-25 degree composition; tighten camera/framing instead",
        },
        "held_out_gate": {
            "apply_to": "focal player plus selected supporting players only",
            "focal_visual_pass_mandatory": True,
            "supporting_visual_pass_mandatory": True,
            "fail_conditions": [
                "missing or doubled limb",
                "identity-changing appearance",
                "obvious temporal smear at freeze",
                "player disappears without real occlusion",
                "major silhouette collapse",
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "focal_matches": matches,
        "action_xy_cm": action_xy.tolist(),
        "supporting_track_ids": [x["track_id"] for x in selected],
        "quality_required_person_count": out["quality_required_person_count"],
        "excluded_person_track_count": out["excluded_person_track_count"],
        "world_action_bounds_cm": world_bounds,
    }, indent=2))


if __name__ == "__main__":
    main()
