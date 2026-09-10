from __future__ import annotations

"""Hardened RF-DETR three-camera diagnostic v10b.

v10 proved that RF-DETR Keypoint Preview extracts strong articulated pose from
our native 960x540 NBA frames, but its first association experiment allowed a
2-view pose bonus to rescue a geometrically implausible player match. Two-view
DLT can manufacture a plausible-looking skeleton for unrelated detections.

v10b therefore makes pose confirmatory, never substitutive:
- the original v9 silhouette/metric association threshold must pass first;
- grossly inconsistent floor anchors are a hard veto;
- a Right-Above-Rim attachment to an existing Left+Broadcast track must pass a
  true three-view RF-DETR joint reprojection gate.

The renderer, source-pixel policy, exact metric static geometry, ball path and
native 960x540 output remain unchanged from v10.
"""

import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import build_three_camera_rfdetr_v10 as v10
from freeze_spin import build_three_camera_semantic_v9 as v9
from freeze_spin import build_three_camera_volumetric_v8 as v8

BROADCAST_MATCHES = {}
TRIPLE_AUDIT = []


def _triple_pose_consistency(primary_i, broadcast_i, rar_i, instances, conf_min=0.20):
    labels = ("Left Above Rim", "Broadcast", "Right Above Rim")
    ids = (primary_i, broadcast_i, rar_i)
    poses = []
    for label, idx in zip(labels, ids):
        if idx is None or idx >= len(instances[label]):
            return {"available": False, "reason": "missing_instance"}
        p = instances[label][idx].get("rfdetr_pose")
        if not p:
            return {"available": False, "reason": f"missing_pose_{label}"}
        poses.append(p)

    joint_rows = []
    for j in range(17):
        obs = {}; confs = {}
        ok = True
        for label, p in zip(labels, poses):
            xy = np.asarray(p["xy"], np.float64)
            cf = np.asarray(p["confidence"], np.float64)
            if j >= len(cf) or cf[j] < conf_min:
                ok = False; break
            obs[label] = xy[j]; confs[label] = float(cf[j])
        if not ok:
            continue
        X = v8.dlt_point(v10.GLOBAL_CAMS, obs)
        if X is None or not np.isfinite(X).all():
            continue
        if not (-320 <= X[0] <= 1250 and -720 <= X[1] <= 720 and -40 <= X[2] <= 390):
            continue
        errs = {}
        for label in labels:
            uv, _, valid = v8.project_metric(v10.GLOBAL_CAMS[label], np.asarray(X).reshape(1, 3))
            if not valid[0]:
                errs = {}; break
            errs[label] = float(np.linalg.norm(uv[0] - obs[label]))
        if not errs:
            continue
        rms = float(np.sqrt(np.mean(np.square(list(errs.values())))))
        joint_rows.append({
            "joint": v10.COCO_NAMES[j],
            "rms_reprojection_px": rms,
            "per_view_error_px": errs,
            "mean_confidence": float(np.mean(list(confs.values()))),
        })

    if not joint_rows:
        return {"available": True, "joint_count": 0, "accepted": False, "reason": "no_three_view_joints"}
    rms = np.asarray([r["rms_reprojection_px"] for r in joint_rows], np.float64)
    med = float(np.median(rms)); p75 = float(np.percentile(rms, 75)); good = int(np.sum(rms <= 15.0))
    accepted = bool(len(joint_rows) >= 4 and good >= 4 and med <= 12.0 and p75 <= 18.0)
    return {
        "available": True,
        "joint_count": int(len(joint_rows)),
        "joints_le_15px": good,
        "median_rms_reprojection_px": med,
        "p75_rms_reprojection_px": p75,
        "accepted": accepted,
        "joints": joint_rows,
    }


def strict_one_to_one(primary_label, other_label, instances, supports):
    mapping, qa = v10._one_to_one_pose(primary_label, other_label, instances, supports)
    hardened = {}
    for row in qa:
        i = int(row["primary_instance"]); j = int(row["other_instance"])
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
        row["v10b_base_geometry_gate"] = {
            "base_score_gt_4_2": base_ok,
            "silhouette_geometry_ok": bool(geom_ok),
            "floor_anchor_le_360cm": bool(floor_ok),
            "pose_plausibility_ok": bool(pose_ok),
        }
        if other_label == "Right Above Rim" and accepted:
            bi = BROADCAST_MATCHES.get(i)
            if bi is not None:
                triple = _triple_pose_consistency(i, bi, j, instances)
                row["three_view_pose_gate"] = triple
                TRIPLE_AUDIT.append({"primary_instance": i, "broadcast_instance": bi, "rar_instance": j, **triple})
                accepted = bool(triple.get("accepted", False))
            else:
                row["three_view_pose_gate"] = {"available": False, "reason": "no_existing_broadcast_match"}
        row["accepted"] = accepted
        if accepted:
            hardened[i] = j
    if other_label == "Broadcast":
        BROADCAST_MATCHES.clear(); BROADCAST_MATCHES.update(hardened)
    return hardened, qa


def main():
    # v10.main will patch v9.one_to_one_match to v10._one_to_one_pose, so replace
    # the module-global function it resolves before entering that main.
    v10._one_to_one_pose = strict_one_to_one
    v10.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("three_camera_v10b")
    p = out / "three_camera_rfdetr_v10_qa.json"
    if p.exists():
        qa = json.loads(p.read_text())
        qa["schema_version"] = "10b"
        qa["status"] = "DIAGNOSTIC_THREE_CAMERA_RFDETR_KEYPOINT_V10B_HARDENED"
        qa["v10b_hardening"] = {
            "pose_cannot_rescue_failed_v9_base_geometry": True,
            "floor_anchor_hard_veto_cm": 360.0,
            "rar_requires_true_three_view_pose_gate_when_broadcast_track_exists": True,
            "triple_pose_audit": TRIPLE_AUDIT,
        }
        qa["render_resolution_policy"] = "native 960x540 only; no upscale"
        (out / "three_camera_rfdetr_v10b_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
        print(json.dumps({"status": qa["status"], "accepted_track_count": qa.get("accepted_track_count"), "triple_pose_audit": TRIPLE_AUDIT}, indent=2), flush=True)


if __name__ == "__main__":
    main()
