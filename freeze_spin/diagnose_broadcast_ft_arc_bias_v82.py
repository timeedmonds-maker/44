from __future__ import annotations

"""v82: A/B the independently audited free-throw semicircle evidence.

The v81 three-point correction remains fixed.  This script compares v81 versus
v82 with the exact same nominal v44 pinhole and v45 Brown model families.  It is
strictly discovery-only: even a <=2 px nominal pinhole result must be followed
by a new full multistart/support-reduction/perturbation validation before any
floor model is promoted.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import diagnose_broadcast_observation_bias_v81 as v81
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--v81", type=Path, required=True)
    ap.add_argument("--v82", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (v44.H, v44.W):
        raise RuntimeError("Expected immutable native 960x540 Broadcast Frame C")

    original = json.loads(args.original.read_text(encoding="utf-8"))
    s81 = json.loads(args.v81.read_text(encoding="utf-8"))
    s82 = json.loads(args.v82.read_text(encoding="utf-8"))
    for spec in (original, s81, s82):
        v81.validate_lock(spec)

    if s81["observations_px"]["three_point_arc"] != s82["observations_px"]["three_point_arc"]:
        raise RuntimeError("v82 changed the sealed v81 three-point correction")
    for key in v44.GROUPS:
        if key == "free_throw_front_semicircle":
            continue
        if s81["observations_px"][key] != s82["observations_px"][key]:
            raise RuntimeError(f"v82 altered non-semicircle evidence: {key}")
    if s81["held_out_indices"] != s82["held_out_indices"]:
        raise RuntimeError("v82 altered the held-out split")

    a = np.asarray(s81["observations_px"]["free_throw_front_semicircle"], dtype=np.float64)
    b = np.asarray(s82["observations_px"]["free_throw_front_semicircle"], dtype=np.float64)
    shifts = np.linalg.norm(b - a, axis=1)

    f0 = v81.fit_nominal(original)
    f81 = v81.fit_nominal(s81)
    f82 = v81.fit_nominal(s82)
    p81 = f81["pinhole_v44"]
    p82 = f82["pinhole_v44"]
    b81 = f81["brown_v45"]
    b82 = f82["brown_v45"]

    max82 = float(p82["max_heldout_feature_p95_px"])
    if max82 <= 2.0:
        interpretation = "CUMULATIVE_CORRECTED_EVIDENCE_RESTORES_NOMINAL_TWO_PX_PINHOLE_CANDIDATE"
    elif max82 < float(p81["max_heldout_feature_p95_px"]):
        interpretation = "FT_ARC_EVIDENCE_BIAS_CONFIRMED_AND_PINHOLE_IMPROVES_BUT_REMAINS_ABOVE_TWO_PX"
    else:
        interpretation = "FT_ARC_REANNOTATION_DOES_NOT_IMPROVE_PINHOLE_MODEL"

    # Native evidence audit overlay: red=v81 stored, green=v82 stripe-centred.
    overlay = image.copy()
    held = set(s81["held_out_indices"]["free_throw_front_semicircle"])
    for i, (p0, p1) in enumerate(zip(a.astype(int), b.astype(int))):
        p0t, p1t = tuple(map(int, p0)), tuple(map(int, p1))
        cv2.line(overlay, p0t, p1t, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, p0t, 5 if i in held else 3, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, p1t, 5 if i in held else 3, (0, 255, 0), 2, cv2.LINE_AA)
        if i in held:
            cv2.putText(overlay, str(i), (p1t[0] + 4, p1t[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(overlay, "red=v81  green=v82 FT stripe-centre  yellow=shift", (18,26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "broadcast_v82_ft_arc_evidence_audit.png"), overlay)

    result = {
        "schema_version": 1,
        "status": "DISCOVERY_ONLY_NO_PROMOTION",
        "game_id": s82["game_id"],
        "event_id": s82["event_id"],
        "camera_label": "Broadcast",
        "test": "v81 three-point-corrected evidence versus v82 cumulative three-point+free-throw-semicircle corrected evidence under identical nominal v44/v45 model families",
        "ft_arc_annotation_shift": {
            "count": int(len(shifts)),
            "per_point_euclidean_px": shifts.tolist(),
            "median_px": float(np.median(shifts)),
            "p95_px": float(np.percentile(shifts,95)),
            "max_px": float(np.max(shifts)),
        },
        "original_v44_evidence": f0,
        "v81_three_point_corrected": f81,
        "v82_cumulative_corrected": f82,
        "comparison_v81_to_v82": {
            "pinhole_max_heldout_p95_improvement_px": float(p81["max_heldout_feature_p95_px"] - p82["max_heldout_feature_p95_px"]),
            "pinhole_pooled_heldout_p95_improvement_px": float(p81["pooled_heldout_p95_px"] - p82["pooled_heldout_p95_px"]),
            "brown_max_heldout_p95_improvement_px": float(b81["max_heldout_feature_p95_px"] - b82["max_heldout_feature_p95_px"]),
            "brown_pooled_heldout_p95_improvement_px": float(b81["pooled_heldout_p95_px"] - b82["pooled_heldout_p95_px"]),
            "brown_grid_p95_distortion_v81_px": float(b81["distortion"]["grid_p95_displacement_px"]),
            "brown_grid_p95_distortion_v82_px": float(b82["distortion"]["grid_p95_displacement_px"]),
        },
        "interpretation": interpretation,
        "permissions": {
            "broadcast_floor_model_promoted": False,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "next_gate": "If v82 reaches <=2 px nominal pinhole held-out p95, run a new full strict pinhole validation on v82 evidence including competitive multistart equivalence, support reduction, and 64 half-pixel perturbations. Do not reuse the historical v44 pass/fail label.",
    }
    (args.out / "broadcast_ft_arc_bias_v82.json").write_text(json.dumps(v44.json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(v44.json_safe({
        "status": result["status"],
        "interpretation": interpretation,
        "ft_arc_shift_median_px": result["ft_arc_annotation_shift"]["median_px"],
        "original_v44_pinhole_max_heldout_p95_px": f0["pinhole_v44"]["max_heldout_feature_p95_px"],
        "v81_pinhole_max_heldout_p95_px": p81["max_heldout_feature_p95_px"],
        "v82_pinhole_max_heldout_p95_px": p82["max_heldout_feature_p95_px"],
        "v82_pinhole_heldout_by_feature_p95_px": {k:v["p95_px"] for k,v in p82["heldout_pixel_error"].items()},
        "v81_brown_grid_p95_distortion_px": b81["distortion"]["grid_p95_displacement_px"],
        "v82_brown_grid_p95_distortion_px": b82["distortion"]["grid_p95_displacement_px"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
