from __future__ import annotations

"""v81: diagnose whether Broadcast three-point annotation bias created false distortion.

This is a discovery-only A/B test on the immutable synchronized Broadcast Frame C.
It preserves the original v44 observations, compares them against the separately
stored v81 re-annotation, and runs the same nominal v44 pinhole and v45 Brown
model families on both.  It deliberately does NOT grant any camera/replay
permission and does not weaken any historical gate.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44
from freeze_spin import solve_frame_c_broadcast_floor_v45 as v45

EXPECTED_SHA256 = "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c"


def pooled_p95(metrics: dict) -> float:
    vals = np.concatenate([np.asarray(metrics[k]["per_point_px"], dtype=np.float64) for k in v44.GROUPS])
    return float(np.percentile(vals, 95))


def validate_lock(spec: dict) -> None:
    lock = spec["freeze_lock"]
    if spec["camera_label"] != "Broadcast":
        raise RuntimeError("Expected Broadcast evidence")
    if lock["authority_camera"] != "Right Slash" or lock["chooser_option"] != "C":
        raise RuntimeError("Evidence is not bound to immutable chooser C")
    if abs(float(lock["right_slash_local_time"]) - 8.275733) > 5e-7 or int(lock["right_slash_decoded_frame_index"]) != 248:
        raise RuntimeError("Authority timing changed")
    if abs(float(lock["broadcast_synchronized_time"]) - 9.194613) > 5e-7 or int(lock["broadcast_decoded_frame_index"]) != 276:
        raise RuntimeError("Broadcast Frame C timing changed")


def fit_nominal(spec: dict) -> dict:
    validate_lock(spec)
    seed = spec["seed_only_correspondences"]
    H_seed, _ = cv2.findHomography(
        np.asarray(seed["world_cm"], dtype=np.float64),
        np.asarray(seed["image_px"], dtype=np.float64),
        method=0,
    )
    if H_seed is None:
        raise RuntimeError("Could not construct seed-only homography")
    h0 = v44.parameter_vector(H_seed)
    train, held = v44.split_groups(spec["observations_px"], spec["held_out_indices"])
    dense = v44.dense_features()

    z44, roots44 = v44.solve_multistart(h0, train, return_roots=True)
    H44 = v44.H_from_z(z44, h0)
    train44 = v44.pixel_metrics(H44, train, dense)
    held44 = v44.pixel_metrics(H44, held, dense)

    q45, roots45 = v45.solve_multistart(h0, train, warm=np.r_[z44, np.zeros(4)], return_roots=True)
    H45, d45 = v45.unpack(q45, h0)
    train45 = v45.pixel_metrics(H45, d45, train, dense)
    held45 = v45.pixel_metrics(H45, d45, held, dense)
    distortion = v45.distortion_diagnostics(d45)

    return {
        "pinhole_v44": {
            "max_heldout_feature_p95_px": float(max(row["p95_px"] for row in held44.values())),
            "pooled_heldout_p95_px": pooled_p95(held44),
            "heldout_pixel_error": held44,
            "training_pixel_error": train44,
            "competitive_root_count_nominal": int(sum(r["median_abs_pixel_residual"] <= min(x["median_abs_pixel_residual"] for x in roots44) + 0.25 for r in roots44)),
        },
        "brown_v45": {
            "max_heldout_feature_p95_px": float(max(row["p95_px"] for row in held45.values())),
            "pooled_heldout_p95_px": pooled_p95(held45),
            "heldout_pixel_error": held45,
            "training_pixel_error": train45,
            "distortion": distortion,
            "competitive_root_count_nominal": int(sum(r["median_abs_pixel_residual"] <= min(x["median_abs_pixel_residual"] for x in roots45) + 0.25 for r in roots45)),
        },
    }


def draw_audit(frame: np.ndarray, original: dict, corrected: dict, out: Path) -> None:
    canvas = frame.copy()
    a = np.asarray(original["observations_px"]["three_point_arc"], dtype=int)
    b = np.asarray(corrected["observations_px"]["three_point_arc"], dtype=int)
    held = set(original["held_out_indices"]["three_point_arc"])
    for i, (p0, p1) in enumerate(zip(a, b)):
        p0t, p1t = tuple(map(int, p0)), tuple(map(int, p1))
        cv2.line(canvas, p0t, p1t, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0t, 5 if i in held else 3, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, p1t, 5 if i in held else 3, (0, 255, 0), 2, cv2.LINE_AA)
        if i in held:
            cv2.putText(canvas, str(i), (p1t[0] + 5, p1t[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "red=v44 stored  green=v81 stripe-centred  yellow=shift", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--corrected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(args.frame))
    if frame is None or frame.shape[:2] != (v44.H, v44.W):
        raise RuntimeError("Expected immutable native 960x540 Broadcast Frame C")
    original = json.loads(args.original.read_text(encoding="utf-8"))
    corrected = json.loads(args.corrected.read_text(encoding="utf-8"))
    validate_lock(original)
    validate_lock(corrected)

    for key in v44.GROUPS:
        if key == "three_point_arc":
            continue
        if original["observations_px"][key] != corrected["observations_px"][key]:
            raise RuntimeError(f"v81 altered non-three-point evidence: {key}")
    if original["held_out_indices"] != corrected["held_out_indices"]:
        raise RuntimeError("v81 altered held-out split")

    a = np.asarray(original["observations_px"]["three_point_arc"], dtype=np.float64)
    b = np.asarray(corrected["observations_px"]["three_point_arc"], dtype=np.float64)
    shifts = np.linalg.norm(b - a, axis=1)

    original_fit = fit_nominal(original)
    corrected_fit = fit_nominal(corrected)

    o44 = original_fit["pinhole_v44"]
    c44 = corrected_fit["pinhole_v44"]
    o45 = original_fit["brown_v45"]
    c45 = corrected_fit["brown_v45"]
    old_disp = float(o45["distortion"]["grid_p95_displacement_px"])
    new_disp = float(c45["distortion"]["grid_p95_displacement_px"])

    if c44["max_heldout_feature_p95_px"] <= 2.0:
        interpretation = "CORRECTED_EVIDENCE_RESTORES_TWO_PX_PINHOLE_CANDIDATE"
    elif new_disp < old_disp * 0.70 and c44["max_heldout_feature_p95_px"] < o44["max_heldout_feature_p95_px"]:
        interpretation = "ANNOTATION_BIAS_MATERIALLY_INFLATED_DISTORTION_BUT_PINHOLE_NOT_YET_RESTORED"
    else:
        interpretation = "ANNOTATION_BIAS_CONFIRMED_BUT_NOT_SUFFICIENT_TO_EXPLAIN_MODEL_GAP"

    draw_audit(frame, original, corrected, args.out / "broadcast_v81_three_point_evidence_audit.png")
    result = {
        "schema_version": 1,
        "status": "DISCOVERY_ONLY_NO_PROMOTION",
        "game_id": original["game_id"],
        "event_id": original["event_id"],
        "camera_label": "Broadcast",
        "immutable_frame_sha256": EXPECTED_SHA256,
        "test": "same nominal v44 pinhole and v45 Brown model families on original v44 observations versus v81 stripe-centred three-point observations",
        "annotation_shift": {
            "count": int(len(shifts)),
            "per_point_euclidean_px": shifts.tolist(),
            "median_px": float(np.median(shifts)),
            "p95_px": float(np.percentile(shifts, 95)),
            "max_px": float(np.max(shifts)),
        },
        "original_v44_evidence": original_fit,
        "corrected_v81_evidence": corrected_fit,
        "comparison": {
            "pinhole_max_heldout_p95_improvement_px": float(o44["max_heldout_feature_p95_px"] - c44["max_heldout_feature_p95_px"]),
            "pinhole_pooled_heldout_p95_improvement_px": float(o44["pooled_heldout_p95_px"] - c44["pooled_heldout_p95_px"]),
            "brown_max_heldout_p95_improvement_px": float(o45["max_heldout_feature_p95_px"] - c45["max_heldout_feature_p95_px"]),
            "brown_pooled_heldout_p95_improvement_px": float(o45["pooled_heldout_p95_px"] - c45["pooled_heldout_p95_px"]),
            "brown_distortion_grid_p95_displacement_original_px": old_disp,
            "brown_distortion_grid_p95_displacement_corrected_px": new_disp,
            "brown_distortion_grid_p95_displacement_reduction_fraction": float((old_disp - new_disp) / old_disp) if old_disp > 1e-9 else None,
        },
        "interpretation": interpretation,
        "permissions": {
            "broadcast_floor_model_promoted": False,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "next_gate": "If corrected evidence materially reduces the apparent distortion, run the full 64-perturbation v44/v45 validation only on the corrected evidence before deciding the Broadcast model family.",
    }
    (args.out / "broadcast_observation_bias_v81.json").write_text(json.dumps(v44.json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(v44.json_safe({
        "status": result["status"],
        "interpretation": interpretation,
        "annotation_shift": result["annotation_shift"],
        "original_pinhole_max_heldout_p95_px": o44["max_heldout_feature_p95_px"],
        "corrected_pinhole_max_heldout_p95_px": c44["max_heldout_feature_p95_px"],
        "original_brown_max_heldout_p95_px": o45["max_heldout_feature_p95_px"],
        "corrected_brown_max_heldout_p95_px": c45["max_heldout_feature_p95_px"],
        "original_brown_grid_p95_distortion_px": old_disp,
        "corrected_brown_grid_p95_distortion_px": new_disp,
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
