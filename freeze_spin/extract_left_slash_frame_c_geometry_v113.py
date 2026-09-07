from __future__ import annotations

"""v113: direct source-pixel Left Slash Frame-C basket geometry.

The accepted v112b event-75 geometry supplies only a local transfer prior.
Frame C is re-observed from its own immutable source pixels using the corrected
red/orange rim-tube centreline model.  The historical v59 hand annotation is
not consumed by this script.

This remains a source-evidence gate only.  Passing v113 cannot promote a metric
camera and cannot authorize novel-view rendering.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import extract_left_slash_event75_geometry_v112 as base
from freeze_spin import extract_left_slash_event75_geometry_v112b as corrected

FRAME_C_SHA256 = "2ced6fbf7108459e4c8acda1d62ab8a4a77a455971287c1fcfcdc6ea7b6a272f"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-c", type=Path, required=True)
    ap.add_argument("--event75-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if base.sha256(args.frame_c) != FRAME_C_SHA256:
        raise SystemExit("immutable Frame C SHA256 mismatch")

    ev = json.loads(args.event75_json.read_text())
    if ev.get("status") != "PASS_LEFT_SLASH_EVENT75_SOURCE_GEOMETRY_V112":
        raise SystemExit("v113 requires a passing v112b event75 source-geometry artifact")
    selected = ev.get("selected")
    if not selected:
        raise SystemExit("v112b artifact has no selected event75 frame")

    image = cv2.imread(str(args.frame_c), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (540, 960):
        raise SystemExit("Frame C missing or not native 960x540")

    H_s2c = np.asarray(selected["H_source_to_frame_c"], dtype=np.float64)
    pred_target = base.perspective(
        H_s2c,
        np.asarray(selected["source_observed_target_inner_corners_px"], dtype=np.float64),
    )
    pred_rim_samples = base.perspective(
        H_s2c,
        np.asarray(selected["rim"]["source_curve_samples_px"], dtype=np.float64),
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target_fit = base.choose_target_lines(gray, pred_target)
    if target_fit is None:
        raise SystemExit("Frame C direct target source fit failed")
    _lines, line_diag, source_target = target_fit
    target_shift = np.linalg.norm(source_target - pred_target, axis=1)

    rim = corrected.extract_rim(image, pred_rim_samples)
    if rim is None:
        raise SystemExit("Frame C direct rim-centreline source fit failed")

    line_angle_max = max(float(d["angle_error_deg"]) for d in line_diag)
    line_mid_max = max(float(d["midline_distance_px"]) for d in line_diag)
    gates = {
        "immutable_frame_c_sha_matches": base.sha256(args.frame_c) == FRAME_C_SHA256,
        "event75_source_geometry_passed": ev.get("status") == "PASS_LEFT_SLASH_EVENT75_SOURCE_GEOMETRY_V112",
        "target_corner_shift_from_transfer_max_at_most_5px": float(target_shift.max()) <= 5.0,
        "target_line_angle_error_max_at_most_12deg": line_angle_max <= 12.0,
        "target_line_mid_distance_max_at_most_10px": line_mid_max <= 10.0,
        "rim_centreline_support_at_least_80": int(rim["source_support_count"]) >= 80,
        "rim_source_fit_p95_at_most_2_5px": float(rim["source_fit_p95_px"]) <= 2.5,
        "rim_heldout_p95_at_most_2_5px": float(rim["heldout_p95_px"]) <= 2.5,
        "rim_center_shift_from_transfer_at_most_7px": float(rim["center_shift_from_transfer_prior_px"]) <= 7.0,
        "rim_major_shift_from_transfer_at_most_10px": float(rim["major_axis_shift_from_transfer_prior_px"]) <= 10.0,
        "rim_minor_shift_from_transfer_at_most_6px": float(rim["minor_axis_shift_from_transfer_prior_px"]) <= 6.0,
        "rim_angle_shift_from_transfer_at_most_12deg": float(rim["angle_shift_from_transfer_prior_deg"]) <= 12.0,
    }
    passed = all(gates.values())

    overlay = args.out / "left_slash_frame_c_source_geometry_v113.png"
    base.draw_overlay(image, pred_target, source_target, rim, overlay)

    payload = {
        "status": "PASS_LEFT_SLASH_FRAME_C_SOURCE_GEOMETRY_V113" if passed else "FAIL_LEFT_SLASH_FRAME_C_SOURCE_GEOMETRY_V113",
        "purpose": "Direct immutable Frame-C target and rim-tube-centreline observation for a later metric identifiability solve",
        "frame_c": args.frame_c.name,
        "frame_c_sha256": base.sha256(args.frame_c),
        "source_image_size": [960, 540],
        "prior_source": {
            "artifact": args.event75_json.name,
            "selected_event75_index": int(selected["index"]),
            "selected_event75_time_s": float(selected["time_s"]),
            "H_event75_source_to_frame_c": H_s2c.tolist(),
            "role": "local search prior only; final observations are source-pixel Frame C evidence",
        },
        "target": {
            "transfer_prior_inner_corners_px": pred_target.tolist(),
            "source_observed_inner_corners_px": source_target.tolist(),
            "corner_shift_from_transfer_px": target_shift.tolist(),
            "corner_shift_max_px": float(target_shift.max()),
            "line_diagnostics": line_diag,
        },
        "rim": rim,
        "gates": gates,
        "permissions": {
            "left_slash_frame_c_source_geometry_allowed": passed,
            "left_slash_metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "supersedes_for_new_metric_work": "freeze_spin/adams_jazz_left_slash_frame_c_metric_v59.json basket observations",
        "guardrail": "v113 is source geometry only. It cannot promote Left Slash or authorize a replay.",
        "overlay": overlay.name,
    }
    out_json = args.out / "left_slash_frame_c_source_geometry_v113.json"
    out_json.write_text(json.dumps(payload, indent=2))

    print(json.dumps({
        "status": payload["status"],
        "target_corner_shift_max_px": payload["target"]["corner_shift_max_px"],
        "rim_source_ellipse": rim["source_ellipse"],
        "rim_support": rim["source_support_count"],
        "rim_source_fit_p95_px": rim["source_fit_p95_px"],
        "rim_heldout_p95_px": rim["heldout_p95_px"],
        "gates": gates,
    }, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
