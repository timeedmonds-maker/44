from __future__ import annotations

"""v84: strict validation of the Broadcast floor homography with NBA line-centre geometry.

v83 established that the remaining v82 pinhole error was caused by a metric
convention bug: NBA court dimensions for the 3PT arc, free-throw circle, free-
throw line and lane width are specified to the OUTSIDE of 2-inch painted lines,
while the image evidence is annotated at each painted stripe centre.  This stage
applies the corresponding one-inch centreline correction and then re-runs the
full v44 robustness protocol unchanged: spatially distributed held-out points,
deterministic multistart roots, pairwise functional-equivalence, support
reduction, and 64 independent half-pixel training-annotation perturbations.

Passing authorizes only the Broadcast floor-plane homography.  It does not
authorize a physical camera centre, a metric event camera, any non-coplanar 3D
reconstruction, or replay rendering.
"""

import json
import shutil
import sys
from pathlib import Path

from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44


FOOT_CM = 30.48
INCH_CM = 2.54

# Official NBA diagrams specify these painted dimensions to the OUTSIDE of
# 2-inch lines.  v82 observations are stripe-centre pixels, so the matching
# metric loci are one inch inward from the outside dimensions.
LINE_WIDTH_IN = 2.0
CENTRE_OFFSET_CM = (LINE_WIDTH_IN / 2.0) * INCH_CM
LINE_AWARE_FT_X_CM = 15.0 * FOOT_CM - CENTRE_OFFSET_CM
LINE_AWARE_FT_R_CM = 6.0 * FOOT_CM - CENTRE_OFFSET_CM
LINE_AWARE_THREE_R_CM = 23.75 * FOOT_CM - CENTRE_OFFSET_CM
LINE_AWARE_PAINT_HALF_CM = 8.0 * FOOT_CM - CENTRE_OFFSET_CM


def _arg_path(flag: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"v84 requires {flag}") from exc


def main() -> None:
    out = _arg_path("--out")

    # Patch only the metric court loci implicated and tested by v83.  All v44
    # solver logic, held-out splits, thresholds and perturbation protocol remain
    # unchanged.
    v44.FT_X_CM = LINE_AWARE_FT_X_CM
    v44.FT_R_CM = LINE_AWARE_FT_R_CM
    v44.THREE_R_CM = LINE_AWARE_THREE_R_CM
    v44.PAINT_HALF_CM = LINE_AWARE_PAINT_HALF_CM

    exit_code = 0
    try:
        v44.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)

    src_json = out / "broadcast_frame_c_floor_v44.json"
    if not src_json.exists():
        raise RuntimeError("v44 strict validator did not emit its JSON evidence")
    payload = json.loads(src_json.read_text(encoding="utf-8"))
    passed = bool(payload.get("permissions", {}).get("broadcast_floor_homography_allowed", False))

    payload["schema_version"] = 2
    payload["status"] = (
        "PASS_BROADCAST_LINE_AWARE_PINHOLE_FLOOR_V84"
        if passed
        else "FAIL_BROADCAST_LINE_AWARE_PINHOLE_FLOOR_V84"
    )
    payload["model"] = "undistorted pinhole floor homography with official NBA painted-line centre geometry"
    payload["method"] = (
        "v44 strict robustness protocol unchanged; v82 native source-visible regulation-paint evidence; "
        "v83-confirmed one-inch centreline correction for NBA 2-inch painted-line outside dimensions"
    )
    payload["validation_lineage"] = {
        "base_strict_solver": "freeze_spin/solve_frame_c_broadcast_floor_v44.py",
        "observation_evidence": "freeze_spin/adams_jazz_frame_c_broadcast_floor_v82.json",
        "discovery_diagnostic": "freeze_spin/diagnose_broadcast_nba_line_width_geometry_v83.py",
        "discovery_run_id": 33945432322,
        "immutable_frame_sha256": "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c",
    }
    payload["official_line_geometry_applied"] = {
        "painted_line_width_in": LINE_WIDTH_IN,
        "centre_offset_from_outside_dimension_in": 1.0,
        "free_throw_line_x_from_backboard_face_ft": LINE_AWARE_FT_X_CM / FOOT_CM,
        "free_throw_circle_centreline_radius_ft": LINE_AWARE_FT_R_CM / FOOT_CM,
        "three_point_arc_centreline_radius_ft": LINE_AWARE_THREE_R_CM / FOOT_CM,
        "lane_boundary_centreline_abs_y_ft": LINE_AWARE_PAINT_HALF_CM / FOOT_CM,
        "unchanged_basket_center_x_from_backboard_face_in": 15.0,
        "unchanged_baseline_x_from_backboard_face_ft": -4.0,
    }
    payload["permissions"] = {
        "broadcast_floor_homography_allowed": passed,
        "broadcast_physical_camera_center_allowed": False,
        "broadcast_metric_event_camera_allowed": False,
        "replay_render_allowed": False,
    }
    payload["failure_policy"] = (
        "Do not relax any robustness gate and do not revive the large Brown-distortion solution. "
        "A v84 failure means the line-aware floor homography is not yet robust enough; inspect the "
        "remaining evidence or metric convention before adding camera degrees of freedom."
    )
    payload["promotion_policy"] = (
        "A v84 pass promotes only this Broadcast floor homography. Physical camera decomposition, "
        "non-coplanar validation and replay authorization require separate later gates."
    )

    dst_json = out / "broadcast_frame_c_floor_v84.json"
    dst_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    src_json.unlink()

    src_overlay = out / "broadcast_frame_c_floor_overlay_v44.png"
    if src_overlay.exists():
        shutil.move(str(src_overlay), str(out / "broadcast_frame_c_floor_overlay_v84.png"))

    print(json.dumps({
        "status": payload["status"],
        "max_heldout_feature_p95_px": payload.get("max_heldout_feature_p95_px"),
        "max_support_reduction_p95_shift_px": payload.get("max_support_reduction_p95_shift_px"),
        "competitive_root_count": payload.get("competitive_root_count"),
        "max_competitive_pairwise_p95_shift_px": payload.get("max_competitive_pairwise_p95_shift_px"),
        "max_half_pixel_p95_shift_px": payload.get("max_half_pixel_p95_shift_px"),
        "gates": payload.get("gates"),
        "permissions": payload["permissions"],
    }, indent=2), flush=True)

    if not passed or exit_code != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
