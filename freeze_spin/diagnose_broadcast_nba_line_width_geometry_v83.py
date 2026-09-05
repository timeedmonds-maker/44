from __future__ import annotations

"""v83: test NBA 2-inch line-width measurement conventions on Broadcast Frame C.

The 2025-26 official NBA court diagram labels the 16 ft lane width, 6 ft
free-throw-circle radius, and 23 ft 9 in three-point radius as OUTSIDE
measurements, while all relevant painted lines are 2 inches wide. The free
throw line is also 2 inches wide; the diagram gives 19 ft to the free throw
line outside and 18 ft 10 in inside from the baseline, so its painted
centreline is 18 ft 11 in from the baseline, i.e. 14 ft 11 in from the
backboard face.

v44/v45 had modeled those regulation dimensions as zero-width curve
centrelines. This diagnostic leaves the v82 image evidence completely fixed
and changes only the metric world-curve constants from outside-edge dimensions
to painted-line centrelines. No permission is granted by this nominal test.
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import diagnose_broadcast_observation_bias_v81 as diag
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44

INCH_CM = 2.54


@contextmanager
def geometry_constants(*, line_aware: bool):
    keys = ["THREE_R_CM", "FT_R_CM", "FT_X_CM", "PAINT_HALF_CM"]
    old = {k: getattr(v44, k) for k in keys}
    try:
        if line_aware:
            # Official diagram outside-edge dimension -> centre of a 2-inch line.
            v44.THREE_R_CM = (23.75 * v44.FOOT_CM) - INCH_CM
            v44.FT_R_CM = (6.0 * v44.FOOT_CM) - INCH_CM
            v44.FT_X_CM = (15.0 * v44.FOOT_CM) - INCH_CM
            v44.PAINT_HALF_CM = (8.0 * v44.FOOT_CM) - INCH_CM
        yield
    finally:
        for k, val in old.items():
            setattr(v44, k, val)


def fit(spec: dict, line_aware: bool) -> dict:
    with geometry_constants(line_aware=line_aware):
        return diag.fit_nominal(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (v44.H, v44.W):
        raise RuntimeError("Expected immutable native 960x540 Broadcast Frame C")
    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    diag.validate_lock(spec)

    legacy = fit(spec, False)
    corrected = fit(spec, True)
    lp = legacy["pinhole_v44"]
    cp = corrected["pinhole_v44"]
    lb = legacy["brown_v45"]
    cb = corrected["brown_v45"]

    maxp = float(cp["max_heldout_feature_p95_px"])
    if maxp <= 2.0:
        interpretation = "OFFICIAL_LINE_WIDTH_CONVENTION_RESTORES_NOMINAL_TWO_PX_PINHOLE_CANDIDATE"
    elif maxp < float(lp["max_heldout_feature_p95_px"]):
        interpretation = "OFFICIAL_LINE_WIDTH_CONVENTION_IMPROVES_PINHOLE_BUT_REMAINS_ABOVE_TWO_PX"
    else:
        interpretation = "OFFICIAL_LINE_WIDTH_CONVENTION_DOES_NOT_IMPROVE_PINHOLE"

    result = {
        "schema_version": 1,
        "status": "DISCOVERY_ONLY_NO_PROMOTION",
        "game_id": spec["game_id"],
        "event_id": spec["event_id"],
        "camera_label": "Broadcast",
        "evidence": "v82 image observations unchanged",
        "official_nba_geometry_interpretation": {
            "all_relevant_lines_width_in": 2.0,
            "lane_width_outside_ft": 16.0,
            "lane_boundary_centre_abs_y_ft": 8.0 - 1.0/12.0,
            "free_throw_circle_radius_outside_ft": 6.0,
            "free_throw_circle_centreline_radius_ft": 6.0 - 1.0/12.0,
            "three_point_arc_radius_outside_ft": 23.75,
            "three_point_arc_centreline_radius_ft": 23.75 - 1.0/12.0,
            "free_throw_line_outside_distance_from_baseline_ft": 19.0,
            "free_throw_line_inside_distance_from_baseline_ft": 18.0 + 10.0/12.0,
            "free_throw_line_centre_distance_from_backboard_face_ft": 14.0 + 11.0/12.0,
            "source": "Official 2025-26 NBA court diagram / Rule No. 1"
        },
        "legacy_zero_width_at_outside_dimensions": legacy,
        "line_width_aware_centreline_geometry": corrected,
        "comparison": {
            "pinhole_max_heldout_p95_improvement_px": float(lp["max_heldout_feature_p95_px"] - cp["max_heldout_feature_p95_px"]),
            "pinhole_pooled_heldout_p95_improvement_px": float(lp["pooled_heldout_p95_px"] - cp["pooled_heldout_p95_px"]),
            "brown_max_heldout_p95_improvement_px": float(lb["max_heldout_feature_p95_px"] - cb["max_heldout_feature_p95_px"]),
            "brown_grid_p95_distortion_legacy_px": float(lb["distortion"]["grid_p95_displacement_px"]),
            "brown_grid_p95_distortion_line_aware_px": float(cb["distortion"]["grid_p95_displacement_px"]),
        },
        "interpretation": interpretation,
        "permissions": {
            "broadcast_floor_model_promoted": False,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "next_gate": "If nominal line-aware pinhole reaches <=2 px max held-out feature p95, run a new full strict validation with the line-aware constants: competitive roots, support reduction, 64 half-pixel perturbations, and visual overlay."
    }
    (args.out / "broadcast_nba_line_width_geometry_v83.json").write_text(json.dumps(v44.json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(v44.json_safe({
        "status": result["status"],
        "interpretation": interpretation,
        "legacy_pinhole_max_heldout_p95_px": lp["max_heldout_feature_p95_px"],
        "line_aware_pinhole_max_heldout_p95_px": cp["max_heldout_feature_p95_px"],
        "line_aware_heldout_by_feature_p95_px": {k:v["p95_px"] for k,v in cp["heldout_pixel_error"].items()},
        "legacy_brown_grid_p95_distortion_px": lb["distortion"]["grid_p95_displacement_px"],
        "line_aware_brown_grid_p95_distortion_px": cb["distortion"]["grid_p95_displacement_px"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
