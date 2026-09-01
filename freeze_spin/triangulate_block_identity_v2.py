from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.triangulate_semantic_points_v1 import camera_map, solve_point
from freeze_spin.triangulate_locked_ball_v1 import project


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--shoulders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cameras = camera_map(args.cameras)
    semantic = json.loads(args.semantic.read_text(encoding="utf-8"))
    ball_report = json.loads(args.ball_report.read_text(encoding="utf-8"))
    spec = json.loads(args.shoulders.read_text(encoding="utf-8"))
    homographies = {
        row["label"]: np.asarray(row["camera_motion_homography_selected_to_anchor"], dtype=np.float64)
        for row in ball_report["views"]
    }

    points = {}
    for name, row in spec["shoulders"].items():
        observations = {}
        sigmas = {}
        for label, item in row["views"].items():
            selected = np.asarray(item["pixel_xy_selected_frame"], dtype=np.float64)
            observations[label] = cv2.perspectiveTransform(
                selected.reshape(1, 1, 2), homographies[label]
            )[0, 0]
            sigmas[label] = float(item["uncertainty_px"])

        X, cost = solve_point(cameras, observations, sigmas)
        views = {}
        max_norm = 0.0
        for label in observations:
            P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
            predicted_anchor = project(P, X)
            error = float(np.linalg.norm(predicted_anchor - observations[label]))
            normalized = error / sigmas[label]
            max_norm = max(max_norm, normalized)
            predicted_selected = cv2.perspectiveTransform(
                predicted_anchor.reshape(1, 1, 2), np.linalg.inv(homographies[label])
            )[0, 0]
            views[label] = {
                "reprojection_error_px": round(error, 4),
                "normalized_error_sigma": round(normalized, 4),
                "predicted_selected_px": [round(float(v), 3) for v in predicted_selected],
            }

        elbow = np.asarray(semantic["points"][row["elbow_point"]]["world_cm"], dtype=np.float64)
        points[name] = {
            "player": row["player"],
            "joint": row["joint"],
            "world_cm": [round(float(v), 6) for v in X],
            "optimizer_cost": round(cost, 8),
            "max_normalized_error_sigma": round(max_norm, 4),
            "views": views,
            "elbow_to_shoulder_cm": round(float(np.linalg.norm(X - elbow)), 4),
        }

    adams = np.asarray(points["adams_block_shoulder"]["world_cm"], dtype=np.float64)
    cissoko = np.asarray(points["cissoko_ball_shoulder"]["world_cm"], dtype=np.float64)
    world_separation = float(np.linalg.norm(adams - cissoko))

    pixel_separation = {}
    for label in ("In Arena", "Left Slash", "Left HandHeld"):
        a = np.asarray(points["adams_block_shoulder"]["views"][label]["predicted_selected_px"], dtype=np.float64)
        c = np.asarray(points["cissoko_ball_shoulder"]["views"][label]["predicted_selected_px"], dtype=np.float64)
        pixel_separation[label] = float(np.linalg.norm(a - c))

    numerical_gate = {
        "upstream_forearm_metric_gate_passed": bool(semantic["metric_gate"]["pass"]),
        "both_shoulders_use_three_views": all(len(row["views"]) >= 3 for row in points.values()),
        "both_shoulders_within_1_5_sigma": all(row["max_normalized_error_sigma"] <= 1.5 for row in points.values()),
        "upper_arm_lengths_plausible": all(20.0 <= row["elbow_to_shoulder_cm"] <= 60.0 for row in points.values()),
        "shoulders_distinct_in_world": world_separation >= 25.0,
        "shoulders_distinct_in_each_identity_view": all(value >= 20.0 for value in pixel_separation.values()),
    }
    numerical_gate["pass"] = bool(all(numerical_gate.values()))

    # IMPORTANT: numerical consistency is necessary but not sufficient for identity.
    # The source-pixel validator owns promotion because it checks independently visible
    # player identity and border/crop validity in the actual locked frames.  This script
    # must never emit a completed identity lock on residuals alone.
    gate = {
        "numerical_triangulation_passed": bool(numerical_gate["pass"]),
        "source_pixel_visual_identity_gate_passed": False,
        "pass": False,
    }

    payload = {
        "status": "candidate_block_arm_geometry_requires_visual_identity_gate",
        "interpretation": (
            "The two provisional forearm chains have numerically consistent candidate shoulders, "
            "but identity is intentionally not promoted here. The independent source-pixel visual "
            "gate must pass before any coarse full-body or novel-view stage may consume these joints."
        ),
        "shoulder_points": points,
        "shoulder_world_separation_cm": round(world_separation, 4),
        "shoulder_selected_frame_separation_px": {key: round(value, 3) for key, value in pixel_separation.items()},
        "numerical_gate": numerical_gate,
        "gate": gate,
        "coarse_full_body_gate": {"pass": False},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not numerical_gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
