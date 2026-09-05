from __future__ import annotations

"""Prove a reusable same-game camera-centre prior from fixed regulation geometry.

The immutable target frame has manually measured NBA target/lane landmarks.  For
independent same-camera event frames, a global static-scene homography is fitted
WITHOUT player/ball landmarks and validated on spatially separated held-out static
features.  Only passing homographies may transport the target's regulation landmarks
back into the source frame.  Each transported source view is then solved as a full
metric camera against the regulation NBA 3D model.

The decisive test is whether those independently solved source camera centres agree
with one another and with the directly annotated target camera.  This script can
therefore authorize a *camera-centre prior only*. It never promotes an event camera
or a rendered replay by itself.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.audit_game_camera_registry_preflight_v1 import best_event_pair
from freeze_spin.solve_nba_geometry_proof_v3 import (
    draw_overlay,
    perturbation_sensitivity,
    solve_camera,
    world_landmarks,
)

W, H = 960, 540


def transform_points(points: dict[str, list[float]], H_target_to_source: np.ndarray) -> dict[str, list[float]]:
    names = list(points)
    arr = np.asarray([points[n] for n in names], dtype=np.float32)
    pred = cv2.perspectiveTransform(arr[:, None, :], H_target_to_source)[:, 0]
    return {n: [float(p[0]), float(p[1])] for n, p in zip(names, pred)}


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def safe(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def solve_view(view: dict, world: dict[str, np.ndarray], *, warm_params=None) -> dict:
    names, obj, obs, rim_samples, board_obs, solved = solve_camera(view, world, warm_params=warm_params)
    rejected, score, params, rmse, center, focal, rim_metrics, board_rmse = solved
    if rejected:
        raise RuntimeError(f"Metric solver rejected {view['label']}")
    return {
        "names": names,
        "obj": obj,
        "obs": obs,
        "rim_samples": rim_samples,
        "board_obs": board_obs,
        "params": np.asarray(params, dtype=np.float64),
        "rmse": float(rmse),
        "center": np.asarray(center, dtype=np.float64),
        "focal": float(focal),
        "score": float(score),
        "rim_metrics": None if rim_metrics is None else [float(x) for x in rim_metrics],
        "board_rmse": None if board_rmse is None else float(board_rmse),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-frame", type=Path, required=True)
    ap.add_argument("--target-landmarks", type=Path, required=True)
    ap.add_argument("--camera-label", default="Left Above Rim")
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-landmark-rmse-px", type=float, default=3.0)
    ap.add_argument("--max-center-pairwise-cm", type=float, default=10.0)
    ap.add_argument("--max-half-pixel-center-shift-cm", type=float, default=75.0)
    ap.add_argument("--min-independent-events", type=int, default=3)
    ap.add_argument("--perturbation-trials", type=int, default=12)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target_image = cv2.imread(str(args.target_frame))
    if target_image is None or target_image.shape[:2] != (H, W):
        raise RuntimeError("Target frame is missing or not native 960x540")

    spec = json.loads(args.target_landmarks.read_text(encoding="utf-8"))
    target_view = next((v for v in spec["views"] if v["label"] == args.camera_label), None)
    if target_view is None:
        raise RuntimeError(f"No {args.camera_label!r} view in {args.target_landmarks}")
    if any(token in json.dumps(target_view).lower() for token in ("player", "ball", "body", "hand", "elbow", "shoulder")):
        raise RuntimeError("Forbidden dynamic anchors in target metric view")

    world = world_landmarks()
    target = solve_view(target_view, world)
    target_shift, target_trials = perturbation_sensitivity(
        target_view, world, target["params"], target["center"], args.perturbation_trials, 260902
    )
    draw_overlay(
        target_image, target["params"], target["names"], target["obj"], target["obs"],
        target["rim_samples"], target["board_obs"], args.out / "target_direct_metric_overlay.png"
    )

    grouped: dict[int, list[Path]] = {}
    prefix = safe(args.camera_label)
    for p in sorted(args.samples.glob(f"{prefix}__event*__s*.png")):
        stem = p.stem
        marker = stem.split("__event", 1)[1].split("__", 1)[0]
        grouped.setdefault(int(marker), []).append(p)
    if len(grouped) < args.min_independent_events:
        raise RuntimeError(f"Only {len(grouped)} independent event sample groups")

    rows = []
    for event_id, paths in sorted(grouped.items()):
        transport = best_event_pair(paths, args.target_frame)
        rec = {
            "event_id": event_id,
            "transport_source": transport.get("source"),
            "transport_status": transport.get("status"),
            "transport_pass": bool(transport.get("pass")),
            "transport_training_inliers": int(transport.get("training_inliers", 0)),
            "transport_training_error": transport.get("training_error"),
            "transport_withheld_error": transport.get("withheld_error"),
            "transport_gates": transport.get("gates"),
        }
        if not transport.get("pass"):
            rows.append(rec)
            continue

        H_source_to_target = np.asarray(transport["H_source_to_target"], dtype=np.float64)
        H_target_to_source = np.linalg.inv(H_source_to_target)
        source_landmarks = transform_points(target_view["landmarks"], H_target_to_source)
        source_view = {
            "index": int(event_id),
            "label": f"{args.camera_label} event {event_id}",
            "principal_point_prior_sigma_px": target_view.get("principal_point_prior_sigma_px", 160.0),
            "principal_point_bound_px": target_view.get("principal_point_bound_px", 350.0),
            "focal_prior_px": target_view.get("focal_prior_px", 900.0),
            "focal_prior_sigma_log": target_view.get("focal_prior_sigma_log", 1.8),
            "landmarks": source_landmarks,
            "notes": "Regulation landmarks transported by independently fitted and held-out-validated static-scene homography; no player/ball anchors.",
        }
        solved = solve_view(source_view, world, warm_params=target["params"])
        max_shift, shifts = perturbation_sensitivity(
            source_view, world, solved["params"], solved["center"], args.perturbation_trials, 260902 + event_id
        )
        src_path = Path(str(transport["source"]))
        source_image = cv2.imread(str(src_path))
        if source_image is None:
            raise RuntimeError(f"Cannot read winning source sample {src_path}")
        draw_overlay(
            source_image, solved["params"], solved["names"], solved["obj"], solved["obs"],
            solved["rim_samples"], solved["board_obs"], args.out / f"event_{event_id}_metric_overlay.png"
        )
        rec.update({
            "metric_landmarks_px": source_landmarks,
            "landmark_rmse_px": solved["rmse"],
            "focal_px": solved["focal"],
            "camera_center_cm": [float(x) for x in solved["center"]],
            "max_half_pixel_camera_center_shift_cm": float(max_shift),
            "half_pixel_camera_center_shifts_cm": [float(x) for x in shifts],
            "source_metric_gate_pass": bool(
                solved["rmse"] <= args.max_landmark_rmse_px
                and max_shift <= args.max_half_pixel_center_shift_cm
            ),
        })
        rows.append(rec)
        print("EVENT", event_id, "RMSE", round(solved["rmse"], 4), "CENTER", np.round(solved["center"], 3).tolist(), "F", round(solved["focal"], 3), "PERT", round(max_shift, 3), flush=True)

    good = [r for r in rows if r.get("transport_pass") and r.get("source_metric_gate_pass")]
    centers = [target["center"]] + [np.asarray(r["camera_center_cm"], float) for r in good]
    pairwise = [center_distance(centers[i], centers[j]) for i in range(len(centers)) for j in range(i + 1, len(centers))]
    median_center = np.median(np.asarray(centers), axis=0) if centers else np.full(3, np.nan)
    deviations = [center_distance(c, median_center) for c in centers]

    gates = {
        "target_direct_metric_rmse_at_most_threshold": target["rmse"] <= args.max_landmark_rmse_px,
        "target_half_pixel_stability_at_most_threshold": target_shift <= args.max_half_pixel_center_shift_cm,
        "independent_source_events_at_least_minimum": len(good) >= args.min_independent_events,
        "all_accepted_source_rmse_at_most_threshold": bool(good) and all(r["landmark_rmse_px"] <= args.max_landmark_rmse_px for r in good),
        "all_accepted_source_half_pixel_stability_at_most_threshold": bool(good) and all(r["max_half_pixel_camera_center_shift_cm"] <= args.max_half_pixel_center_shift_cm for r in good),
        "max_pairwise_camera_center_distance_at_most_threshold": bool(pairwise) and max(pairwise) <= args.max_center_pairwise_cm,
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_CENTER_PRIOR" if passed else "FAIL_CENTER_PRIOR",
        "game_id": "0022500301",
        "camera_label": args.camera_label,
        "coordinate_system": "basket-local regulation NBA centimetres: +X from board toward court, +Y across board, +Z upward",
        "method": "direct target metric geometry + independent same-game static-scene homography transfer + repeated regulation metric solves",
        "guardrail": "This proof may authorize a reusable physical camera-centre prior only. It does not by itself promote a target event camera, foreground reconstruction or replay render.",
        "thresholds": {
            "max_landmark_rmse_px": args.max_landmark_rmse_px,
            "max_camera_center_pairwise_cm": args.max_center_pairwise_cm,
            "max_half_pixel_camera_center_shift_cm": args.max_half_pixel_center_shift_cm,
            "min_independent_events": args.min_independent_events,
            "perturbation_trials": args.perturbation_trials,
        },
        "target_direct_metric": {
            "landmark_rmse_px": target["rmse"],
            "focal_px": target["focal"],
            "camera_center_cm": [float(x) for x in target["center"]],
            "max_half_pixel_camera_center_shift_cm": float(target_shift),
        },
        "independent_event_results": rows,
        "accepted_independent_event_count": len(good),
        "accepted_centers_including_target_count": len(centers),
        "median_camera_center_cm": [float(x) for x in median_center],
        "camera_center_deviation_from_median_cm": deviations,
        "max_camera_center_deviation_from_median_cm": max(deviations) if deviations else None,
        "max_pairwise_camera_center_distance_cm": max(pairwise) if pairwise else None,
        "gates": gates,
        "center_prior_allowed": passed,
        "metric_camera_promotion_allowed": False,
        "replay_render_allowed": False,
    }
    (args.out / "game_camera_center_proof_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "accepted_independent_event_count": len(good),
        "median_camera_center_cm": payload["median_camera_center_cm"],
        "max_pairwise_camera_center_distance_cm": payload["max_pairwise_camera_center_distance_cm"],
        "center_prior_allowed": passed,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
