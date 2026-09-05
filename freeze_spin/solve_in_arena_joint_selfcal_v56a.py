from __future__ import annotations

"""v56a: joint In-Arena rotational self-calibration preflight.

This deliberately does not reuse the v52 homography as an exact pinhole
camera.  The immutable target state is fitted to its original regulation floor
lines, restricted-area pixels and mount-excluded regulation rim.  Five v53b
pixel-verified optical states constrain the target intrinsics through the
pure-rotation image homographies of a shared physical camera centre.  Every
state keeps its own rotation/focal length and a bounded crop offset.

The run is diagnostic-only.  It cannot promote a metric camera or replay.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.diagnose_in_arena_shared_center_v54a import (
    action_volume,
    decompose_floor_homography,
    norm_h,
)
from freeze_spin.prove_in_arena_noncoplanar_center_v54 import (
    RIM_CENTER,
    TARGET_RIM_SEED,
    extract_rim_ellipse,
    json_safe,
    project_point,
    refine_transfer,
    sha256,
)
from freeze_spin.solve_in_arena_direct_camera_v55a import (
    FREE_THROW_RADIUS,
    FREE_THROW_X,
    RESTRICTED_RADIUS,
    bounds as target_bounds,
    circle_signed_distances,
    floor_circle,
    functional_p95,
    line_distances,
    nearest_curve_distances,
    optimize as optimize_target,
    project,
    rim_circle,
    starts as target_starts,
)


W, H = 960, 540


def intrinsic(focal: float, pp: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[focal, 0.0, pp[0]], [0.0, focal, pp[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def unpack(x: np.ndarray, keys: list[str]):
    target = np.r_[x[3:6], x[:3], x[6:9]]
    blocks = x[9:].reshape(len(keys) - 1, 6)
    states = {"target": target}
    for index, key in enumerate(keys[1:]):
        block = blocks[index]
        states[key] = np.r_[
            block[:3],
            x[:3],
            block[3],
            x[7:9] + block[4:6],
        ]
    return states, blocks


def relative_homography(target_parameters: np.ndarray, state_parameters: np.ndarray):
    rt, _ = cv2.Rodrigues(target_parameters[:3].reshape(3, 1))
    rs, _ = cv2.Rodrigues(state_parameters[:3].reshape(3, 1))
    kt = intrinsic(math.exp(float(target_parameters[6])), target_parameters[7:9])
    ks = intrinsic(math.exp(float(state_parameters[6])), state_parameters[7:9])
    return norm_h(ks @ rs @ rt.T @ np.linalg.inv(kt))


def transfer_points(h: np.ndarray, points: np.ndarray):
    q = (h @ np.column_stack([points, np.ones(len(points))]).T).T
    return q[:, :2] / q[:, 2:3]


def spatial_subset(points: np.ndarray, count: int = 16):
    """Deterministic farthest-point subset with broad target-frame support."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return np.arange(len(points))
    selected = [int(np.argmin(np.linalg.norm(points - [W / 2, H / 2], axis=1)))]
    minimum = np.linalg.norm(points - points[selected[0]], axis=1)
    for _ in range(count - 1):
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(points - points[index], axis=1))
    return np.asarray(selected, dtype=int)


def make_start(target_parameters, keys, transfers):
    rows = [target_parameters[3:6], target_parameters[:3], target_parameters[6:9]]
    rotation, _ = cv2.Rodrigues(target_parameters[:3].reshape(3, 1))
    target_h = intrinsic(
        math.exp(float(target_parameters[6])), target_parameters[7:9]
    ) @ np.column_stack(
        [
            rotation[:, 0],
            rotation[:, 1],
            -rotation @ target_parameters[3:6],
        ]
    )
    for key in keys[1:]:
        h_state = norm_h(transfers[key] @ target_h)
        decomposed = decompose_floor_homography(
            h_state, target_parameters[7:9]
        )
        if decomposed is None:
            raise RuntimeError(f"cannot initialize optical state {key}")
        focal, _, rvec = decomposed
        rows.append(np.r_[rvec, math.log(focal), 0.0, 0.0])
    return np.concatenate(rows)


def solve(
    start,
    keys,
    lines,
    restricted,
    rim_centers,
    transfer_correspondences,
    max_nfev=3500,
):
    count = len(keys) - 1
    target_lo, target_hi = target_bounds()
    lower = np.r_[
        target_lo[3:6],
        target_lo[:3],
        target_lo[6:9],
        np.tile([-10.0, -10.0, -10.0, math.log(300.0), -60.0, -60.0], count),
    ]
    upper = np.r_[
        target_hi[3:6],
        target_hi[:3],
        target_hi[6:9],
        np.tile([10.0, 10.0, 10.0, math.log(9000.0), 60.0, 60.0], count),
    ]
    start = np.clip(np.asarray(start, dtype=float), lower + 1e-6, upper - 1e-6)

    def residual(x):
        states, blocks = unpack(x, keys)
        target = states["target"]
        line_error, _ = line_distances(target, lines)
        restricted_error = circle_signed_distances(
            target, restricted, RIM_CENTER[0], RESTRICTED_RADIUS
        )
        target_rim, target_depth = project(target, RIM_CENTER[None, :])
        values = [
            line_error,
            restricted_error,
            (target_rim[0] - rim_centers["target"]) * 2.0,
            (target[7:9] - np.asarray([W / 2.0, H / 2.0])) / 350.0,
            np.asarray([(target[6] - math.log(3000.0)) / 3.0]),
            np.minimum(target_depth - 20.0, 0.0).ravel() * 5.0,
        ]
        for index, key in enumerate(keys[1:]):
            state = states[key]
            predicted_rim, rim_depth = project(state, RIM_CENTER[None, :])
            source, observed = transfer_correspondences[key]
            predicted_h = relative_homography(target, state)
            predicted = transfer_points(predicted_h, source)
            # Sixteen correspondences represent the eight independent degrees
            # of freedom of one measured homography.  Avoid pretending they are
            # 32 unrelated metric observations.
            values.append((predicted - observed).ravel() * 0.5)
            values.append((predicted_rim[0] - rim_centers[key]) * 2.0)
            values.append(blocks[index, 4:6] / 25.0)
            values.append(np.minimum(rim_depth - 20.0, 0.0).ravel() * 5.0)
        return np.concatenate(values)

    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    states, blocks = unpack(result.x, keys)
    return {
        "x": result.x,
        "states": states,
        "blocks": blocks,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "optimality": float(result.optimality),
    }


def summarize(solution, keys, lines, restricted, ft, rim_centers, rim_ellipses, transfer_data):
    states = solution["states"]
    target = states["target"]
    line_error, _ = line_distances(target, lines)
    restricted_error = nearest_curve_distances(
        restricted,
        project(target, floor_circle(RIM_CENTER[0], RESTRICTED_RADIUS))[0],
    )
    ft_error = nearest_curve_distances(
        ft, project(target, floor_circle(FREE_THROW_X, FREE_THROW_RADIUS))[0]
    )
    state_rows = {}
    rim_errors = []
    transfer_errors = []
    for key in keys:
        parameters = states[key]
        rim_uv, _ = project(parameters, RIM_CENTER[None, :])
        rim_error = float(np.linalg.norm(rim_uv[0] - rim_centers[key]))
        rim_errors.append(rim_error)
        row = {
            "focal_px": math.exp(float(parameters[6])),
            "principal_point_px": parameters[7:9].tolist(),
            "rvec": parameters[:3].tolist(),
            "rim_center_observed_px": rim_centers[key].tolist(),
            "rim_center_predicted_px": rim_uv[0].tolist(),
            "rim_center_error_px": rim_error,
        }
        if key != "target":
            source, observed = transfer_data[key]
            predicted = transfer_points(relative_homography(target, parameters), source)
            errors = np.linalg.norm(predicted - observed, axis=1)
            transfer_errors.extend(errors.tolist())
            row["rotation_homography_rms_px"] = float(np.sqrt(np.mean(errors**2)))
            row["rotation_homography_p95_px"] = float(np.percentile(errors, 95))
        state_rows[key] = row
    projected_rim = project(target, rim_circle())[0]
    ellipse = cv2.fitEllipse(projected_rim.astype(np.float32))
    projected_axes = np.sort(np.asarray(ellipse[1], dtype=float))[::-1]
    observed_axes = rim_ellipses["target"]["axes"]
    return {
        "camera_center_cm": target[3:6].tolist(),
        "target_focal_px": math.exp(float(target[6])),
        "target_principal_point_px": target[7:9].tolist(),
        "line_p95_px": float(np.percentile(np.abs(line_error), 95)),
        "restricted_p95_px": float(np.percentile(restricted_error, 95)),
        "free_throw_holdout_p95_px": float(np.percentile(ft_error, 95)),
        "max_rim_center_error_px": max(rim_errors),
        "rim_axis_error_major_minor_px": np.abs(projected_axes - observed_axes).tolist(),
        "rotation_homography_p95_px": float(np.percentile(transfer_errors, 95)),
        "states": state_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--floor-proof", type=Path, required=True)
    ap.add_argument("--family-proof", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root-limit", type=int, default=9)
    ap.add_argument("--max-nfev", type=int, default=3500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    target_image = cv2.imread(str(args.target))
    floor = json.loads(args.floor_proof.read_text(encoding="utf-8"))
    family = json.loads(args.family_proof.read_text(encoding="utf-8"))
    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    if target_image is None or target_image.shape[:2] != (H, W):
        raise RuntimeError("target must be native 960x540")
    if floor.get("status") != "PASS_IN_ARENA_FLOOR_V52":
        raise RuntimeError("v52 floor is not sealed PASS")
    if family.get("status") != "PASS_IN_ARENA_STATIC_SCENE_FAMILY_V53B":
        raise RuntimeError("v53b family is not sealed PASS")
    if sha256(args.target) != family["target_sha256_png"]:
        raise RuntimeError("target differs from sealed v53b pixels")

    lines = {
        key: np.asarray(value, dtype=float)
        for key, value in spec["training_line_segments_px"].items()
    }
    restricted = np.asarray(
        spec["heldout_curves_px"]["restricted_area_arc"], dtype=float
    )
    ft = np.asarray(
        spec["heldout_curves_px"]["free_throw_circle_dashed_half"], dtype=float
    )
    keys = ["target"]
    images = {"target": target_image}
    transfers = {}
    transfer_data = {}
    rim_centers = {}
    rim_ellipses = {}
    rim_diagnostics = {}

    center, axes, angle, _, diagnostic = extract_rim_ellipse(
        target_image, TARGET_RIM_SEED
    )
    rim_centers["target"] = center
    rim_ellipses["target"] = {"axes": axes, "angle": angle}
    rim_diagnostics["target"] = diagnostic

    for row in family["selected_candidates"]:
        key = f"event_{int(row['event_probe'])}"
        image = cv2.imread(str(args.states / row["file"]))
        sealed = norm_h(np.asarray(row["H_target_to_state"], dtype=float))
        transfer, p, q, transfer_diagnostic = refine_transfer(
            target_image, image, sealed
        )
        subset = spatial_subset(p)
        p, q = p[subset], q[subset]
        keys.append(key)
        images[key] = image
        transfers[key] = transfer
        transfer_data[key] = (p, q)
        predicted = project_point(transfer, TARGET_RIM_SEED)
        center, axes, angle, _, diagnostic = extract_rim_ellipse(image, predicted)
        diagnostic["transfer_refinement"] = transfer_diagnostic
        rim_centers[key] = center
        rim_ellipses[key] = {"axes": axes, "angle": angle}
        rim_diagnostics[key] = diagnostic

    h0 = norm_h(np.asarray(floor["floor_homography_world_to_image"], dtype=float))
    target_roots = []
    for start in target_starts(h0):
        parameters, cost, _, _ = optimize_target(
            start, lines, restricted, rim_centers["target"], max_nfev=1800
        )
        target_roots.append((cost, parameters))
    target_roots.sort(key=lambda row: row[0])
    roots = []
    for _, target_parameters in target_roots[: args.root_limit]:
        roots.append(
            solve(
                make_start(target_parameters, keys, transfers),
                keys,
                lines,
                restricted,
                rim_centers,
                transfer_data,
                max_nfev=args.max_nfev,
            )
        )
    roots.sort(key=lambda row: row["cost"])
    competitive = [row for row in roots if row["cost"] <= roots[0]["cost"] * 1.05]
    best = competitive[0]
    summaries = [
        summarize(
            row, keys, lines, restricted, ft, rim_centers, rim_ellipses, transfer_data
        )
        for row in roots
    ]
    root_pairs = []
    volume = action_volume()
    for left in range(len(competitive)):
        for right in range(left + 1, len(competitive)):
            a = competitive[left]["states"]["target"]
            b = competitive[right]["states"]["target"]
            root_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "camera_center_shift_cm": float(
                        np.linalg.norm(a[3:6] - b[3:6])
                    ),
                    "action_volume_p95_px": functional_p95(a, b, volume),
                }
            )
    qa = summaries[0]
    payload = json_safe(
        {
            "status": "DIAGNOSTIC_IN_ARENA_JOINT_SELFCAL_V56A",
            "version": "v56a",
            "game_id": "0022500301",
            "target_event_id": 489,
            "method": "target source regulation lines + restricted curve + mount-excluded rim; five v53b pure-rotation optical-state homographies; shared centre/target principal point with state-specific pan-tilt-focal and bounded crop",
            "guardrail": "Diagnostic only; no metric-camera or replay permission regardless of numerical result.",
            "best": qa,
            "multistart": {
                "root_count": len(roots),
                "competitive_root_count": len(competitive),
                "max_competitive_center_shift_cm": max(
                    (row["camera_center_shift_cm"] for row in root_pairs), default=0.0
                ),
                "max_competitive_action_volume_p95_px": max(
                    (row["action_volume_p95_px"] for row in root_pairs), default=0.0
                ),
                "pairs": root_pairs,
                "roots": [
                    {"cost": row["cost"], "nfev": row["nfev"], "qa": summary}
                    for row, summary in zip(roots, summaries)
                ],
            },
            "rim_source_pixel_diagnostics": rim_diagnostics,
            "permissions": {
                "physical_camera_center_allowed": False,
                "metric_event_camera_allowed": False,
                "replay_render_allowed": False,
            },
        }
    )
    path = args.out / "in_arena_joint_selfcal_v56a.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "camera_center_cm": qa["camera_center_cm"],
                "principal_point_px": qa["target_principal_point_px"],
                "line_p95_px": qa["line_p95_px"],
                "restricted_p95_px": qa["restricted_p95_px"],
                "free_throw_holdout_p95_px": qa["free_throw_holdout_p95_px"],
                "max_rim_center_error_px": qa["max_rim_center_error_px"],
                "rotation_homography_p95_px": qa["rotation_homography_p95_px"],
                "competitive_root_count": len(competitive),
                "max_root_center_shift_cm": payload["multistart"][
                    "max_competitive_center_shift_cm"
                ],
                "max_root_action_volume_p95_px": payload["multistart"][
                    "max_competitive_action_volume_p95_px"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
