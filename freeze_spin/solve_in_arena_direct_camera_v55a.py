from __future__ import annotations

"""v55a: direct physical-camera diagnostic for the In-Arena event frame.

Unlike v54, this does not force the sealed v52 line intersection homography to
be an exact pinhole homography.  It fits the original regulation line families
as lines, the restricted-area curve as source pixels, and one independently
extracted elevated rim centre.  The free-throw curve and full rim ellipse stay
held out.  The result is diagnostic-only until multistart, support-removal and
half-pixel functional stability all pass.
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
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
    RIM_RADIUS,
    TARGET_RIM_SEED,
    extract_rim_ellipse,
    json_safe,
)


W, H = 960, 540
FT = 30.48
BASELINE_X = -4.0 * FT
FREE_THROW_X = 15.0 * FT
LANE_HALF = 8.0 * FT
RESTRICTED_RADIUS = 4.0 * FT
FREE_THROW_RADIUS = 6.0 * FT


def project(parameters: np.ndarray, xyz: np.ndarray):
    rvec = parameters[:3]
    center = parameters[3:6]
    focal = math.exp(float(parameters[6]))
    pp = parameters[7:9]
    rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    camera = (rotation @ (xyz - center).T).T
    uv = np.column_stack(
        [
            focal * camera[:, 0] / camera[:, 2] + pp[0],
            focal * camera[:, 1] / camera[:, 2] + pp[1],
        ]
    )
    return uv, camera


def floor_circle(center_x: float, radius: float, samples=721):
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    return np.column_stack(
        [
            center_x + radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros(len(theta)),
        ]
    )


def rim_circle(samples=721):
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    return np.column_stack(
        [
            RIM_CENTER[0] + RIM_RADIUS * np.cos(theta),
            RIM_CENTER[1] + RIM_RADIUS * np.sin(theta),
            np.full(len(theta), RIM_CENTER[2]),
        ]
    )


def line_world(name: str):
    if name == "far_lane_sideline":
        return np.asarray(
            [[BASELINE_X, -LANE_HALF, 0.0], [FREE_THROW_X, -LANE_HALF, 0.0]]
        )
    if name == "near_lane_sideline":
        return np.asarray(
            [[BASELINE_X, LANE_HALF, 0.0], [FREE_THROW_X, LANE_HALF, 0.0]]
        )
    if name == "free_throw_line":
        return np.asarray(
            [[FREE_THROW_X, -LANE_HALF, 0.0], [FREE_THROW_X, LANE_HALF, 0.0]]
        )
    if name == "baseline":
        return np.asarray(
            [[BASELINE_X, -LANE_HALF, 0.0], [BASELINE_X, LANE_HALF, 0.0]]
        )
    raise KeyError(name)


def image_line(points: np.ndarray):
    line = np.cross(np.r_[points[0], 1.0], np.r_[points[1], 1.0])
    norm = float(np.hypot(line[0], line[1]))
    if norm <= 1e-12:
        raise RuntimeError("degenerate projected line")
    return line / norm


def line_distances(parameters, observations):
    values = []
    rows = {}
    for name, observed in observations.items():
        predicted, camera = project(parameters, line_world(name))
        line = image_line(predicted)
        distance = observed @ line[:2] + line[2]
        values.extend(distance.tolist())
        rows[name] = {
            "signed_endpoint_distances_px": distance.tolist(),
            "rms_px": float(np.sqrt(np.mean(distance**2))),
            "minimum_camera_depth_cm": float(np.min(camera[:, 2])),
            "predicted_line_endpoints_px": predicted.tolist(),
        }
    return np.asarray(values), rows


def nearest_curve_distances(observed: np.ndarray, predicted: np.ndarray):
    return np.sqrt(
        np.sum((observed[:, None, :] - predicted[None, :, :]) ** 2, axis=2)
    ).min(axis=1)


def circle_signed_distances(
    parameters: np.ndarray,
    observed: np.ndarray,
    center_x: float,
    radius: float,
):
    """First-order Euclidean distance from image points to a projected circle.

    Evaluating the circle as a conic keeps the optimization residual smooth.
    The earlier nearest-point lookup changed its winning sample discontinuously
    and caused otherwise useful roots to exhaust the optimizer budget.
    """
    rvec = parameters[:3]
    center = parameters[3:6]
    focal = math.exp(float(parameters[6]))
    cx, cy = parameters[7:9]
    rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    translation = -rotation @ center
    intrinsic = np.asarray(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    floor_h = intrinsic @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation]
    )
    circle = np.asarray(
        [
            [1.0, 0.0, -center_x],
            [0.0, 1.0, 0.0],
            [-center_x, 0.0, center_x**2 - radius**2],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(floor_h)
    conic = inverse.T @ circle @ inverse
    homogeneous = np.column_stack([observed, np.ones(len(observed))])
    value = np.einsum("ni,ij,nj->n", homogeneous, conic, homogeneous)
    gradient = 2.0 * (homogeneous @ conic.T)[:, :2]
    scale = np.linalg.norm(gradient, axis=1)
    return value / np.maximum(scale, 1e-12)


def residual(parameters, lines, restricted_obs, rim_center_obs):
    line_error, _ = line_distances(parameters, lines)
    restricted_error = circle_signed_distances(
        parameters,
        restricted_obs,
        RIM_CENTER[0],
        RESTRICTED_RADIUS,
    )
    _, restricted_camera = project(
        parameters, floor_circle(RIM_CENTER[0], RESTRICTED_RADIUS, 33)
    )
    rim_predicted, rim_camera = project(parameters, RIM_CENTER[None, :])
    rim_error = rim_predicted[0] - rim_center_obs
    minimum_depth = min(
        float(np.min(restricted_camera[:, 2])), float(rim_camera[0, 2])
    )
    depth_penalty = max(0.0, 20.0 - minimum_depth) / 2.0
    pp = parameters[7:9]
    priors = np.asarray(
        [
            (pp[0] - W / 2.0) / 350.0,
            (pp[1] - H / 2.0) / 350.0,
            (parameters[6] - math.log(3000.0)) / 3.0,
            depth_penalty,
        ]
    )
    return np.r_[line_error, restricted_error, rim_error * 2.0, priors]


def bounds():
    lower = np.r_[
        [-10.0, -10.0, -10.0],
        [-1000.0, -7000.0, 200.0],
        math.log(300.0),
        [-500.0, -500.0],
    ]
    upper = np.r_[
        [10.0, 10.0, 10.0],
        [6000.0, 7000.0, 3000.0],
        math.log(9000.0),
        [1460.0, 1040.0],
    ]
    return lower, upper


def optimize(start, lines, restricted_obs, rim_center_obs, max_nfev=3000):
    lower, upper = bounds()
    start = np.clip(start, lower + 1e-6, upper - 1e-6)
    result = least_squares(
        residual,
        start,
        args=(lines, restricted_obs, rim_center_obs),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    return result.x, float(result.cost), int(result.nfev), float(result.optimality)


def starts(floor_h):
    rows = []
    for pp in (
        (480.0, 270.0),
        (200.0, 0.0),
        (480.0, 0.0),
        (760.0, 0.0),
        (200.0, 540.0),
        (480.0, 540.0),
        (760.0, 540.0),
        (-250.0, 500.0),
        (1100.0, -100.0),
    ):
        result = decompose_floor_homography(floor_h, np.asarray(pp, dtype=float))
        if result is None:
            continue
        focal, center, rvec = result
        rows.append(np.r_[rvec, center, math.log(focal), pp])
    return rows


def summarize(parameters, lines, restricted_obs, ft_obs, rim_center_obs, observed_ellipse):
    line_error, line_rows = line_distances(parameters, lines)
    restricted_uv, restricted_camera = project(
        parameters, floor_circle(RIM_CENTER[0], RESTRICTED_RADIUS)
    )
    ft_uv, ft_camera = project(
        parameters, floor_circle(FREE_THROW_X, FREE_THROW_RADIUS)
    )
    rim_uv, rim_camera = project(parameters, rim_circle())
    rim_center_uv, _ = project(parameters, RIM_CENTER[None, :])
    restricted_error = nearest_curve_distances(restricted_obs, restricted_uv)
    ft_error = nearest_curve_distances(ft_obs, ft_uv)
    rim_center_error = float(np.linalg.norm(rim_center_uv[0] - rim_center_obs))
    projected_ellipse = cv2.fitEllipse(rim_uv.astype(np.float32))
    projected_axes = np.sort(np.asarray(projected_ellipse[1], float))[::-1]
    observed_axes = observed_ellipse["axes"]
    angle_error = min(
        abs(float(projected_ellipse[2]) - observed_ellipse["angle"]),
        abs(float(projected_ellipse[2]) - observed_ellipse["angle"] + 180.0),
        abs(float(projected_ellipse[2]) - observed_ellipse["angle"] - 180.0),
    )
    center = parameters[3:6]
    focal = math.exp(float(parameters[6]))
    pp = parameters[7:9]
    return {
        "camera_center_cm": center.tolist(),
        "rvec": parameters[:3].tolist(),
        "focal_px": focal,
        "principal_point_px": pp.tolist(),
        "line_endpoint_rms_px": float(np.sqrt(np.mean(line_error**2))),
        "line_endpoint_p95_px": float(np.percentile(np.abs(line_error), 95)),
        "line_families": line_rows,
        "restricted_fit_rms_px": float(np.sqrt(np.mean(restricted_error**2))),
        "restricted_fit_p95_px": float(np.percentile(restricted_error, 95)),
        "free_throw_holdout_rms_px": float(np.sqrt(np.mean(ft_error**2))),
        "free_throw_holdout_p95_px": float(np.percentile(ft_error, 95)),
        "rim_center_error_px": rim_center_error,
        "rim_center_predicted_px": rim_center_uv[0].tolist(),
        "rim_center_observed_px": rim_center_obs.tolist(),
        "rim_projected_axes_major_minor_px": projected_axes.tolist(),
        "rim_observed_axes_major_minor_px": observed_axes.tolist(),
        "rim_major_axis_error_px": float(abs(projected_axes[0] - observed_axes[0])),
        "rim_minor_axis_error_px": float(abs(projected_axes[1] - observed_axes[1])),
        "rim_angle_error_deg": angle_error,
        "minimum_geometry_depth_cm": min(
            float(np.min(restricted_camera[:, 2])),
            float(np.min(ft_camera[:, 2])),
            float(np.min(rim_camera[:, 2])),
        ),
        "camera_distance_from_basket_cm": float(np.linalg.norm(center - RIM_CENTER)),
    }


def functional_p95(a, b, volume):
    ua, ca = project(a, volume)
    ub, cb = project(b, volume)
    valid = (ca[:, 2] > 20.0) & (cb[:, 2] > 20.0)
    if int(valid.sum()) < 20:
        return float("inf")
    return float(np.percentile(np.linalg.norm(ua[valid] - ub[valid], axis=1), 95))


def draw_overlay(image, parameters, lines, restricted_obs, ft_obs, rim_points, out):
    result = image.copy()
    colours = {
        "far_lane_sideline": (0, 255, 255),
        "near_lane_sideline": (0, 255, 255),
        "free_throw_line": (0, 165, 255),
        "baseline": (0, 165, 255),
    }
    for name, observed in lines.items():
        predicted, _ = project(parameters, line_world(name))
        cv2.line(
            result,
            tuple(np.round(predicted[0]).astype(int)),
            tuple(np.round(predicted[1]).astype(int)),
            colours[name],
            2,
            cv2.LINE_AA,
        )
        for point in observed:
            cv2.circle(result, tuple(np.round(point).astype(int)), 5, (255, 0, 0), 2)
    for world, observed, colour in (
        (floor_circle(RIM_CENTER[0], RESTRICTED_RADIUS), restricted_obs, (0, 255, 0)),
        (floor_circle(FREE_THROW_X, FREE_THROW_RADIUS), ft_obs, (0, 0, 255)),
        (rim_circle(), rim_points, (255, 0, 255)),
    ):
        predicted, _ = project(parameters, world)
        cv2.polylines(
            result,
            [np.round(predicted).astype(np.int32)],
            True,
            colour,
            2,
            cv2.LINE_AA,
        )
        for point in observed.astype(int):
            cv2.circle(result, tuple(point), 2, (255, 255, 255), -1)
    cv2.imwrite(str(out), result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--floor-proof", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturbation-trials", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("v55a expects native 960x540 In-Arena frame")
    floor = json.loads(args.floor_proof.read_text(encoding="utf-8"))
    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    if floor.get("status") != "PASS_IN_ARENA_FLOOR_V52":
        raise RuntimeError("v52 floor is not sealed PASS")
    floor_h = norm_h(np.asarray(floor["floor_homography_world_to_image"], float))
    lines = {
        key: np.asarray(value, dtype=float)
        for key, value in spec["training_line_segments_px"].items()
    }
    restricted_obs = np.asarray(
        spec["heldout_curves_px"]["restricted_area_arc"], dtype=float
    )
    ft_obs = np.asarray(
        spec["heldout_curves_px"]["free_throw_circle_dashed_half"], dtype=float
    )
    rim_center, rim_axes, rim_angle, rim_points, rim_diagnostic = extract_rim_ellipse(
        image, TARGET_RIM_SEED
    )
    observed_ellipse = {"center": rim_center, "axes": rim_axes, "angle": rim_angle}

    def root_job(start):
        parameters, cost, nfev, optimality = optimize(
            start, lines, restricted_obs, rim_center
        )
        qa = summarize(
            parameters, lines, restricted_obs, ft_obs, rim_center, observed_ellipse
        )
        return {
            "parameters": parameters,
            "cost": cost,
            "nfev": nfev,
            "optimality": optimality,
            "qa": qa,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        roots = list(pool.map(root_job, starts(floor_h)))
    roots.sort(key=lambda row: row["cost"])
    best = roots[0]
    competitive = [
        row
        for row in roots
        if row["cost"] <= min(item["cost"] for item in roots) * 1.05 + 1e-8
    ]
    volume = action_volume()
    root_pairs = []
    for left in range(len(competitive)):
        for right in range(left + 1, len(competitive)):
            root_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "action_volume_p95_px": functional_p95(
                        competitive[left]["parameters"],
                        competitive[right]["parameters"],
                        volume,
                    ),
                }
            )

    support_rows = []
    for dropped in list(lines) + ["restricted"]:
        use_lines = {key: value for key, value in lines.items() if key != dropped}
        use_restricted = (
            restricted_obs[::2] if dropped == "restricted" else restricted_obs
        )
        parameters, _, _, _ = optimize(
            best["parameters"],
            use_lines,
            use_restricted,
            rim_center,
            max_nfev=1800,
        )
        support_rows.append(
            {
                "dropped": dropped,
                "camera_center_shift_cm": float(
                    np.linalg.norm(parameters[3:6] - best["parameters"][3:6])
                ),
                "action_volume_p95_shift_px": functional_p95(
                    best["parameters"], parameters, volume
                ),
            }
        )

    rng = np.random.default_rng(550903)
    perturb_inputs = []
    for trial in range(args.perturbation_trials):
        perturb_inputs.append(
            (
                trial,
                {
                    key: value + rng.uniform(-0.5, 0.5, value.shape)
                    for key, value in lines.items()
                },
                restricted_obs + rng.uniform(-0.5, 0.5, restricted_obs.shape),
                rim_center + rng.uniform(-0.5, 0.5, 2),
            )
        )

    def perturb_job(item):
        trial, perturbed_lines, perturbed_restricted, perturbed_rim = item
        parameters, _, _, _ = optimize(
            best["parameters"],
            perturbed_lines,
            perturbed_restricted,
            perturbed_rim,
            max_nfev=1200,
        )
        return {
            "trial": trial,
            "camera_center_shift_cm": float(
                np.linalg.norm(parameters[3:6] - best["parameters"][3:6])
            ),
            "action_volume_p95_shift_px": functional_p95(
                best["parameters"], parameters, volume
            ),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        perturbations = list(pool.map(perturb_job, perturb_inputs))
    perturbations.sort(key=lambda row: row["trial"])

    qa = best["qa"]
    pp = np.asarray(qa["principal_point_px"])
    center = np.asarray(qa["camera_center_cm"])
    gates = {
        "line_endpoint_rms_at_most_2px": qa["line_endpoint_rms_px"] <= 2.0,
        "line_endpoint_p95_at_most_3px": qa["line_endpoint_p95_px"] <= 3.0,
        "restricted_fit_p95_at_most_3px": qa["restricted_fit_p95_px"] <= 3.0,
        "rim_center_at_most_2px": qa["rim_center_error_px"] <= 2.0,
        "free_throw_holdout_p95_at_most_6_5px": qa[
            "free_throw_holdout_p95_px"
        ] <= 6.5,
        "rim_major_axis_holdout_at_most_7px": qa["rim_major_axis_error_px"] <= 7.0,
        "rim_minor_axis_holdout_at_most_5px": qa["rim_minor_axis_error_px"] <= 5.0,
        "principal_point_within_160px_crop_margin": (
            -160.0 <= pp[0] <= W + 160.0 and -160.0 <= pp[1] <= H + 160.0
        ),
        "camera_height_5m_to_25m": 500.0 <= center[2] <= 2500.0,
        "camera_outside_playing_width": abs(center[1]) >= 25.0 * FT,
        "camera_distance_under_60m": qa["camera_distance_from_basket_cm"] <= 6000.0,
        "competitive_roots_functionally_equivalent": max(
            (row["action_volume_p95_px"] for row in root_pairs), default=0.0
        ) <= 0.5,
        "support_removal_action_volume_at_most_2px": max(
            row["action_volume_p95_shift_px"] for row in support_rows
        ) <= 2.0,
        "half_pixel_center_shift_at_most_25cm": max(
            row["camera_center_shift_cm"] for row in perturbations
        ) <= 25.0,
        "half_pixel_action_volume_at_most_2px": max(
            row["action_volume_p95_shift_px"] for row in perturbations
        ) <= 2.0,
    }
    passed = bool(all(gates.values()))
    report = json_safe(
        {
            "status": "PASS_IN_ARENA_DIRECT_CAMERA_V55A"
            if passed
            else "FAIL_IN_ARENA_DIRECT_CAMERA_V55A",
            "version": "v55a",
            "game_id": "0022500301",
            "event_id": 489,
            "camera": "In Arena static-scene family v53b target state",
            "method": "direct physical pinhole fit to four regulation line families + restricted-area source curve + elevated regulation rim centre; v52 homography used only for deterministic starts; free-throw curve and full rim ellipse held out",
            "guardrail": "Diagnostic only. A pass does not promote the event camera or authorize replay until independently reproduced and paired with the v53b family/physical-centre evidence.",
            "rim_source_pixel_diagnostic": rim_diagnostic,
            "best": {
                "cost": best["cost"],
                "nfev": best["nfev"],
                "optimality": best["optimality"],
                **qa,
            },
            "multistart": {
                "root_count": len(roots),
                "competitive_root_count": len(competitive),
                "max_action_volume_p95_px": max(
                    (row["action_volume_p95_px"] for row in root_pairs), default=0.0
                ),
                "pairs": root_pairs,
                "roots": [
                    {"cost": row["cost"], "qa": row["qa"]} for row in roots
                ],
            },
            "support_removal": support_rows,
            "half_pixel_perturbation": {
                "trial_count": len(perturbations),
                "max_camera_center_shift_cm": max(
                    row["camera_center_shift_cm"] for row in perturbations
                ),
                "p95_camera_center_shift_cm": float(
                    np.percentile(
                        [row["camera_center_shift_cm"] for row in perturbations], 95
                    )
                ),
                "max_action_volume_p95_shift_px": max(
                    row["action_volume_p95_shift_px"] for row in perturbations
                ),
                "trials": perturbations,
            },
            "gates": gates,
            "permissions": {
                "metric_event_camera_allowed": False,
                "two_metric_cameras_allowed": False,
                "replay_render_allowed": False,
            },
        }
    )
    (args.out / "in_arena_direct_camera_v55a.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    draw_overlay(
        image,
        best["parameters"],
        lines,
        restricted_obs,
        ft_obs,
        rim_points,
        args.out / "in_arena_direct_camera_overlay_v55a.png",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "camera_center_cm": qa["camera_center_cm"],
                "principal_point_px": qa["principal_point_px"],
                "line_p95_px": qa["line_endpoint_p95_px"],
                "restricted_p95_px": qa["restricted_fit_p95_px"],
                "rim_center_error_px": qa["rim_center_error_px"],
                "free_throw_holdout_p95_px": qa["free_throw_holdout_p95_px"],
                "max_support_action_volume_p95_px": max(
                    row["action_volume_p95_shift_px"] for row in support_rows
                ),
                "max_half_pixel_center_shift_cm": report["half_pixel_perturbation"][
                    "max_camera_center_shift_cm"
                ],
                "max_half_pixel_action_volume_p95_px": report[
                    "half_pixel_perturbation"
                ]["max_action_volume_p95_shift_px"],
                "gates": report["gates"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
