from __future__ import annotations

"""v54: fail-closed In-Arena shared physical camera-centre proof.

Inputs are the sealed v52 metric floor, the sealed v53b same-static-scene
family, the immutable event-489 target frame, and the selected v53a native
frames.  Source pixels independently determine one rim-centre observation in
every state.  Nothing is transported from the target rim into a source state.

The solve allows state-specific pan/tilt/zoom and bounded crop/principal-point
movement around a shared optical prior.  Passing can authorize only the
physical camera centre.  The event camera and replay remain forbidden.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.diagnose_in_arena_shared_center_v54a import (
    action_volume,
    decompose_floor_homography,
    floor_observations,
    norm_h,
    project_camera,
    project_h,
)
from freeze_spin.prove_in_arena_floor_v52 import (
    corners_from_segments,
    homography_from_segments,
)


W, H = 960, 540
FT = 30.48
INCH = 2.54
RIM_CENTER = np.asarray([15.0 * INCH, 0.0, 10.0 * FT], dtype=np.float64)
RIM_RADIUS = 9.0 * INCH
TARGET_RIM_SEED = np.asarray([544.0, 123.0], dtype=np.float64)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sift_matches(a: np.ndarray, b: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.012)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    good = []
    for first, second in cv2.BFMatcher().knnMatch(da, db, k=2):
        if first.distance < 0.70 * second.distance:
            good.append(first)
    return (
        np.float32([ka[item.queryIdx].pt for item in good]),
        np.float32([kb[item.trainIdx].pt for item in good]),
    )


def homography_error(h: np.ndarray, source: np.ndarray, target: np.ndarray):
    predicted = cv2.perspectiveTransform(source[:, None, :].astype(np.float32), h)[:, 0]
    return np.linalg.norm(predicted - target, axis=1)


def refine_transfer(
    target: np.ndarray, state: np.ndarray, sealed_h: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    source_points, state_points = sift_matches(target, state)
    if len(source_points) < 100:
        raise RuntimeError("insufficient SIFT matches for v54 transfer refinement")
    seed_error = homography_error(sealed_h, source_points, state_points)
    support = seed_error <= 3.0
    if int(support.sum()) < 80:
        raise RuntimeError("insufficient sealed-v53b-compatible static support")
    refined, mask = cv2.findHomography(
        source_points[support],
        state_points[support],
        cv2.RANSAC,
        1.2,
        maxIters=30000,
        confidence=0.999,
    )
    if refined is None:
        raise RuntimeError("v54 transfer refinement failed")
    inliers = mask.ravel().astype(bool)
    p = source_points[support][inliers].astype(np.float64)
    q = state_points[support][inliers].astype(np.float64)
    refined = norm_h(refined)
    error = homography_error(refined, p.astype(np.float32), q.astype(np.float32))
    xs = p[:, 0]
    ys = p[:, 1]
    cells = len(
        set(
            (
                min(3, max(0, int(x / (W / 4)))),
                min(2, max(0, int(y / (H / 3)))),
            )
            for x, y in p
        )
    )
    diagnostic = {
        "raw_matches": int(len(source_points)),
        "sealed_transform_support": int(support.sum()),
        "refined_inliers": int(len(p)),
        "refined_rms_px": float(np.sqrt(np.mean(error**2))),
        "refined_p95_px": float(np.percentile(error, 95)),
        "target_support_bbox_fraction": float(
            ((xs.max() - xs.min()) * (ys.max() - ys.min())) / (W * H)
        ),
        "target_support_grid_cells": cells,
    }
    if (
        diagnostic["refined_inliers"] < 80
        or diagnostic["refined_p95_px"] > 1.5
        or diagnostic["target_support_bbox_fraction"] < 0.25
        or cells < 5
    ):
        raise RuntimeError(f"v54 refined transfer gate failed: {diagnostic}")
    return refined, p, q, diagnostic


def project_point(h: np.ndarray, point: np.ndarray) -> np.ndarray:
    q = h @ np.r_[point, 1.0]
    return q[:2] / q[2]


def extract_rim_ellipse(image: np.ndarray, predicted_center: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yy, xx = np.indices(image.shape[:2])
    roi = (
        (np.abs(xx - predicted_center[0]) <= 78)
        & (np.abs(yy - predicted_center[1]) <= 48)
    )
    orange = (
        (((hsv[:, :, 0] <= 18) | (hsv[:, :, 0] >= 172)))
        & (hsv[:, :, 1] >= 110)
        & (hsv[:, :, 2] >= 100)
        & roi
    ).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(orange, 8)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[index]]
        if not (120 <= area <= 1400 and 28 <= width <= 90 and 8 <= height <= 42):
            continue
        distance = float(np.linalg.norm(centroids[index] - predicted_center))
        if distance <= 35:
            candidates.append((distance, -area, index))
    if not candidates:
        raise RuntimeError(f"no regulation-rim colour component near {predicted_center}")
    _, _, selected = min(candidates)
    component_points = np.column_stack(np.where(labels == selected))[:, ::-1].astype(
        np.float32
    )
    if len(component_points) < 5:
        raise RuntimeError("rim component cannot support ellipse")
    # The board-side orange mounting block is connected to the regulation ring
    # in these compressed frames.  Fitting the complete component biases the
    # centre roughly 10--15 px toward the board.  The mount is distinguishable
    # without a fitted camera: it is a run of nearly full-height orange columns
    # on the right side of the selected component.  Retain the sparse ring arc,
    # include detached antialiased pixels immediately to its left, and exclude
    # that dense mount run before fitting the diagnostic ellipse.
    x0, y0, width, height = [int(v) for v in stats[selected, :4]]
    column_counts = {
        x: int(np.sum(component_points[:, 0] == x))
        for x in range(x0, x0 + width)
    }
    dense_columns = [
        x for x, value in column_counts.items() if value >= 0.72 * height
    ]
    mount_cutoff = None
    for x in dense_columns:
        if x >= x0 + 0.45 * width and x + 1 in dense_columns:
            mount_cutoff = x
            break
    if mount_cutoff is None:
        raise RuntimeError("cannot separate regulation rim from orange mount")
    ring_mask = (
        (orange.astype(bool))
        & (xx >= x0 - 8)
        & (xx < mount_cutoff)
        & (yy >= y0)
        & (yy < y0 + height)
    )
    points = np.column_stack(np.where(ring_mask))[:, ::-1].astype(np.float32)
    if len(points) < 80:
        raise RuntimeError("mount-excluded rim pixels are insufficient")
    ellipse = cv2.fitEllipse(points)
    center = np.asarray(ellipse[0], dtype=np.float64)
    axes = np.sort(np.asarray(ellipse[1], dtype=np.float64))[::-1]
    angle = float(ellipse[2])
    return center, axes, angle, points, {
        "predicted_search_center_px": predicted_center.tolist(),
        "component_area_px_before_mount_exclusion": int(len(component_points)),
        "ring_pixel_count_after_mount_exclusion": int(len(points)),
        "component_centroid_px": centroids[selected].tolist(),
        "component_bbox_xywh": [int(v) for v in stats[selected, :4]],
        "mount_cutoff_x_px_exclusive": int(mount_cutoff),
        "mount_exclusion_rule": "first two-column run at >=72% component height in right 55%; retain source-orange pixels from x0-8 through the preceding column within the component y-span",
        "ellipse_center_px": center.tolist(),
        "ellipse_axes_major_minor_px": axes.tolist(),
        "ellipse_opencv_angle_deg": angle,
    }


def sparse_floor_observations(h: np.ndarray):
    xyz, uv = floor_observations(h)
    selected = []
    for row in range(3):
        for column in range(4):
            center = np.asarray([(column + 0.5) * W / 4.0, (row + 0.5) * H / 3.0])
            in_cell = (
                (uv[:, 0] >= column * W / 4.0)
                & (uv[:, 0] < (column + 1) * W / 4.0)
                & (uv[:, 1] >= row * H / 3.0)
                & (uv[:, 1] < (row + 1) * H / 3.0)
            )
            choices = np.where(in_cell)[0]
            if len(choices):
                selected.append(int(choices[np.argmin(np.linalg.norm(uv[choices] - center, axis=1))]))
    if len(selected) < 6:
        order = np.argsort(np.linalg.norm(uv - np.asarray([W / 2.0, H / 2.0]), axis=1))
        selected = list(dict.fromkeys(selected + order[: 8 - len(selected)].tolist()))
    return xyz[selected], uv[selected]


def make_start(keys, homographies, seed_pp):
    pp = np.asarray(seed_pp, dtype=np.float64)
    centers = []
    blocks = []
    for key in keys:
        decomposed = decompose_floor_homography(homographies[key], pp)
        if decomposed is None:
            return None
        focal, center, rvec = decomposed
        centers.append(center)
        blocks.append([math.log(focal), *rvec, 0.0, 0.0])
    return np.r_[np.median(np.asarray(centers), axis=0), pp, np.asarray(blocks).ravel()]


def unpack(x: np.ndarray, count: int):
    return x[:3], x[3:5], x[5:].reshape(count, 6)


def solve(
    keys,
    homographies,
    floor_data,
    rim_centers,
    *,
    seed_pp,
    warm=None,
    max_nfev=12000,
):
    count = len(keys)
    start = np.asarray(warm, dtype=np.float64) if warm is not None else make_start(keys, homographies, seed_pp)
    if start is None:
        return None
    lower = np.r_[
        [-1000.0, -7000.0, 200.0],
        [-250.0, -250.0],
        np.tile([math.log(300.0), -10.0, -10.0, -10.0, -60.0, -60.0], count),
    ]
    upper = np.r_[
        [6000.0, 7000.0, 3000.0],
        [1210.0, 790.0],
        np.tile([math.log(9000.0), 10.0, 10.0, 10.0, 60.0, 60.0], count),
    ]
    start = np.clip(start, lower + 1e-6, upper - 1e-6)

    def residual(x):
        center, common_pp, blocks = unpack(x, count)
        values = []
        for index, key in enumerate(keys):
            block = blocks[index]
            focal = math.exp(float(block[0]))
            pp = common_pp + block[4:6]
            xyz, observed = floor_data[key]
            predicted, depth = project_camera(center, pp, focal, block[1:4], xyz)
            values.append((predicted - observed).ravel())
            values.append(np.minimum(depth - 10.0, 0.0) * 10.0)
            rim_predicted, rim_depth = project_camera(
                center, pp, focal, block[1:4], RIM_CENTER[None, :]
            )
            # Floor points are samples from one 8-DOF homography, not dozens of
            # independent measurements. Give the direct non-coplanar point its
            # appropriate structural weight without fabricating extra pixels.
            values.append((rim_predicted[0] - rim_centers[key]) * 2.0)
            values.append(np.minimum(rim_depth - 10.0, 0.0) * 10.0)
            values.append(block[4:6] / 25.0)
        values.append((common_pp - np.asarray([W / 2.0, H / 2.0])) / 350.0)
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
    center, common_pp, blocks = unpack(result.x, count)
    states = {}
    floor_errors = []
    rim_errors = []
    for index, key in enumerate(keys):
        block = blocks[index]
        focal = math.exp(float(block[0]))
        pp = common_pp + block[4:6]
        xyz, observed = floor_data[key]
        floor_predicted, depth = project_camera(center, pp, focal, block[1:4], xyz)
        ferr = np.linalg.norm(floor_predicted - observed, axis=1)
        rim_predicted, _ = project_camera(center, pp, focal, block[1:4], RIM_CENTER[None, :])
        rerr = float(np.linalg.norm(rim_predicted[0] - rim_centers[key]))
        floor_errors.extend(ferr.tolist())
        rim_errors.append(rerr)
        states[key] = {
            "focal_px": focal,
            "principal_point_px": pp.tolist(),
            "principal_point_delta_px": block[4:6].tolist(),
            "rvec": block[1:4].tolist(),
            "floor_point_count": int(len(ferr)),
            "floor_rms_px": float(np.sqrt(np.mean(ferr**2))),
            "floor_p95_px": float(np.percentile(ferr, 95)),
            "rim_center_observed_px": rim_centers[key].tolist(),
            "rim_center_predicted_px": rim_predicted[0].tolist(),
            "rim_center_error_px": rerr,
            "minimum_floor_depth_cm": float(np.min(depth)),
        }
    return {
        "x": result.x,
        "cost": float(result.cost),
        "camera_center_cm": center,
        "common_principal_point_px": common_pp,
        "blocks": blocks,
        "floor_rms_px": float(np.sqrt(np.mean(np.square(floor_errors)))),
        "floor_p95_px": float(np.percentile(floor_errors, 95)),
        "max_rim_center_error_px": max(rim_errors),
        "states": states,
        "nfev": int(result.nfev),
        "optimality": float(result.optimality),
    }


def rim_circle_points():
    theta = np.linspace(0.0, 2.0 * math.pi, 721, endpoint=False)
    return np.column_stack(
        [
            RIM_CENTER[0] + RIM_RADIUS * np.cos(theta),
            RIM_CENTER[1] + RIM_RADIUS * np.sin(theta),
            np.full(len(theta), RIM_CENTER[2]),
        ]
    )


def angle_distance(a: float, b: float) -> float:
    return min(abs(a - b), abs(a - b + 180.0), abs(a - b - 180.0))


def rim_shape_diagnostics(solution, keys, observed_ellipses):
    rows = {}
    circle = rim_circle_points()
    for index, key in enumerate(keys):
        block = solution["blocks"][index]
        pp = solution["common_principal_point_px"] + block[4:6]
        projected, _ = project_camera(
            solution["camera_center_cm"], pp, math.exp(float(block[0])), block[1:4], circle
        )
        ellipse = cv2.fitEllipse(projected.astype(np.float32))
        center = np.asarray(ellipse[0], dtype=np.float64)
        axes = np.sort(np.asarray(ellipse[1], dtype=np.float64))[::-1]
        observed = observed_ellipses[key]
        rows[key] = {
            "projected_center_px": center.tolist(),
            "projected_axes_major_minor_px": axes.tolist(),
            "projected_opencv_angle_deg": float(ellipse[2]),
            "observed_axes_major_minor_px": observed["axes"].tolist(),
            "observed_opencv_angle_deg": observed["angle"],
            "major_axis_error_px": float(abs(axes[0] - observed["axes"][0])),
            "minor_axis_error_px": float(abs(axes[1] - observed["axes"][1])),
            "ellipse_angle_error_deg": angle_distance(float(ellipse[2]), observed["angle"]),
        }
    return rows


def target_projection(solution, keys, points):
    index = keys.index("target")
    block = solution["blocks"][index]
    pp = solution["common_principal_point_px"] + block[4:6]
    return project_camera(
        solution["camera_center_cm"], pp, math.exp(float(block[0])), block[1:4], points
    )[0]


def draw_overlay(image, solution, keys, rim_points, out):
    result = image.copy()
    index = keys.index("target")
    block = solution["blocks"][index]
    pp = solution["common_principal_point_px"] + block[4:6]
    circle, _ = project_camera(
        solution["camera_center_cm"],
        pp,
        math.exp(float(block[0])),
        block[1:4],
        rim_circle_points(),
    )
    for point in np.round(circle).astype(int):
        if 0 <= point[0] < W and 0 <= point[1] < H:
            cv2.circle(result, tuple(point), 1, (255, 0, 255), -1, cv2.LINE_AA)
    for point in rim_points.astype(int):
        cv2.circle(result, tuple(point), 1, (0, 255, 255), -1, cv2.LINE_AA)
    observed = solution["states"]["target"]["rim_center_observed_px"]
    predicted = solution["states"]["target"]["rim_center_predicted_px"]
    cv2.circle(result, tuple(np.round(observed).astype(int)), 6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(result, tuple(np.round(predicted).astype(int)), 4, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), result)


def serializable_solution(solution):
    return {
        "cost": solution["cost"],
        "camera_center_cm": solution["camera_center_cm"].tolist(),
        "common_principal_point_px": solution["common_principal_point_px"].tolist(),
        "floor_rms_px": solution["floor_rms_px"],
        "floor_p95_px": solution["floor_p95_px"],
        "max_rim_center_error_px": solution["max_rim_center_error_px"],
        "nfev": solution["nfev"],
        "optimality": solution["optimality"],
        "states": solution["states"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--floor-proof", type=Path, required=True)
    ap.add_argument("--family-proof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturbation-trials", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    floor = json.loads(args.floor_proof.read_text(encoding="utf-8"))
    family = json.loads(args.family_proof.read_text(encoding="utf-8"))
    if floor.get("status") != "PASS_IN_ARENA_FLOOR_V52":
        raise RuntimeError("v52 floor proof is not sealed PASS")
    if family.get("status") != "PASS_IN_ARENA_STATIC_SCENE_FAMILY_V53B":
        raise RuntimeError("v53b family proof is not sealed PASS")
    target = cv2.imread(str(args.target))
    if target is None or target.shape[:2] != (H, W):
        raise RuntimeError("target must be native 960x540")
    if sha256(args.target) != family["target_sha256_png"]:
        raise RuntimeError("v54 target hash differs from sealed v53b target")

    target_h = norm_h(np.asarray(floor["floor_homography_world_to_image"], float))
    keys = ["target"]
    homographies = {"target": target_h}
    transfers = {}
    correspondences = {}
    transfer_diagnostics = {}
    images = {"target": target}
    source_files = {"target": args.target.name}

    for row in family["selected_candidates"]:
        event = int(row["event_probe"])
        key = f"event_{event}"
        path = args.states / row["file"]
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (H, W):
            raise RuntimeError(f"missing native selected state {path}")
        sealed_transfer = norm_h(np.asarray(row["H_target_to_state"], float))
        refined, p, q, diagnostic = refine_transfer(target, image, sealed_transfer)
        keys.append(key)
        transfers[key] = refined
        correspondences[key] = (p, q)
        transfer_diagnostics[key] = diagnostic
        homographies[key] = norm_h(refined @ target_h)
        images[key] = image
        source_files[key] = row["file"]

    rim_centers = {}
    observed_ellipses = {}
    rim_pixels = {}
    rim_diagnostics = {}
    for key in keys:
        predicted = TARGET_RIM_SEED if key == "target" else project_point(transfers[key], TARGET_RIM_SEED)
        center, axes, angle, points, diagnostic = extract_rim_ellipse(images[key], predicted)
        rim_centers[key] = center
        observed_ellipses[key] = {"center": center, "axes": axes, "angle": angle}
        rim_pixels[key] = points
        rim_diagnostics[key] = diagnostic

    floor_data = {key: sparse_floor_observations(homographies[key]) for key in keys}
    roots = []
    for seed in (
        (480.0, 270.0),
        (300.0, 0.0),
        (600.0, 0.0),
        (300.0, 500.0),
        (600.0, 500.0),
        (0.0, 0.0),
        (900.0, 540.0),
    ):
        solution = solve(
            keys,
            homographies,
            floor_data,
            rim_centers,
            seed_pp=seed,
        )
        if solution is not None:
            roots.append(solution)
    if not roots:
        raise RuntimeError("no v54 roots")
    roots.sort(key=lambda item: item["cost"])
    competitive = [item for item in roots if item["cost"] <= roots[0]["cost"] * 1.05 + 1e-8]
    best = competitive[0]

    volume = action_volume()
    root_pairs = []
    center_spreads = []
    for left in range(len(competitive)):
        for right in range(left + 1, len(competitive)):
            center_spreads.append(
                float(
                    np.linalg.norm(
                        competitive[left]["camera_center_cm"]
                        - competitive[right]["camera_center_cm"]
                    )
                )
            )
            difference = np.linalg.norm(
                target_projection(competitive[left], keys, volume)
                - target_projection(competitive[right], keys, volume),
                axis=1,
            )
            root_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "p95_px": float(np.percentile(difference, 95)),
                    "max_px": float(np.max(difference)),
                }
            )

    leave_one_out = {}
    for dropped in keys[1:]:
        subset = [key for key in keys if key != dropped]
        indices = [keys.index(key) for key in subset]
        warm = np.r_[
            best["camera_center_cm"],
            best["common_principal_point_px"],
            best["blocks"][indices].ravel(),
        ]
        solution = solve(
            subset,
            homographies,
            floor_data,
            rim_centers,
            seed_pp=best["common_principal_point_px"],
            warm=warm,
            max_nfev=8000,
        )
        leave_one_out[dropped] = {
            "camera_center_shift_cm": float(
                np.linalg.norm(solution["camera_center_cm"] - best["camera_center_cm"])
            ),
            "floor_p95_px": solution["floor_p95_px"],
            "max_rim_center_error_px": solution["max_rim_center_error_px"],
        }

    rng = np.random.default_rng(540903)
    perturbations = []
    training_segments = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in floor["training_line_segments_px"].items()
    }
    for trial in range(args.perturbation_trials):
        perturbed_segments = {
            key: value + rng.uniform(-0.5, 0.5, value.shape)
            for key, value in training_segments.items()
        }
        perturbed_target_h, _ = homography_from_segments(perturbed_segments)
        perturbed_h = {"target": norm_h(perturbed_target_h)}
        for key in keys[1:]:
            p, q = correspondences[key]
            pj = p + rng.uniform(-0.5, 0.5, p.shape)
            qj = q + rng.uniform(-0.5, 0.5, q.shape)
            transfer, _ = cv2.findHomography(pj.astype(np.float32), qj.astype(np.float32), 0)
            if transfer is None:
                raise RuntimeError("perturbed v54 transfer failed")
            perturbed_h[key] = norm_h(transfer @ perturbed_h["target"])
        perturbed_floor = {
            key: sparse_floor_observations(perturbed_h[key]) for key in keys
        }
        perturbed_rim = {
            key: value + rng.uniform(-0.5, 0.5, 2) for key, value in rim_centers.items()
        }
        solution = solve(
            keys,
            perturbed_h,
            perturbed_floor,
            perturbed_rim,
            seed_pp=best["common_principal_point_px"],
            warm=best["x"],
            max_nfev=6000,
        )
        shift = float(np.linalg.norm(solution["camera_center_cm"] - best["camera_center_cm"]))
        target_difference = np.linalg.norm(
            target_projection(solution, keys, volume) - target_projection(best, keys, volume),
            axis=1,
        )
        perturbations.append(
            {
                "trial": trial,
                "camera_center_shift_cm": shift,
                "target_action_volume_p95_shift_px": float(np.percentile(target_difference, 95)),
                "target_action_volume_max_shift_px": float(np.max(target_difference)),
            }
        )

    rim_shapes = rim_shape_diagnostics(best, keys, observed_ellipses)
    camera_center = best["camera_center_cm"]
    state_pp = [np.asarray(value["principal_point_px"]) for value in best["states"].values()]
    state_pp_spread = max(
        float(np.linalg.norm(state_pp[left] - state_pp[right]))
        for left in range(len(state_pp))
        for right in range(left + 1, len(state_pp))
    )
    gates = {
        "competitive_roots_functionally_equivalent": max(
            (row["p95_px"] for row in root_pairs), default=0.0
        ) <= 0.5,
        "competitive_centres_within_1cm": max(center_spreads, default=0.0) <= 1.0,
        "floor_p95_at_most_2px": best["floor_p95_px"] <= 2.0,
        "all_rim_centres_at_most_2px": best["max_rim_center_error_px"] <= 2.0,
        "state_principal_point_spread_at_most_80px": state_pp_spread <= 80.0,
        "no_state_principal_delta_near_bound": all(
            np.linalg.norm(value["principal_point_delta_px"], ord=np.inf) <= 55.0
            for value in best["states"].values()
        ),
        "camera_height_5m_to_25m": 500.0 <= camera_center[2] <= 2500.0,
        "camera_outside_playing_width": abs(camera_center[1]) >= 25.0 * FT,
        "camera_distance_under_60m": float(np.linalg.norm(camera_center)) <= 6000.0,
        "leave_one_state_center_shift_at_most_25cm": max(
            row["camera_center_shift_cm"] for row in leave_one_out.values()
        ) <= 25.0,
        "half_pixel_center_shift_at_most_15cm": max(
            row["camera_center_shift_cm"] for row in perturbations
        ) <= 15.0,
        "half_pixel_action_volume_p95_at_most_1px": max(
            row["target_action_volume_p95_shift_px"] for row in perturbations
        ) <= 1.0,
        "rim_major_axis_holdout_at_most_7px": max(
            row["major_axis_error_px"] for row in rim_shapes.values()
        ) <= 7.0,
        "rim_minor_axis_holdout_at_most_5px": max(
            row["minor_axis_error_px"] for row in rim_shapes.values()
        ) <= 5.0,
    }
    passed = bool(all(gates.values()))
    report = {
        "schema_version": 1,
        "status": "PASS_IN_ARENA_NONCOPLANAR_CENTER_V54" if passed else "FAIL_IN_ARENA_NONCOPLANAR_CENTER_V54",
        "version": "v54",
        "game_id": "0022500301",
        "target_event_id": 489,
        "camera_family": "In Arena static-scene cluster v53b",
        "method": "sealed v52 metric floor + independently source-pixel-extracted regulation rim centres in six v53b-compatible optical states; shared centre with state-specific pan/tilt/focal and bounded crop variation",
        "guardrail": "A pass authorizes only an In-Arena physical camera-centre prior. Target event-camera intrinsics/extrinsics require a separate fixed-centre proof; replay rendering remains forbidden.",
        "source_files": source_files,
        "transfer_diagnostics": transfer_diagnostics,
        "rim_source_pixel_diagnostics": rim_diagnostics,
        "best_solution": serializable_solution(best),
        "rim_shape_holdout_diagnostics": rim_shapes,
        "multistart": {
            "root_count": len(roots),
            "competitive_root_count": len(competitive),
            "competitive_center_max_pairwise_cm": max(center_spreads, default=0.0),
            "target_action_volume_max_pairwise_p95_px": max(
                (row["p95_px"] for row in root_pairs), default=0.0
            ),
            "target_action_volume_max_pairwise_max_px": max(
                (row["max_px"] for row in root_pairs), default=0.0
            ),
            "pairs": root_pairs,
            "roots": [serializable_solution(root) for root in roots],
        },
        "state_principal_point_max_pairwise_px": state_pp_spread,
        "leave_one_state_out": leave_one_out,
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
            "max_target_action_volume_p95_shift_px": max(
                row["target_action_volume_p95_shift_px"] for row in perturbations
            ),
            "trials": perturbations,
        },
        "gates": gates,
        "permissions": {
            "physical_camera_center_allowed": passed,
            "metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "next_gate": (
            "Hold this physical centre fixed and solve the immutable event-489 optical state against v52 floor plus direct rim/board geometry; require held-out basket, multistart and half-pixel 3D-volume stability."
            if passed
            else "Reject this v54 formulation or the incompatible states; do not promote In Arena and do not render replay."
        ),
    }
    report = json_safe(report)
    (args.out / "in_arena_noncoplanar_center_v54.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    draw_overlay(
        target,
        best,
        keys,
        rim_pixels["target"],
        args.out / "in_arena_target_rim_overlay_v54.png",
    )
    print(
        json.dumps(
            json_safe({
                "status": report["status"],
                "camera_center_cm": report["best_solution"]["camera_center_cm"],
                "floor_p95_px": best["floor_p95_px"],
                "max_rim_center_error_px": best["max_rim_center_error_px"],
                "state_principal_point_max_pairwise_px": state_pp_spread,
                "max_leave_one_state_center_shift_cm": max(
                    row["camera_center_shift_cm"] for row in leave_one_out.values()
                ),
                "max_half_pixel_center_shift_cm": report["half_pixel_perturbation"][
                    "max_camera_center_shift_cm"
                ],
                "max_half_pixel_action_volume_p95_px": report[
                    "half_pixel_perturbation"
                ]["max_target_action_volume_p95_shift_px"],
                "gates": gates,
            }),
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
