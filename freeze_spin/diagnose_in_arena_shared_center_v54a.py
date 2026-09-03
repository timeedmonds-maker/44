from __future__ import annotations

"""v54a diagnostic: can the v53b In-Arena family determine one camera centre?

This diagnostic composes the sealed v52 metric floor homography with each
independently validated v53b target-to-state static-scene homography.  It then
fits one shared physical camera centre while allowing every optical state its
own rotation, focal length and a bounded principal-point/crop offset.

The output is deliberately non-promoting.  Its purpose is to expose any
remaining 3D ambiguity before direct rim/backboard observations enter v54.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


W, H = 960, 540
FT = 30.48


def norm_h(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return h / h[2, 2]


def project_h(h: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ph = np.column_stack([points_xy, np.ones(len(points_xy))])
    q = (h @ ph.T).T
    uv = q[:, :2] / q[:, 2:3]
    return uv, q[:, 2]


def floor_observations(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(
        [
            [x, y]
            for x in np.linspace(-4.0 * FT, 34.0 * FT, 18)
            for y in np.linspace(-25.0 * FT, 25.0 * FT, 21)
        ],
        dtype=np.float64,
    )
    uv, depth = project_h(h, xy)
    keep = (
        (depth > 0)
        & (uv[:, 0] > 16)
        & (uv[:, 0] < W - 16)
        & (uv[:, 1] > 16)
        & (uv[:, 1] < H - 16)
    )
    xyz = np.column_stack([xy[keep], np.zeros(int(keep.sum()))])
    return xyz, uv[keep]


def k_matrix(focal: float, pp: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[focal, 0.0, pp[0]], [0.0, focal, pp[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def decompose_floor_homography(h: np.ndarray, pp: np.ndarray):
    h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]
    cx, cy = pp
    a1 = np.asarray([h1[0] - cx * h1[2], h1[1] - cy * h1[2]])
    a2 = np.asarray([h2[0] - cx * h2[2], h2[1] - cy * h2[2]])
    focal_sq = []
    if abs(float(h1[2] * h2[2])) > 1e-12:
        q = -float(a1 @ a2) / float(h1[2] * h2[2])
        if q > 0:
            focal_sq.append(q)
    denominator = float(h1[2] ** 2 - h2[2] ** 2)
    if abs(denominator) > 1e-12:
        q = -float(a1 @ a1 - a2 @ a2) / denominator
        if q > 0:
            focal_sq.append(q)
    if not focal_sq:
        return None
    focal = math.sqrt(float(np.median(focal_sq)))
    ki = np.linalg.inv(k_matrix(focal, pp))
    q1, q2, q3 = ki @ h1, ki @ h2, ki @ h3
    scale = 2.0 / (np.linalg.norm(q1) + np.linalg.norm(q2))
    r0 = np.column_stack([scale * q1, scale * q2, np.cross(scale * q1, scale * q2)])
    u, _, vt = np.linalg.svd(r0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    center = -rotation.T @ (scale * q3)
    rvec, _ = cv2.Rodrigues(rotation)
    return focal, center, rvec.ravel()


def project_camera(
    center: np.ndarray,
    pp: np.ndarray,
    focal: float,
    rvec: np.ndarray,
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera = (rotation @ (xyz - center).T).T
    uv = np.column_stack(
        [
            focal * camera[:, 0] / camera[:, 2] + pp[0],
            focal * camera[:, 1] / camera[:, 2] + pp[1],
        ]
    )
    return uv, camera[:, 2]


def unpack(x: np.ndarray, count: int):
    return x[:3], x[3:5], x[5:].reshape(count, 6)


def make_start(keys: list[str], homographies: dict[str, np.ndarray], seed_pp):
    centers = []
    blocks = []
    pp = np.asarray(seed_pp, dtype=np.float64)
    for key in keys:
        decomposed = decompose_floor_homography(homographies[key], pp)
        if decomposed is None:
            return None
        focal, center, rvec = decomposed
        centers.append(center)
        blocks.append([math.log(focal), *rvec, 0.0, 0.0])
    return np.r_[np.median(np.asarray(centers), axis=0), pp, np.asarray(blocks).ravel()]


def solve(
    keys: list[str],
    homographies: dict[str, np.ndarray],
    observations: dict[str, tuple[np.ndarray, np.ndarray]],
    seed_pp,
    principal_delta_sigma: float,
):
    start = make_start(keys, homographies, seed_pp)
    if start is None:
        return None
    count = len(keys)
    lower = np.r_[
        [-10000.0, -10000.0, 50.0],
        [-500.0, -500.0],
        np.tile([math.log(180.0), -10.0, -10.0, -10.0, -150.0, -150.0], count),
    ]
    upper = np.r_[
        [10000.0, 10000.0, 5000.0],
        [1460.0, 1040.0],
        np.tile([math.log(12000.0), 10.0, 10.0, 10.0, 150.0, 150.0], count),
    ]
    start = np.clip(start, lower + 1e-6, upper - 1e-6)

    def residual(x):
        center, common_pp, blocks = unpack(x, count)
        values = []
        for index, key in enumerate(keys):
            focal = math.exp(float(blocks[index, 0]))
            rvec = blocks[index, 1:4]
            delta_pp = blocks[index, 4:6]
            xyz, observed = observations[key]
            predicted, depth = project_camera(center, common_pp + delta_pp, focal, rvec, xyz)
            values.append((predicted - observed).ravel())
            values.append(np.minimum(depth - 10.0, 0.0) * 10.0)
            values.append(delta_pp / principal_delta_sigma)
        values.append((common_pp - np.asarray([W / 2.0, H / 2.0])) / 300.0)
        return np.concatenate(values)

    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=12000,
    )
    center, common_pp, blocks = unpack(result.x, count)
    states = {}
    all_pixel_errors = []
    for index, key in enumerate(keys):
        focal = math.exp(float(blocks[index, 0]))
        rvec = blocks[index, 1:4]
        state_pp = common_pp + blocks[index, 4:6]
        xyz, observed = observations[key]
        predicted, depth = project_camera(center, state_pp, focal, rvec, xyz)
        error = np.linalg.norm(predicted - observed, axis=1)
        all_pixel_errors.extend(error.tolist())
        states[key] = {
            "focal_px": focal,
            "principal_point_px": state_pp.tolist(),
            "principal_point_delta_px": blocks[index, 4:6].tolist(),
            "rvec": rvec.tolist(),
            "floor_point_count": int(len(error)),
            "floor_rms_px": float(np.sqrt(np.mean(error**2))),
            "floor_p95_px": float(np.percentile(error, 95)),
            "minimum_camera_depth_cm": float(np.min(depth)),
        }
    return {
        "x": result.x,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "status": int(result.status),
        "nfev": int(result.nfev),
        "camera_center_cm": center,
        "common_principal_point_px": common_pp,
        "blocks": blocks,
        "states": states,
        "floor_rms_px": float(np.sqrt(np.mean(np.square(all_pixel_errors)))),
        "floor_p95_px": float(np.percentile(all_pixel_errors, 95)),
    }


def action_volume() -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in np.linspace(-30.0, 250.0, 8)
            for y in np.linspace(-180.0, 180.0, 9)
            for z in np.linspace(20.0, 350.0, 8)
        ],
        dtype=np.float64,
    )


def target_projection(solution: dict, keys: list[str], xyz: np.ndarray) -> np.ndarray:
    index = keys.index("target")
    block = solution["blocks"][index]
    pp = solution["common_principal_point_px"] + block[4:6]
    uv, _ = project_camera(
        solution["camera_center_cm"], pp, math.exp(float(block[0])), block[1:4], xyz
    )
    return uv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-proof", type=Path, required=True)
    ap.add_argument("--family-proof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--principal-delta-sigma-px", type=float, default=35.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    floor = json.loads(args.floor_proof.read_text(encoding="utf-8"))
    family = json.loads(args.family_proof.read_text(encoding="utf-8"))
    if floor.get("status") != "PASS_IN_ARENA_FLOOR_V52":
        raise RuntimeError("sealed v52 floor proof is not accepted")
    if family.get("status") != "PASS_IN_ARENA_STATIC_SCENE_FAMILY_V53B":
        raise RuntimeError("sealed v53b static-scene family is not accepted")

    target_h = norm_h(np.asarray(floor["floor_homography_world_to_image"], dtype=np.float64))
    homographies = {"target": target_h}
    for row in family["selected_candidates"]:
        key = f"event_{int(row['event_probe'])}"
        transfer = norm_h(np.asarray(row["H_target_to_state"], dtype=np.float64))
        homographies[key] = norm_h(transfer @ target_h)
    keys = list(homographies)
    observations = {key: floor_observations(value) for key, value in homographies.items()}

    roots = []
    for seed_pp in (
        (480.0, 270.0),
        (420.0, 180.0),
        (540.0, 180.0),
        (420.0, 390.0),
        (540.0, 390.0),
        (300.0, 270.0),
        (660.0, 270.0),
    ):
        solution = solve(
            keys, homographies, observations, seed_pp, args.principal_delta_sigma_px
        )
        if solution is not None:
            roots.append(solution)
    if not roots:
        raise RuntimeError("no valid shared-centre roots")
    roots.sort(key=lambda item: (item["cost"], item["floor_rms_px"]))
    competitive = [root for root in roots if root["cost"] <= roots[0]["cost"] * 1.05 + 1e-6]

    volume = action_volume()
    root_projection_pairs = []
    for left in range(len(competitive)):
        for right in range(left + 1, len(competitive)):
            a = target_projection(competitive[left], keys, volume)
            b = target_projection(competitive[right], keys, volume)
            distance = np.linalg.norm(a - b, axis=1)
            root_projection_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "p95_px": float(np.percentile(distance, 95)),
                    "max_px": float(np.max(distance)),
                }
            )
    center_pairs = []
    for left in range(len(competitive)):
        for right in range(left + 1, len(competitive)):
            center_pairs.append(
                float(
                    np.linalg.norm(
                        competitive[left]["camera_center_cm"]
                        - competitive[right]["camera_center_cm"]
                    )
                )
            )

    def serializable(root: dict):
        return {
            "cost": root["cost"],
            "optimality": root["optimality"],
            "optimizer_status": root["status"],
            "nfev": root["nfev"],
            "camera_center_cm": root["camera_center_cm"].tolist(),
            "common_principal_point_px": root["common_principal_point_px"].tolist(),
            "floor_rms_px": root["floor_rms_px"],
            "floor_p95_px": root["floor_p95_px"],
            "states": root["states"],
        }

    payload = {
        "status": "DIAGNOSTIC_IN_ARENA_SHARED_CENTER_V54A",
        "version": "v54a",
        "game_id": "0022500301",
        "target_event_id": 489,
        "method": "sealed v52 metric floor composed through independently validated v53b static-scene homographies; one shared physical centre; state-specific rotation/focal/principal-point crop offset",
        "guardrail": "Diagnostic only. No rim or backboard observation is fitted, no metric camera is promoted, and replay remains forbidden.",
        "principal_delta_sigma_px": args.principal_delta_sigma_px,
        "state_keys": keys,
        "root_count": len(roots),
        "competitive_root_count": len(competitive),
        "roots": [serializable(root) for root in roots],
        "competitive_center_max_pairwise_cm": max(center_pairs) if center_pairs else 0.0,
        "competitive_target_action_volume_projection": {
            "max_pairwise_p95_px": max((row["p95_px"] for row in root_projection_pairs), default=0.0),
            "max_pairwise_max_px": max((row["max_px"] for row in root_projection_pairs), default=0.0),
            "pairs": root_projection_pairs,
        },
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
        "next_gate": "Add direct, source-pixel rim/backboard observations in target and selected states; require held-out non-coplanar agreement, leave-one-state-out stability and half-pixel perturbation stability without relaxing the v52/v53b gates.",
    }
    (args.out / "in_arena_shared_center_v54a.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "root_count": payload["root_count"],
                "competitive_root_count": payload["competitive_root_count"],
                "best_center_cm": payload["roots"][0]["camera_center_cm"],
                "best_floor_rms_px": payload["roots"][0]["floor_rms_px"],
                "best_floor_p95_px": payload["roots"][0]["floor_p95_px"],
                "competitive_center_max_pairwise_cm": payload[
                    "competitive_center_max_pairwise_cm"
                ],
                "competitive_action_volume_max_p95_px": payload[
                    "competitive_target_action_volume_projection"
                ]["max_pairwise_p95_px"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
