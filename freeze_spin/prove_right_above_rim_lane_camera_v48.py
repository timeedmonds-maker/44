from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.solve_right_above_rim_fixed_geometry_v1 import (
    FOOT_CM,
    IMAGE_H,
    IMAGE_W,
    RIM_RADIUS_CM,
    RIM_X_CM,
    RIM_Z_CM,
    RESTRICTED_RADIUS_CM,
    circle,
    look_at_rvec,
    nearest_curve_distances,
    project,
)

LANE_HALF_CM = 8.0 * FOOT_CM
BASELINE_X_CM = -4.0 * FOOT_CM
FT_X_CM = 15.0 * FOOT_CM


def lane_sideline(y_cm: float, samples: int = 360) -> np.ndarray:
    x = np.linspace(BASELINE_X_CM, FT_X_CM, samples)
    return np.column_stack([x, np.full_like(x, y_cm), np.zeros_like(x)]).astype(np.float64)


def geometry_curves(samples: int = 360) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        circle(RIM_X_CM, RIM_Z_CM, RIM_RADIUS_CM, samples),
        circle(RIM_X_CM, 0.0, RESTRICTED_RADIUS_CM, samples),
        lane_sideline(-LANE_HALF_CM, samples),
        lane_sideline(+LANE_HALF_CM, samples),
    )


def residual_for(
    p: np.ndarray,
    rim_obs: np.ndarray,
    restricted_obs: np.ndarray,
    left_lane_obs: np.ndarray,
    right_lane_obs: np.ndarray,
) -> np.ndarray:
    rim3, restricted3, left3, right3 = geometry_curves(360)
    rim_uv, rim_cam, _ = project(p, rim3)
    restricted_uv, restricted_cam, _ = project(p, restricted3)
    left_uv, left_cam, _ = project(p, left3)
    right_uv, right_cam, _ = project(p, right3)

    rim_d = nearest_curve_distances(rim_obs, rim_uv)
    restricted_d = nearest_curve_distances(restricted_obs, restricted_uv)
    left_d = nearest_curve_distances(left_lane_obs, left_uv)
    right_d = nearest_curve_distances(right_lane_obs, right_uv)

    depth_min = min(
        float(np.min(rim_cam[:, 2])),
        float(np.min(restricted_cam[:, 2])),
        float(np.min(left_cam[:, 2])),
        float(np.min(right_cam[:, 2])),
    )
    depth_penalty = max(0.0, 20.0 - depth_min) / 2.0
    priors = np.asarray([
        (p[7] - IMAGE_W / 2.0) / 100.0,
        (p[8] - IMAGE_H / 2.0) / 100.0,
        (p[6] - math.log(1000.0)) / 1.5,
        depth_penalty,
    ])
    return np.concatenate([rim_d, restricted_d / 1.5, left_d, right_d, priors])


def bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.r_[[-np.inf] * 3, [-1000.0, -1000.0, 250.0], math.log(150.0), 300.0, 100.0]
    upper = np.r_[[np.inf] * 3, [1000.0, 1000.0, 3000.0], math.log(4000.0), 660.0, 440.0]
    return lower, upper


def optimize(
    p0: np.ndarray,
    rim_obs: np.ndarray,
    restricted_obs: np.ndarray,
    left_lane_obs: np.ndarray,
    right_lane_obs: np.ndarray,
    max_nfev: int = 2200,
) -> tuple[np.ndarray, float]:
    lower, upper = bounds()
    fit = least_squares(
        lambda p: residual_for(p, rim_obs, restricted_obs, left_lane_obs, right_lane_obs),
        p0,
        bounds=(lower, upper),
        loss='soft_l1',
        f_scale=2.0,
        x_scale='jac',
        max_nfev=max_nfev,
    )
    return fit.x, float(fit.cost)


def summarize(
    p: np.ndarray,
    rim_obs: np.ndarray,
    restricted_obs: np.ndarray,
    left_lane_obs: np.ndarray,
    right_lane_obs: np.ndarray,
) -> dict:
    rim3, restricted3, left3, right3 = geometry_curves(720)
    rim_uv, rim_cam, R = project(p, rim3)
    restricted_uv, restricted_cam, _ = project(p, restricted3)
    left_uv, left_cam, _ = project(p, left3)
    right_uv, right_cam, _ = project(p, right3)
    rim_d = nearest_curve_distances(rim_obs, rim_uv)
    restricted_d = nearest_curve_distances(restricted_obs, restricted_uv)
    left_d = nearest_curve_distances(left_lane_obs, left_uv)
    right_d = nearest_curve_distances(right_lane_obs, right_uv)
    all_d = np.r_[rim_d, restricted_d, left_d, right_d]
    plausible = (
        min(
            float(np.min(rim_cam[:, 2])),
            float(np.min(restricted_cam[:, 2])),
            float(np.min(left_cam[:, 2])),
            float(np.min(right_cam[:, 2])),
        ) > 20.0
        and 150.0 <= math.exp(float(p[6])) <= 4000.0
        and -1000.0 <= float(p[3]) <= 1000.0
        and -1000.0 <= float(p[4]) <= 1000.0
        and 250.0 <= float(p[5]) <= 3000.0
    )
    return {
        'plausible': bool(plausible),
        'combined_anchor_rms_px': float(np.sqrt(np.mean(all_d ** 2))),
        'combined_anchor_p95_px': float(np.percentile(all_d, 95)),
        'rim_curve_rms_px': float(np.sqrt(np.mean(rim_d ** 2))),
        'restricted_curve_rms_px': float(np.sqrt(np.mean(restricted_d ** 2))),
        'left_lane_rms_px': float(np.sqrt(np.mean(left_d ** 2))),
        'right_lane_rms_px': float(np.sqrt(np.mean(right_d ** 2))),
        'camera_center_world_cm': [float(v) for v in p[3:6]],
        'focal_px': float(math.exp(float(p[6]))),
        'principal_point_px': [float(p[7]), float(p[8])],
        'R_world_to_camera': R.tolist(),
    }


def deterministic_starts() -> list[np.ndarray]:
    rows = [
        (20.0, -50.0, 550.0, 1100.0),
        (20.0, 50.0, 550.0, 1100.0),
        (0.0, 0.0, 650.0, 1300.0),
        (-120.0, -120.0, 700.0, 900.0),
        (120.0, 120.0, 700.0, 900.0),
        (-80.0, 100.0, 850.0, 1500.0),
        (80.0, -100.0, 850.0, 1500.0),
    ]
    out = []
    for cx, cy, cz, focal in rows:
        center = np.asarray([cx, cy, cz], dtype=np.float64)
        out.append(np.r_[
            look_at_rvec(center, np.asarray([RIM_X_CM, 0.0, 150.0])),
            center,
            math.log(focal),
            IMAGE_W / 2.0,
            IMAGE_H / 2.0,
        ])
    return out


def action_volume() -> np.ndarray:
    pts = []
    for x in (-90.0, 0.0, 75.0, 150.0, 225.0, 300.0):
        for y in (-220.0, -110.0, 0.0, 110.0, 220.0):
            for z in (0.0, 100.0, 200.0, 304.8, 380.0):
                pts.append([x, y, z])
    return np.asarray(pts, dtype=np.float64)


def functional_p95(a: np.ndarray, b: np.ndarray, volume: np.ndarray) -> float:
    ua, ca, _ = project(a, volume)
    ub, cb, _ = project(b, volume)
    valid = (ca[:, 2] > 20.0) & (cb[:, 2] > 20.0)
    if int(np.sum(valid)) < 20:
        return float('inf')
    d = np.linalg.norm(ua[valid] - ub[valid], axis=1)
    return float(np.percentile(d, 95))


def draw_overlay(image: np.ndarray, p: np.ndarray, spec: dict, out: Path) -> None:
    rim3, restricted3, left3, right3 = geometry_curves(720)
    rim_uv = project(p, rim3)[0]
    restricted_uv = project(p, restricted3)[0]
    left_uv = project(p, left3)[0]
    right_uv = project(p, right3)[0]
    overlay = image.copy()
    cv2.polylines(overlay, [np.round(rim_uv).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(overlay, [np.round(restricted_uv).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(overlay, [np.round(left_uv).astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(overlay, [np.round(right_uv).astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
    for key, colour in (
        ('rim_inner_edge_samples_px', (0, 0, 255)),
        ('restricted_area_centerline_samples_px', (255, 0, 255)),
        ('left_lane_sideline_samples_px', (255, 0, 0)),
        ('right_lane_sideline_samples_px', (255, 0, 0)),
    ):
        for q in np.asarray(spec[key], dtype=int):
            cv2.circle(overlay, tuple(q), 3, colour, -1, cv2.LINE_AA)
    cv2.imwrite(str(out), overlay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--observations', type=Path, required=True)
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.observations.read_text(encoding='utf-8'))
    registry = json.loads(args.registry.read_text(encoding='utf-8'))
    image = cv2.imread(str(args.image))
    if image is None or image.shape[:2] != (IMAGE_H, IMAGE_W):
        raise RuntimeError('Expected immutable native 960x540 Right Above Rim Frame C')

    rim_obs = np.asarray(spec['rim_inner_edge_samples_px'], dtype=np.float64)
    restricted_obs = np.asarray(spec['restricted_area_centerline_samples_px'], dtype=np.float64)
    left_lane_obs = np.asarray(spec['left_lane_sideline_samples_px'], dtype=np.float64)
    right_lane_obs = np.asarray(spec['right_lane_sideline_samples_px'], dtype=np.float64)
    volume = action_volume()
    gate_cfg = spec['gates']

    def root_job(item: tuple[int, np.ndarray]) -> dict:
        i, p0 = item
        p, cost = optimize(p0, rim_obs, restricted_obs, left_lane_obs, right_lane_obs)
        qa = summarize(p, rim_obs, restricted_obs, left_lane_obs, right_lane_obs)
        return {'index': i, 'p': p, 'cost': cost, 'qa': qa}

    starts = list(enumerate(deterministic_starts()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        roots = list(pool.map(root_job, starts))
    roots.sort(key=lambda r: (not r['qa']['plausible'], r['qa']['combined_anchor_rms_px'], r['cost']))
    base = roots[0]
    base_p = base['p']
    base_qa = base['qa']

    competitive = [
        r for r in roots
        if r['qa']['plausible']
        and r['qa']['combined_anchor_rms_px'] <= base_qa['combined_anchor_rms_px'] + 0.35
        and r['qa']['combined_anchor_p95_px'] <= float(gate_cfg['max_combined_anchor_p95_px'])
    ]
    root_pairwise = []
    max_root_shift = 0.0
    for i in range(len(competitive)):
        for j in range(i + 1, len(competitive)):
            s = functional_p95(competitive[i]['p'], competitive[j]['p'], volume)
            max_root_shift = max(max_root_shift, s)
            root_pairwise.append({'a': competitive[i]['index'], 'b': competitive[j]['index'], 'action_volume_p95_shift_px': s})

    def support_job(k: int) -> float:
        rkeep = np.asarray([i % 4 != k for i in range(len(rim_obs))])
        ckeep = np.asarray([i % 4 != k for i in range(len(restricted_obs))])
        lkeep = np.asarray([i % 4 != k for i in range(len(left_lane_obs))])
        qkeep = np.asarray([i % 4 != k for i in range(len(right_lane_obs))])
        p, _ = optimize(
            base_p,
            rim_obs[rkeep],
            restricted_obs[ckeep],
            left_lane_obs[lkeep],
            right_lane_obs[qkeep],
            max_nfev=1800,
        )
        return functional_p95(base_p, p, volume)

    with ThreadPoolExecutor(max_workers=4) as pool:
        support_shifts = list(pool.map(support_job, range(4)))
    max_support_shift = max(support_shifts)

    rng = np.random.default_rng(20260903)
    perturb_trials = 32
    perturb_inputs = []
    for i in range(perturb_trials):
        perturb_inputs.append((
            i,
            rim_obs + rng.uniform(-0.5, 0.5, size=rim_obs.shape),
            restricted_obs + rng.uniform(-0.5, 0.5, size=restricted_obs.shape),
            left_lane_obs + rng.uniform(-0.5, 0.5, size=left_lane_obs.shape),
            right_lane_obs + rng.uniform(-0.5, 0.5, size=right_lane_obs.shape),
        ))

    base_center = base_p[3:6].copy()

    def perturb_job(item: tuple) -> tuple[int, float, float]:
        i, rr, cc, ll, qq = item
        p, _ = optimize(base_p, rr, cc, ll, qq, max_nfev=1600)
        return i, float(np.linalg.norm(p[3:6] - base_center)), functional_p95(base_p, p, volume)

    with ThreadPoolExecutor(max_workers=4) as pool:
        perturb_rows = list(pool.map(perturb_job, perturb_inputs))
    perturb_rows.sort(key=lambda x: x[0])
    perturb_center_shifts = [x[1] for x in perturb_rows]
    perturb_projection_shifts = [x[2] for x in perturb_rows]

    first_center = np.asarray(registry['cameras']['Left Above Rim']['physical_camera_center_prior_cm'], dtype=np.float64)
    second_center = np.asarray(base_qa['camera_center_world_cm'], dtype=np.float64)
    distinct_baseline_cm = float(np.linalg.norm(second_center - first_center))

    gates = {
        'native_immutable_frame_geometry': True,
        'player_ball_pose_evidence_excluded': True,
        'asymmetric_regulation_lane_anchors_present': True,
        'strict_anchor_fit': bool(
            base_qa['plausible']
            and base_qa['combined_anchor_rms_px'] <= float(gate_cfg['max_combined_anchor_rms_px'])
            and base_qa['combined_anchor_p95_px'] <= float(gate_cfg['max_combined_anchor_p95_px'])
        ),
        'at_least_three_competitive_multistart_roots': len(competitive) >= 3,
        'competitive_roots_functionally_equivalent': max_root_shift <= float(gate_cfg['max_competitive_root_action_volume_p95_shift_px']),
        'support_reduction_action_volume_p95_at_most_two_px': max_support_shift <= float(gate_cfg['max_support_reduction_action_volume_p95_shift_px']),
        'half_pixel_camera_center_shift_at_most_75cm': max(perturb_center_shifts) <= float(gate_cfg['max_half_pixel_camera_center_shift_cm']),
        'half_pixel_action_volume_p95_at_most_two_px': max(perturb_projection_shifts) <= float(gate_cfg['max_half_pixel_action_volume_p95_shift_px']),
        'genuinely_distinct_from_left_above_rim': distinct_baseline_cm >= float(gate_cfg['min_distinct_camera_baseline_cm']),
    }
    passed = all(gates.values())

    report = {
        'status': 'PASS_RIGHT_ABOVE_RIM_LANE_CAMERA_V48' if passed else 'FAIL_RIGHT_ABOVE_RIM_LANE_CAMERA_V48',
        'method': 'regulation rim + restricted-area circle + asymmetric 16-foot lane sidelines; multistart/support/half-pixel functional proof',
        'base_camera': base_qa,
        'competitive_root_count': len(competitive),
        'max_competitive_pairwise_action_volume_p95_shift_px': max_root_shift,
        'root_pairwise': root_pairwise,
        'support_reduction_action_volume_p95_shifts_px': support_shifts,
        'max_support_reduction_action_volume_p95_shift_px': max_support_shift,
        'perturbation_trials': perturb_trials,
        'max_half_pixel_camera_center_shift_cm': max(perturb_center_shifts),
        'p95_half_pixel_camera_center_shift_cm': float(np.percentile(perturb_center_shifts, 95)),
        'max_half_pixel_action_volume_p95_shift_px': max(perturb_projection_shifts),
        'p95_half_pixel_action_volume_p95_shift_px': float(np.percentile(perturb_projection_shifts, 95)),
        'left_above_rim_accepted_center_cm': [float(v) for v in first_center],
        'right_above_rim_candidate_center_cm': [float(v) for v in second_center],
        'distinct_camera_baseline_cm': distinct_baseline_cm,
        'gates': gates,
        'permissions': {
            'right_above_rim_metric_event_camera_allowed': bool(passed),
            'two_distinct_physical_cameras_validated': bool(passed),
            'static_novel_view_allowed': False,
            'replay_render_allowed': False,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'right_above_rim_lane_camera_v48.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    draw_overlay(image, base_p, spec, args.out / 'right_above_rim_lane_overlay_v48.png')
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
