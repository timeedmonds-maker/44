from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.prove_right_above_rim_functional_camera_v47 import (
    IMAGE_H,
    IMAGE_W,
    action_volume,
    deterministic_starts,
    draw_overlay,
    functional_p95,
    optimize,
    summarize,
)


def root_task(payload):
    idx, p0, rim_obs, restricted_obs = payload
    p, cost = optimize(p0, rim_obs, restricted_obs, max_nfev=2200)
    return idx, p, cost, summarize(p, rim_obs, restricted_obs)


def support_task(payload):
    base_p, rim_obs, restricted_obs, k, volume = payload
    rkeep = np.asarray([i % 4 != k for i in range(len(rim_obs))])
    ckeep = np.asarray([i % 4 != k for i in range(len(restricted_obs))])
    p, _ = optimize(base_p, rim_obs[rkeep], restricted_obs[ckeep], max_nfev=1800)
    return k, functional_p95(base_p, p, volume)


def perturb_task(payload):
    idx, base_p, base_center, rr, cc, volume = payload
    p, _ = optimize(base_p, rr, cc, max_nfev=1600)
    center_shift = float(np.linalg.norm(p[3:6] - base_center))
    proj_shift = functional_p95(base_p, p, volume)
    return idx, center_shift, proj_shift


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
    volume = action_volume()
    workers = max(1, min(4, int(os.cpu_count() or 1)))

    root_inputs = [(i, p0, rim_obs, restricted_obs) for i, p0 in enumerate(deterministic_starts())]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        root_results = list(ex.map(root_task, root_inputs))
    roots = [{'index': i, 'p': p, 'cost': cost, 'qa': qa} for i, p, cost, qa in root_results]
    roots.sort(key=lambda r: (not r['qa']['plausible'], r['qa']['combined_curve_rms_px'], r['cost']))
    base = roots[0]
    base_p = base['p']
    base_qa = base['qa']

    competitive = [
        r for r in roots
        if r['qa']['plausible']
        and r['qa']['combined_curve_rms_px'] <= base_qa['combined_curve_rms_px'] + 0.35
        and r['qa']['combined_curve_p95_px'] <= float(spec['strict_max_combined_curve_p95_px'])
    ]
    root_pairwise = []
    max_root_shift = 0.0
    for i in range(len(competitive)):
        for j in range(i + 1, len(competitive)):
            s = functional_p95(competitive[i]['p'], competitive[j]['p'], volume)
            max_root_shift = max(max_root_shift, s)
            root_pairwise.append({'a': competitive[i]['index'], 'b': competitive[j]['index'], 'action_volume_p95_shift_px': s})

    support_inputs = [(base_p, rim_obs, restricted_obs, k, volume) for k in range(4)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        support_results = list(ex.map(support_task, support_inputs))
    support_results.sort(key=lambda x: x[0])
    support_shifts = [float(x[1]) for x in support_results]
    max_support_shift = max(support_shifts)

    rng = np.random.default_rng(20260903)
    perturb_trials = 32
    base_center = base_p[3:6].copy()
    perturb_inputs = []
    for idx in range(perturb_trials):
        rr = rim_obs + rng.uniform(-0.5, 0.5, size=rim_obs.shape)
        cc = restricted_obs + rng.uniform(-0.5, 0.5, size=restricted_obs.shape)
        perturb_inputs.append((idx, base_p, base_center, rr, cc, volume))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        perturb_results = list(ex.map(perturb_task, perturb_inputs))
    perturb_results.sort(key=lambda x: x[0])
    perturb_center_shifts = [float(x[1]) for x in perturb_results]
    perturb_projection_shifts = [float(x[2]) for x in perturb_results]

    first_center = np.asarray(registry['cameras']['Left Above Rim']['physical_camera_center_prior_cm'], dtype=np.float64)
    second_center = np.asarray(base_qa['camera_center_world_cm'], dtype=np.float64)
    distinct_baseline_cm = float(np.linalg.norm(second_center - first_center))

    gates = {
        'native_immutable_frame_geometry': True,
        'player_ball_pose_evidence_excluded': True,
        'strict_curve_fit': bool(
            base_qa['plausible']
            and base_qa['combined_curve_rms_px'] <= float(spec['strict_max_combined_curve_rms_px'])
            and base_qa['combined_curve_p95_px'] <= float(spec['strict_max_combined_curve_p95_px'])
        ),
        'at_least_three_competitive_multistart_roots': len(competitive) >= 3,
        'competitive_roots_functionally_equivalent': max_root_shift <= 0.5,
        'support_reduction_action_volume_p95_at_most_two_px': max_support_shift <= 2.0,
        'half_pixel_camera_center_shift_at_most_75cm': max(perturb_center_shifts) <= float(spec['strict_max_half_pixel_camera_center_shift_cm']),
        'half_pixel_action_volume_p95_at_most_two_px': max(perturb_projection_shifts) <= 2.0,
        'genuinely_distinct_from_left_above_rim': distinct_baseline_cm >= 50.0,
    }
    passed = all(gates.values())

    report = {
        'status': 'PASS_RIGHT_ABOVE_RIM_FUNCTIONAL_CAMERA_V47' if passed else 'FAIL_RIGHT_ABOVE_RIM_FUNCTIONAL_CAMERA_V47',
        'execution': {'parallel_workers': workers, 'mathematical_gates_identical_to_serial_v47': True},
        'method': 'regulation 18-inch rim circle + regulation 4-foot restricted-area floor circle; deterministic multistart, support reduction and half-pixel functional stability',
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
    (args.out / 'right_above_rim_functional_camera_v47.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    draw_overlay(image, base_p, spec, args.out / 'right_above_rim_functional_overlay_v47.png')
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
