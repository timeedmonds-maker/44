from __future__ import annotations

"""v60: direct non-coplanar physical-camera proof for immutable Left Slash Frame C.

This deliberately abandons the failed v59 same-game transfer route.  The only
metric evidence is regulation basket geometry measured directly in the immutable
Frame C: target opening on the backboard plane plus the 18-inch rim on its
horizontal plane.  Dynamic player/ball pixels are forbidden.

Promotion requires the direct solution to remain functionally equivalent across
widely different focal-prior starts, target-corner deletion, and independent
+/-0.5 pixel perturbations.  Passing v60 may promote this exact event camera and
its physical centre; it does not authorize replay rendering.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.solve_nba_geometry_proof_v3 import (
    draw_overlay,
    perturbation_sensitivity,
    project,
    solve_camera,
    world_landmarks,
)

W, H = 960, 540


def solve_one(view: dict, world: dict[str, np.ndarray]):
    names, obj, obs, rim_samples, board_obs, solved = solve_camera(view, world)
    rejected, score, params, rmse, center, focal, rim_metrics, board_rmse = solved
    if rejected:
        raise RuntimeError('metric solver rejected solution')
    return {
        'names': names,
        'obj': obj,
        'obs': obs,
        'rim_samples': rim_samples,
        'board_obs': board_obs,
        'params': np.asarray(params, float),
        'rmse': float(rmse),
        'center': np.asarray(center, float),
        'focal': float(focal),
        'rim_metrics': None if rim_metrics is None else np.asarray(rim_metrics, float),
    }


def action_volume() -> np.ndarray:
    xs = np.linspace(-60.0, 650.0, 8)
    ys = np.linspace(-320.0, 320.0, 9)
    zs = np.asarray([0.0, 90.0, 180.0, 270.0, 360.0, 430.0])
    return np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def projection_delta(a: np.ndarray, b: np.ndarray, volume: np.ndarray) -> dict:
    pa = project(a, volume)[0]
    pb = project(b, volume)[0]
    d = np.linalg.norm(pa - pb, axis=1)
    return {
        'median_px': float(np.median(d)),
        'p95_px': float(np.percentile(d, 95)),
        'max_px': float(np.max(d)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', type=Path, required=True)
    ap.add_argument('--observations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError('immutable Left Slash frame missing or not 960x540')
    payload = json.loads(args.observations.read_text())
    base_view = next(v for v in payload['views'] if v['label'] == 'Left Slash')
    text = json.dumps(base_view).lower()
    if any(t in text for t in ('player', 'ball', 'hand', 'body', 'shoulder', 'elbow')):
        raise RuntimeError('dynamic landmark contamination')

    world = world_landmarks()
    nominal = solve_one(base_view, world)
    draw_overlay(image, nominal['params'], nominal['names'], nominal['obj'], nominal['obs'],
                 nominal['rim_samples'], nominal['board_obs'], args.out / 'left_slash_v60_overlay.png')

    perturb_max, perturb_shifts = perturbation_sensitivity(
        base_view, world, nominal['params'], nominal['center'], 32, 260903
    )

    volume = action_volume()
    prior_roots = []
    for fp in (300.0, 450.0, 650.0, 900.0, 1300.0, 1800.0, 2500.0, 3500.0):
        v = json.loads(json.dumps(base_view))
        v['focal_prior_px'] = fp
        v['focal_prior_sigma_log'] = 3.0
        s = solve_one(v, world)
        prior_roots.append({
            'focal_prior_px': fp,
            'focal_px': s['focal'],
            'center_cm': [float(x) for x in s['center']],
            'center_shift_from_nominal_cm': float(np.linalg.norm(s['center'] - nominal['center'])),
            'projection_delta_from_nominal': projection_delta(nominal['params'], s['params'], volume),
            '_params': s['params'],
        })

    leave_one = []
    names = list(base_view['landmarks'])
    for drop in names:
        v = json.loads(json.dumps(base_view))
        del v['landmarks'][drop]
        s = solve_one(v, world)
        leave_one.append({
            'dropped_landmark': drop,
            'focal_px': s['focal'],
            'center_cm': [float(x) for x in s['center']],
            'center_shift_from_nominal_cm': float(np.linalg.norm(s['center'] - nominal['center'])),
            'projection_delta_from_nominal': projection_delta(nominal['params'], s['params'], volume),
        })

    centers = np.asarray([r['center_cm'] for r in prior_roots], float)
    pairwise = [float(np.linalg.norm(centers[i]-centers[j])) for i in range(len(centers)) for j in range(i+1,len(centers))]
    rim = nominal['rim_metrics']
    gates = {
        'nominal_landmark_rmse_at_most_3px': nominal['rmse'] <= 3.0,
        'nominal_rim_center_error_at_most_3px': rim is not None and float(rim[0]) <= 3.0,
        'nominal_rim_major_error_at_most_4px': rim is not None and float(rim[1]) <= 4.0,
        'nominal_rim_minor_error_at_most_2px': rim is not None and float(rim[2]) <= 2.0,
        'nominal_rim_angle_error_at_most_3deg': rim is not None and float(rim[3]) <= 3.0,
        'half_pixel_center_shift_at_most_10cm': float(perturb_max) <= 10.0,
        'focal_prior_root_pairwise_center_spread_at_most_5cm': bool(pairwise) and max(pairwise) <= 5.0,
        'all_focal_prior_roots_volume_p95_at_most_0_5px': all(r['projection_delta_from_nominal']['p95_px'] <= 0.5 for r in prior_roots),
        'all_leave_one_center_shifts_at_most_10cm': all(r['center_shift_from_nominal_cm'] <= 10.0 for r in leave_one),
        'all_leave_one_volume_p95_at_most_1px': all(r['projection_delta_from_nominal']['p95_px'] <= 1.0 for r in leave_one),
    }
    passed = bool(all(gates.values()))
    for r in prior_roots:
        r.pop('_params', None)
    report = {
        'status': 'PASS_LEFT_SLASH_DIRECT_PHYSICAL_CAMERA_V60' if passed else 'FAIL_LEFT_SLASH_DIRECT_PHYSICAL_CAMERA_V60',
        'game_id': '0022500301',
        'camera_label': 'Left Slash',
        'frame': args.frame.name,
        'method': 'direct regulation target-plane + rim-plane metric camera; no same-game centre transfer',
        'nominal': {
            'landmark_rmse_px': nominal['rmse'],
            'focal_px': nominal['focal'],
            'camera_center_basket_local_cm': [float(x) for x in nominal['center']],
            'rim_metrics': None if rim is None else {
                'center_error_px': float(rim[0]),
                'major_axis_error_px': float(rim[1]),
                'minor_axis_error_px': float(rim[2]),
                'angle_error_deg': float(rim[3]),
            },
        },
        'half_pixel_perturbation': {
            'trials': 32,
            'max_camera_center_shift_cm': float(perturb_max),
            'p95_camera_center_shift_cm': float(np.percentile(perturb_shifts,95)) if perturb_shifts else None,
        },
        'focal_prior_multistart_roots': prior_roots,
        'max_pairwise_focal_prior_root_center_distance_cm': max(pairwise) if pairwise else None,
        'leave_one_target_landmark': leave_one,
        'gates': gates,
        'permissions': {
            'left_slash_frame_c_metric_event_camera_allowed': passed,
            'left_slash_physical_camera_center_allowed': passed,
            'static_two_camera_novel_view_allowed': False,
            'replay_render_allowed': False,
        },
        'guardrail': 'Passing promotes only the exact immutable Left Slash Frame C physical/event camera. Static two-camera novel-view QA is the next gate before any replay render.',
    }
    (args.out / 'left_slash_frame_c_direct_v60.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
