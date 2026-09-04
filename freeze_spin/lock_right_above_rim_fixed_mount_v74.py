from __future__ import annotations

"""v74: operationalize the certified Right Above Rim camera as a fixed-centre PTZ anchor.

The direct-over-rim feed is physically mounted at the basket. v73 already proved the
metric physical centre and the exact Frame C optical state. v74 does not re-fit or
relax that geometry. It freezes the physical centre, emits the explicit K/R/t/P
camera matrices for the immutable Frame C, and defines the only variables that may
change in later clips from this mount: orientation, focal length and image/crop
principal point.

Cross-arena reuse of the exact numeric mount offset remains fail-closed until a
second arena/rig is checked. For the current Utah basket and this game, the centre
is a hard anchor and must never be allowed to wander to explain zoom/crop changes.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

RIM_CENTER_CM = np.array([15.0 * 2.54, 0.0, 10.0 * 30.48], dtype=float)
BACKBOARD_PLANE_X_CM = 0.0


def matrix_payload(q73: dict) -> dict:
    C = np.asarray(q73['physical_center_cm'], dtype=float)
    state = q73['target_frame_c']
    f = float(state['focal_px'])
    cx, cy = map(float, state['principal_point_px'])
    rvec = np.asarray(state['rvec'], dtype=float)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    t = -R @ C
    K = np.asarray([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)
    P = K @ np.column_stack([R, t])
    optical_axis_world = R.T @ np.asarray([0.0, 0.0, 1.0])
    return {
        'K': K.tolist(),
        'R_world_to_camera': R.tolist(),
        't_world_to_camera_cm': t.tolist(),
        'P_world_cm_to_image_homogeneous': P.tolist(),
        'optical_axis_world_unit': optical_axis_world.tolist(),
        'camera_center_basket_local_cm': C.tolist(),
        'camera_height_above_floor_cm': float(C[2]),
        'camera_height_above_rim_cm': float(C[2] - RIM_CENTER_CM[2]),
        'camera_setback_from_backboard_plane_cm': float(C[0] - BACKBOARD_PLANE_X_CM),
        'camera_lateral_offset_from_basket_centerline_cm': float(C[1]),
        'horizontal_offset_from_rim_center_cm': float(np.linalg.norm((C - RIM_CENTER_CM)[:2])),
        'distance_to_rim_center_cm': float(np.linalg.norm(C - RIM_CENTER_CM)),
    }


def reprojection_equivalence(q73: dict, matrices: dict) -> float:
    """Check that emitted matrix P exactly reproduces the v73 pinhole state."""
    C = np.asarray(q73['physical_center_cm'], dtype=float)
    state = q73['target_frame_c']
    f = float(state['focal_px'])
    cx, cy = map(float, state['principal_point_px'])
    rvec = np.asarray(state['rvec'], dtype=float)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    P = np.asarray(matrices['P_world_cm_to_image_homogeneous'], dtype=float)

    # Deterministic 3-D action volume around the basket. This is not training data.
    pts = np.asarray(
        [[x, y, z] for x in np.linspace(-30.0, 250.0, 8)
                   for y in np.linspace(-180.0, 180.0, 9)
                   for z in np.linspace(20.0, 350.0, 8)],
        dtype=float,
    )
    cam = (R @ (pts - C).T).T
    uv_direct = np.column_stack([
        f * cam[:, 0] / cam[:, 2] + cx,
        f * cam[:, 1] / cam[:, 2] + cy,
    ])
    h = (P @ np.column_stack([pts, np.ones(len(pts))]).T).T
    uv_matrix = h[:, :2] / h[:, 2:3]
    return float(np.max(np.linalg.norm(uv_direct - uv_matrix, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v73-report', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    q73 = json.loads(args.v73_report.read_text())
    if q73.get('status') != 'PASS_RIGHT_ABOVE_RIM_METRIC_CAMERA_V73':
        raise RuntimeError('v74 requires the certified v73 Right Above Rim report')
    if q73.get('camera_label') != 'Right Above Rim':
        raise RuntimeError('camera label mismatch')
    if q73.get('game_id') != '0022500301':
        raise RuntimeError('game provenance mismatch')
    if int(q73.get('intended_basketball_event', -1)) != 486:
        raise RuntimeError('basketball-event provenance mismatch')
    if int(q73.get('frame_c_source_clip_event', -1)) != 489:
        raise RuntimeError('Frame C source-clip provenance mismatch')
    perms = q73.get('permissions', {})
    if not perms.get('physical_camera_center_allowed') or not perms.get('metric_event_camera_allowed'):
        raise RuntimeError('v73 camera permissions are not intact')
    if not all(q73.get('gates', {}).values()):
        raise RuntimeError('v73 certification gates are not all true')

    matrices = matrix_payload(q73)
    max_equivalence_error = reprojection_equivalence(q73, matrices)
    C = np.asarray(q73['physical_center_cm'], dtype=float)

    gates = {
        'v73_metric_camera_certification_intact': True,
        'physical_center_finite': bool(np.all(np.isfinite(C))),
        'current_frame_matrix_reproduces_v73_projection_at_1e_8px': max_equivalence_error <= 1e-8,
        # This is a structural sanity check only, not a re-fit: the certified centre
        # is essentially on the basket centreline, as expected for the fixed rim rig.
        'certified_center_within_2cm_of_basket_centerline': abs(float(C[1])) <= 2.0,
    }
    passed = bool(all(gates.values()))

    report = {
        'schema_version': 1,
        'status': 'PASS_RIGHT_ABOVE_RIM_FIXED_MOUNT_ANCHOR_V74' if passed else 'FAIL_RIGHT_ABOVE_RIM_FIXED_MOUNT_ANCHOR_V74',
        'camera_label': 'Right Above Rim',
        'camera_role': 'direct-over-rim fixed-centre PTZ anchor',
        'game_id': '0022500301',
        'intended_basketball_event': 486,
        'frame_c_source_clip_event': 489,
        'source_certification': {
            'report': args.v73_report.name,
            'status': q73['status'],
        },
        'fixed_mount_model': {
            'hard_locked_across_current_rig_states': ['physical_camera_center_basket_local_cm'],
            'solve_per_clip_state': ['orientation_rvec', 'focal_px', 'principal_point_or_digital_crop_px'],
            'forbidden_compensation': 'Do not move the physical camera centre to explain zoom/crop/PTZ changes.',
            'cross_arena_exact_numeric_transfer_allowed': False,
            'cross_arena_note': 'The fixed-mount architecture is reusable, but the exact numeric mount offset must be checked once per distinct physical rig/arena before transfer.',
        },
        'current_frame_c_camera': matrices,
        'matrix_equivalence_max_error_px': max_equivalence_error,
        'gates': gates,
        'permissions': {
            'fixed_center_ptz_anchor_allowed': passed,
            'current_frame_projection_matrix_allowed': passed,
            'cross_arena_exact_mount_transfer_allowed': False,
            'static_novel_view_allowed': False,
            'replay_render_allowed': False,
        },
        'guardrail': 'This camera is solved as a fixed-centre PTZ anchor for the current basket/rig. It strengthens reconstruction but does not by itself unlock Freeview rendering.',
    }
    out = args.out / 'right_above_rim_fixed_mount_anchor_v74.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
