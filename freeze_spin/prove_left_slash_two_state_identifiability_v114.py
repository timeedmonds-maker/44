from __future__ import annotations

"""v114: two-state metric-identifiability gate for Left Slash.

Inputs are source-pixel geometry only:
- event 75 from passing v112b;
- immutable Frame C from passing v113.

Both states share one physical optical centre while orientation, focal length
and principal point/crop may vary per state.  The observed rim is the metal
rim-tube centreline.  The tube cross-section radius is therefore solved as a
shared nuisance parameter; this script does not silently equate that centreline
with the regulation 18-inch inner opening.

The decisive gate is multistart physical-root uniqueness.  If multiple
competitive roots explain the two source states, Left Slash remains unpromoted
and a third independent same-game state is required.  No replay permission can
be granted here.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

INCH_CM = 2.54
FOOT_CM = 30.48
RIM_INNER_RADIUS_CM = 9.0 * INCH_CM
RIM_CENTER_X_CM = 15.0 * INCH_CM
RIM_TOP_Z_CM = 10.0 * FOOT_CM
TARGET_INNER_HALF_W_CM = (24.0 - 4.0) * INCH_CM / 2.0
TARGET_INNER_BOTTOM_Z_CM = RIM_TOP_Z_CM + 2.0 * INCH_CM
TARGET_INNER_TOP_Z_CM = RIM_TOP_Z_CM + 16.0 * INCH_CM

TARGET_WORLD = np.asarray([
    [0.0, -TARGET_INNER_HALF_W_CM, TARGET_INNER_TOP_Z_CM],
    [0.0, +TARGET_INNER_HALF_W_CM, TARGET_INNER_TOP_Z_CM],
    [0.0, +TARGET_INNER_HALF_W_CM, TARGET_INNER_BOTTOM_Z_CM],
    [0.0, -TARGET_INNER_HALF_W_CM, TARGET_INNER_BOTTOM_Z_CM],
], dtype=np.float64)


def look_at_rvec(center: np.ndarray) -> np.ndarray:
    aim = np.asarray([RIM_CENTER_X_CM, 0.0, RIM_TOP_Z_CM + 10.0], dtype=np.float64)
    z = aim - np.asarray(center, dtype=np.float64)
    z /= np.linalg.norm(z)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R = np.vstack([x, y, z])
    return cv2.Rodrigues(R)[0].ravel()


def unpack(z: np.ndarray):
    C = z[:3]
    a = z[3:9]
    b = z[9:15]
    tube_radius_cm = float(z[15])
    return C, a, b, tube_radius_cm


def projection_matrix(center: np.ndarray, state: np.ndarray):
    rvec = state[:3]
    focal = float(np.exp(state[3]))
    cx, cy = map(float, state[4:6])
    R = cv2.Rodrigues(rvec.reshape(3, 1))[0]
    t = -R @ center
    K = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K @ np.c_[R, t], R, t, focal


def project(center: np.ndarray, state: np.ndarray, obj: np.ndarray):
    P, R, t, focal = projection_matrix(center, state)
    cam = (R @ obj.T).T + t
    uv = np.column_stack([
        focal * cam[:, 0] / cam[:, 2] + state[4],
        focal * cam[:, 1] / cam[:, 2] + state[5],
    ])
    return uv, cam


def rim_conic(center: np.ndarray, state: np.ndarray, tube_radius_cm: float) -> np.ndarray:
    # Regulation evidence defines the 18-inch inside opening and 10-ft top.
    # Source pixels observe the metal-tube centreline, so its radius grows and
    # its centreline height drops by the unknown tube cross-section radius.
    centreline_radius = RIM_INNER_RADIUS_CM + tube_radius_cm
    centreline_z = RIM_TOP_Z_CM - tube_radius_cm
    P, _, _, _ = projection_matrix(center, state)
    H = np.c_[P[:, 0], P[:, 1], P[:, 2] * centreline_z + P[:, 3]]
    Qw = np.asarray([
        [1.0, 0.0, -RIM_CENTER_X_CM],
        [0.0, 1.0, 0.0],
        [-RIM_CENTER_X_CM, 0.0, RIM_CENTER_X_CM ** 2 - centreline_radius ** 2],
    ], dtype=np.float64)
    Hi = np.linalg.inv(H)
    Q = Hi.T @ Qw @ Hi
    return 0.5 * (Q + Q.T)


def sampson_distance(Q: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(points, dtype=np.float64), np.ones(len(points))]
    f = np.einsum('ni,ij,nj->n', p, Q, p)
    g = (2.0 * (p @ Q.T))[:, :2]
    return f / np.maximum(np.linalg.norm(g, axis=1), 1e-9)


def ring_cardinals(tube_radius_cm: float) -> np.ndarray:
    r = RIM_INNER_RADIUS_CM + tube_radius_cm
    z = RIM_TOP_Z_CM - tube_radius_cm
    return np.asarray([
        [RIM_CENTER_X_CM + r, 0.0, z],
        [RIM_CENTER_X_CM - r, 0.0, z],
        [RIM_CENTER_X_CM, +r, z],
        [RIM_CENTER_X_CM, -r, z],
    ], dtype=np.float64)


def split_support(points: np.ndarray):
    # Extraction order follows the ellipse-normal sampling order.  Sparse train
    # support keeps the nonlinear solve fast; all unused source points remain
    # independent validation evidence.
    p = np.asarray(points, dtype=np.float64)
    train = p[::8]
    held = np.asarray([x for i, x in enumerate(p) if i % 8 != 0], dtype=np.float64)
    return train, held


def residual(z, target_c, rim_train_c, target_e, rim_train_e):
    C, sc, se, tube = unpack(z)
    tc, cam_tc = project(C, sc, TARGET_WORLD)
    te, cam_te = project(C, se, TARGET_WORLD)
    qc = rim_conic(C, sc, tube)
    qe = rim_conic(C, se, tube)
    rows = [
        (tc - target_c).ravel(),
        (te - target_e).ravel(),
        sampson_distance(qc, rim_train_c),
        sampson_distance(qe, rim_train_e),
        # Weak crop/principal-point regularity only.  It cannot select a root
        # whose actual source geometry is worse.
        np.asarray([
            (sc[4] - 480.0) / 180.0, (sc[5] - 270.0) / 180.0,
            (se[4] - 480.0) / 180.0, (se[5] - 270.0) / 180.0,
        ]),
    ]
    card = ring_cardinals(tube)
    _, cam_rc = project(C, sc, card)
    _, cam_re = project(C, se, card)
    depth = np.r_[cam_tc[:, 2], cam_te[:, 2], cam_rc[:, 2], cam_re[:, 2]]
    rows.append(np.minimum(depth - 20.0, 0.0) / 5.0)
    return np.concatenate(rows)


def bounds():
    lo = np.r_[
        [-5000.0, -5000.0, -1000.0],
        [-10.0, -10.0, -10.0, math.log(150.0), -300.0, -300.0],
        [-10.0, -10.0, -10.0, math.log(150.0), -300.0, -300.0],
        0.0,
    ]
    hi = np.r_[
        [5000.0, 5000.0, 4000.0],
        [10.0, 10.0, 10.0, math.log(5000.0), 1260.0, 840.0],
        [10.0, 10.0, 10.0, math.log(5000.0), 1260.0, 840.0],
        2.5,
    ]
    return lo, hi


def seed(center, focal_c=1000.0, focal_e=1500.0, tube=0.8):
    C = np.asarray(center, dtype=np.float64)
    rv = look_at_rvec(C)
    return np.r_[C, rv, math.log(focal_c), 480.0, 270.0, rv, math.log(focal_e), 480.0, 270.0, tube]


def solve(z0, target_c, train_c, target_e, train_e):
    lo, hi = bounds()
    x0 = np.minimum(np.maximum(np.asarray(z0, dtype=np.float64), lo + 1e-6), hi - 1e-6)
    opt = least_squares(
        lambda z: residual(z, target_c, train_c, target_e, train_e),
        x0, bounds=(lo, hi), loss='soft_l1', f_scale=1.0, x_scale='jac',
        diff_step=1e-5, max_nfev=1400,
    )
    return opt


def state_metrics(z, target_c, held_c, target_e, held_e):
    C, sc, se, tube = unpack(z)
    tc, _ = project(C, sc, TARGET_WORLD)
    te, _ = project(C, se, TARGET_WORLD)
    dc = np.abs(sampson_distance(rim_conic(C, sc, tube), held_c))
    de = np.abs(sampson_distance(rim_conic(C, se, tube), held_e))
    return {
        'camera_center_cm': C.tolist(),
        'tube_radius_cm_nuisance': float(tube),
        'frame_c': {
            'focal_px': float(np.exp(sc[3])), 'principal_point_px': sc[4:6].tolist(),
            'target_rmse_px': float(np.sqrt(np.mean(np.sum((tc-target_c)**2, axis=1)))),
            'rim_heldout_count': int(len(dc)), 'rim_heldout_median_px': float(np.median(dc)),
            'rim_heldout_p95_px': float(np.percentile(dc,95)),
        },
        'event75': {
            'focal_px': float(np.exp(se[3])), 'principal_point_px': se[4:6].tolist(),
            'target_rmse_px': float(np.sqrt(np.mean(np.sum((te-target_e)**2, axis=1)))),
            'rim_heldout_count': int(len(de)), 'rim_heldout_median_px': float(np.median(de)),
            'rim_heldout_p95_px': float(np.percentile(de,95)),
        },
    }


def draw_state(image, z, state_index, target_obs, rim_obs, out):
    C, sc, se, tube = unpack(z)
    s = sc if state_index == 0 else se
    im = image.copy()
    tuv, _ = project(C, s, TARGET_WORLD)
    cv2.polylines(im, [np.round(tuv).astype(np.int32)], True, (0,255,0), 2, cv2.LINE_AA)
    cv2.polylines(im, [np.round(target_obs).astype(np.int32)], True, (0,215,255), 1, cv2.LINE_AA)
    r = RIM_INNER_RADIUS_CM + tube
    zz = RIM_TOP_Z_CM - tube
    th = np.linspace(0.0, 2.0*np.pi, 361)
    ring = np.c_[RIM_CENTER_X_CM+r*np.cos(th), r*np.sin(th), np.full_like(th,zz)]
    ruv, _ = project(C, s, ring)
    cv2.polylines(im, [np.round(ruv).astype(np.int32)], True, (255,0,255), 2, cv2.LINE_AA)
    for p in np.round(rim_obs[::max(1,len(rim_obs)//80)]).astype(np.int32):
        cv2.circle(im, tuple(p), 1, (255,255,0), -1, cv2.LINE_AA)
    cv2.imwrite(str(out), im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame-c', type=Path, required=True)
    ap.add_argument('--frame-c-geometry', type=Path, required=True)
    ap.add_argument('--event75-frame', type=Path, required=True)
    ap.add_argument('--event75-geometry', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fc = json.loads(args.frame_c_geometry.read_text())
    ev = json.loads(args.event75_geometry.read_text())
    if fc.get('status') != 'PASS_LEFT_SLASH_FRAME_C_SOURCE_GEOMETRY_V113':
        raise SystemExit('v114 requires passing v113 Frame-C source geometry')
    if ev.get('status') != 'PASS_LEFT_SLASH_EVENT75_SOURCE_GEOMETRY_V112':
        raise SystemExit('v114 requires passing v112b event75 source geometry')

    target_c = np.asarray(fc['target']['source_observed_inner_corners_px'], dtype=np.float64)
    rim_c = np.asarray(fc['rim']['source_edge_support_px'], dtype=np.float64)
    target_e = np.asarray(ev['selected']['source_observed_target_inner_corners_px'], dtype=np.float64)
    rim_e = np.asarray(ev['selected']['rim']['source_edge_support_px'], dtype=np.float64)
    train_c, held_c = split_support(rim_c)
    train_e, held_e = split_support(rim_e)

    seeds = [
        ([800,0,700],800,1300), ([1200,500,800],1000,1500),
        ([1600,500,800],1200,1800), ([2000,0,1000],1300,2000),
        ([1200,-500,1000],1000,1500), ([1600,-500,1200],1200,1800),
        ([800,800,1400],900,1400), ([2000,800,1400],1400,2200),
    ]
    roots = []
    for i, (C0, f1, f2) in enumerate(seeds):
        try:
            opt = solve(seed(C0,f1,f2), target_c, train_c, target_e, train_e)
            m = state_metrics(opt.x, target_c, held_c, target_e, held_e)
            source_good = (
                m['frame_c']['target_rmse_px'] <= 3.0 and m['event75']['target_rmse_px'] <= 3.0
                and m['frame_c']['rim_heldout_p95_px'] <= 3.0 and m['event75']['rim_heldout_p95_px'] <= 3.0
            )
            roots.append({
                'seed_index': i, 'cost': float(opt.cost), 'nfev': int(opt.nfev),
                'optimality': float(opt.optimality), 'success_flag': bool(opt.success),
                'source_geometry_good': bool(source_good), 'metrics': m,
                '_z': opt.x,
            })
        except Exception as e:
            roots.append({'seed_index':i,'error':repr(e),'source_geometry_good':False})

    good = [r for r in roots if r.get('source_geometry_good') and '_z' in r]
    good.sort(key=lambda r:r['cost'])
    nominal = good[0] if good else None
    competitive = []
    pairwise = []
    if nominal is not None:
        cap = nominal['cost'] * 1.10 + 2.0
        competitive = [r for r in good if r['cost'] <= cap]
        centers = [np.asarray(r['metrics']['camera_center_cm'],float) for r in competitive]
        pairwise = [float(np.linalg.norm(centers[i]-centers[j])) for i in range(len(centers)) for j in range(i+1,len(centers))]

    # Preserve the earlier Left-Slash root-uniqueness standard rather than
    # weakening it to make the new solve green.
    max_spread = max(pairwise) if pairwise else 0.0
    gates = {
        'at_least_one_source_geometry_root': nominal is not None,
        'competitive_multistart_roots_share_center_within_5cm': nominal is not None and max_spread <= 5.0,
        'nominal_frame_c_target_rmse_at_most_3px': nominal is not None and nominal['metrics']['frame_c']['target_rmse_px'] <= 3.0,
        'nominal_event75_target_rmse_at_most_3px': nominal is not None and nominal['metrics']['event75']['target_rmse_px'] <= 3.0,
        'nominal_frame_c_rim_heldout_p95_at_most_3px': nominal is not None and nominal['metrics']['frame_c']['rim_heldout_p95_px'] <= 3.0,
        'nominal_event75_rim_heldout_p95_at_most_3px': nominal is not None and nominal['metrics']['event75']['rim_heldout_p95_px'] <= 3.0,
    }
    passed = all(gates.values())

    if nominal is not None:
        imc = cv2.imread(str(args.frame_c), cv2.IMREAD_COLOR)
        ime = cv2.imread(str(args.event75_frame), cv2.IMREAD_COLOR)
        if imc is not None:
            draw_state(imc, nominal['_z'], 0, target_c, rim_c, args.out/'left_slash_v114_frame_c_overlay.png')
        if ime is not None:
            draw_state(ime, nominal['_z'], 1, target_e, rim_e, args.out/'left_slash_v114_event75_overlay.png')

    for r in roots:
        r.pop('_z',None)
    report = {
        'status': 'PASS_LEFT_SLASH_TWO_STATE_IDENTIFIABILITY_V114' if passed else 'FAIL_LEFT_SLASH_TWO_STATE_IDENTIFIABILITY_V114',
        'method': 'shared physical optical centre; per-state pose/focal/principal point; target opening + rim-tube-centreline source evidence; tube radius solved as nuisance',
        'frame_c_source_geometry': args.frame_c_geometry.name,
        'event75_source_geometry': args.event75_geometry.name,
        'rim_model': {
            'regulation_inside_radius_cm': RIM_INNER_RADIUS_CM,
            'regulation_top_z_cm': RIM_TOP_Z_CM,
            'tube_radius_is_unknown_nuisance': True,
            'tube_radius_bound_cm': [0.0,2.5],
        },
        'support': {
            'frame_c_total': int(len(rim_c)), 'frame_c_train': int(len(train_c)), 'frame_c_heldout': int(len(held_c)),
            'event75_total': int(len(rim_e)), 'event75_train': int(len(train_e)), 'event75_heldout': int(len(held_e)),
        },
        'nominal': None if nominal is None else nominal['metrics'],
        'competitive_root_count': len(competitive),
        'competitive_root_center_pairwise_max_cm': max_spread if nominal is not None else None,
        'roots': roots,
        'gates': gates,
        'permissions': {
            'left_slash_metric_camera_allowed': passed,
            'counts_as_fourth_metric_camera': False,
            'four_camera_cross_consistency_allowed': passed,
            'replay_render_allowed': False,
        },
        'next_if_fail': 'Add a third independent same-game Left Slash source state; do not weaken the 5 cm root-uniqueness gate.',
        'next_if_pass': 'Run support-removal and +/-0.5 px robustness plus three-camera-registry distinctness before promoting camera #4.',
        'guardrail': 'v114 cannot authorize replay rendering even if it passes.',
    }
    (args.out/'left_slash_two_state_identifiability_v114.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
