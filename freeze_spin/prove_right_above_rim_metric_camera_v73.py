from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from freeze_spin import prove_right_above_rim_physical_event_v70 as v70


EXPECTED_V72_FALSE_GATES = {
    'zoom_pp_envelope_event225_heldout_floor_p95_at_most_2px',
    'zoom_pp_envelope_nominal_state_separation_at_most_5px',
    'half_pixel_state_pp_separation_at_most_5px',
}


def pack(cx, cy, radius, theta, pe, pt):
    return np.r_[cx, cy, radius, theta, pe[0], pe[3:6], pt[0], pt[3:6]].astype(float)


def unpack(x):
    cx, cy, radius, theta = x[:4]
    d = radius * np.array([math.cos(theta), math.sin(theta)], float)
    pe = np.r_[x[4], cx + d[0], cy + d[1], x[5:8]]
    pt = np.r_[x[8], cx - d[0], cy - d[1], x[9:12]]
    return pe, pt


def pp_pair(x):
    pe, pt = unpack(x)
    return pe[1:3], pt[1:3]


def pstats_dense(obs, key, p, C, n):
    pr, _ = v70.project_fixed(p, C, v70.curve(key, n))
    d = cKDTree(pr).query(obs)[0]
    return {
        'count': int(len(d)),
        'median_px': float(np.median(d)),
        'p95_px': float(np.percentile(d, 95)),
        'max_px': float(np.max(d)),
    }


def dense_heldout(ete, erte, tte, pe, pt, C, n):
    event = {k: pstats_dense(ete[k], k, pe, C, n) for k in ete}
    event['rim'] = pstats_dense(erte, 'rim', pe, C, n)
    target = {k: pstats_dense(tte[k], k, pt, C, n) for k in tte}
    return event, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v72-report', type=Path, required=True)
    ap.add_argument('--event225-frame', type=Path, required=True)
    ap.add_argument('--target-frame', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    q72 = json.loads(a.v72_report.read_text())
    if q72.get('status') != 'FAIL_RIGHT_ABOVE_RIM_ZOOM_PP_ENVELOPE_V72':
        raise RuntimeError('v73 expects the exact v72 fail-closed diagnostic report')
    false_gates = {k for k, v in q72.get('gates', {}).items() if v is False}
    if false_gates != EXPECTED_V72_FALSE_GATES:
        raise RuntimeError(f'v72 failure signature changed: {sorted(false_gates)}')
    if not all(v is True for k, v in q72['gates'].items() if k not in EXPECTED_V72_FALSE_GATES):
        raise RuntimeError('v72 prerequisite evidence is not intact')
    if q72.get('game_id') != '0022500301':
        raise RuntimeError('v72 game provenance mismatch')
    if int(q72.get('intended_basketball_event', -1)) != 486:
        raise RuntimeError('v72 intended basketball-event provenance mismatch')
    if int(q72.get('frame_c_source_clip_event', -1)) != 489:
        raise RuntimeError('v72 Frame C source-clip provenance mismatch')
    if v70.sha256(a.event225_frame) != v70.EV_SHA:
        raise RuntimeError('immutable event225 SHA mismatch')
    if v70.sha256(a.target_frame) != v70.TG_SHA:
        raise RuntimeError('immutable Frame C SHA mismatch')

    C = np.array(q72['physical_center_cm'], float)
    ie = cv2.imread(str(a.event225_frame))
    it = cv2.imread(str(a.target_frame))
    if ie is None or it is None:
        raise RuntimeError('could not read immutable source frames')

    oe, re = v70.extract_event(ie)
    etr, ete = v70.split_dict(oe, {'left': 0, 'right': 1, 'ft': 2, 'restricted': 3})
    ertr, erte = v70.split_rim(re, 1)
    ot, rt = v70.extract_target(it)
    ttr, tte = v70.split_dict(ot, {'left': 0, 'right': 1, 'ft': 2})
    trtr, trte = v70.split_rim(rt, 3)
    ttr['rim'] = trtr
    tte['rim'] = trte

    def residual(x, event_data=etr, event_rim=ertr, target_data=ttr):
        pe, pt = unpack(x)
        out = []
        for k in ['left', 'right', 'ft', 'restricted']:
            pr, _ = v70.project_fixed(pe, C, v70.CUR[k])
            out.append(v70.nv(event_data[k], pr) * (1.3 if k == 'restricted' else 1.0))
        pr, _ = v70.project_fixed(pe, C, v70.CUR['rim'])
        out.append(v70.nv(event_rim, pr) * 1.15)
        for k in ['left', 'right', 'ft', 'rim']:
            pr, _ = v70.project_fixed(pt, C, v70.CUR[k])
            out.append(v70.nv(target_data[k], pr) * (1.25 if k == 'rim' else 1.0))
        return np.concatenate(out)

    pe72 = np.r_[
        math.log(q72['event225']['focal_px']),
        q72['event225']['principal_point_px'],
        q72['event225']['rvec'],
    ]
    pt72 = np.r_[
        math.log(q72['target_frame_c']['focal_px']),
        q72['target_frame_c']['principal_point_px'],
        q72['target_frame_c']['rvec'],
    ]
    pp_e72 = np.array(q72['event225']['principal_point_px'], float)
    pp_t72 = np.array(q72['target_frame_c']['principal_point_px'], float)
    base72 = np.array(q72['shared_base_principal_point_px'], float)
    direction = pp_e72 - pp_t72
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    theta0 = math.atan2(float(direction[1]), float(direction[0]))

    # v72 bounded dx and dy independently to +/-2 px. That box permits a
    # Euclidean state separation of 2*sqrt(2^2+2^2)=5.657 px, so its stated
    # <=5 px envelope was not actually enforced by the solver. v73 uses the
    # radial half-separation directly: radius<=2.5 => ||PP_e-PP_t||<=5 px.
    lo = np.r_[100.0, 50.0, 0.0, -math.pi, math.log(150), [-10.0] * 3, math.log(150), [-10.0] * 3]
    hi = np.r_[850.0, 520.0, 2.5, math.pi, math.log(2500), [10.0] * 3, math.log(2500), [10.0] * 3]

    starts = []
    for pp in [base72, base72 + np.array([1.0, 0.0])]:
        for radius in [0.0, 1.5, 2.45]:
            for ef, tf in [(1.0, 1.0), (0.97, 1.03), (1.03, 0.97)]:
                pe = pe72.copy()
                pt = pt72.copy()
                pe[0] += math.log(ef)
                pt[0] += math.log(tf)
                starts.append(pack(pp[0], pp[1], radius, theta0, pe, pt))

    rows = []
    for s in starts:
        o = least_squares(
            residual,
            s,
            bounds=(lo, hi),
            loss='soft_l1',
            f_scale=1.2,
            x_scale='jac',
            max_nfev=900,
        )
        pe, pt = unpack(o.x)
        _, ze = v70.project_fixed(pe, C, v70.ACTION)
        _, zt = v70.project_fixed(pt, C, v70.ACTION)
        valid = (np.all(ze > 0) or np.all(ze < 0)) and (np.all(zt > 0) or np.all(zt < 0))
        if valid:
            rows.append((float(o.cost), o.x))
    if not rows:
        raise RuntimeError('no physically valid v73 roots')
    rows.sort(key=lambda z: z[0])

    best_cost, xb = rows[0]
    peb, ptb = unpack(xb)
    ppeb, pptb = pp_pair(xb)
    pp_separation = float(np.linalg.norm(ppeb - pptb))

    target_nominal = v70.project_fixed(ptb, C, v70.ACTION)[0]
    competitive = []
    for cost, x in rows:
        if cost > best_cost + 1.0:
            break
        _, pt = unpack(x)
        ppe, ppt = pp_pair(x)
        competitive.append({
            'cost': cost,
            **v70.dstat(target_nominal, v70.project_fixed(pt, C, v70.ACTION)[0]),
            'principal_point_separation_px': float(np.linalg.norm(ppe - ppt)),
        })

    # Preserve the old coarse-grid held-out values as a diagnostic. The v72
    # event-floor miss was only 0.019 px on the left family and 0.0025 px after
    # the hard Euclidean constraint; therefore final acceptance is evaluated
    # against numerically converged metric curves rather than the sparse
    # 720/900-point proxy used during optimization.
    event_held_coarse = {k: v70.pstats(ete[k], k, peb, False, C) for k in ete}
    event_held_coarse['rim'] = v70.pstats(erte, 'rim', peb, False, C)
    target_held_coarse = {k: v70.pstats(tte[k], k, ptb, False, C) for k in tte}

    event_held_5760, target_held_5760 = dense_heldout(ete, erte, tte, peb, ptb, C, 5760)
    event_held_11520, target_held_11520 = dense_heldout(ete, erte, tte, peb, ptb, C, 11520)
    convergence = {}
    for scope, a0, a1 in [
        ('event225', event_held_5760, event_held_11520),
        ('frame_c', target_held_5760, target_held_11520),
    ]:
        for k in a0:
            convergence[f'{scope}_{k}'] = abs(a0[k]['p95_px'] - a1[k]['p95_px'])
    max_convergence_delta = float(max(convergence.values()))

    rng = np.random.default_rng(7313)
    pert = []
    for _ in range(64):
        ed = {k: v + rng.uniform(-0.5, 0.5, v.shape) for k, v in etr.items()}
        er = ertr + rng.uniform(-0.5, 0.5, ertr.shape)
        td = {k: v + rng.uniform(-0.5, 0.5, v.shape) for k, v in ttr.items()}
        o = least_squares(
            lambda x: residual(x, ed, er, td),
            xb,
            bounds=(lo, hi),
            loss='soft_l1',
            f_scale=1.2,
            x_scale='jac',
            max_nfev=1200,
        )
        pe, pt = unpack(o.x)
        ppe, ppt = pp_pair(o.x)
        d = v70.dstat(target_nominal, v70.project_fixed(pt, C, v70.ACTION)[0])
        d['max_target_heldout_p95_px'] = float(max(v70.pstats(tte[k], k, pt, False, C)['p95_px'] for k in tte))
        d['shared_base_pp_shift_px'] = float(np.linalg.norm(o.x[:2] - xb[:2]))
        d['principal_point_separation_px'] = float(np.linalg.norm(ppe - ppt))
        d['half_separation_radius_px'] = float(o.x[2])
        pert.append(d)

    max_comp_p95 = float(max(x['p95_px'] for x in competitive))
    max_pert_p95 = float(max(x['p95_px'] for x in pert))
    max_pert_held = float(max(x['max_target_heldout_p95_px'] for x in pert))
    max_base_shift = float(max(x['shared_base_pp_shift_px'] for x in pert))
    max_pert_pp_sep = float(max(x['principal_point_separation_px'] for x in pert))
    max_pert_radius = float(max(x['half_separation_radius_px'] for x in pert))

    event_floor_dense_max = float(max(event_held_11520[k]['p95_px'] for k in ['left', 'right', 'ft', 'restricted']))
    event_rim_dense = float(event_held_11520['rim']['p95_px'])
    target_dense_max = float(max(x['p95_px'] for x in target_held_11520.values()))

    gates = {
        'v72_prerequisite_evidence_intact': True,
        'immutable_event225_and_frame_c': True,
        'correct_provenance_game_0022500301_block_486_frame_c_source_489': True,
        'euclidean_pp_half_separation_radius_at_most_2_5px': float(xb[2]) <= 2.5 + 1e-9,
        'euclidean_pp_nominal_state_separation_at_most_5px': pp_separation <= 5.0 + 1e-9,
        'dense_curve_evaluation_converged_p95_delta_at_most_0_01px': max_convergence_delta <= 0.01,
        'dense_event225_heldout_floor_p95_at_most_2px': event_floor_dense_max <= 2.0,
        'dense_event225_heldout_rim_p95_at_most_2px': event_rim_dense <= 2.0,
        'dense_frame_c_heldout_p95_at_most_2px': target_dense_max <= 2.0,
        'competitive_roots_frame_c_action_p95_at_most_0_5px': max_comp_p95 <= 0.5,
        'half_pixel_frame_c_action_p95_at_most_2px': max_pert_p95 <= 2.0,
        'half_pixel_frame_c_heldout_p95_at_most_2_5px': max_pert_held <= 2.5,
        'half_pixel_shared_base_pp_shift_at_most_5px': max_base_shift <= 5.0,
        'half_pixel_pp_half_separation_radius_at_most_2_5px': max_pert_radius <= 2.5 + 1e-9,
        'half_pixel_state_pp_separation_at_most_5px': max_pert_pp_sep <= 5.0 + 1e-9,
    }
    passed = all(gates.values())
    status = 'PASS_RIGHT_ABOVE_RIM_METRIC_CAMERA_V73' if passed else 'FAIL_RIGHT_ABOVE_RIM_METRIC_CAMERA_V73'
    permissions = {
        'physical_camera_center_allowed': passed,
        'metric_event_camera_allowed': passed,
        'static_novel_view_allowed': False,
        'replay_render_allowed': False,
    }

    report = {
        'schema_version': 1,
        'status': status,
        'camera_label': 'Right Above Rim',
        'game_id': '0022500301',
        'intended_basketball_event': 486,
        'intended_basketball_clock': 'Q3 04:11.00',
        'intended_basketball_description': 'MISS B. Sensabaugh driving DUNK - blocked by Steven Adams',
        'frame_c_source_clip_event': 489,
        'physical_center_cm': C.tolist(),
        'baseline_to_left_above_rim_cm': q72['baseline_to_left_above_rim_cm'],
        'model': 'fixed v70 physical centre; exact Euclidean <=5px inter-state principal-point envelope via half-separation radius<=2.5px; independent focal length and orientation per PTZ optical state; pinhole projection; numerically converged held-out metric-curve evaluation',
        'v72_failure_diagnosis': {
            'parameterization_bug': 'v72 bounded dx and dy independently to +/-2px, which permits up to 5.656854px Euclidean state separation despite the intended <=5px envelope',
            'numerical_evaluation_issue': 'the remaining event225 floor miss is evaluated against a sparse sampled metric curve; v73 retains the same 2px threshold and proves convergence with 5760 and 11520 samples',
            'v72_false_gates': sorted(EXPECTED_V72_FALSE_GATES),
        },
        'shared_base_principal_point_px': xb[:2].tolist(),
        'half_separation_radius_px': float(xb[2]),
        'half_separation_theta_rad': float(xb[3]),
        'nominal_state_principal_point_separation_px': pp_separation,
        'event225': {
            'principal_point_px': ppeb.tolist(),
            'focal_px': float(np.exp(peb[0])),
            'rvec': peb[3:6].tolist(),
            'heldout_coarse_proxy': event_held_coarse,
            'heldout_dense_5760': event_held_5760,
            'heldout_dense_11520': event_held_11520,
        },
        'target_frame_c': {
            'principal_point_px': pptb.tolist(),
            'focal_px': float(np.exp(ptb[0])),
            'rvec': ptb[3:6].tolist(),
            'heldout_coarse_proxy': target_held_coarse,
            'heldout_dense_5760': target_held_5760,
            'heldout_dense_11520': target_held_11520,
            'competitive_roots': competitive,
            'perturbation_64': {
                'max_action_p95_px': max_pert_p95,
                'max_action_max_px': float(max(x['max_px'] for x in pert)),
                'max_heldout_p95_px': max_pert_held,
                'max_shared_base_pp_shift_px': max_base_shift,
                'max_state_pp_separation_px': max_pert_pp_sep,
                'max_half_separation_radius_px': max_pert_radius,
            },
        },
        'dense_curve_convergence': {
            'samples_low': 5760,
            'samples_high': 11520,
            'p95_delta_px_by_family': convergence,
            'max_p95_delta_px': max_convergence_delta,
        },
        'competitive_root_count': len(competitive),
        'gates': gates,
        'permissions': permissions,
        'next_gate': 'If PASS, Right Above Rim is formally promoted as metric camera #2. Static novel-view rendering remains separately gated; next step is the deterministic static Freeview validation.',
    }

    (a.out / 'right_above_rim_metric_camera_v73.json').write_text(json.dumps(report, indent=2))
    v70.draw(ie, a.out / 'right_above_rim_event225_overlay_v73.png', peb, C=C, full=False, held={**ete, 'rim': erte})
    v70.draw(it, a.out / 'right_above_rim_frame_c_overlay_v73.png', ptb, C=C, full=False, held=tte)

    print(json.dumps({
        'status': status,
        'best_cost': best_cost,
        'shared_base_principal_point_px': xb[:2].tolist(),
        'event225_principal_point_px': ppeb.tolist(),
        'frame_c_principal_point_px': pptb.tolist(),
        'half_separation_radius_px': float(xb[2]),
        'nominal_state_pp_separation_px': pp_separation,
        'coarse_event225_floor_p95_max_px': float(max(event_held_coarse[k]['p95_px'] for k in ['left', 'right', 'ft', 'restricted'])),
        'dense_event225_floor_p95_max_px': event_floor_dense_max,
        'dense_event225_rim_p95_px': event_rim_dense,
        'dense_frame_c_heldout_p95_max_px': target_dense_max,
        'dense_curve_max_p95_delta_px': max_convergence_delta,
        'competitive_root_count': len(competitive),
        'max_competitive_target_action_p95_px': max_comp_p95,
        'target_perturbation': report['target_frame_c']['perturbation_64'],
        'gates': gates,
        'permissions': permissions,
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
