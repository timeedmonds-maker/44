from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin import prove_right_above_rim_physical_event_v70 as v70

MAX_STATE_PP_SEPARATION_PX = 5.0
MAX_HALF_SEPARATION_PX = MAX_STATE_PP_SEPARATION_PX / 2.0
NUMERIC_EPS_PX = 1e-6


def delta_from(x: np.ndarray) -> np.ndarray:
    rho = float(x[2])
    theta = float(x[3])
    return rho * np.array([math.cos(theta), math.sin(theta)], dtype=float)


def pack(cx, cy, delta, pe, pt):
    delta = np.asarray(delta, dtype=float)
    rho = float(np.linalg.norm(delta))
    theta = float(math.atan2(delta[1], delta[0])) if rho > 1e-12 else 0.0
    return np.r_[cx, cy, rho, theta, pe[0], pe[3:6], pt[0], pt[3:6]].astype(float)


def unpack(x):
    cx, cy = x[:2]
    d = delta_from(x)
    pe = np.r_[x[4], cx + d[0], cy + d[1], x[5:8]]
    pt = np.r_[x[8], cx - d[0], cy - d[1], x[9:12]]
    return pe, pt


def pp_pair(x):
    base = np.asarray(x[:2], dtype=float)
    d = delta_from(x)
    return base + d, base - d


def state_separation(x) -> float:
    a, b = pp_pair(x)
    return float(np.linalg.norm(a - b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v70-report', type=Path, required=True)
    ap.add_argument('--v71-report', type=Path, required=True)
    ap.add_argument('--v72-report', type=Path, required=True)
    ap.add_argument('--chooser-options', type=Path, required=True)
    ap.add_argument('--event225-frame', type=Path, required=True)
    ap.add_argument('--target-frame', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    q70 = json.loads(a.v70_report.read_text())
    q71 = json.loads(a.v71_report.read_text())
    q72 = json.loads(a.v72_report.read_text())
    chooser = json.loads(a.chooser_options.read_text())

    if q70.get('status') != 'FAIL_RIGHT_ABOVE_RIM_PHYSICAL_EVENT_V70':
        raise RuntimeError('v73 expects exact v70 diagnostic evidence')
    if q71.get('status') != 'FAIL_RIGHT_ABOVE_RIM_JOINT_INTRINSICS_V71':
        raise RuntimeError('v73 expects exact v71 one-gate-fail evidence')
    if q72.get('status') != 'FAIL_RIGHT_ABOVE_RIM_ZOOM_PP_ENVELOPE_V72':
        raise RuntimeError('v73 expects exact v72 three-gate-fail evidence')
    if v70.sha256(a.event225_frame) != v70.EV_SHA:
        raise RuntimeError('immutable event225 SHA mismatch')
    if v70.sha256(a.target_frame) != v70.TG_SHA:
        raise RuntimeError('immutable Frame C SHA mismatch')

    physical_gate_names = [
        'immutable_event225_and_target_frames',
        'accepted_v69_floor_input',
        'event225_all_heldout_floor_p95_at_most_2px',
        'event225_heldout_inner_rim_p95_at_most_2px',
        'event225_competitive_physical_roots_action_p95_at_most_1px',
        'event225_half_pixel_center_shift_at_most_25cm',
        'event225_half_pixel_action_p95_at_most_2px',
        'event225_half_pixel_heldout_floor_p95_at_most_2_5px',
        'event225_half_pixel_heldout_rim_p95_at_most_2_5px',
        'distinct_from_left_above_rim_by_at_least_50cm',
    ]
    if not all(q70['gates'].get(k) is True for k in physical_gate_names):
        raise RuntimeError('v70 physical-centre evidence is not intact')

    v71_required = [
        'v70_physical_centre_evidence_intact',
        'immutable_event225_and_frame_c',
        'shared_principal_point_model_fits_event225_rim_p95_at_most_2px',
        'shared_principal_point_model_fits_frame_c_heldout_p95_at_most_2px',
        'joint_competitive_roots_frame_c_action_p95_at_most_0_5px',
        'joint_half_pixel_frame_c_action_p95_at_most_2px',
        'joint_half_pixel_frame_c_heldout_p95_at_most_2_5px',
        'joint_half_pixel_shared_pp_shift_at_most_5px',
    ]
    if not all(q71['gates'].get(k) is True for k in v71_required):
        raise RuntimeError('v71 target-identifiability breakthrough is not intact')
    if q71['gates'].get('shared_principal_point_model_fits_event225_heldout_floor_p95_at_most_2px') is not False:
        raise RuntimeError('v73 expects the exact v71 held-out floor miss')

    expected_v72_false = {
        'zoom_pp_envelope_event225_heldout_floor_p95_at_most_2px',
        'zoom_pp_envelope_nominal_state_separation_at_most_5px',
        'half_pixel_state_pp_separation_at_most_5px',
    }
    actual_v72_false = {k for k, v in q72.get('gates', {}).items() if v is False}
    if actual_v72_false != expected_v72_false:
        raise RuntimeError(f'unexpected v72 failure surface: {sorted(actual_v72_false)}')
    if q72.get('physical_center_cm') != q70.get('physical_center_cm'):
        raise RuntimeError('v72 physical centre does not match v70')

    if str(chooser.get('event', {}).get('game_id')) != '0022500301' or int(chooser.get('event', {}).get('event_id', -1)) != 489:
        raise RuntimeError('Frame C chooser source provenance mismatch')
    opts = [o for o in chooser.get('options', []) if o.get('camera') == 'Right Above Rim']
    if len(opts) != 1 or opts[0].get('file') != 'L_Right_Above_Rim_8.562013s_frame0257.png':
        raise RuntimeError('Frame C Right Above Rim chooser option mismatch')

    C = np.array(q70['physical_center_cm'], float)
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

    pe71 = np.r_[math.log(q71['event225']['focal_px']), q71['shared_principal_point_px'], q71['event225']['rvec']]
    pt71 = np.r_[math.log(q71['target_frame_c']['focal_px']), q71['shared_principal_point_px'], q71['target_frame_c']['rvec']]
    pp70e = np.array(q70['event225']['principal_point_px'], float)
    pp70t = np.array(q70['target_frame_c']['principal_point_px'], float)
    direction = pp70e - pp70t
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    pp71 = np.array(q71['shared_principal_point_px'], float)

    # v72 used independent dx/dy bounds of +/-2 px, which permitted a full
    # Euclidean state separation of 2*sqrt(2^2+2^2)=5.657 px despite a 5 px
    # gate. v73 encodes that same 5 px physical requirement directly as a
    # radial half-separation rho <= 2.5 px. The threshold is not relaxed.
    lo = np.r_[100.0, 50.0, 0.0, -math.pi, math.log(150), [-10.0] * 3, math.log(150), [-10.0] * 3]
    hi = np.r_[850.0, 520.0, MAX_HALF_SEPARATION_PX, math.pi, math.log(2500), [10.0] * 3, math.log(2500), [10.0] * 3]

    starts = []
    pp_starts = [pp71, (pp70e + pp70t) / 2.0, pp71 + np.array([1.0, 0.0]), pp71 + np.array([-1.0, 0.0])]
    for pp in pp_starts:
        for mag in [0.05, 0.5, 1.0, 1.5, 1.9, 2.49]:
            d = direction * mag
            for ef, tf in [(1.0, 1.0), (0.97, 1.03), (1.03, 0.97)]:
                pe = pe71.copy(); pt = pt71.copy()
                pe[0] += math.log(ef); pt[0] += math.log(tf)
                starts.append(pack(pp[0], pp[1], d, pe, pt))

    rows = []
    for s in starts:
        o = least_squares(
            residual, s, bounds=(lo, hi), loss='soft_l1', f_scale=1.2,
            x_scale='jac', max_nfev=2200,
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
    pp_separation = state_separation(xb)

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
            'half_separation_radius_px': float(x[2]),
        })

    event_held = {k: v70.pstats(ete[k], k, peb, False, C) for k in ete}
    event_held['rim'] = v70.pstats(erte, 'rim', peb, False, C)
    target_held = {k: v70.pstats(tte[k], k, ptb, False, C) for k in tte}

    rng = np.random.default_rng(7312)
    pert = []
    for _ in range(64):
        ed = {k: v + rng.uniform(-0.5, 0.5, v.shape) for k, v in etr.items()}
        er = ertr + rng.uniform(-0.5, 0.5, ertr.shape)
        td = {k: v + rng.uniform(-0.5, 0.5, v.shape) for k, v in ttr.items()}
        o = least_squares(
            lambda x: residual(x, ed, er, td), xb, bounds=(lo, hi),
            loss='soft_l1', f_scale=1.2, x_scale='jac', max_nfev=1200,
        )
        pe, pt = unpack(o.x)
        d = v70.dstat(target_nominal, v70.project_fixed(pt, C, v70.ACTION)[0])
        d['max_target_heldout_p95_px'] = float(max(v70.pstats(tte[k], k, pt, False, C)['p95_px'] for k in tte))
        d['shared_base_pp_shift_px'] = float(np.linalg.norm(o.x[:2] - xb[:2]))
        d['principal_point_separation_px'] = state_separation(o.x)
        d['half_separation_radius_px'] = float(o.x[2])
        pert.append(d)

    max_comp_p95 = float(max(x['p95_px'] for x in competitive))
    max_pert_p95 = float(max(x['p95_px'] for x in pert))
    max_pert_held = float(max(x['max_target_heldout_p95_px'] for x in pert))
    max_base_shift = float(max(x['shared_base_pp_shift_px'] for x in pert))
    max_pert_pp_sep = float(max(x['principal_point_separation_px'] for x in pert))
    max_pert_rho = float(max(x['half_separation_radius_px'] for x in pert))

    gates = {
        'v70_physical_centre_evidence_intact': True,
        'v71_target_identifiability_breakthrough_intact': True,
        'v72_failure_surface_reproduced_exactly': True,
        'immutable_event225_and_frame_c': True,
        'frame_c_source_clip_proven_game_0022500301_event_489': True,
        'radial_pp_model_enforces_full_state_separation_at_most_5px': xb[2] <= MAX_HALF_SEPARATION_PX + NUMERIC_EPS_PX and pp_separation <= MAX_STATE_PP_SEPARATION_PX + NUMERIC_EPS_PX,
        'radial_pp_event225_heldout_floor_p95_at_most_2px': max(event_held[k]['p95_px'] for k in ['left', 'right', 'ft', 'restricted']) <= 2.0,
        'radial_pp_event225_rim_p95_at_most_2px': event_held['rim']['p95_px'] <= 2.0,
        'radial_pp_frame_c_heldout_p95_at_most_2px': max(x['p95_px'] for x in target_held.values()) <= 2.0,
        'competitive_roots_frame_c_action_p95_at_most_0_5px': max_comp_p95 <= 0.5,
        'half_pixel_frame_c_action_p95_at_most_2px': max_pert_p95 <= 2.0,
        'half_pixel_frame_c_heldout_p95_at_most_2_5px': max_pert_held <= 2.5,
        'half_pixel_shared_base_pp_shift_at_most_5px': max_base_shift <= 5.0,
        'half_pixel_state_pp_separation_at_most_5px': max_pert_rho <= MAX_HALF_SEPARATION_PX + NUMERIC_EPS_PX and max_pert_pp_sep <= MAX_STATE_PP_SEPARATION_PX + NUMERIC_EPS_PX,
    }
    passed = all(gates.values())
    status = 'PASS_RIGHT_ABOVE_RIM_RADIAL_PP_V73' if passed else 'FAIL_RIGHT_ABOVE_RIM_RADIAL_PP_V73'
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
        'frame_c_source_clip_note': 'The immutable synchronized Frame C was extracted from the official multi-angle clip package addressed as event 489; event 486 is the intended basketball block represented by the replay frame.',
        'physical_center_cm': C.tolist(),
        'baseline_to_left_above_rim_cm': q70['baseline_to_left_above_rim_cm'],
        'model': 'fixed physical centre; common sensor principal-point base plus radial half-separation rho<=2.5px so full inter-state principal-point separation is mathematically <=5px; independent focal length and orientation per PTZ optical state',
        'v72_parameterization_diagnosis': 'v72 bounded dx and dy independently to +/-2px, which permits 5.657px Euclidean full state separation. v73 encodes the unchanged 5px gate directly as a radial half-separation bound.',
        'max_full_state_principal_point_separation_px': MAX_STATE_PP_SEPARATION_PX,
        'shared_base_principal_point_px': xb[:2].tolist(),
        'half_separation_radius_px': float(xb[2]),
        'half_separation_angle_rad': float(xb[3]),
        'event225': {
            'principal_point_px': ppeb.tolist(),
            'focal_px': float(np.exp(peb[0])),
            'rvec': peb[3:6].tolist(),
            'heldout': event_held,
        },
        'target_frame_c': {
            'principal_point_px': pptb.tolist(),
            'focal_px': float(np.exp(ptb[0])),
            'rvec': ptb[3:6].tolist(),
            'heldout': target_held,
            'competitive_roots': competitive,
            'perturbation_64': {
                'max_action_p95_px': max_pert_p95,
                'max_action_max_px': float(max(x['max_px'] for x in pert)),
                'max_heldout_p95_px': max_pert_held,
                'max_shared_base_pp_shift_px': max_base_shift,
                'max_state_pp_separation_px': max_pert_pp_sep,
                'max_half_separation_radius_px': max_pert_rho,
            },
        },
        'nominal_state_principal_point_separation_px': pp_separation,
        'independent_v70_principal_points_px': {'event225': pp70e.tolist(), 'target_frame_c': pp70t.tolist()},
        'v71_shared_principal_point_px': q71['shared_principal_point_px'],
        'v72_failed_gates': sorted(expected_v72_false),
        'gates': gates,
        'permissions': permissions,
        'next_gate': 'If PASS, Right Above Rim is formally promoted as camera #2. Static novel-view rendering still remains gated until >=4 distinct metric cameras and exact visual-state alignment are proven.',
    }

    (a.out / 'right_above_rim_radial_pp_v73.json').write_text(json.dumps(report, indent=2) + '\n')
    v70.draw(ie, a.out / 'right_above_rim_event225_overlay_v73.png', peb, C=C, full=False, held={**ete, 'rim': erte})
    v70.draw(it, a.out / 'right_above_rim_frame_c_overlay_v73.png', ptb, C=C, full=False, held=tte)
    print(json.dumps({
        'status': status,
        'shared_base_principal_point_px': xb[:2].tolist(),
        'half_separation_radius_px': float(xb[2]),
        'event225_principal_point_px': ppeb.tolist(),
        'frame_c_principal_point_px': pptb.tolist(),
        'nominal_state_pp_separation_px': pp_separation,
        'event225_heldout': event_held,
        'target_heldout': target_held,
        'competitive_root_count': len(competitive),
        'max_competitive_target_action_p95_px': max_comp_p95,
        'target_perturbation': report['target_frame_c']['perturbation_64'],
        'gates': gates,
        'permissions': permissions,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
