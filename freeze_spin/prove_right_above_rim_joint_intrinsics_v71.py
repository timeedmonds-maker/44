from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin import prove_right_above_rim_physical_event_v70 as v70


def pack_shared(cx, cy, pe, pt):
    return np.r_[cx, cy, pe[0], pe[3:6], pt[0], pt[3:6]].astype(float)


def unpack_shared(x):
    cx, cy = x[:2]
    pe = np.r_[x[2], cx, cy, x[3:6]]
    pt = np.r_[x[6], cx, cy, x[7:10]]
    return pe, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v70-report', type=Path, required=True)
    ap.add_argument('--event225-frame', type=Path, required=True)
    ap.add_argument('--target-frame', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    q70 = json.loads(a.v70_report.read_text())
    if q70.get('status') != 'FAIL_RIGHT_ABOVE_RIM_PHYSICAL_EVENT_V70':
        raise RuntimeError('v71 expects the exact v70 one-gate-fail diagnostic report')
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
        pe, pt = unpack_shared(x)
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

    pe70 = np.r_[
        math.log(q70['event225']['focal_px']),
        q70['event225']['principal_point_px'],
        q70['event225']['rvec'],
    ]
    pt70 = np.r_[
        math.log(q70['target_frame_c']['focal_px']),
        q70['target_frame_c']['principal_point_px'],
        q70['target_frame_c']['rvec'],
    ]
    pp_e = np.array(q70['event225']['principal_point_px'], float)
    pp_t = np.array(q70['target_frame_c']['principal_point_px'], float)
    pp_mid = (pp_e + pp_t) / 2.0

    lo = np.r_[100.0, 50.0, math.log(150), [-10.0] * 3, math.log(150), [-10.0] * 3]
    hi = np.r_[850.0, 520.0, math.log(2500), [10.0] * 3, math.log(2500), [10.0] * 3]

    starts = []
    pp_starts = [pp_mid, pp_e, pp_t, np.array([480.0, 270.0])]
    for pp in pp_starts:
        for ef in [0.92, 1.0, 1.08]:
            for tf in [0.92, 1.0, 1.08]:
                pe = pe70.copy(); pt = pt70.copy()
                pe[0] += math.log(ef); pt[0] += math.log(tf)
                starts.append(pack_shared(pp[0], pp[1], pe, pt))

    rows = []
    for s in starts:
        o = least_squares(
            residual, s, bounds=(lo, hi), loss='soft_l1', f_scale=1.2,
            x_scale='jac', max_nfev=2200,
        )
        pe, pt = unpack_shared(o.x)
        _, ze = v70.project_fixed(pe, C, v70.ACTION)
        _, zt = v70.project_fixed(pt, C, v70.ACTION)
        valid = (np.all(ze > 0) or np.all(ze < 0)) and (np.all(zt > 0) or np.all(zt < 0))
        if valid:
            rows.append((float(o.cost), o.x))
    if not rows:
        raise RuntimeError('no physically valid joint-intrinsics roots')
    rows.sort(key=lambda z: z[0])
    best_cost, xb = rows[0]
    peb, ptb = unpack_shared(xb)

    target_nominal = v70.project_fixed(ptb, C, v70.ACTION)[0]
    competitive = []
    for cost, x in rows:
        if cost > best_cost + 1.0:
            break
        _, pt = unpack_shared(x)
        competitive.append({'cost': cost, **v70.dstat(target_nominal, v70.project_fixed(pt, C, v70.ACTION)[0])})

    event_held = {k: v70.pstats(ete[k], k, peb, False, C) for k in ete}
    event_held['rim'] = v70.pstats(erte, 'rim', peb, False, C)
    target_held = {k: v70.pstats(tte[k], k, ptb, False, C) for k in tte}

    rng = np.random.default_rng(7111)
    pert = []
    for _ in range(64):
        ed = {k: v + rng.uniform(-0.5, 0.5, v.shape) for k, v in etr.items()}
        er = ertr + rng.uniform(-0.5, 0.5, ertr.shape)
        td = {k: v + rng.uniform(-0.5, 0.5, v.shape) for k, v in ttr.items()}
        o = least_squares(
            lambda x: residual(x, ed, er, td), xb, bounds=(lo, hi),
            loss='soft_l1', f_scale=1.2, x_scale='jac', max_nfev=1200,
        )
        pe, pt = unpack_shared(o.x)
        d = v70.dstat(target_nominal, v70.project_fixed(pt, C, v70.ACTION)[0])
        d['max_target_heldout_p95_px'] = float(max(v70.pstats(tte[k], k, pt, False, C)['p95_px'] for k in tte))
        d['shared_pp_shift_px'] = float(np.linalg.norm(o.x[:2] - xb[:2]))
        pert.append(d)

    max_comp_p95 = float(max(x['p95_px'] for x in competitive))
    max_pert_p95 = float(max(x['p95_px'] for x in pert))
    max_pert_held = float(max(x['max_target_heldout_p95_px'] for x in pert))
    max_pp_shift = float(max(x['shared_pp_shift_px'] for x in pert))

    gates = {
        'v70_physical_centre_evidence_intact': True,
        'immutable_event225_and_frame_c': True,
        'shared_principal_point_model_fits_event225_heldout_floor_p95_at_most_2px': max(event_held[k]['p95_px'] for k in ['left','right','ft','restricted']) <= 2.0,
        'shared_principal_point_model_fits_event225_rim_p95_at_most_2px': event_held['rim']['p95_px'] <= 2.0,
        'shared_principal_point_model_fits_frame_c_heldout_p95_at_most_2px': max(x['p95_px'] for x in target_held.values()) <= 2.0,
        'joint_competitive_roots_frame_c_action_p95_at_most_0_5px': max_comp_p95 <= 0.5,
        'joint_half_pixel_frame_c_action_p95_at_most_2px': max_pert_p95 <= 2.0,
        'joint_half_pixel_frame_c_heldout_p95_at_most_2_5px': max_pert_held <= 2.5,
        'joint_half_pixel_shared_pp_shift_at_most_5px': max_pp_shift <= 5.0,
        'correct_target_provenance_adams_block_game_0022500301_event_486': True,
    }
    passed = all(gates.values())
    status = 'PASS_RIGHT_ABOVE_RIM_JOINT_INTRINSICS_V71' if passed else 'FAIL_RIGHT_ABOVE_RIM_JOINT_INTRINSICS_V71'
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
        'target_event': 486,
        'target_clock': 'Q3 04:11.00',
        'target_description': 'MISS B. Sensabaugh driving DUNK - blocked by Steven Adams',
        'physical_center_cm': C.tolist(),
        'baseline_to_left_above_rim_cm': q70['baseline_to_left_above_rim_cm'],
        'model': 'fixed physical centre; shared principal point across two PTZ optical states; independent focal length and orientation per state',
        'shared_principal_point_px': xb[:2].tolist(),
        'event225': {
            'focal_px': float(np.exp(peb[0])),
            'rvec': peb[3:6].tolist(),
            'heldout': event_held,
        },
        'target_frame_c': {
            'focal_px': float(np.exp(ptb[0])),
            'rvec': ptb[3:6].tolist(),
            'heldout': target_held,
            'competitive_roots': competitive,
            'perturbation_64': {
                'max_action_p95_px': max_pert_p95,
                'max_action_max_px': float(max(x['max_px'] for x in pert)),
                'max_heldout_p95_px': max_pert_held,
                'max_shared_pp_shift_px': max_pp_shift,
            },
        },
        'independent_v70_principal_points_px': {'event225': pp_e.tolist(), 'target_frame_c': pp_t.tolist()},
        'gates': gates,
        'permissions': permissions,
        'next_gate': 'If PASS, Right Above Rim is formally promoted as camera #2. Static novel-view rendering remains separately gated.',
    }

    (a.out / 'right_above_rim_joint_intrinsics_v71.json').write_text(json.dumps(report, indent=2))
    v70.draw(ie, a.out / 'right_above_rim_event225_joint_overlay_v71.png', peb, C=C, full=False, held={**ete, 'rim': erte})
    v70.draw(it, a.out / 'right_above_rim_frame_c_joint_overlay_v71.png', ptb, C=C, full=False, held=tte)
    print(json.dumps({
        'status': status,
        'shared_principal_point_px': xb[:2].tolist(),
        'event225_heldout': event_held,
        'target_heldout': target_held,
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
