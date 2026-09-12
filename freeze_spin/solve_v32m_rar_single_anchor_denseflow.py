from __future__ import annotations

"""v32m: single clean RAR anchor + dense temporal tracking.

v32l proved dense flow survives t+00, but multi-anchor COCO labels disagree on
limbs (including left/right swaps), so consensus collapses to face points.
v32m deliberately chooses the clearest Adams RF-DETR pose at real RAR t+04,
tracks that single coherent articulated pose backward through real frames, then
uses Broadcast and Left as independent geometric validation.  No rendering or
pixel synthesis occurs.
"""

import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l


def single_anchor_consensus(track_rows):
    out, audit = {}, {}
    if not track_rows:
        return out, audit
    row = track_rows[0]
    for j in range(17):
        ok = bool(row['valid'][j])
        audit[j] = {
            'accepted': ok,
            'support': 1 if ok else 0,
            'anchors': [int(row['rel'])] if ok else [],
            'median_xy': row['t0_xy'][j].tolist() if ok else None,
            'single_anchor_primary': True,
        }
        if ok:
            out[j] = np.asarray(row['t0_xy'][j], float)
    return out, audit


def main():
    v32k.ANCHOR_RELS = (4,)
    v32k.track_back = v32l.dense_track_back
    v32k.consensus_tracks = single_anchor_consensus

    base_code = 0
    try:
        v32k.main()
    except SystemExit as e:
        base_code = int(e.code or 0)

    oi = sys.argv.index('--out')
    out = Path(sys.argv[oi + 1])
    p = out / 'v32k_temporal_deblend_qa.json'
    if not p.exists():
        raise SystemExit(base_code or 9)
    q = json.loads(p.read_text())

    a = q.get('anchor_audit', {}).get('4', {})
    ranked = a.get('ranked_candidates', [])
    top = ranked[0] if ranked else {}
    left = q.get('left_best_validation', {})
    gates = {
        't04_identity_dark_fraction_ge_0_50': float(top.get('dark_fraction', 0)) >= 0.50,
        't04_identity_torso_dark_ge_0_55': float(top.get('torso_dark_score', 0)) >= 0.55,
        't04_identity_center_distance_le_90px': float(top.get('center_distance_px', 999)) <= 90.0,
        'tracked_t0_joints_ge_10': int(q.get('consensus_joint_count', 0)) >= 10,
        'triangulated_joints_ge_9': int(q.get('triangulated_joint_count', 0)) >= 9,
        'measured_bones_ge_6': int(q.get('measured_bone_count', 0)) >= 6,
        'bone_plausible_fraction_ge_0_70': float(q.get('bone_plausible_fraction', 0)) >= 0.70,
        'median_two_view_reprojection_le_20px': float(q.get('median_two_view_reprojection_px', 999)) <= 20.0,
        'left_validation_joints_ge_6': int(left.get('joint_count', 0)) >= 6,
        'left_validation_median_le_25px': float(left.get('median_px', 999)) <= 25.0,
        'left_validation_p75_le_40px': float(left.get('p75_px', 999)) <= 40.0,
    }
    passed = all(gates.values())
    q['version'] = 'v32m_rar_single_anchor_denseflow'
    q['status'] = 'PASS_V32M_SINGLE_ANCHOR_ARTICULATED_POSITION' if passed else 'FAIL_CLOSED_V32M_SINGLE_ANCHOR_ARTICULATED_POSITION'
    q['temporal_method'] = 'RF-DETR Adams pose at real RAR t+04; DIS dense bidirectional tracking to t+00; Broadcast triangulation; Left independent validation'
    q['v32m_strict_gate'] = gates
    q['surface_stage_unlocked'] = bool(passed)
    (out / 'v32m_single_anchor_qa.json').write_text(json.dumps(q, indent=2))
    print(json.dumps({'status': q['status'], 'gates': gates, 'base_status': q.get('gate', {})}, indent=2), flush=True)
    if not passed:
        raise SystemExit(4)


if __name__ == '__main__':
    main()
