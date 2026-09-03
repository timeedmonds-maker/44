from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from freeze_spin.prove_right_above_rim_lane_camera_v48 import (
    action_volume,
    deterministic_starts,
    functional_p95,
    optimize,
    summarize,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--observations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.observations.read_text(encoding='utf-8'))
    rim = np.asarray(spec['rim_inner_edge_samples_px'], dtype=np.float64)
    restricted = np.asarray(spec['restricted_area_centerline_samples_px'], dtype=np.float64)
    left = np.asarray(spec['left_lane_sideline_samples_px'], dtype=np.float64)
    right = np.asarray(spec['right_lane_sideline_samples_px'], dtype=np.float64)
    volume = action_volume()

    def one(item):
        i, p0 = item
        p, cost = optimize(p0, rim, restricted, left, right, max_nfev=900)
        qa = summarize(p, rim, restricted, left, right)
        return {'index': i, 'p': p, 'cost': cost, 'qa': qa}

    with ThreadPoolExecutor(max_workers=4) as pool:
        roots = list(pool.map(one, list(enumerate(deterministic_starts()))))
    roots.sort(key=lambda r: (not r['qa']['plausible'], r['qa']['combined_anchor_rms_px'], r['cost']))
    base = roots[0]
    competitive = [
        r for r in roots
        if r['qa']['plausible']
        and r['qa']['combined_anchor_rms_px'] <= base['qa']['combined_anchor_rms_px'] + 0.35
        and r['qa']['combined_anchor_p95_px'] <= float(spec['gates']['max_combined_anchor_p95_px'])
    ]
    pairwise = []
    max_shift = 0.0
    for i in range(len(competitive)):
        for j in range(i + 1, len(competitive)):
            s = functional_p95(competitive[i]['p'], competitive[j]['p'], volume)
            pairwise.append({'a': competitive[i]['index'], 'b': competitive[j]['index'], 'p95_px': s})
            max_shift = max(max_shift, s)

    report = {
        'status': 'DIAGNOSTIC_ONLY_NO_PROMOTION',
        'base': base['qa'],
        'root_summaries': [
            {
                'index': r['index'],
                'combined_anchor_rms_px': r['qa']['combined_anchor_rms_px'],
                'combined_anchor_p95_px': r['qa']['combined_anchor_p95_px'],
                'camera_center_world_cm': r['qa']['camera_center_world_cm'],
                'focal_px': r['qa']['focal_px'],
                'principal_point_px': r['qa']['principal_point_px'],
            }
            for r in roots
        ],
        'competitive_root_count': len(competitive),
        'max_competitive_pairwise_action_volume_p95_shift_px': max_shift,
        'pairwise': pairwise,
        'lane_anchors_appear_to_resolve_v47_rotational_ambiguity': bool(len(competitive) >= 3 and max_shift <= 0.5),
        'permissions': {
            'camera_promotion_allowed': False,
            'static_novel_view_allowed': False,
            'replay_render_allowed': False,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'right_above_rim_lane_roots_v48a.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
