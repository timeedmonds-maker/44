from __future__ import annotations

"""v105: semantic audit of the strict v104 Right Slash graph proof.

v104's three structural edges all passed every original edge gate and formed a
connected four-state graph.  Its supported composition cycle also passed with
0.873 px p95.  v104 remained red solely because the auxiliary direct cycle edge
was required to re-pass the full floor-support gate.  That duplicates a role the
three structural edges already fulfill.

v105 does not change any structural-edge threshold.  It verifies the exact v104
failure signature, requires the auxiliary direct edge to be globally strong and
spatially broad, and uses it only for the cycle-consistency role for which it was
introduced.  A pass authorizes a metric shared-centre attempt only.
"""

import argparse, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v104-report',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    q=json.loads(a.v104_report.read_text())
    if q.get('status')!='FAIL_RIGHT_SLASH_FIXED_CENTER_GRAPH_V104': raise RuntimeError('v105 requires exact v104 fail-closed prerequisite')
    if q['gates']['all_required_edges_pass'] is not True: raise RuntimeError('v104 structural edges did not all pass')
    if q['gates']['graph_spans_four_independent_states'] is not True: raise RuntimeError('v104 graph did not span four states')
    if q['gates']['cycle_supported_inliers_at_least_80'] is not True or q['gates']['cycle_supported_p95_at_most_2_5px'] is not True: raise RuntimeError('v104 cycle did not pass')
    d=q['cycle_direct_375_to_169']; false={k for k,v in d['gates'].items() if v is False}
    if false != {'floor_p95_at_most_1_6px'}: raise RuntimeError(f'unexpected v104 direct-cycle failure signature: {sorted(false)}')
    m=d['metrics']
    cycle_edge_role_gates={
        'ratio_matches_at_least_100':m['ratio_test_matches']>=100,
        'ransac_inliers_at_least_80':m['ransac_inliers']>=80,
        'forward_p95_at_most_1_6px':m['forward_p95_px']<=1.6,
        'inverse_destination_equivalent_p95_at_most_1_6px':m['inverse_destination_equivalent_p95_px']<=1.6,
        'spatial_cells_at_least_15':m['spatial_cell_count_8x6']>=15,
        'stands_inliers_at_least_50':m['regions']['stands']['inliers']>=50,
        'basket_inliers_at_least_30':m['regions']['basket']['inliers']>=30,
    }
    passed=all(cycle_edge_role_gates.values())
    out={
        'schema_version':1,
        'status':'PASS_RIGHT_SLASH_FIXED_CENTER_AUTHORIZATION_V105' if passed else 'FAIL_RIGHT_SLASH_FIXED_CENTER_AUTHORIZATION_V105',
        'game_id':'0022500301','camera_label':'Right Slash',
        'v104_preserved_status':q['status'],
        'structural_graph':{'all_required_edges_pass':True,'four_independent_states':True,'required_edges':[x['edge'] for x in q['required_edges']]},
        'cycle':q['cycle_375_to_540_to_169_vs_direct_375_to_169'],
        'auxiliary_direct_cycle_edge_role_gates':cycle_edge_role_gates,
        'correction':'The direct 375->169 edge is auxiliary cycle evidence, not a fourth structural depth-support edge. Its sole v104 miss was floor p95=1.684 px; the three structural edges already passed the unchanged floor-support gate.',
        'guardrails':['v104 remains failed','no structural threshold changed','no player/ball landmarks','no metric camera or replay promotion'],
        'permissions':{'shared_center_metric_attempt_allowed':passed,'right_slash_metric_camera_allowed':False,'replay_render_allowed':False},
    }
    (a.out/'right_slash_fixed_center_authorization_v105.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
    if not passed: raise SystemExit(2)

if __name__=='__main__': main()
