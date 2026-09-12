from __future__ import annotations

"""v32n: v32m with a hard focal action-region identity lock.

v32m exposed one specific failure: at RAR t-02 the appearance score selected a
distant dark-uniform player ~312 px from the known Adams action region, while a
17-joint candidate only ~19 px from the B32 focal region was available.  v32n
keeps every v32m geometry/tracking/consensus gate unchanged and strengthens only
identity selection: candidates more than 125 px from the B32 focal-region centre
are ineligible unless no local candidate exists.
"""

import json
from pathlib import Path
import numpy as np
from freeze_spin import solve_v32m_bidirectional_temporal_pose as base


def spatially_locked_choose(img,dets,target):
    rows=[]
    for i,d in enumerate(dets):
        c=np.array([(d['box'][0]+d['box'][2])/2,(d['box'][1]+d['box'][3])/2])
        dist=float(np.linalg.norm(c-target))
        dark=base.v32j.dark_fraction(img,d['box'])
        td=base.torso_dark(img,d)
        n=int(np.sum(d['conf']>=.2))
        score=2.7*dark+2.3*td+.35*d['det_conf']+.035*n-.006*dist
        rows.append({'index':i,'score':float(score),'dark_fraction':float(dark),
                     'torso_dark':float(td),'confident_joints':n,'center_distance_px':dist,
                     'inside_focal_radius_125px':bool(dist<=125.0)})
    rows.sort(key=lambda x:x['score'],reverse=True)
    local=[r for r in rows if r['inside_focal_radius_125px']]
    chosen=(local[0] if local else rows[0])
    return chosen['index'],rows


def main():
    base.choose=spatially_locked_choose
    rc=0
    try:
        base.main()
    except SystemExit as e:
        rc=int(e.code or 0)
    # Preserve base evidence but add the stronger v32n provenance explicitly.
    try:
        oi=__import__('sys').argv.index('--out')
        out=Path(__import__('sys').argv[oi+1])
        p=out/'v32m_bidirectional_temporal_qa.json'
        if p.exists():
            q=json.loads(p.read_text())
            q['v32n_spatial_identity_lock']={
                'hard_focal_radius_px':125.0,
                'reason':'v32m t-02 selected a distant dark player despite a local 17-joint candidate',
                'geometry_changed':False,
                'tracking_changed':False,
                'consensus_thresholds_changed':False,
                'visual_QA_still_required':True,
            }
            p.write_text(json.dumps(q,indent=2))
            (out/'v32n_spatially_locked_qa.json').write_text(json.dumps(q,indent=2))
    finally:
        if rc: raise SystemExit(rc)

if __name__=='__main__': main()
