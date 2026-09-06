from __future__ import annotations

"""v106a: fast broad-root audit of the exact v106 event416 metric model.

This does not alter any v106 observation or geometry. It caps each optimizer run
so we can quickly determine whether widely separated physical starts collapse
onto the same useful basin while the deeper v106 sweep continues.
"""

import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from freeze_spin import solve_right_slash_event416_metric_v106 as v106


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frame',type=Path,required=True); ap.add_argument('--v105',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if v106.sha256(a.frame)!=v106.FRAME_SHA: raise RuntimeError('immutable event416 f06 SHA mismatch')
    auth=json.loads(a.v105.read_text())
    if auth.get('status')!='PASS_RIGHT_SLASH_FIXED_CENTER_AUTHORIZATION_V105': raise RuntimeError('v105 prerequisite missing')
    frame=cv2.imread(str(a.frame)); rim=v106.rim_observations(frame)
    train={}; held={}
    for key,arr,off in [
        ('target_top',v106.TARGET_OBS['target_top'],0),('target_left',v106.TARGET_OBS['target_left'],1),('target_right',v106.TARGET_OBS['target_right'],2),
        ('free_throw_line',v106.FT_LINE_OBS,3),('free_throw_front',v106.FT_FRONT_OBS,0),('restricted_arc',v106.RESTRICT_OBS,1),('rim',rim,2)]:
        train[key],held[key]=v106.split_obs(np.asarray(arr,float),off)
    lo=np.r_[[-5000.,-5000.,50.],math.log(150.),-2000.,-1500.,[-10.]*3]
    hi=np.r_[[5000.,5000.,2000.],math.log(5000.),3000.,2000.,[10.]*3]
    specs=[([1000,-1400,220],700),([1000,1400,220],900),([1600,-600,260],1000),([1600,600,260],1400),([2200,-1000,380],1300),([2200,1000,380],1700),([1500,0,650],900),([2800,0,650],1800)]
    roots=[]
    for i,(C0,f0) in enumerate(specs):
        C0=np.asarray(C0,float); s=np.r_[C0,math.log(float(f0)),480.,270.,v106.lookat_rvec(C0)]
        try:
            o=least_squares(lambda p:v106.residual(p,train),s,bounds=(lo,hi),loss='soft_l1',f_scale=1.2,x_scale='jac',max_nfev=1400)
            p=np.asarray(o.x,float)
            _,depth=v106.project(p,np.vstack([v106.FT_CURVE[::60],v106.RESTRICT_CURVE[::60],v106.RIM_CURVE[::60],np.vstack(list(v106.TARGET_WORLD.values()))]))
            physical=bool(np.isfinite(p).all() and np.all(depth>20.))
            row={'index':i,'cost':float(o.cost),'physical':physical}
            if physical:
                row.update(center_cm=p[:3].tolist(),focal_px=float(np.exp(p[3])),principal_point_px=p[4:6].tolist(),params=p.tolist(),median_abs_train_residual_px=float(np.median(np.abs(v106.residual(p,train)))))
            roots.append(row)
            print('V106A ROOT',i,'physical',physical,'cost',round(float(o.cost),3),'center',np.round(p[:3],2).tolist(),flush=True)
        except Exception as e:
            roots.append({'index':i,'error':repr(e)}); print('V106A ROOT',i,'ERROR',repr(e),flush=True)
    physical=[r for r in roots if r.get('physical')]
    if not physical: raise RuntimeError('no physical coarse roots')
    physical.sort(key=lambda r:r['cost']); best=physical[0]; pb=np.asarray(best['params'],float); ref=v106.dense_action_projection(pb)
    comp=[]
    for r in physical:
        p=np.asarray(r['params'],float); d=np.linalg.norm(v106.dense_action_projection(p)-ref,axis=1)
        comp.append({'index':r['index'],'cost':r['cost'],'center_cm':r['center_cm'],'center_shift_cm':float(np.linalg.norm(p[:3]-pb[:3])),'action_projection_p95_shift_px':float(np.percentile(d,95))})
    heldm={
        'target_top':v106.metric_line(pb,'target_top',held['target_top']),'target_left':v106.metric_line(pb,'target_left',held['target_left']),'target_right':v106.metric_line(pb,'target_right',held['target_right']),
        'free_throw_line':v106.metric_line(pb,'free_throw_line',held['free_throw_line']),'free_throw_front':v106.metric_curve(pb,v106.FT_CURVE,held['free_throw_front']),
        'restricted_arc':v106.metric_curve(pb,v106.RESTRICT_CURVE,held['restricted_arc']),'rim':v106.metric_curve(pb,v106.RIM_CURVE,held['rim'])}
    v106.draw_overlay(frame,pb,a.out/'right_slash_event416_metric_overlay_v106a.png',held)
    report={'schema_version':1,'status':'RIGHT_SLASH_EVENT416_COARSE_ROOT_AUDIT_V106A','best_root':best,'roots':roots,'competitive_root_comparison':comp,'heldout_metrics':heldm,
            'guardrails':['identical v106 observations and regulation model','coarse optimizer cap only','discovery-only','no metric camera or replay promotion'],
            'permissions':{'shared_center_metric_attempt_allowed':True,'right_slash_metric_camera_allowed':False,'replay_render_allowed':False}}
    (a.out/'right_slash_event416_metric_v106a.json').write_text(json.dumps(report,indent=2)+'\n')
    print('V106A BEST',np.round(pb[:3],3).tolist(),'f',round(float(np.exp(pb[3])),3),'pp',np.round(pb[4:6],3).tolist(),flush=True)
    print('V106A HELDOUT',{k:round(v['p95_px'],3) for k,v in heldm.items()},flush=True)

if __name__=='__main__': main()
