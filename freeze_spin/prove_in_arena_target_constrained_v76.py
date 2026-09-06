from __future__ import annotations
"""v76: fail-closed In-Arena metric camera using new target-plane evidence."""
import argparse,json,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from freeze_spin import solve_in_arena_joint_selfcal_v56a as b
from freeze_spin import solve_in_arena_direct_camera_v55a as d
from freeze_spin.prove_in_arena_noncoplanar_center_v54 import RIM_CENTER,TARGET_RIM_SEED,extract_rim_ellipse,json_safe,project_point,refine_transfer,sha256
from freeze_spin.solve_nba_geometry_proof_v3 import world_landmarks
W,H=960,540

def solve(start,keys,lines,restricted,rims,transfers,obj,obs,weight,nfev):
    n=len(keys)-1; lo,hi=d.bounds()
    lower=np.r_[lo[3:6],lo[:3],lo[6:9],np.tile([-10,-10,-10,math.log(300),-60,-60],n)]
    upper=np.r_[hi[3:6],hi[:3],hi[6:9],np.tile([10,10,10,math.log(9000),60,60],n)]
    start=np.clip(np.asarray(start,float),lower+1e-6,upper-1e-6)
    def residual(x):
        states,blocks=b.unpack(x,keys); target=states['target']
        le,_=d.line_distances(target,lines)
        re=d.circle_signed_distances(target,restricted,RIM_CENTER[0],d.RESTRICTED_RADIUS)
        ruv,rd=d.project(target,RIM_CENTER[None,:]); tuv,td=d.project(target,obj)
        out=[le,re,(ruv[0]-rims['target'])*2,(tuv-obs).ravel()*weight,
             (target[7:9]-[W/2,H/2])/350,[(target[6]-math.log(3000))/3],
             np.minimum(rd-20,0).ravel()*5,np.minimum(td-20,0).ravel()*5]
        for i,key in enumerate(keys[1:]):
            state=states[key]; suv,sd=d.project(state,RIM_CENTER[None,:]); p,q=transfers[key]
            pred=b.transfer_points(b.relative_homography(target,state),p)
            out += [(pred-q).ravel()*.5,(suv[0]-rims[key])*2,blocks[i,4:6]/25,np.minimum(sd-20,0).ravel()*5]
        return np.concatenate(out)
    r=least_squares(residual,start,bounds=(lower,upper),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=nfev)
    states,blocks=b.unpack(r.x,keys)
    return {'x':r.x,'states':states,'blocks':blocks,'cost':float(r.cost),'nfev':int(r.nfev)}

def angle(points):
    e=cv2.fitEllipse(np.asarray(points,np.float32)); a0,a1=e[1]; q=float(e[2]) if a0>=a1 else float((e[2]+90)%180)
    return q-180 if q>=90 else q

def summarize(s,keys,lines,restricted,ft,rims,ellipses,transfers,obj,obs):
    q=b.summarize(s,keys,lines,restricted,ft,rims,ellipses,transfers); p=s['states']['target']
    uv,_=d.project(p,obj); er=np.linalg.norm(uv-obs,axis=1)
    q['target_corners']={'observed_px':obs.tolist(),'predicted_px':uv.tolist(),'per_corner_error_px':er.tolist(),'rmse_px':float(np.sqrt(np.mean(er**2))),'max_px':float(er.max())}
    ruv,_=d.project(p,d.rim_circle()); q['rim_angle_holdout_error_deg']=abs(float((angle(ruv)-ellipses['target']['canonical_angle_deg']+90)%180-90))
    q['camera_distance_from_basket_cm']=float(np.linalg.norm(np.asarray(q['camera_center_cm'])-RIM_CENTER)); return q

def overlay(image,p,lines,restricted,ft,rimpts,obj,obs,allobj,path):
    out=image.copy()
    for name,pts in lines.items():
        uv,_=d.project(p,d.line_world(name)); cv2.line(out,tuple(np.round(uv[0]).astype(int)),tuple(np.round(uv[1]).astype(int)),(0,255,255),2,cv2.LINE_AA)
        for x in pts: cv2.circle(out,tuple(np.round(x).astype(int)),4,(255,0,0),2,cv2.LINE_AA)
    for world,pix,c in ((d.floor_circle(RIM_CENTER[0],d.RESTRICTED_RADIUS),restricted,(0,255,0)),(d.floor_circle(d.FREE_THROW_X,d.FREE_THROW_RADIUS),ft,(0,0,255)),(d.rim_circle(),rimpts,(255,0,255))):
        uv,_=d.project(p,world); cv2.polylines(out,[np.round(uv).astype(np.int32)],True,c,2,cv2.LINE_AA)
        for x in np.round(pix).astype(int): cv2.circle(out,tuple(x),2,(255,255,255),-1,cv2.LINE_AA)
    poly,_=d.project(p,allobj); cv2.polylines(out,[np.round(poly).astype(np.int32)],True,(0,255,0),2,cv2.LINE_AA)
    pred,_=d.project(p,obj)
    for a,z in zip(obs,pred):
        cv2.circle(out,tuple(np.round(a).astype(int)),5,(0,215,255),2,cv2.LINE_AA); cv2.circle(out,tuple(np.round(z).astype(int)),3,(0,255,0),-1,cv2.LINE_AA)
    cv2.imwrite(str(path),out)

def main():
    ap=argparse.ArgumentParser()
    for x in ('target','states','floor-proof','family-proof','floor-observations','target-observations','out'): ap.add_argument('--'+x,type=Path,required=True)
    ap.add_argument('--root-limit',type=int,default=9); ap.add_argument('--max-nfev',type=int,default=5000); ap.add_argument('--perturbation-trials',type=int,default=12); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(a.target)); floor=json.load(open(a.floor_proof)); fam=json.load(open(a.family_proof)); fs=json.load(open(a.floor_observations)); ts=json.load(open(a.target_observations)); t=ts['thresholds']
    if image is None or image.shape[:2]!=(H,W): raise RuntimeError('v76 target must be native 960x540')
    if floor.get('status')!='PASS_IN_ARENA_FLOOR_V52' or fam.get('status')!='PASS_IN_ARENA_STATIC_SCENE_FAMILY_V53B': raise RuntimeError('unsealed upstream evidence')
    expected=ts['freeze_lock']['sha256_png']
    if sha256(a.target)!=expected or fam['target_sha256_png']!=expected: raise RuntimeError('immutable target mismatch')
    names=list(ts['target_inner_corners_px']); world=world_landmarks(); obj=np.asarray([world[n] for n in names],float); obs=np.asarray([ts['target_inner_corners_px'][n] for n in names],float); allobj=np.asarray([world[n] for n in ('target_inner_top_left','target_inner_top_right','target_inner_bottom_right','target_inner_bottom_left')],float)
    lines={k:np.asarray(v,float) for k,v in fs['training_line_segments_px'].items()}; restricted=np.asarray(fs['heldout_curves_px']['restricted_area_arc'],float); ft=np.asarray(fs['heldout_curves_px']['free_throw_circle_dashed_half'],float)
    keys=['target']; transfers={}; rims={}; ellipses={}; diags={}
    c,axes,ang,rimpts,diag=extract_rim_ellipse(image,TARGET_RIM_SEED); rims['target']=c; ellipses['target']={'axes':axes,'angle':ang,'canonical_angle_deg':angle(rimpts)}; diags['target']=diag
    xfers={}
    for row in fam['selected_candidates']:
        key=f"event_{int(row['event_probe'])}"; state=cv2.imread(str(a.states/row['file']))
        if state is None or state.shape[:2]!=(H,W): raise RuntimeError('missing v53a state '+row['file'])
        h,p,q,xd=refine_transfer(image,state,b.norm_h(np.asarray(row['H_target_to_state'],float))); take=b.spatial_subset(p); p,q=p[take],q[take]; xfers[key]=h; transfers[key]=(p,q); keys.append(key)
        pred=project_point(h,TARGET_RIM_SEED); cc,ax,aa,_,dd=extract_rim_ellipse(state,pred); dd['transfer_refinement']=xd; rims[key]=cc; ellipses[key]={'axes':ax,'angle':aa,'canonical_angle_deg':0}; diags[key]=dd
    h0=b.norm_h(np.asarray(floor['floor_homography_world_to_image'],float)); seeds=[]
    for s0 in d.starts(h0):
        p,cost,_,_=d.optimize(s0,lines,restricted,rims['target'],max_nfev=1800); seeds.append((cost,p))
    seeds.sort(key=lambda z:z[0]); roots=[solve(b.make_start(p,keys,xfers),keys,lines,restricted,rims,transfers,obj,obs,float(ts['target_corner_weight']),a.max_nfev) for _,p in seeds[:a.root_limit]]; roots.sort(key=lambda z:z['cost'])
    sums=[summarize(r,keys,lines,restricted,ft,rims,ellipses,transfers,obj,obs) for r in roots]; best=roots[0]; qa=sums[0]; comp=[r for r in roots if r['cost']<=roots[0]['cost']*1.05+1e-8]; vol=b.action_volume(); pairs=[]
    for i in range(len(comp)):
        for j in range(i+1,len(comp)):
            p0,p1=comp[i]['states']['target'],comp[j]['states']['target']; pairs.append({'left':i,'right':j,'camera_center_shift_cm':float(np.linalg.norm(p0[3:6]-p1[3:6])),'action_volume_p95_px':d.functional_p95(p0,p1,vol)})
    center=np.asarray(qa['camera_center_cm']); pp=np.asarray(qa['target_principal_point_px']); axeserr=qa['rim_axis_error_major_minor_px']
    gates={
      'target_corner_rmse':qa['target_corners']['rmse_px']<=t['max_target_corner_rmse_px'],'target_corner_max':qa['target_corners']['max_px']<=t['max_target_corner_max_px'],'floor_line_p95':qa['line_p95_px']<=t['max_floor_line_p95_px'],'restricted_p95':qa['restricted_p95_px']<=t['max_restricted_p95_px'],'free_throw_holdout':qa['free_throw_holdout_p95_px']<=t['max_free_throw_holdout_p95_px'],'rim_center':qa['max_rim_center_error_px']<=t['max_rim_center_error_px'],'rim_major_holdout':axeserr[0]<=t['max_rim_major_axis_holdout_error_px'],'rim_minor_holdout':axeserr[1]<=t['max_rim_minor_axis_holdout_error_px'],'rim_angle_holdout':qa['rim_angle_holdout_error_deg']<=t['max_rim_angle_holdout_error_deg'],'rotation_homography':qa['rotation_homography_p95_px']<=t['max_rotation_homography_p95_px'],'principal_point_crop':-t['principal_point_crop_margin_px']<=pp[0]<=W+t['principal_point_crop_margin_px'] and -t['principal_point_crop_margin_px']<=pp[1]<=H+t['principal_point_crop_margin_px'],'camera_height':t['min_camera_height_cm']<=center[2]<=t['max_camera_height_cm'],'camera_outside_width':abs(center[1])>=t['min_abs_camera_y_cm'],'camera_distance':qa['camera_distance_from_basket_cm']<=t['max_camera_distance_from_basket_cm'],'minimum_roots':len(roots)>=t['minimum_multistart_root_count'],'root_center_spread':max((x['camera_center_shift_cm'] for x in pairs),default=0)<=t['max_competitive_root_center_shift_cm'],'root_action_volume':max((x['action_volume_p95_px'] for x in pairs),default=0)<=t['max_competitive_root_action_volume_p95_px']}
    leave=[]; support=[]; perturb=[]
    if all(gates.values()):
        def lj(i):
            keep=[j for j in range(len(obj)) if j!=i]; r=solve(best['x'],keys,lines,restricted,rims,transfers,obj[keep],obs[keep],float(ts['target_corner_weight']),1600); p=r['states']['target']; return {'dropped_corner':names[i],'camera_center_shift_cm':float(np.linalg.norm(p[3:6]-best['states']['target'][3:6])),'action_volume_p95_shift_px':d.functional_p95(best['states']['target'],p,vol)}
        with ThreadPoolExecutor(max_workers=3) as pool: leave=list(pool.map(lj,range(len(obj))))
        def sj(drop):
            ll={k:v for k,v in lines.items() if k!=drop}; rr=restricted[::2] if drop=='restricted' else restricted; r=solve(best['x'],keys,ll,rr,rims,transfers,obj,obs,float(ts['target_corner_weight']),1400); p=r['states']['target']; return {'dropped':drop,'camera_center_shift_cm':float(np.linalg.norm(p[3:6]-best['states']['target'][3:6])),'action_volume_p95_shift_px':d.functional_p95(best['states']['target'],p,vol)}
        with ThreadPoolExecutor(max_workers=4) as pool: support=list(pool.map(sj,list(lines)+['restricted']))
        rng=np.random.default_rng(760905); delta=float(ts['perturbation_half_pixel']); jobs=[]
        for i in range(a.perturbation_trials): jobs.append((i,{k:v+rng.uniform(-delta,delta,v.shape) for k,v in lines.items()},restricted+rng.uniform(-delta,delta,restricted.shape),{k:v+rng.uniform(-delta,delta,v.shape) for k,v in rims.items()},obs+rng.uniform(-delta,delta,obs.shape)))
        def pj(z):
            i,ll,rr,ri,oo=z; r=solve(best['x'],keys,ll,rr,ri,transfers,obj,oo,float(ts['target_corner_weight']),1200); p=r['states']['target']; return {'trial':i,'camera_center_shift_cm':float(np.linalg.norm(p[3:6]-best['states']['target'][3:6])),'action_volume_p95_shift_px':d.functional_p95(best['states']['target'],p,vol)}
        with ThreadPoolExecutor(max_workers=4) as pool: perturb=list(pool.map(pj,jobs))
        gates.update({'leave_center':max(x['camera_center_shift_cm'] for x in leave)<=t['max_leave_one_target_center_shift_cm'],'leave_volume':max(x['action_volume_p95_shift_px'] for x in leave)<=t['max_leave_one_target_action_volume_p95_px'],'support_center':max(x['camera_center_shift_cm'] for x in support)<=t['max_support_removal_center_shift_cm'],'support_volume':max(x['action_volume_p95_shift_px'] for x in support)<=t['max_support_removal_action_volume_p95_px'],'half_pixel_center':max(x['camera_center_shift_cm'] for x in perturb)<=t['max_half_pixel_center_shift_cm'],'half_pixel_volume':max(x['action_volume_p95_shift_px'] for x in perturb)<=t['max_half_pixel_action_volume_p95_px']})
    else: gates.update({k:False for k in ('leave_center','leave_volume','support_center','support_volume','half_pixel_center','half_pixel_volume')})
    passed=bool(all(gates.values())); report=json_safe({'status':'PASS_IN_ARENA_TARGET_CONSTRAINED_METRIC_CAMERA_V76' if passed else 'FAIL_IN_ARENA_TARGET_CONSTRAINED_METRIC_CAMERA_V76','version':'v76','game_id':'0022500301','event_id':489,'camera':'In Arena','method':'sealed v52 floor + three native target-opening corners + source rim centres + five sealed v53b optical states; full target rim ellipse and free-throw curve held out','guardrail':ts['guardrail'],'immutable_target_sha256':expected,'best':qa,'multistart':{'root_count':len(roots),'competitive_root_count':len(comp),'max_competitive_center_shift_cm':max((x['camera_center_shift_cm'] for x in pairs),default=0),'max_competitive_action_volume_p95_px':max((x['action_volume_p95_px'] for x in pairs),default=0),'pairs':pairs,'roots':[{'cost':r['cost'],'nfev':r['nfev'],'qa':q} for r,q in zip(roots,sums)]},'leave_one_target_corner':leave,'support_removal':support,'half_pixel_perturbation':{'trial_count':len(perturb),'trials':perturb},'rim_source_pixel_diagnostics':diags,'thresholds':t,'gates':gates,'permissions':{'physical_camera_center_allowed':passed,'metric_event_camera_allowed':passed,'replay_render_allowed':False}})
    open(a.out/'in_arena_target_constrained_v76.json','w').write(json.dumps(report,indent=2)+'\n'); overlay(image,best['states']['target'],lines,restricted,ft,rimpts,obj,obs,allobj,a.out/'in_arena_target_constrained_v76_overlay.png'); print(json.dumps({'status':report['status'],'camera_center_cm':qa['camera_center_cm'],'focal_px':qa['target_focal_px'],'principal_point_px':qa['target_principal_point_px'],'target_corner_rmse_px':qa['target_corners']['rmse_px'],'line_p95_px':qa['line_p95_px'],'free_throw_holdout_p95_px':qa['free_throw_holdout_p95_px'],'competitive_roots':len(comp),'max_root_center_shift_cm':report['multistart']['max_competitive_center_shift_cm'],'gates':gates},indent=2),flush=True)
if __name__=='__main__': main()
