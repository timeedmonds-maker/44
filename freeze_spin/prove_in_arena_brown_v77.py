from __future__ import annotations
"""v77: fail-closed distortion-aware In-Arena metric camera certification candidate."""
import argparse,json,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from freeze_spin import solve_in_arena_joint_selfcal_v56a as b
from freeze_spin import solve_in_arena_direct_camera_v55a as d
from freeze_spin.prove_in_arena_noncoplanar_center_v54 import RIM_CENTER,TARGET_RIM_SEED,extract_rim_ellipse,project_point,refine_transfer,sha256
from freeze_spin.solve_nba_geometry_proof_v3 import world_landmarks
W,H,SCALE=960,540,480.0
DB=np.asarray([0.08,0.04,0.01,0.01],float)

def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    return v

def distort(uv,pp,k):
    uv=np.asarray(uv,float); pp=np.asarray(pp,float); xy=(uv-pp[None,:])/SCALE; x,y=xy[:,0],xy[:,1]; k1,k2,p1,p2=np.asarray(k,float); r2=x*x+y*y; rad=1+k1*r2+k2*r2*r2
    xd=x*rad+2*p1*x*y+p2*(r2+2*x*x); yd=y*rad+p1*(r2+2*y*y)+2*p2*x*y
    return np.column_stack([xd,yd])*SCALE+pp[None,:]

def undistort(uv,pp,k,it=10):
    uv=np.asarray(uv,float); pp=np.asarray(pp,float); xyd=(uv-pp[None,:])/SCALE; xd,yd=xyd[:,0],xyd[:,1]; x,y=xd.copy(),yd.copy(); k1,k2,p1,p2=np.asarray(k,float)
    for _ in range(it):
        r2=x*x+y*y; rad=1+k1*r2+k2*r2*r2; dx=2*p1*x*y+p2*(r2+2*x*x); dy=p1*(r2+2*y*y)+2*p2*x*y; rad=np.where(np.abs(rad)<1e-6,1e-6,rad); x=(xd-dx)/rad; y=(yd-dy)/rad
    return np.column_stack([x,y])*SCALE+pp[None,:]

def project(p,xyz,k):
    uv,cam=d.project(p,xyz); return distort(uv,p[7:9],k),cam

def line_train_residual(p,lines,k):
    u={name:undistort(obs,p[7:9],k) for name,obs in lines.items()}; return d.line_distances(p,u)[0]

def circle_train_residual(p,obs,cx,r,k): return d.circle_signed_distances(p,undistort(obs,p[7:9],k),cx,r)

def dense_line(name,n=161):
    z=d.line_world(name); a=np.linspace(0,1,n)[:,None]; return z[0]*(1-a)+z[1]*a

def nearest(obs,pred): return d.nearest_curve_distances(np.asarray(obs,float),np.asarray(pred,float))

def func_p95(pa,ka,pb,kb,vol):
    ua,ca=project(pa,vol,ka); ub,cb=project(pb,vol,kb); ok=(ca[:,2]>20)&(cb[:,2]>20)
    return float(np.percentile(np.linalg.norm(ua[ok]-ub[ok],axis=1),95)) if int(ok.sum())>=20 else float('inf')

def angle(points):
    e=cv2.fitEllipse(np.asarray(points,np.float32)); a0,a1=e[1]; q=float(e[2]) if a0>=a1 else float((e[2]+90)%180); return q-180 if q>=90 else q

def diagnostics(states,k):
    xs=np.linspace(0,W-1,25); ys=np.linspace(0,H-1,15); grid=np.asarray([(x,y) for y in ys for x in xs],float); rows={}; global_max=0.; minrad=1e9; maxrad=-1e9; mindet=1e9; maxdet=-1e9
    for key,p in states.items():
        warped=distort(grid,p[7:9],k); disp=np.linalg.norm(warped-grid,axis=1); eps=.5; xp=distort(grid+[eps,0],p[7:9],k); xm=distort(grid-[eps,0],p[7:9],k); yp=distort(grid+[0,eps],p[7:9],k); ym=distort(grid-[0,eps],p[7:9],k); dx=(xp-xm)/(2*eps); dy=(yp-ym)/(2*eps); det=dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0]
        xy=(grid-p[7:9][None,:])/SCALE; r2=np.sum(xy*xy,axis=1); rad=1+k[0]*r2+k[1]*r2*r2; row={'max_displacement_px':float(disp.max()),'p95_displacement_px':float(np.percentile(disp,95)),'min_radial_scale':float(rad.min()),'max_radial_scale':float(rad.max()),'min_jacobian_det':float(det.min()),'max_jacobian_det':float(det.max())}; rows[key]=row; global_max=max(global_max,row['max_displacement_px']); minrad=min(minrad,row['min_radial_scale']); maxrad=max(maxrad,row['max_radial_scale']); mindet=min(mindet,row['min_jacobian_det']); maxdet=max(maxdet,row['max_jacobian_det'])
    return {'coefficients':{'k1':float(k[0]),'k2':float(k[1]),'p1':float(k[2]),'p2':float(k[3])},'abs_fraction_of_bound':(np.abs(k)/DB).tolist(),'max_abs_fraction_of_bound':float(np.max(np.abs(k)/DB)),'global_max_displacement_px':global_max,'global_min_radial_scale':minrad,'global_max_radial_scale':maxrad,'global_min_jacobian_det':mindet,'global_max_jacobian_det':maxdet,'per_state':rows}

def unpack_full(x,keys):
    return (*b.unpack(np.asarray(x[:-4],float),keys),np.asarray(x[-4:],float))

def solve(start,keys,lines,restricted,rims,transfers,obj,obs,weight,nfev):
    n=len(keys)-1; lo,hi=d.bounds(); lower=np.r_[lo[3:6],lo[:3],lo[6:9],np.tile([-10,-10,-10,math.log(300),-60,-60],n),-DB]; upper=np.r_[hi[3:6],hi[:3],hi[6:9],np.tile([10,10,10,math.log(9000),60,60],n),DB]; start=np.clip(np.asarray(start,float),lower+1e-7,upper-1e-7)
    def residual(x):
        states,blocks,k=unpack_full(x,keys); target=states['target']; le=line_train_residual(target,lines,k); re=circle_train_residual(target,restricted,RIM_CENTER[0],d.RESTRICTED_RADIUS,k); ruv,rd=project(target,RIM_CENTER[None,:],k); tuv,td=project(target,obj,k)
        out=[le,re,(ruv[0]-rims['target'])*2,(tuv-obs).ravel()*weight,(target[7:9]-[W/2,H/2])/350,[(target[6]-math.log(3000))/3],(k/DB)*.01,np.minimum(rd-20,0).ravel()*5,np.minimum(td-20,0).ravel()*5]
        for i,key in enumerate(keys[1:]):
            state=states[key]; suv,sd=project(state,RIM_CENTER[None,:],k); p,q=transfers[key]; pu=undistort(p,target[7:9],k); pred_u=b.transfer_points(b.relative_homography(target,state),pu); pred=distort(pred_u,state[7:9],k)
            out += [(pred-q).ravel()*.5,(suv[0]-rims[key])*2,blocks[i,4:6]/25,np.minimum(sd-20,0).ravel()*5]
        return np.concatenate(out)
    r=least_squares(residual,start,bounds=(lower,upper),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=nfev); states,blocks,k=unpack_full(r.x,keys); return {'x':r.x,'states':states,'blocks':blocks,'distortion':k,'cost':float(r.cost),'nfev':int(r.nfev)}

def summarize(sol,keys,lines,restricted,ft,rims,ellipses,transfers,obj,obs):
    states,k=sol['states'],sol['distortion']; target=states['target']; line_err=[]
    for name,px in lines.items(): line_err.extend(nearest(px,project(target,dense_line(name),k)[0]).tolist())
    re=nearest(restricted,project(target,d.floor_circle(RIM_CENTER[0],d.RESTRICTED_RADIUS),k)[0]); fe=nearest(ft,project(target,d.floor_circle(d.FREE_THROW_X,d.FREE_THROW_RADIUS),k)[0]); rim_errors=[]; transfer_errors=[]; rows={}
    for key,p in states.items():
        ruv,_=project(p,RIM_CENTER[None,:],k); er=float(np.linalg.norm(ruv[0]-rims[key])); rim_errors.append(er); row={'focal_px':math.exp(float(p[6])),'principal_point_px':p[7:9].tolist(),'rvec':p[:3].tolist(),'rim_center_observed_px':rims[key].tolist(),'rim_center_predicted_px':ruv[0].tolist(),'rim_center_error_px':er}
        if key!='target':
            src,q=transfers[key]; su=undistort(src,target[7:9],k); pred=distort(b.transfer_points(b.relative_homography(target,p),su),p[7:9],k); ee=np.linalg.norm(pred-q,axis=1); transfer_errors.extend(ee.tolist()); row['rotation_homography_rms_px']=float(np.sqrt(np.mean(ee**2))); row['rotation_homography_p95_px']=float(np.percentile(ee,95))
        rows[key]=row
    rim_uv,_=project(target,d.rim_circle(),k); e=cv2.fitEllipse(rim_uv.astype(np.float32)); axes=np.sort(np.asarray(e[1],float))[::-1]; obsaxes=np.asarray(ellipses['target']['axes'],float); uv,_=project(target,obj,k); ce=np.linalg.norm(uv-obs,axis=1)
    return {'camera_center_cm':target[3:6].tolist(),'target_focal_px':math.exp(float(target[6])),'target_principal_point_px':target[7:9].tolist(),'line_p95_px':float(np.percentile(line_err,95)),'restricted_p95_px':float(np.percentile(re,95)),'free_throw_holdout_p95_px':float(np.percentile(fe,95)),'max_rim_center_error_px':max(rim_errors),'rim_axis_error_major_minor_px':np.abs(axes-obsaxes).tolist(),'rim_angle_holdout_error_deg':abs(float((angle(rim_uv)-ellipses['target']['canonical_angle_deg']+90)%180-90)),'rotation_homography_p95_px':float(np.percentile(transfer_errors,95)),'states':rows,'target_corners':{'observed_px':obs.tolist(),'predicted_px':uv.tolist(),'per_corner_error_px':ce.tolist(),'rmse_px':float(np.sqrt(np.mean(ce**2))),'max_px':float(ce.max())},'camera_distance_from_basket_cm':float(np.linalg.norm(target[3:6]-RIM_CENTER)),'distortion':diagnostics(states,k)}

def overlay(image,p,k,lines,restricted,ft,rimpts,obj,obs,allobj,path):
    out=image.copy()
    for name,pts in lines.items():
        uv,_=project(p,dense_line(name),k); cv2.polylines(out,[np.round(uv).astype(np.int32)],False,(0,255,255),2,cv2.LINE_AA)
        for x in pts: cv2.circle(out,tuple(np.round(x).astype(int)),4,(255,0,0),2,cv2.LINE_AA)
    for world,pix,c in ((d.floor_circle(RIM_CENTER[0],d.RESTRICTED_RADIUS),restricted,(0,255,0)),(d.floor_circle(d.FREE_THROW_X,d.FREE_THROW_RADIUS),ft,(0,0,255)),(d.rim_circle(),rimpts,(255,0,255))):
        uv,_=project(p,world,k); cv2.polylines(out,[np.round(uv).astype(np.int32)],True,c,2,cv2.LINE_AA)
        for x in np.round(pix).astype(int): cv2.circle(out,tuple(x),2,(255,255,255),-1,cv2.LINE_AA)
    poly,_=project(p,allobj,k); cv2.polylines(out,[np.round(poly).astype(np.int32)],True,(0,255,0),2,cv2.LINE_AA); pred,_=project(p,obj,k)
    for a,z in zip(obs,pred): cv2.circle(out,tuple(np.round(a).astype(int)),5,(0,215,255),2,cv2.LINE_AA); cv2.circle(out,tuple(np.round(z).astype(int)),3,(0,255,0),-1,cv2.LINE_AA)
    cv2.imwrite(str(path),out)

def main():
    ap=argparse.ArgumentParser()
    for x in ('target','states','floor-proof','family-proof','floor-observations','target-observations','out'): ap.add_argument('--'+x,type=Path,required=True)
    ap.add_argument('--root-limit',type=int,default=9); ap.add_argument('--max-nfev',type=int,default=6500); ap.add_argument('--perturbation-trials',type=int,default=12); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(a.target)); floor=json.load(open(a.floor_proof)); fam=json.load(open(a.family_proof)); fs=json.load(open(a.floor_observations)); ts=json.load(open(a.target_observations)); t=ts['thresholds']; dm=ts['distortion_model']
    if image is None or image.shape[:2]!=(H,W): raise RuntimeError('v77 target must be native 960x540')
    if floor.get('status')!='PASS_IN_ARENA_FLOOR_V52' or fam.get('status')!='PASS_IN_ARENA_STATIC_SCENE_FAMILY_V53B': raise RuntimeError('unsealed upstream evidence')
    expected=ts['freeze_lock']['sha256_png'];
    if sha256(a.target)!=expected or fam['target_sha256_png']!=expected: raise RuntimeError('immutable target mismatch')
    names=list(ts['target_inner_corners_px']); world=world_landmarks(); obj=np.asarray([world[n] for n in names],float); obs=np.asarray([ts['target_inner_corners_px'][n] for n in names],float); allobj=np.asarray([world[n] for n in ('target_inner_top_left','target_inner_top_right','target_inner_bottom_right','target_inner_bottom_left')],float)
    lines={k:np.asarray(v,float) for k,v in fs['training_line_segments_px'].items()}; restricted=np.asarray(fs['heldout_curves_px']['restricted_area_arc'],float); ft=np.asarray(fs['heldout_curves_px']['free_throw_circle_dashed_half'],float); keys=['target']; transfers={}; rims={}; ellipses={}; diags={}; xfers={}
    c,axes,ang,rimpts,diag=extract_rim_ellipse(image,TARGET_RIM_SEED); rims['target']=c; ellipses['target']={'axes':axes,'angle':ang,'canonical_angle_deg':angle(rimpts)}; diags['target']=diag
    for row in fam['selected_candidates']:
        key=f"event_{int(row['event_probe'])}"; im=cv2.imread(str(a.states/row['file']))
        if im is None or im.shape[:2]!=(H,W): raise RuntimeError('missing v53a state '+row['file'])
        h,p,q,xd=refine_transfer(image,im,b.norm_h(np.asarray(row['H_target_to_state'],float))); take=b.spatial_subset(p); p,q=p[take],q[take]; xfers[key]=h; transfers[key]=(p,q); keys.append(key); pred=project_point(h,TARGET_RIM_SEED); cc,ax,aa,rp,dd=extract_rim_ellipse(im,pred); dd['transfer_refinement']=xd; rims[key]=cc; ellipses[key]={'axes':ax,'angle':aa,'canonical_angle_deg':angle(rp)}; diags[key]=dd
    h0=b.norm_h(np.asarray(floor['floor_homography_world_to_image'],float)); base=[]
    for s0 in d.starts(h0):
        p,cost,_,_=d.optimize(s0,lines,restricted,rims['target'],max_nfev=1800); base.append((cost,p))
    base.sort(key=lambda z:z[0]); patterns=[np.zeros(4),[.02,0,0,0],[-.02,0,0,0],[0,.01,0,0],[0,-.01,0,0],[0,0,.003,0],[0,0,-.003,0],[0,0,0,.003],[0,0,0,-.003]]; starts=[]
    for i,pat in enumerate(patterns[:a.root_limit]): starts.append(np.r_[b.make_start(base[i%min(3,len(base))][1],keys,xfers),np.asarray(pat,float)])
    roots=[solve(s,keys,lines,restricted,rims,transfers,obj,obs,float(ts['target_corner_weight']),a.max_nfev) for s in starts]; roots.sort(key=lambda z:z['cost']); sums=[summarize(r,keys,lines,restricted,ft,rims,ellipses,transfers,obj,obs) for r in roots]; best=roots[0]; qa=sums[0]; comp=[r for r in roots if r['cost']<=roots[0]['cost']*1.05+1e-8]; vol=b.action_volume(); pairs=[]
    for i in range(len(comp)):
        for j in range(i+1,len(comp)):
            pairs.append({'left':i,'right':j,'camera_center_shift_cm':float(np.linalg.norm(comp[i]['states']['target'][3:6]-comp[j]['states']['target'][3:6])),'action_volume_p95_px':func_p95(comp[i]['states']['target'],comp[i]['distortion'],comp[j]['states']['target'],comp[j]['distortion'],vol),'distortion_l2_shift':float(np.linalg.norm(comp[i]['distortion']-comp[j]['distortion']))})
    center=np.asarray(qa['camera_center_cm']); pp=np.asarray(qa['target_principal_point_px']); axeserr=qa['rim_axis_error_major_minor_px']; dg=qa['distortion']; plausible=dg['max_abs_fraction_of_bound']<dm['max_abs_fraction_of_coefficient_bound'] and dg['global_max_displacement_px']<=dm['max_grid_displacement_px'] and dg['global_min_radial_scale']>=dm['min_radial_scale'] and dg['global_max_radial_scale']<=dm['max_radial_scale'] and dg['global_min_jacobian_det']>=dm['min_jacobian_det'] and dg['global_max_jacobian_det']<=dm['max_jacobian_det']
    gates={'target_corner_rmse':qa['target_corners']['rmse_px']<=t['max_target_corner_rmse_px'],'target_corner_max':qa['target_corners']['max_px']<=t['max_target_corner_max_px'],'floor_line_p95':qa['line_p95_px']<=t['max_floor_line_p95_px'],'restricted_p95':qa['restricted_p95_px']<=t['max_restricted_p95_px'],'free_throw_holdout':qa['free_throw_holdout_p95_px']<=t['max_free_throw_holdout_p95_px'],'rim_center':qa['max_rim_center_error_px']<=t['max_rim_center_error_px'],'rim_major_holdout':axeserr[0]<=t['max_rim_major_axis_holdout_error_px'],'rim_minor_holdout':axeserr[1]<=t['max_rim_minor_axis_holdout_error_px'],'rim_angle_holdout':qa['rim_angle_holdout_error_deg']<=t['max_rim_angle_holdout_error_deg'],'rotation_homography':qa['rotation_homography_p95_px']<=t['max_rotation_homography_p95_px'],'principal_point_crop':-t['principal_point_crop_margin_px']<=pp[0]<=W+t['principal_point_crop_margin_px'] and -t['principal_point_crop_margin_px']<=pp[1]<=H+t['principal_point_crop_margin_px'],'camera_height':t['min_camera_height_cm']<=center[2]<=t['max_camera_height_cm'],'camera_outside_width':abs(center[1])>=t['min_abs_camera_y_cm'],'camera_distance':qa['camera_distance_from_basket_cm']<=t['max_camera_distance_from_basket_cm'],'minimum_roots':len(roots)>=t['minimum_multistart_root_count'],'root_center_spread':max((x['camera_center_shift_cm'] for x in pairs),default=0)<=t['max_competitive_root_center_shift_cm'],'root_action_volume':max((x['action_volume_p95_px'] for x in pairs),default=0)<=t['max_competitive_root_action_volume_p95_px'],'distortion_physically_mild_nonfolding':plausible}
    leave=[]; support=[]; perturb=[]
    if all(gates.values()):
        def lj(i):
            keep=[j for j in range(len(obj)) if j!=i]; r=solve(best['x'],keys,lines,restricted,rims,transfers,obj[keep],obs[keep],float(ts['target_corner_weight']),1800); return {'dropped_corner':names[i],'camera_center_shift_cm':float(np.linalg.norm(r['states']['target'][3:6]-best['states']['target'][3:6])),'action_volume_p95_shift_px':func_p95(best['states']['target'],best['distortion'],r['states']['target'],r['distortion'],vol)}
        with ThreadPoolExecutor(max_workers=3) as pool: leave=list(pool.map(lj,range(len(obj))))
        def sj(drop):
            ll={k:v for k,v in lines.items() if k!=drop}; rr=restricted[::2] if drop=='restricted' else restricted; r=solve(best['x'],keys,ll,rr,rims,transfers,obj,obs,float(ts['target_corner_weight']),1600); return {'dropped':drop,'camera_center_shift_cm':float(np.linalg.norm(r['states']['target'][3:6]-best['states']['target'][3:6])),'action_volume_p95_shift_px':func_p95(best['states']['target'],best['distortion'],r['states']['target'],r['distortion'],vol)}
        with ThreadPoolExecutor(max_workers=4) as pool: support=list(pool.map(sj,list(lines)+['restricted']))
        rng=np.random.default_rng(770905); delta=float(ts['perturbation_half_pixel']); jobs=[]
        for i in range(a.perturbation_trials): jobs.append((i,{k:v+rng.uniform(-delta,delta,v.shape) for k,v in lines.items()},restricted+rng.uniform(-delta,delta,restricted.shape),{k:v+rng.uniform(-delta,delta,v.shape) for k,v in rims.items()},obs+rng.uniform(-delta,delta,obs.shape)))
        def pj(z):
            i,ll,rr,ri,oo=z; r=solve(best['x'],keys,ll,rr,ri,transfers,obj,oo,float(ts['target_corner_weight']),1400); return {'trial':i,'camera_center_shift_cm':float(np.linalg.norm(r['states']['target'][3:6]-best['states']['target'][3:6])),'action_volume_p95_shift_px':func_p95(best['states']['target'],best['distortion'],r['states']['target'],r['distortion'],vol)}
        with ThreadPoolExecutor(max_workers=4) as pool: perturb=list(pool.map(pj,jobs))
        gates.update({'leave_center':max(x['camera_center_shift_cm'] for x in leave)<=t['max_leave_one_target_center_shift_cm'],'leave_volume':max(x['action_volume_p95_shift_px'] for x in leave)<=t['max_leave_one_target_action_volume_p95_px'],'support_center':max(x['camera_center_shift_cm'] for x in support)<=t['max_support_removal_center_shift_cm'],'support_volume':max(x['action_volume_p95_shift_px'] for x in support)<=t['max_support_removal_action_volume_p95_px'],'half_pixel_center':max(x['camera_center_shift_cm'] for x in perturb)<=t['max_half_pixel_center_shift_cm'],'half_pixel_volume':max(x['action_volume_p95_shift_px'] for x in perturb)<=t['max_half_pixel_action_volume_p95_px']})
    else: gates.update({k:False for k in ('leave_center','leave_volume','support_center','support_volume','half_pixel_center','half_pixel_volume')})
    passed=bool(all(gates.values())); overlay(image,best['states']['target'],best['distortion'],lines,restricted,ft,rimpts,obj,obs,allobj,a.out/'in_arena_brown_v77_overlay.png'); report={'status':'PASS_IN_ARENA_BROWN_METRIC_CAMERA_V77' if passed else 'FAIL_IN_ARENA_BROWN_METRIC_CAMERA_V77','version':'v77','game_id':'0022500301','event_id':489,'camera':'In Arena','method':'v76 immutable evidence plus one shared tightly bounded Brown-Conrady distortion vector; all v76 geometric and robustness gates unchanged','guardrail':ts['guardrail'],'immutable_target_sha256':expected,'best':qa,'multistart':{'root_count':len(roots),'competitive_root_count':len(comp),'max_competitive_center_shift_cm':max((x['camera_center_shift_cm'] for x in pairs),default=0),'max_competitive_action_volume_p95_px':max((x['action_volume_p95_px'] for x in pairs),default=0),'pairs':pairs,'roots':[{'cost':r['cost'],'nfev':r['nfev'],'qa':q} for r,q in zip(roots,sums)]},'leave_one_target_corner':leave,'support_removal':support,'half_pixel_perturbation':{'trial_count':len(perturb),'trials':perturb},'rim_source_pixel_diagnostics':diags,'thresholds':t,'distortion_model_lock':dm,'gates':gates,'permissions':{'physical_camera_center_allowed':passed,'metric_event_camera_allowed':passed,'replay_render_allowed':False}}
    (a.out/'in_arena_brown_v77.json').write_text(json.dumps(safe(report),indent=2),encoding='utf-8'); print(json.dumps(safe({'status':report['status'],'camera_center_cm':qa['camera_center_cm'],'target_focal_px':qa['target_focal_px'],'target_principal_point_px':qa['target_principal_point_px'],'target_corner_rmse_px':qa['target_corners']['rmse_px'],'floor_line_p95_px':qa['line_p95_px'],'restricted_p95_px':qa['restricted_p95_px'],'free_throw_holdout_p95_px':qa['free_throw_holdout_p95_px'],'max_rim_center_error_px':qa['max_rim_center_error_px'],'rotation_homography_p95_px':qa['rotation_homography_p95_px'],'distortion':dg,'competitive_roots':len(comp),'max_root_center_shift_cm':report['multistart']['max_competitive_center_shift_cm'],'gates':gates,'permissions':report['permissions']}),indent=2))
if __name__=='__main__': main()
