from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from freeze_spin import prove_frame_c_left_above_rim_noncoplanar_functional_camera_v41 as v41

# v42 changes only the held-out target-event evidence: replace the single manually
# specified rim-centre pixel with directly observed source pixels from the visible
# right/underside of the regulation rim. The occluded/ambiguous left side and the
# backboard target rectangle are excluded. No v26 target pixels enter the fit.
TARGET_RIM_ROI=(455,505,175,180)
TARGET_RIM_MIN_POINTS=20
SEEDS=[(0.,0.),(-20.,0.),(20.,0.),(0.,-20.),(0.,20.)]

def target_rim_pixels(im:np.ndarray)->np.ndarray:
    x0,x1,y0,y1=TARGET_RIM_ROI
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[y0:y1,x0:x1]
    h,s,v=cv2.split(hsv)
    m=(((h<=20)|(h>=170))&(s>=70)&(v>=60)).astype(np.uint8)
    ys,xs=np.where(m)
    pts=np.c_[xs+x0,ys+y0].astype(float)
    # Spatially bin adjacent source pixels so antialiasing/thickness does not
    # overweight one edge location. These remain real observed pixels.
    bins={}
    for p in pts: bins.setdefault((int(p[0]//2),int(p[1]//2)),[]).append(p)
    out=np.array([np.mean(q,axis=0) for q in bins.values()],float)
    if len(out)<TARGET_RIM_MIN_POINTS:
        raise RuntimeError(f'target visible rim support sparse: {len(out)}')
    return out

def json_safe(x):
    if isinstance(x,dict):return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target-frame',type=Path,required=True)
    ap.add_argument('--same-game-samples',type=Path,required=True)
    ap.add_argument('--target-clip-samples',type=Path,required=True)
    ap.add_argument('--target-manifest',type=Path,required=True)
    ap.add_argument('--floor-proof',type=Path,required=True)
    ap.add_argument('--wide-court',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)

    floor=json.loads(a.floor_proof.read_text()); wide=json.loads(a.wide_court.read_text())
    Ht=v41.normh(np.array(floor['floor_homography_world_to_image'],float))
    target_im=cv2.imread(str(a.target_frame))
    if target_im is None or target_im.shape[:2]!=(v41.H,v41.W): raise RuntimeError('immutable target frame missing/wrong size')
    target_rim=target_rim_pixels(target_im)

    # Independent game-camera centre: exactly v41, event 489 excluded.
    sel={}; Hs={}; rims={}
    for e in v41.EVENTS:
        r=v41.best(sorted(a.same_game_samples.glob(f'Left_Above_Rim__event{e:04d}__s*.png')),a.target_frame)
        sel[e]=r; Hs[e]=v41.normh(np.linalg.inv(r['H'])@Ht)
        rims[e]=v41.rim_pixels(cv2.imread(r['source']),e)
    center_roots=[]
    for pp in [(458,450),(456,249),(480,370),(430,450),(500,420)]:
        x,rms=v41.solve_center(Hs,rims,v41.EVENTS,seed_pp=pp)
        center_roots.append({'seed_pp':pp,'x':x,'rms':rms})
    center_roots.sort(key=lambda z:z['rms'])
    base=center_roots[0]['x']; C=base[:3]; gamepp=base[3:5]
    center_spread=max(np.linalg.norm(x['x'][:3]-y['x'][:3]) for i,x in enumerate(center_roots) for y in center_roots[i+1:])
    gamepp_spread=max(np.linalg.norm(x['x'][3:5]-y['x'][3:5]) for i,x in enumerate(center_roots) for y in center_roots[i+1:])
    loo={}
    for omit in v41.EVENTS:
        ev=[e for e in v41.EVENTS if e!=omit]; x,_=v41.solve_center(Hs,rims,ev)
        loo[str(omit)]={'center_shift_cm':float(np.linalg.norm(x[:3]-C)),'pp_shift_px':float(np.linalg.norm(x[3:5]-gamepp))}

    obs={k:np.array(v,float) for k,v in wide['observations_px'].items()}
    held={k:set(v) for k,v in wide['held_out_indices'].items()}
    def metric_res(core):
        pp=core[4:6]; par=np.r_[core[3],core[:3]]; rows=[]
        for n,oo in obs.items():
            pred=v41.project(C,pp,par,v41.CURVES[n])
            for i,p in enumerate(oo):
                if i not in held[n]: rows.extend(v41.nearest_res(pred,p))
        # Full visible real rim segment, not a single centre pixel.
        rows.extend(v41.nearvec(v41.project(C,pp,par,v41.RIM),target_rim))
        return np.asarray(rows,float)

    d=v41.decomp(Ht,459,430); metric0=np.r_[d[2],math.log(d[0]),459.,430.]
    manifest=json.loads(a.target_manifest.read_text()); meta={x['file']:x for x in manifest['samples']}
    pairs=[]
    for p in sorted(a.target_clip_samples.glob('Left_Above_Rim_target_event__*.png')):
        q=v41.clip_pair(p,a.target_frame,meta)
        if q is not None:pairs.append(q)
    estimates=[]
    for pr in pairs:
        sp=v41.init_source(pr,metric0); estimates.append((pr,sp,float(np.exp(sp[0]))))
    med=float(np.median([x[2] for x in estimates])); settled=[x[0] for x in estimates if abs(x[2]-med)/med<=.01]
    if len(settled)<6:raise RuntimeError('settled static support')

    lo=np.r_[[-10]*3,math.log(250),100,50]; hi=np.r_[[10]*3,math.log(2500),850,520]
    for _ in settled:
        lo=np.r_[lo,math.log(150),[-10]*3]; hi=np.r_[hi,math.log(4000),[10]*3]
    def seed(core):return np.concatenate([core]+[v41.init_source(pr,core) for pr in settled])
    def residual(x):
        co=x[:6]; rows=[metric_res(co)]; off=6
        for pr in settled:
            rows.append((v41.pair_project(pr['p'],co,x[off:off+4])-pr['q']).ravel()); off+=4
        return np.concatenate(rows)

    roots=[]
    for dx,dy in SEEDS:
        c=metric0.copy(); c[4]+=dx; c[5]+=dy
        o=least_squares(residual,seed(c),bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=10000)
        r=residual(o.x)
        roots.append({'seed_pp_offset_px':[dx,dy],'x':np.asarray(o.x,float),'mean_square_residual':float(np.mean(r*r)),'cost':float(o.cost),'optimality':float(o.optimality),'success':bool(o.success)})
    roots.sort(key=lambda z:z['mean_square_residual']); best=roots[0]['x']; core=best[:6]
    pp=core[4:6]; par=np.r_[core[3],core[:3]]

    tr=[]; ho=[]
    for n,oo in obs.items():
        pred=v41.project(C,pp,par,v41.CURVES[n])
        for i,p in enumerate(oo):
            e=float(np.min(np.linalg.norm(pred-p,axis=1))); (ho if i in held[n] else tr).append(e)
    rim_e=np.linalg.norm(v41.project(C,pp,par,v41.RIM)[None,:,:]-target_rim[:,None,:],axis=2).min(axis=1)
    static=[]; off=6
    for pr in settled:
        sp=best[off:off+4];off+=4
        ew=np.linalg.norm(v41.pair_project(pr['pw'],core,sp)-pr['qw'],axis=1)
        static.append({'frame':pr['name'],'p95_px':float(np.percentile(ew,95)),'median_px':float(np.median(ew))})

    functional=[]
    for i,ri in enumerate(roots):
        x=ri['x']; ua=v41.project(C,x[4:6],np.r_[x[3],x[:3]],v41.ACTION)
        for j,rj in enumerate(roots[i+1:],i+1):
            y=rj['x']; ub=v41.project(C,y[4:6],np.r_[y[3],y[:3]],v41.ACTION)
            m=(ua[:,0]>0)&(ua[:,0]<v41.W)&(ua[:,1]>0)&(ua[:,1]<v41.H)&(ub[:,0]>0)&(ub[:,0]<v41.W)&(ub[:,1]>0)&(ub[:,1]<v41.H)
            dd=np.linalg.norm(ua[m]-ub[m],axis=1)
            functional.append({'i':i,'j':j,'p95_px':float(np.percentile(dd,95)),'max_px':float(dd.max())})
    fe95=max(x['p95_px'] for x in functional); femax=max(x['max_px'] for x in functional)
    pp_spread=max(np.linalg.norm(x['x'][4:6]-y['x'][4:6]) for i,x in enumerate(roots) for y in roots[i+1:])

    gates={
      'center_root_spread':center_spread<=1.0,
      'center_loo':max(x['center_shift_cm'] for x in loo.values())<=15.0,
      'target_floor_heldout':float(np.percentile(ho,95))<=1.5,
      'target_visible_rim_p95':float(np.percentile(rim_e,95))<=1.0,
      'target_static_heldout':max(x['p95_px'] for x in static)<=3.0,
      'functional_root_p95':fe95<=0.5,
      'functional_root_max':femax<=0.75,
    }
    passed=all(gates.values())
    report={
      'schema_version':1,'status':'PASS_VISIBLE_RIM_FUNCTIONAL_CAMERA_V42' if passed else 'FAIL_VISIBLE_RIM_FUNCTIONAL_CAMERA_V42',
      'game_id':'0022500301','event_id':489,'camera_label':'Left Above Rim',
      'method':'v41 independent full-rim same-game centre + immutable Frame C raw wide court + directly observed visible regulation-rim segment + settled same-clip static background',
      'camera_center_cm':C.tolist(),'center_multistart_max_cm':center_spread,'game_level_pp_diagnostic_px':gamepp.tolist(),'game_pp_multistart_max_px':gamepp_spread,
      'leave_one_event_out':loo,'selected_samples':{str(e):Path(sel[e]['source']).name for e in v41.EVENTS},
      'target_visible_rim':{'roi_xyxy':list(TARGET_RIM_ROI),'observed_point_count':len(target_rim),'observed_px':target_rim.tolist(),'p95_error_px':float(np.percentile(rim_e,95)),'max_error_px':float(rim_e.max())},
      'target_camera':{'rvec':core[:3].tolist(),'focal_px':float(np.exp(core[3])),'principal_point_px':core[4:6].tolist(),'parameter_pp_root_spread_px':float(pp_spread)},
      'target_metric':{'floor_train_p95_px':float(np.percentile(tr,95)),'floor_heldout_p95_px':float(np.percentile(ho,95))},
      'settled_static':{'count':len(settled),'median_initial_source_focal_px':med,'heldout':static},
      'multistart_roots':[{'seed_pp_offset_px':r['seed_pp_offset_px'],'mean_square_residual':r['mean_square_residual'],'cost':r['cost'],'optimality':r['optimality'],'success':r['success'],'target_focal_px':float(np.exp(r['x'][3])),'target_pp_px':r['x'][4:6].tolist()} for r in roots],
      'functional_root_equivalence':{'max_p95_px':fe95,'max_px':femax,'pairs':functional},
      'gates':gates,'permissions':{'physical_camera_center_allowed':passed,'metric_event_camera_allowed':passed,'replay_render_allowed':False},
      'retired_constraint':'legacy backboard inner-target pixels excluded; target rim centre singleton replaced by real visible rim segment'
    }
    report=json_safe(report); (a.out/'left_above_rim_visible_rim_functional_camera_v42.json').write_text(json.dumps(report,indent=2)+'\n')
    # Diagnostic overlay only.
    out=target_im.copy(); pred=v41.project(C,pp,par,v41.RIM)
    for p in target_rim:cv2.circle(out,tuple(np.round(p).astype(int)),2,(0,255,0),-1)
    for p in pred:
        if 430<p[0]<525 and 150<p[1]<200:cv2.circle(out,tuple(np.round(p).astype(int)),1,(255,0,255),-1)
    cv2.imwrite(str(a.out/'target_visible_rim_overlay_v42.png'),out)
    print(json.dumps(report,indent=2)); raise SystemExit(0 if passed else 2)
if __name__=='__main__':main()
