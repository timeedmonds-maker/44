from __future__ import annotations

"""Right Slash v102 mutual-correspondence multi-hub pair graph.

Failure-mode repair only. v101 showed that independent one-way SIFT sets can each
contain different descriptor outliers: a geometrically excellent H can pass in one
direction while a few unrelated withheld descriptors explode the opposite p90.
This version does not relax any transfer threshold. It changes correspondence
construction to require mutual Lowe-ratio agreement, then applies the SAME training
and held-out pixel gates. Multiple candidate hubs are tested so event 540 is not
privileged. Passing still authorizes only an intrinsics prior, never a metric camera.
"""

import argparse, json, math, re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from scipy.optimize import least_squares

W,H=960,540
EVENT_RE=re.compile(r"event_(\d+)_frames$")
HUBS=[(15,'f03.png'),(300,'f03.png'),(410,'f00.png'),(415,'f06.png'),(540,'f02.png'),(690,'f04.png')]


def event_id(p:Path)->int:
    m=EVENT_RE.search(p.parent.name)
    return int(m.group(1)) if m else -1

def action_core(xy):
    x,y=xy[:,0],xy[:,1]
    return (x>.20*W)&(x<.80*W)&(y>.48*H)&(y<.98*H)

def stats(e):
    if not len(e): return {'n':0,'median_px':None,'p90_px':None,'p95_px':None}
    return {'n':int(len(e)),'median_px':float(np.median(e)),'p90_px':float(np.percentile(e,90)),'p95_px':float(np.percentile(e,95))}

def K(f,pp):
    return np.asarray([[f,0,pp[0]],[0,f,pp[1]],[0,0,1.]],float)

class Cache:
    def __init__(self,paths):
        sift=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015)
        self.f={}
        for p in paths:
            im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape!=(H,W): continue
            kp,d=sift.detectAndCompute(im,None)
            self.f[p]=(kp,d)
    def mutual(self,a,b):
        ka,da=self.f.get(a,(None,None)); kb,db=self.f.get(b,(None,None))
        if da is None or db is None: return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
        bf=cv2.BFMatcher(cv2.NORM_L2)
        ab=bf.knnMatch(da,db,k=2); ba=bf.knnMatch(db,da,k=2)
        gab={m.queryIdx:m for m,n in ab if m.distance<.72*n.distance}
        gba={m.queryIdx:m for m,n in ba if m.distance<.72*n.distance}
        ms=[]
        for qi,m in gab.items():
            r=gba.get(m.trainIdx)
            if r is not None and r.trainIdx==qi: ms.append(m)
        if not ms: return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
        return np.float32([ka[m.queryIdx].pt for m in ms]),np.float32([kb[m.trainIdx].pt for m in ms])

def audit(cache,a,b):
    p,q=cache.mutual(a,b)
    rec={'source':str(a),'target':str(b),'mutual_match_count':int(len(p)),'pass':False}
    if len(p)<30: rec['status']='insufficient_mutual_matches'; return rec
    xa,ya=p[:,0],p[:,1]; xb,yb=q[:,0],q[:,1]
    tg=((ya<.46*H)|(xa<.14*W)|(xa>.86*W))&((yb<.46*H)|(xb<.14*W)|(xb>.86*W))
    train=tg&~action_core(p)&~action_core(q)
    held=~train&~action_core(p)&~action_core(q)
    rec['training_count']=int(train.sum()); rec['withheld_count']=int(held.sum())
    if int(train.sum())<12: rec['status']='insufficient_background_training'; return rec
    M,mask=cv2.findHomography(p[train],q[train],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
    if M is None or mask is None: rec['status']='homography_failed'; return rec
    ii=mask.ravel().astype(bool)
    pred=cv2.perspectiveTransform(p[:,None,:],M)[:,0]
    e=np.linalg.norm(pred-q,axis=1)
    tr=e[np.where(train)[0][ii]]; wh=e[held]
    trs,whs=stats(tr),stats(wh)
    gates={
      'training_inliers_at_least_24':int(ii.sum())>=24,
      'training_p95_at_most_1_5px':trs['p95_px'] is not None and trs['p95_px']<=1.5,
      'withheld_matches_at_least_10':whs['n']>=10,
      'withheld_median_at_most_2_5px':whs['median_px'] is not None and whs['median_px']<=2.5,
      'withheld_p90_at_most_4px':whs['p90_px'] is not None and whs['p90_px']<=4.0,
    }
    rec.update({'training_inliers':int(ii.sum()),'training_error':trs,'withheld_error':whs,'gates':gates})
    rec['pass']=bool(all(gates.values())); rec['status']='mutual_transfer_pass' if rec['pass'] else 'transfer_rejected'
    if rec['pass']: rec['H_source_to_target']=M.tolist()
    return rec

def rotation_residual(x,rows):
    pp=x[:2]; ft=math.exp(float(x[2])); out=[]
    for i,row in enumerate(rows):
        fs=math.exp(float(x[3+i])); M=np.linalg.inv(K(ft,pp))@np.asarray(row['H_source_to_target'],float)@K(fs,pp)
        det=float(np.linalg.det(M))
        if abs(det)<1e-12: out.extend([100.]*7); continue
        M=M/np.cbrt(abs(det)); M=-M if det<0 else M
        A=M.T@M-np.eye(3)
        out.extend((5*A[np.triu_indices(3)]).tolist()); out.append(5*(np.linalg.det(M)-1))
    out.extend([(pp[0]-W/2)/350.,(pp[1]-H/2)/350.,(math.log(ft)-math.log(550.))/1.8])
    return np.asarray(out,float)
def selfcal(rows,seed=None):
    n=len(rows); seeds=[seed] if seed else [(480.,270.,350.),(480.,330.,550.),(520.,300.,700.),(440.,300.,700.),(500.,290.,1000.)]
    lo=np.r_[0.,0.,math.log(150.),np.repeat(math.log(150.),n)]; hi=np.r_[960.,540.,math.log(4000.),np.repeat(math.log(4000.),n)]
    best=None; bs=1e99
    for sx,sy,sf in seeds:
        x0=np.r_[sx,sy,math.log(sf),np.repeat(math.log(sf),n)]
        o=least_squares(lambda z:rotation_residual(z,rows),x0,bounds=(lo,hi),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=12000)
        s=float(np.mean(rotation_residual(o.x,rows)**2))
        if np.isfinite(s) and s<bs: bs,best=s,o.x
    if best is None: raise RuntimeError('selfcal failed')
    return best

def radial_diag(row,pp,ft,fs):
    Hm=np.asarray(row['H_source_to_target'],float); M=np.linalg.inv(K(ft,pp))@Hm@K(fs,pp)
    det=np.linalg.det(M); M=M/np.cbrt(abs(det)); M=-M if det<0 else M
    U,_,Vt=np.linalg.svd(M); R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    Hr=K(ft,pp)@R@np.linalg.inv(K(fs,pp)); Hr/=Hr[2,2]
    xs=np.linspace(45,W-45,15); ys=np.linspace(35,H-35,10)
    p=np.asarray([[x,y] for y in ys for x in xs],np.float32); p=p[~action_core(p)]
    q0=cv2.perspectiveTransform(p[:,None,:],Hm)[:,0]; q1=cv2.perspectiveTransform(p[:,None,:],Hr)[:,0]
    r=q1-q0; v=q0-pp; radius=np.linalg.norm(v,axis=1); u=v/np.maximum(radius[:,None],1e-9)
    rr=np.sum(r*u,axis=1); tt=r[:,0]*(-u[:,1])+r[:,1]*u[:,0]
    return {'radial_corr':float(np.corrcoef(radius,rr)[0,1]),'radial_slope_px_per_px':float(np.polyfit(radius,rr,1)[0]),
            'median_abs_radial_px':float(np.median(np.abs(rr))),'median_abs_tangential_px':float(np.median(np.abs(tt)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--min-events',type=int,default=4); ap.add_argument('--max-loo-pp-shift-px',type=float,default=8.0); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    paths=sorted(a.bank.glob('event_*_frames/f*.png')); cache=Cache(paths)
    hub_rows=[]; all_edges=[]
    for he,hf in HUBS:
        hub=a.bank/f'event_{he}_frames'/hf
        if hub not in cache.f: continue
        by_event=defaultdict(list)
        for p in paths:
            if event_id(p)==he: continue
            z=audit(cache,p,hub); rec={'hub_event':he,'hub_frame':hf,'source_event':event_id(p),'source_frame':p.name,**z}; all_edges.append(rec)
            if z.get('pass'):
                score=(z['withheld_error']['p90_px'],z['withheld_error']['median_px'],-z['training_inliers'])
                by_event[event_id(p)].append((score,rec))
        best=[]
        for eid,rows in by_event.items(): rows.sort(key=lambda x:x[0]); best.append(rows[0][1])
        best.sort(key=lambda r:r['source_event'])
        hub_rows.append({'hub_event':he,'hub_frame':hf,'independent_passing_event_count':len(best),'best_edges':best})
    hub_rows.sort(key=lambda r:r['independent_passing_event_count'],reverse=True)
    besthub=hub_rows[0] if hub_rows else {'independent_passing_event_count':0,'best_edges':[]}
    report={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','method':'mutual Lowe-ratio correspondences; unchanged v1 transfer gates; multi-hub search; rotational self-calibration only after >=4 independent events','guardrail':'Diagnostic/intrinsics-prior only. No metric event camera or render promotion.','hubs':hub_rows,'all_edges':all_edges,'selected_hub':{'event':besthub.get('hub_event'),'frame':besthub.get('hub_frame')},'independent_passing_event_count':besthub.get('independent_passing_event_count',0)}
    if besthub.get('independent_passing_event_count',0)>=a.min_events:
        rows=besthub['best_edges']; x=selfcal(rows); pp=x[:2]; ft=math.exp(float(x[2])); loo=[]
        for eid in [r['source_event'] for r in rows]:
            sub=[r for r in rows if r['source_event']!=eid]; y=selfcal(sub,seed=(float(pp[0]),float(pp[1]),ft)); loo.append({'held_out_event':eid,'principal_point_px':y[:2].tolist(),'shift_px':float(np.linalg.norm(y[:2]-pp))})
        maxloo=max(z['shift_px'] for z in loo); rd=[{'event_id':r['source_event'],**radial_diag(r,pp,ft,math.exp(float(x[3+i])))} for i,r in enumerate(rows)]
        slopes=np.asarray([r['radial_slope_px_per_px'] for r in rd]); same=float(max(np.mean(slopes>0),np.mean(slopes<0))) if len(slopes) else 0.; coherent=bool(same>=.75 and np.median(np.abs(slopes))>=.002)
        gates={'independent_events_at_least_4':len(rows)>=a.min_events,'leave_one_whole_event_out_pp_shift_at_most_8px':maxloo<=a.max_loo_pp_shift_px}
        passed=bool(all(gates.values())); report.update({'status':'PASS_RIGHT_SLASH_INTRINSICS_PRIOR_V102' if passed else 'FAIL_RIGHT_SLASH_INTRINSICS_V102','shared_principal_point_px':pp.tolist(),'hub_focal_px':ft,'source_focal_px':[float(math.exp(v)) for v in x[3:]],'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':maxloo,'radial_residual_diagnostic':rd,'radial_slope_same_sign_fraction':same,'coherent_radial_distortion_pattern_detected':coherent,'gates':gates,'principal_point_prior_allowed':passed})
    else:
        report.update({'status':'FAIL_RIGHT_SLASH_PAIR_GRAPH_V102','gates':{'independent_events_at_least_4':False},'principal_point_prior_allowed':False})
    report['metric_event_camera_allowed']=False; report['replay_render_allowed']=False
    def conv(o):
        if isinstance(o,np.generic): return o.item()
        if isinstance(o,np.ndarray): return o.tolist()
        raise TypeError(type(o).__name__)
    (a.out/'right_slash_pair_graph_v102.json').write_text(json.dumps(report,indent=2,default=conv)+'\n')
    print(json.dumps({'status':report['status'],'selected_hub':report['selected_hub'],'events':report['independent_passing_event_count'],'pp':report.get('shared_principal_point_px'),'max_loo':report.get('max_leave_one_event_out_pp_shift_px'),'radial':report.get('coherent_radial_distortion_pattern_detected'),'hub_counts':[(h['hub_event'],h['independent_passing_event_count']) for h in hub_rows]},indent=2,default=conv),flush=True)

if __name__=='__main__': main()
