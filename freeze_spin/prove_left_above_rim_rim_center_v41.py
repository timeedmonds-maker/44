from __future__ import annotations

"""v41: independent non-coplanar Left Above Rim game-camera centre proof.

Uses accepted v35 target floor homography plus source-pixel floor transfer from
four independent same-game events and one regulation-defined 3D rim centre per
event. Event 489 is completely excluded from the shared-centre fit. The retired
v26 baseline anchors, v26 backboard target and v1/v36 centre are not fit inputs.
Passing promotes only a game-level physical centre; event-camera/replay remain false.
"""

import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

W,H=960,540
FT=30.48


def apply_h(Hm,p):
    z=cv2.perspectiveTransform(np.asarray(p,np.float32)[:,None,:],np.asarray(Hm,float))[:,0]
    return z.astype(float)


def sift(a,b):
    s=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015)
    ka,da=s.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None)
    kb,db=s.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None:return np.empty((0,2)),np.empty((0,2))
    good=[]
    for m,n in cv2.BFMatcher().knnMatch(da,db,k=2):
        if m.distance<.72*n.distance:good.append(m)
    return np.float64([ka[m.queryIdx].pt for m in good]),np.float64([kb[m.trainIdx].pt for m in good])


def floor_transfer(src,target):
    p,q=sift(src,target)
    m=(p[:,1]>245)&(q[:,1]>245)&(p[:,0]>80)&(p[:,0]<880)&(q[:,0]>80)&(q[:,0]<880)
    if int(m.sum())<40:raise RuntimeError('insufficient floor matches')
    Hm,mask=cv2.findHomography(p[m],q[m],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
    if Hm is None:raise RuntimeError('floor transfer homography failed')
    ii=mask.ravel().astype(bool); pp,qq=p[m][ii],q[m][ii]
    e=np.linalg.norm(apply_h(Hm,pp)-qq,axis=1)
    if len(e)<35 or np.percentile(e,95)>1.5:raise RuntimeError('floor transfer gate failed')
    return Hm,pp,qq,{'raw_matches':int(len(p)),'floor_candidates':int(m.sum()),'inliers':int(ii.sum()),'train_p95_px':float(np.percentile(e,95))}


def grid_obs(Hm):
    P=np.array([[x,y,0.] for x in np.linspace(-4*FT,30*FT,12) for y in np.linspace(-25*FT,25*FT,15)],float)
    ph=np.c_[P[:,:2],np.ones(len(P))];z=(Hm@ph.T).T;uv=z[:,:2]/z[:,2,None]
    m=(z[:,2]>0)&(uv[:,0]>20)&(uv[:,0]<940)&(uv[:,1]>20)&(uv[:,1]<520)
    return P[m],uv[m]


def K(f,pp):return np.array([[f,0,pp[0]],[0,f,pp[1]],[0,0,1.]],float)


def decomp(Hm,cx,cy):
    h1,h2,h3=Hm[:,0],Hm[:,1],Hm[:,2]
    a1=np.array([h1[0]-cx*h1[2],h1[1]-cy*h1[2]])
    a2=np.array([h2[0]-cx*h2[2],h2[1]-cy*h2[2]])
    cand=[]
    if abs(h1[2]*h2[2])>1e-12:
        x=-(a1@a2)/(h1[2]*h2[2])
        if x>0:cand.append(x)
    d=h1[2]**2-h2[2]**2
    if abs(d)>1e-12:
        x=-(a1@a1-a2@a2)/d
        if x>0:cand.append(x)
    if not cand:raise RuntimeError('homography decomposition failed')
    f=math.sqrt(float(np.median(cand)));ki=np.linalg.inv(K(f,[cx,cy]))
    q1,q2,q3=ki@h1,ki@h2,ki@h3;lam=2/(np.linalg.norm(q1)+np.linalg.norm(q2))
    r0=np.c_[lam*q1,lam*q2,np.cross(lam*q1,lam*q2)];u,_,v=np.linalg.svd(r0);R=u@v
    if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
    rv,_=cv2.Rodrigues(R);C=-R.T@(lam*q3)
    return f,C,rv.ravel()


def project(C,pp,f,rv,P):
    R,_=cv2.Rodrigues(np.asarray(rv,float).reshape(3,1));z=(R@(P-C).T).T
    return np.c_[f*z[:,0]/z[:,2]+pp[0],f*z[:,1]/z[:,2]+pp[1]]


def solve(keys,Hs,grid,rims,warm=None,max_nfev=15000):
    n=len(keys)
    if warm is None:
        pp0=np.array([456.0,400.0]);cs=[];blocks=[]
        for e in keys:
            f,C,rv=decomp(Hs[e],*pp0);cs.append(C);blocks.append([math.log(f),*rv])
        x0=np.r_[np.mean(cs,axis=0),pp0,np.asarray(blocks).ravel()]
    else:
        C0=np.asarray(warm[:3]);pp0=np.asarray(warm[3:5]);blocks=[]
        for e in keys:
            f,_,rv=decomp(Hs[e],*pp0);blocks.append([math.log(f),*rv])
        x0=np.r_[C0,pp0,np.asarray(blocks).ravel()]
    lo=np.r_[[-5000,-3000,50],[100,50],np.tile(np.r_[math.log(250),[-10,-10,-10]],n)]
    hi=np.r_[[5000,3000,1500],[850,520],np.tile(np.r_[math.log(2500),[10,10,10]],n)]
    def un(x):return x[:3],x[3:5],x[5:].reshape(n,4)
    def fun(x):
        C,pp,b=un(x);out=[]
        for i,e in enumerate(keys):
            f=np.exp(b[i,0]);rv=b[i,1:];P,U=grid[e]
            out.append((project(C,pp,f,rv,P)-U).ravel())
            out.append((project(C,pp,f,rv,np.asarray([rims['world_cm']],float))[0]-np.asarray(rims['obs'][e],float)).ravel())
        return np.concatenate(out)
    opt=least_squares(fun,x0,bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=max_nfev)
    C,pp,b=un(opt.x)
    return {'x':opt.x,'C':C,'pp':pp,'blocks':b,'rms':float(np.sqrt(np.mean(fun(opt.x)**2)))}


def metrics(sol,keys,grid,rims):
    out={}
    for i,e in enumerate(keys):
        f=np.exp(sol['blocks'][i,0]);rv=sol['blocks'][i,1:];P,U=grid[e]
        fe=np.linalg.norm(project(sol['C'],sol['pp'],f,rv,P)-U,axis=1)
        rp=project(sol['C'],sol['pp'],f,rv,np.asarray([rims['world_cm']],float))[0]
        out[e]={'focal_px':float(f),'floor_p95_px':float(np.percentile(fe,95)),'rim_center_error_px':float(np.linalg.norm(rp-np.asarray(rims['obs'][e],float))),'rim_predicted_px':rp.tolist()}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target-frame',type=Path,required=True);ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--floor-proof',type=Path,required=True);ap.add_argument('--rim-observations',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--perturbation-trials',type=int,default=24);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    floor=json.loads(a.floor_proof.read_text());spec=json.loads(a.rim_observations.read_text())
    if floor.get('status')!='PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35':raise RuntimeError('v35 floor not accepted')
    Ht=np.asarray(floor['floor_homography_world_to_image'],float);target=cv2.imread(str(a.target_frame))
    if target is None or target.shape[:2]!=(H,W):raise RuntimeError('bad immutable target frame')
    obsrows={int(x['event_id']):x for x in spec['observations']};keys=sorted(obsrows);Hs={};grid={};diag={};corr={}
    for e in keys:
        wanted=obsrows[e]['sample_file'];p=a.samples/wanted
        if not p.exists():raise RuntimeError(f'missing locked sample {wanted}')
        src=cv2.imread(str(p));Hst,sp,tp,d=floor_transfer(src,target);Hs[e]=np.linalg.inv(Hst)@Ht;Hs[e]/=Hs[e][2,2];grid[e]=grid_obs(Hs[e]);diag[e]=d;corr[e]=(sp,tp)
    rims={'world_cm':spec['regulation_rim_center_world_cm'],'obs':{e:obsrows[e]['rim_center_px'] for e in keys}}
    full=solve(keys,Hs,grid,rims);per=metrics(full,keys,grid,rims)
    loo={}
    for drop in keys:
        sub=[e for e in keys if e!=drop];s=solve(sub,Hs,grid,rims,warm=full['x'],max_nfev=8000)
        loo[drop]={'center_shift_cm':float(np.linalg.norm(s['C']-full['C'])),'pp_shift_px':float(np.linalg.norm(s['pp']-full['pp']))}
    rng=np.random.default_rng(20260903);pert=[]
    for _ in range(a.perturbation_trials):
        Hp={};gp={};ro={}
        for e in keys:
            sp,tp=corr[e]
            spp=sp+rng.uniform(-.5,.5,sp.shape);tpp=tp+rng.uniform(-.5,.5,tp.shape)
            Hst,_=cv2.findHomography(spp.astype(np.float32),tpp.astype(np.float32),0)
            if Hst is None:raise RuntimeError('perturbed transfer fit failed')
            Hw=np.linalg.inv(Hst)@Ht;Hw/=Hw[2,2];Hp[e]=Hw;gp[e]=grid_obs(Hw)
            ro[e]=(np.asarray(rims['obs'][e])+rng.uniform(-.5,.5,2)).tolist()
        s=solve(keys,Hp,gp,{'world_cm':rims['world_cm'],'obs':ro},warm=full['x'],max_nfev=6000)
        pert.append({'center_shift_cm':float(np.linalg.norm(s['C']-full['C'])),'pp_shift_px':float(np.linalg.norm(s['pp']-full['pp']))})
    maxfloor=max(x['floor_p95_px'] for x in per.values());maxrim=max(x['rim_center_error_px'] for x in per.values());maxloo=max(x['center_shift_cm'] for x in loo.values());maxpp=max(x['pp_shift_px'] for x in loo.values());maxpc=max(x['center_shift_cm'] for x in pert);maxppp=max(x['pp_shift_px'] for x in pert)
    gates={'shared_rms':full['rms']<=0.75,'floor_p95':maxfloor<=2.0,'rim_center':maxrim<=4.0,'loo_center':maxloo<=5.0,'loo_pp':maxpp<=6.0,'perturb_center':maxpc<=5.0,'perturb_pp':maxppp<=6.0}
    passed=all(gates.values())
    rep={'schema_version':2,'status':'PASS_NONCOPLANAR_GAME_CAMERA_CENTER_V41' if passed else 'FAIL_NONCOPLANAR_GAME_CAMERA_CENTER_V41','game_id':'0022500301','camera_label':'Left Above Rim','method':'independent same-game source floor transfer + regulation 3D rim centre; target event 489 excluded','perturbation_method':'jitter real source/target SIFT inlier coordinates by +/-0.5 px, refit each transfer homography, then perturb rim observations +/-0.5 px','camera_center_cm':full['C'].tolist(),'shared_principal_point_diagnostic_px':full['pp'].tolist(),'shared_rms_px':full['rms'],'selected_samples':{str(e):obsrows[e]['sample_file'] for e in keys},'transfer_diagnostics':{str(e):diag[e] for e in keys},'per_event':{str(e):per[e] for e in keys},'leave_one_event_out':{str(e):loo[e] for e in keys},'perturbation':{'trials':len(pert),'max_center_shift_cm':maxpc,'max_pp_shift_px':maxppp,'p95_center_shift_cm':float(np.percentile([x['center_shift_cm'] for x in pert],95)),'p95_pp_shift_px':float(np.percentile([x['pp_shift_px'] for x in pert],95))},'gates':gates,'deprecated_constraints':['v26 baseline floor anchors','v26 backboard target as camera fit constraint','v1 centre','v36 floor-only centre if this v41 proof passes'],'permissions':{'physical_camera_center_allowed':passed,'principal_point_prior_allowed':False,'metric_event_camera_allowed':False,'replay_render_allowed':False}}
    (a.out/'left_above_rim_noncoplanar_center_v41.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps(rep,indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
