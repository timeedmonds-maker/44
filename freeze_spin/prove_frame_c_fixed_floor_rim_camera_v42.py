from __future__ import annotations

"""v42 exact Frame C camera: fixed accepted v35 floor homography + regulation rim.

The v35 homography is a durable floor primitive and is not re-fit here. v41 must
first independently authorize the game-level physical centre using other events.
Event 489 contributes only its accepted v35 floor homography and a source-pixel
rim-centre observation. The legacy backboard target is diagnostic-only.
"""

import argparse,json,math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

FT=30.48; IN=2.54
RIM=np.array([[15*IN,0.,10*FT]],float)
SEEDS=((480.,270.),(440.,450.),(500.,450.),(460.,380.),(480.,500.),(456.,478.))


def K(f,pp):return np.array([[f,0,pp[0]],[0,f,pp[1]],[0,0,1.]],float)

def decomp(H,cx,cy):
    h1,h2,h3=H[:,0],H[:,1],H[:,2];a1=np.array([h1[0]-cx*h1[2],h1[1]-cy*h1[2]]);a2=np.array([h2[0]-cx*h2[2],h2[1]-cy*h2[2]]);c=[]
    if abs(h1[2]*h2[2])>1e-12:
        z=-(a1@a2)/(h1[2]*h2[2]);
        if z>0:c.append(z)
    d=h1[2]**2-h2[2]**2
    if abs(d)>1e-12:
        z=-(a1@a1-a2@a2)/d
        if z>0:c.append(z)
    if not c:raise RuntimeError('decomposition failed')
    f=math.sqrt(float(np.median(c)));ki=np.linalg.inv(K(f,[cx,cy]));q1,q2,q3=ki@h1,ki@h2,ki@h3;l=2/(np.linalg.norm(q1)+np.linalg.norm(q2));r0=np.c_[l*q1,l*q2,np.cross(l*q1,l*q2)];u,_,v=np.linalg.svd(r0);R=u@v
    if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
    rv,_=cv2.Rodrigues(R);return f,rv.ravel()

def project(C,pp,f,rv,P):
    R,_=cv2.Rodrigues(rv.reshape(3,1));z=(R@(P-C).T).T
    return np.c_[f*z[:,0]/z[:,2]+pp[0],f*z[:,1]/z[:,2]+pp[1]]

def grid(H):
    P=np.array([[x,y,0.] for x in np.linspace(-4*FT,30*FT,12) for y in np.linspace(-25*FT,25*FT,15)],float);ph=np.c_[P[:,:2],np.ones(len(P))];z=(H@ph.T).T;U=z[:,:2]/z[:,2,None];m=(z[:,2]>0)&(U[:,0]>20)&(U[:,0]<940)&(U[:,1]>20)&(U[:,1]<520);return P[m],U[m]
def fit(C,H,rim,seed,warm=None,U_override=None):
    P,U=grid(H);U=U if U_override is None else U_override
    f0,rv0=decomp(H,*seed);x0=np.r_[rv0,math.log(f0),seed] if warm is None else np.asarray(warm,float)
    lo=np.r_[[-10]*3,math.log(250),100,50];hi=np.r_[[10]*3,math.log(2500),850,520]
    def fun(x):
        rv=x[:3];f=np.exp(x[3]);pp=x[4:6]
        return np.r_[(project(C,pp,f,rv,P)-U).ravel(),project(C,pp,f,rv,RIM)[0]-rim]
    o=least_squares(fun,x0,bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=12000);return o.x,float(np.mean(fun(o.x)**2)),P,U

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v41',type=Path,required=True);ap.add_argument('--floor-proof',type=Path,required=True);ap.add_argument('--rim-observations',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--perturbation-trials',type=int,default=24);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    v41=json.loads(a.v41.read_text());floor=json.loads(a.floor_proof.read_text());rs=json.loads(a.rim_observations.read_text())
    if v41.get('status')!='PASS_NONCOPLANAR_GAME_CAMERA_CENTER_V41' or not v41['permissions']['physical_camera_center_allowed']:raise RuntimeError('v41 centre not authorized')
    if floor.get('status')!='PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35':raise RuntimeError('v35 floor not accepted')
    C=np.asarray(v41['camera_center_cm'],float);H=np.asarray(floor['floor_homography_world_to_image'],float);rim=np.asarray(rs['held_out_target_event']['rim_center_px_diagnostic_only'],float)
    roots=[]
    for s in SEEDS:
        x,cost,P,U=fit(C,H,rim,s);roots.append({'seed_pp':list(s),'x':x,'cost':cost,'focal_px':float(np.exp(x[3])),'pp_px':x[4:6].tolist()})
    roots.sort(key=lambda z:z['cost']);best=roots[0];x=np.asarray(best['x']);f=float(np.exp(x[3]));pp=x[4:6];rv=x[:3];P,U=grid(H);fe=np.linalg.norm(project(C,pp,f,rv,P)-U,axis=1);re=float(np.linalg.norm(project(C,pp,f,rv,RIM)[0]-rim))
    pp_spread=max(float(np.linalg.norm(np.asarray(a0['pp_px'])-np.asarray(b0['pp_px']))) for i,a0 in enumerate(roots) for b0 in roots[i+1:]);ff=[r['focal_px'] for r in roots];fspread=(max(ff)-min(ff))/np.mean(ff)
    rng=np.random.default_rng(20260903);pc=float(v41['perturbation']['max_center_shift_cm']);pert=[]
    for _ in range(a.perturbation_trials):
        d=rng.normal(size=3);d/=max(np.linalg.norm(d),1e-12);Cq=C+d*rng.uniform(0,pc);Uq=U+rng.uniform(-.5,.5,U.shape);rq=rim+rng.uniform(-.5,.5,2);q,_,_,_=fit(Cq,H,rq,tuple(pp),warm=x,U_override=Uq);fq=float(np.exp(q[3]));pert.append({'pp_shift_px':float(np.linalg.norm(q[4:6]-pp)),'focal_fraction':abs(fq-f)/f})
    maxpps=max(z['pp_shift_px'] for z in pert);maxffs=max(z['focal_fraction'] for z in pert)
    gates={'root_pp':pp_spread<=0.05,'root_focal':fspread<=0.0001,'fixed_floor_p95':float(np.percentile(fe,95))<=0.5,'rim_center':re<=1.5,'perturb_pp':maxpps<=10.0,'perturb_focal':maxffs<=0.005}
    passed=all(gates.values())
    rep={'schema_version':1,'status':'PASS_FIXED_FLOOR_RIM_EVENT_CAMERA_V42' if passed else 'FAIL_FIXED_FLOOR_RIM_EVENT_CAMERA_V42','game_id':'0022500301','event_id':489,'camera_label':'Left Above Rim','physical_center_cm':C.tolist(),'camera':{'rvec':rv.tolist(),'focal_px':f,'principal_point_px':pp.tolist()},'method':'v41 independent physical centre + fixed accepted v35 floor homography + regulation 3D rim centre','fixed_floor':{'grid_points':int(len(fe)),'rms_px':float(np.sqrt(np.mean(fe**2))),'p95_px':float(np.percentile(fe,95)),'max_px':float(np.max(fe))},'rim_center':{'observed_px':rim.tolist(),'predicted_px':project(C,pp,f,rv,RIM)[0].tolist(),'error_px':re},'multistart':{'roots':[{k:v for k,v in r.items() if k!='x'} for r in roots],'max_pp_pairwise_px':pp_spread,'focal_spread_fraction':fspread},'perturbation':{'trials':len(pert),'source_center_radius_cm':pc,'max_pp_shift_px':maxpps,'max_focal_fraction':maxffs},'legacy_backboard_target_policy':'diagnostic only; excluded from fit and permissions','gates':gates,'permissions':{'metric_event_camera_allowed':passed,'replay_render_allowed':False}}
    (a.out/'frame_c_left_above_rim_fixed_floor_rim_camera_v42.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
