from __future__ import annotations
import argparse,json,math,hashlib
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44
from freeze_spin import solve_broadcast_direct_target_lines_v87 as v87

TARGET_KEYS=('target_top','target_left','target_right')

def sha256(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def p_from(C,rv,logf,cx,cy):
 R=cv2.Rodrigues(np.asarray(rv,float).reshape(3,1))[0]; t=-R@np.asarray(C,float)
 return np.r_[rv,t,logf,cx,cy]

def unpack(z,n):
 C=z[:3]; cx,cy=z[-2:]; ps=[]; o=3
 for _ in range(n):
  ps.append(p_from(C,z[o:o+3],z[o+3],cx,cy)); o+=4
 return C,ps

def state_residual(p,s):
 rows=[]; H=v87.floor_homography(p)
 for k,pts in s.get('floor_train_px',{}).items():
  a=np.asarray(pts,float)
  if len(a): rows.append(v44.signed_pixel_residual(H,k,a))
 for k in TARGET_KEYS:
  a=np.asarray(s.get('target_line_samples_px',{}).get(k,[]),float)
  if len(a):
   uv,_=v87.project3(p,v87.world_target_lines()[k]); rows.append(v87.signed_line_distance(a,uv[0],uv[1]))
 return np.concatenate(rows) if rows else np.array([1e6])
def residual(z,states):
 _,ps=unpack(z,len(states)); return np.concatenate([state_residual(p,s) for p,s in zip(ps,states)])
def bounds(n):
 lo=np.r_[[-20000.]*3, np.tile(np.r_[[-10.]*3,math.log(150.)],n), -1000.,-1000.]
 hi=np.r_[[20000.]*3, np.tile(np.r_[[10.]*3,math.log(8000.)],n), 2000.,2000.]
 return lo,hi
def solve(z0,states,max_nfev=30000):
 lo,hi=bounds(len(states)); q=least_squares(lambda z:residual(z,states),z0,bounds=(lo,hi),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=max_nfev)
 return q.x,float(q.cost)
def metrics(p,s):
 out={}
 for k,pts in s.get('floor_holdout_px',{}).items():
  a=np.asarray(pts,float); d=np.abs(v44.signed_pixel_residual(v87.floor_homography(p),k,a)); out['floor_'+k]={'p95_px':float(np.percentile(d,95)),'max_px':float(d.max())}
 for k in TARGET_KEYS:
  a=np.asarray(s.get('target_holdout_px',{}).get(k,[]),float)
  if len(a):
   uv,_=v87.project3(p,v87.world_target_lines()[k]); d=np.abs(v87.signed_line_distance(a,uv[0],uv[1])); out['target_'+k]={'p95_px':float(np.percentile(d,95)),'max_px':float(d.max())}
 rim=np.asarray(s.get('heldout_rim_samples_px',[]),float)
 if len(rim):
  th=np.linspace(0,2*np.pi,1601,endpoint=False); P=np.c_[15*2.54+9*2.54*np.cos(th),9*2.54*np.sin(th),np.full_like(th,10*30.48)]
  uv,_=v87.project3(p,P); d=np.sqrt(((rim[:,None,:]-uv[None,:,:])**2).sum(2)).min(1); out['rim']={'p95_px':float(np.percentile(d,95)),'max_px':float(d.max())}
 return out
def maxp(m): return max([v['p95_px'] for v in m.values()] or [999.])
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--spec',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--perturbation-trials',type=int,default=64); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 sp=json.loads(a.spec.read_text()); states=sp['states']; n=len(states); assert n>=3
 for s in states:
  if s.get('image_path') and s.get('image_sha256'): assert sha256(s['image_path'])==s['image_sha256']
 seed=np.asarray(sp['seed'],float); z,cost=solve(seed,states); C,ps=unpack(z,n)
 nominal=[metrics(p,s) for p,s in zip(ps,states)]
 rng=np.random.default_rng(1070301); roots=[]
 for i in range(8):
  zz=z.copy(); zz[:3]+=rng.uniform(-400,400,3); zz[-2:]+=rng.uniform(-120,120,2)
  for j in range(n): zz[3+4*j+3]+=math.log(rng.uniform(.7,1.3))
  try: r,cc=solve(zz,states,18000); Cr,_=unpack(r,n); roots.append(float(np.linalg.norm(Cr-C)))
  except Exception: roots.append(1e9)
 support=[]; fam=[]
 for si,s in enumerate(states):
  fam += [(si,'floor',k) for k,v in s.get('floor_train_px',{}).items() if len(v)]
  fam += [(si,'target',k) for k,v in s.get('target_line_samples_px',{}).items() if len(v)]
 for si,typ,k in fam:
  ss=json.loads(json.dumps(states)); key='floor_train_px' if typ=='floor' else 'target_line_samples_px'; ss[si][key][k]=[]
  try: r,_=solve(z,ss,15000); Cr,_=unpack(r,n); support.append({'state':states[si]['event'],'family':f'{typ}:{k}','center_shift_cm':float(np.linalg.norm(Cr-C))})
  except Exception: support.append({'state':states[si]['event'],'family':f'{typ}:{k}','center_shift_cm':1e9})
 pert=[]
 for t in range(a.perturbation_trials):
  rr=np.random.default_rng(107000+t); ss=json.loads(json.dumps(states))
  for s in ss:
   for key in ('floor_train_px','target_line_samples_px'):
    for k,v in s.get(key,{}).items():
     A=np.asarray(v,float)
     if len(A): s[key][k]=(A+rr.choice([-.5,.5],size=A.shape)).tolist()
  try: r,_=solve(z,ss,12000); Cr,_=unpack(r,n); pert.append(float(np.linalg.norm(Cr-C)))
  except Exception: pert.append(1e9)
 gates={'nominal_holdout_p95_le_2px':max(maxp(m) for m in nominal)<=2.0,'multistart_center_shift_le_75cm':max(roots)<=75.0,'support_reduction_center_shift_le_75cm':max(x['center_shift_cm'] for x in support)<=75.0 if support else False,'half_pixel_center_shift_le_75cm':max(pert)<=75.0}
 status='PASS_RIGHT_SLASH_SHARED_CENTER_V107' if all(gates.values()) else 'FAIL_RIGHT_SLASH_SHARED_CENTER_V107'
 out={'status':status,'shared_center_cm':C.tolist(),'shared_principal_point_px':z[-2:].tolist(),'focal_px':[float(np.exp(p[6])) for p in ps],'nominal':nominal,'multistart_max_center_shift_cm':max(roots),'support_reduction':support,'support_max_center_shift_cm':max(x['center_shift_cm'] for x in support) if support else None,'half_pixel_max_center_shift_cm':max(pert),'gates':gates,'permissions':{'metric_camera_allowed':False,'replay_render_allowed':False}}
 (a.out/'right_slash_shared_center_v107.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
 if status.startswith('FAIL'): raise SystemExit(2)
if __name__=='__main__': main()
