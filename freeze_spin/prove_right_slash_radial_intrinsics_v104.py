from __future__ import annotations
"""Right Slash v104: k1-only radial intrinsics failure-mode repair.

Reuses v102 mutual correspondences and UNCHANGED transfer gates. Fits only v102
training inliers with shared sensor PP + one shared Brown k1, per-state focal/rotation.
Held-out correspondences, whole-event LOO and half-pixel perturbation are independent
promotion gates. Passing grants an intrinsics prior only; metric camera/render stay off.
"""
import argparse,json,math
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from scipy.optimize import least_squares
from freeze_spin.diagnose_right_slash_pair_graph_v102 import Cache,HUBS,audit,event_id,action_core,stats,K
W,H=960,540

def payload(cache,src,hub):
 p,q=cache.mutual(src,hub); xa,ya=p[:,0],p[:,1];xb,yb=q[:,0],q[:,1]
 tg=((ya<.46*H)|(xa<.14*W)|(xa>.86*W))&((yb<.46*H)|(xb<.14*W)|(xb>.86*W))
 tr=tg&~action_core(p)&~action_core(q); wh=~tr&~action_core(p)&~action_core(q)
 M,mask=cv2.findHomography(p[tr],q[tr],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
 ii=mask.ravel().astype(bool); ids=np.where(tr)[0][ii]
 return {'fit_source_px':p[ids].astype(float),'fit_target_px':q[ids].astype(float),
         'held_source_px':p[wh].astype(float),'held_target_px':q[wh].astype(float),
         'H':np.asarray(M,float)}

def nearest_R(Hm,pp,ft,fs):
 M=np.linalg.inv(K(ft,pp))@Hm@K(fs,pp);d=float(np.linalg.det(M));M=M/np.cbrt(abs(d));M=-M if d<0 else M
 U,_,V=np.linalg.svd(M);R=U@V
 if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@V
 rv,_=cv2.Rodrigues(R);return rv.ravel()

def project(src,pp,fs,ft,k1,rv):
 Ks=K(fs,pp); d=np.asarray([k1,0,0,0,0],float)
 xu=cv2.undistortPoints(src.reshape(-1,1,2).astype(np.float64),Ks,d).reshape(-1,2)
 rays=np.c_[xu,np.ones(len(xu))];R,_=cv2.Rodrigues(rv);y=(R@rays.T).T
 yu=y[:,:2]/y[:,2:3];r2=np.sum(yu*yu,axis=1,keepdims=True);yd=yu*(1+k1*r2)
 return yd*ft+pp

def unpack(x,n):
 pp=x[:2];ft=math.exp(float(x[2]));k1=float(x[3]);st=[]
 for i in range(n):
  o=4+4*i;st.append((math.exp(float(x[o])),x[o+1:o+4]))
 return pp,ft,k1,st

def residual(x,rows,jitter=None):
 pp,ft,k1,st=unpack(x,len(rows));out=[]
 for i,r in enumerate(rows):
  a=r['geom']['fit_source_px'];b=r['geom']['fit_target_px']
  if jitter is not None:a=a+jitter[i][0];b=b+jitter[i][1]
  pred=project(a,pp,st[i][0],ft,k1,st[i][1]);out.append(((pred-b)/math.sqrt(max(len(a),1)/80.)).ravel())
 out.append(np.asarray([(pp[0]-480)/350.,(pp[1]-270)/350.,(math.log(ft)-math.log(700))/1.8,k1]))
 return np.concatenate(out)

def seed(rows,pp,ft,k1):
 a=[pp[0],pp[1],math.log(ft),k1]
 for r in rows:
  fs=max(300.,min(3000.,ft*1.3));a += [math.log(fs),*nearest_R(r['geom']['H'],np.asarray(pp,float),ft,fs)]
 return np.asarray(a,float)

def solve(rows,warm=None,jitter=None,full_multistart=True):
 n=len(rows);starts=[]
 if warm is not None:starts.append(np.asarray(warm,float))
 specs=[((480,270),900,-.2),((500,285),1400,.2),((480,300),1200,-.4),((520,280),1600,.4)] if full_multistart else [((480,270),1000,0.)]
 if warm is None or full_multistart:
  for pp,f,k in specs:starts.append(seed(rows,pp,f,k))
 lo=np.r_[0.,0.,math.log(150.),-1.,np.tile([math.log(150.),-np.inf,-np.inf,-np.inf],n)]
 hi=np.r_[960.,540.,math.log(4000.),1.,np.tile([math.log(4000.),np.inf,np.inf,np.inf],n)]
 best=None;bs=1e99
 for x0 in starts:
  try:o=least_squares(lambda z:residual(z,rows,jitter),x0,bounds=(lo,hi),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=15000)
  except Exception:continue
  s=float(np.mean(residual(o.x,rows,jitter)**2))
  if np.isfinite(s) and s<bs:bs,best=s,o.x
 if best is None:raise RuntimeError('v104 radial solve failed')
 return best

def subwarm(full,rows,sub):
 ix={(r['source_event'],r['source_frame']):i for i,r in enumerate(rows)};a=[*full[:4]]
 for r in sub:
  o=4+4*ix[(r['source_event'],r['source_frame'])];a += list(full[o:o+4])
 return np.asarray(a,float)

def qa(x,rows):
 pp,ft,k1,st=unpack(x,len(rows));out=[];ok=True
 for i,r in enumerate(rows):
  g=r['geom'];tr=np.linalg.norm(project(g['fit_source_px'],pp,st[i][0],ft,k1,st[i][1])-g['fit_target_px'],axis=1)
  wh=np.linalg.norm(project(g['held_source_px'],pp,st[i][0],ft,k1,st[i][1])-g['held_target_px'],axis=1);ts,ws=stats(tr),stats(wh)
  gates={'training_p95_at_most_1_5px':ts['p95_px']<=1.5,'withheld_matches_at_least_10':ws['n']>=10,
         'withheld_median_at_most_2_5px':ws['median_px']<=2.5,'withheld_p90_at_most_4px':ws['p90_px']<=4.}
  ok &= all(gates.values());out.append({'event_id':r['source_event'],'frame':r['source_frame'],'source_focal_px':st[i][0],'training':ts,'withheld':ws,'gates':gates})
 return out,bool(ok)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--perturbation-trials',type=int,default=12);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 cv2.setRNGSeed(0);cv2.setNumThreads(1);paths=sorted(a.bank.glob('event_*_frames/f*.png'));c=Cache(paths);hubs=[]
 for he,hf in HUBS:
  h=a.bank/f'event_{he}_frames'/hf
  if h not in c.f:continue
  by=defaultdict(list)
  for p in paths:
   if event_id(p)==he:continue
   z=audit(c,p,h)
   if z.get('pass'):by[event_id(p)].append(((z['withheld_error']['p90_px'],z['withheld_error']['median_px'],-z['training_inliers']),p,z))
  best=[]
  for e,rr in by.items():rr.sort(key=lambda x:x[0]);_,p,z=rr[0];best.append({'source_event':e,'source_frame':p.name,'source':str(p),'audit':z,'geom':payload(c,p,h)})
  best.sort(key=lambda r:r['source_event']);hubs.append({'hub_event':he,'hub_frame':hf,'count':len(best),'rows':best})
 hubs.sort(key=lambda r:r['count'],reverse=True);bh=hubs[0] if hubs else {'count':0,'rows':[]};rows=bh['rows']
 rep={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','model':'shared PP + shared Brown k1 only; k2=p1=p2=0; event focal/rotation','selected_hub':{'event':bh.get('hub_event'),'frame':bh.get('hub_frame')},'hub_counts':[(h['hub_event'],h['count']) for h in hubs],'independent_passing_event_count':len(rows),'metric_event_camera_allowed':False,'replay_render_allowed':False}
 if len(rows)<4:rep.update({'status':'FAIL_RIGHT_SLASH_RADIAL_INTRINSICS_V104_INSUFFICIENT_EVENTS','principal_point_prior_allowed':False,'gates':{'independent_events_at_least_4':False}})
 else:
  x=solve(rows);pp,ft,k1,st=unpack(x,len(rows));q,qp=qa(x,rows);focals=[ft]+[s[0] for s in st];fbound=all(155<f<3950 for f in focals)
  loo=[]
  for i,r in enumerate(rows):
   sub=[z for j,z in enumerate(rows) if j!=i];y=solve(sub,subwarm(x,rows,sub),full_multistart=True);yp,yf,yk,_=unpack(y,len(sub));loo.append({'held_out_event':r['source_event'],'principal_point_px':yp.tolist(),'pp_shift_px':float(np.linalg.norm(yp-pp)),'hub_focal_px':yf,'k1':yk,'k1_same_sign':bool(np.sign(yk)==np.sign(k1) and abs(yk)>1e-8)})
  ml=max(r['pp_shift_px'] for r in loo);sign=all(r['k1_same_sign'] for r in loo);mags=[abs(k1)]+[abs(r['k1']) for r in loo];ratio=max(mags)/max(min(mags),1e-8)
  rng=np.random.default_rng(104);pert=[];mpp=0.;mf=0.;psign=True
  for t in range(a.perturbation_trials):
   jit=[]
   for r in rows:
    n=len(r['geom']['fit_source_px']);jit.append((rng.uniform(-.5,.5,(n,2)),rng.uniform(-.5,.5,(n,2))))
   y=solve(rows,x,jit,full_multistart=False);yp,yf,yk,_=unpack(y,len(rows));ds=float(np.linalg.norm(yp-pp));df=abs(yf-ft)/ft;same=bool(np.sign(yk)==np.sign(k1) and abs(yk)>1e-8);mpp=max(mpp,ds);mf=max(mf,df);psign &= same;pert.append({'trial':t,'pp_shift_px':ds,'hub_focal_fraction_shift':df,'k1':yk,'k1_same_sign':same})
  gates={'independent_events_at_least_4':len(rows)>=4,'physical_model_pixel_gates':qp,'focals_not_on_bounds':fbound,'loo_pp_at_most_8px':ml<=8.,'loo_k1_sign_stable':sign,'loo_k1_magnitude_ratio_at_most_2x':ratio<=2.,'half_pixel_pp_at_most_5px':mpp<=5.,'half_pixel_hub_focal_at_most_5pct':mf<=.05,'half_pixel_k1_sign_stable':psign};passed=bool(all(gates.values()))
  rep.update({'status':'PASS_RIGHT_SLASH_RADIAL_INTRINSICS_PRIOR_V104' if passed else 'FAIL_RIGHT_SLASH_RADIAL_INTRINSICS_V104','shared_principal_point_px':pp.tolist(),'shared_k1_normalized_brown':k1,'hub_focal_px':ft,'source_states':[{'event_id':r['source_event'],'frame':r['source_frame'],'focal_px':st[i][0]} for i,r in enumerate(rows)],'physical_model_qa':q,'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':ml,'k1_loo_max_min_ratio':ratio,'perturbation_trials':pert,'max_half_pixel_pp_shift_px':mpp,'max_half_pixel_hub_focal_fraction':mf,'gates':gates,'principal_point_prior_allowed':passed})
 def cvt(o):
  if isinstance(o,np.generic):return o.item()
  if isinstance(o,np.ndarray):return o.tolist()
  raise TypeError(type(o).__name__)
 (a.out/'right_slash_radial_intrinsics_v104.json').write_text(json.dumps(rep,indent=2,default=cvt)+'\n');print(json.dumps({'status':rep['status'],'hub':rep['selected_hub'],'events':rep['independent_passing_event_count'],'pp':rep.get('shared_principal_point_px'),'k1':rep.get('shared_k1_normalized_brown'),'max_loo_pp':rep.get('max_leave_one_event_out_pp_shift_px'),'k1_ratio':rep.get('k1_loo_max_min_ratio'),'max_half_pixel_pp':rep.get('max_half_pixel_pp_shift_px'),'gates':rep.get('gates')},indent=2,default=cvt),flush=True)
if __name__=='__main__':main()
