from __future__ import annotations

"""Right Slash v103: oriented pair graph using the existing one-direction v1 gate.

For each source/hub pair, accept the edge only if a complete existing transfer audit
passes in at least one direction.  If only hub->source passes, invert that validated
homography to obtain source->hub.  Numerical transfer thresholds are unchanged.
This is deliberately distinct from v102's extra mutual-correspondence research gate.
Rotational self-calibration and whole-event LOO remain diagnostic/intrinsics-only.
"""

import argparse,json,math,re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from scipy.optimize import least_squares

W,H=960,540
EVENT_RE=re.compile(r'event_(\d+)_frames$')
HUBS=[(15,'f03.png'),(300,'f03.png'),(410,'f00.png'),(415,'f06.png'),(540,'f02.png'),(690,'f04.png')]

def eid(p):
 m=EVENT_RE.search(p.parent.name); return int(m.group(1)) if m else -1

def ac(x):
 a,b=x[:,0],x[:,1]; return (a>.20*W)&(a<.80*W)&(b>.48*H)&(b<.98*H)

def es(e):
 if not len(e): return {'n':0,'median_px':None,'p90_px':None,'p95_px':None}
 return {'n':int(len(e)),'median_px':float(np.median(e)),'p90_px':float(np.percentile(e,90)),'p95_px':float(np.percentile(e,95))}

class Cache:
 def __init__(self,paths):
  s=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015); self.d={}
  for p in paths:
   im=cv2.imread(str(p));
   if im is None or im.shape[:2]!=(H,W): continue
   k,d=s.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None); self.d[p]=(k,d)
 def pts(self,a,b):
  ka,da=self.d.get(a,(None,None)); kb,db=self.d.get(b,(None,None))
  if da is None or db is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
  raw=cv2.BFMatcher().knnMatch(da,db,k=2); good=[m for m,n in raw if m.distance<.72*n.distance]
  best={}
  for m in good:
   if m.trainIdx not in best or m.distance<best[m.trainIdx].distance: best[m.trainIdx]=m
  good=list(best.values())
  if not good:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
  return np.float32([ka[m.queryIdx].pt for m in good]),np.float32([kb[m.trainIdx].pt for m in good])

def audit(c,a,b):
 p,q=c.pts(a,b); r={'pass':False,'match_count':int(len(p))}
 if len(p)<30:r['status']='insufficient_matches';return r
 xa,ya=p[:,0],p[:,1];xb,yb=q[:,0],q[:,1]
 tg=((ya<.46*H)|(xa<.14*W)|(xa>.86*W))&((yb<.46*H)|(xb<.14*W)|(xb>.86*W)); tr=tg&~ac(p)&~ac(q); wh=~tr&~ac(p)&~ac(q)
 r['training_count']=int(tr.sum());r['withheld_count']=int(wh.sum())
 if int(tr.sum())<12:r['status']='insufficient_background_training';return r
 M,ma=cv2.findHomography(p[tr],q[tr],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
 if M is None or ma is None:r['status']='homography_failed';return r
 ii=ma.ravel().astype(bool);pred=cv2.perspectiveTransform(p[:,None,:],M)[:,0];e=np.linalg.norm(pred-q,axis=1);te=e[np.where(tr)[0][ii]];we=e[wh];ts,ws=es(te),es(we)
 g={'training_inliers_at_least_24':int(ii.sum())>=24,'training_p95_at_most_1_5px':ts['p95_px'] is not None and ts['p95_px']<=1.5,'withheld_matches_at_least_10':ws['n']>=10,'withheld_median_at_most_2_5px':ws['median_px'] is not None and ws['median_px']<=2.5,'withheld_p90_at_most_4px':ws['p90_px'] is not None and ws['p90_px']<=4.0}
 r.update({'training_inliers':int(ii.sum()),'training_error':ts,'withheld_error':ws,'gates':g});r['pass']=bool(all(g.values()));r['status']='pass' if r['pass'] else 'reject'
 if r['pass']:r['H']=M.tolist()
 return r

def oriented(c,s,h):
 f=audit(c,s,h)
 if f.get('pass'): return {'pass':True,'accepted_direction':'source_to_hub','accepted_audit':f,'H_source_to_hub':f['H']}
 r=audit(c,h,s)
 if r.get('pass'):
  M=np.linalg.inv(np.asarray(r['H'],float));M/=M[2,2]
  return {'pass':True,'accepted_direction':'hub_to_source_inverted','accepted_audit':r,'H_source_to_hub':M.tolist(),'rejected_forward':f}
 return {'pass':False,'forward':f,'reverse':r}

def K(f,p):return np.asarray([[f,0,p[0]],[0,f,p[1]],[0,0,1.]],float)
def residual(x,rows):
 pp=x[:2];ft=math.exp(float(x[2]));o=[]
 for i,r in enumerate(rows):
  fs=math.exp(float(x[3+i]));M=np.linalg.inv(K(ft,pp))@np.asarray(r['H_source_to_hub'],float)@K(fs,pp);d=float(np.linalg.det(M))
  if abs(d)<1e-12:o.extend([100.]*7);continue
  M=M/np.cbrt(abs(d));M=-M if d<0 else M;A=M.T@M-np.eye(3);o.extend((5*A[np.triu_indices(3)]).tolist());o.append(5*(np.linalg.det(M)-1))
 o.extend([(pp[0]-W/2)/350.,(pp[1]-H/2)/350.,(math.log(ft)-math.log(550.))/1.8]);return np.asarray(o,float)
def solve(rows,seed=None):
 n=len(rows);ss=[seed] if seed else [(480.,270.,350.),(480.,330.,550.),(520.,300.,700.),(440.,300.,700.),(500.,290.,1000.)];lo=np.r_[0.,0.,math.log(150.),np.repeat(math.log(150.),n)];hi=np.r_[960.,540.,math.log(4000.),np.repeat(math.log(4000.),n)];best=None;bs=1e99
 for sx,sy,sf in ss:
  x0=np.r_[sx,sy,math.log(sf),np.repeat(math.log(sf),n)];z=least_squares(lambda x:residual(x,rows),x0,bounds=(lo,hi),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=12000);sc=float(np.mean(residual(z.x,rows)**2))
  if np.isfinite(sc) and sc<bs:bs,best=sc,z.x
 if best is None:raise RuntimeError('selfcal failed')
 return best

def radial(r,pp,ft,fs):
 Hm=np.asarray(r['H_source_to_hub'],float);M=np.linalg.inv(K(ft,pp))@Hm@K(fs,pp);d=np.linalg.det(M);M=M/np.cbrt(abs(d));M=-M if d<0 else M;U,_,V=np.linalg.svd(M);R=U@V
 if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@V
 Hr=K(ft,pp)@R@np.linalg.inv(K(fs,pp));Hr/=Hr[2,2];xs=np.linspace(45,W-45,15);ys=np.linspace(35,H-35,10);p=np.asarray([[x,y] for y in ys for x in xs],np.float32);p=p[~ac(p)];q0=cv2.perspectiveTransform(p[:,None,:],Hm)[:,0];q1=cv2.perspectiveTransform(p[:,None,:],Hr)[:,0];rrr=q1-q0;v=q0-pp;rad=np.linalg.norm(v,axis=1);u=v/np.maximum(rad[:,None],1e-9);rv=np.sum(rrr*u,axis=1);tv=rrr[:,0]*(-u[:,1])+rrr[:,1]*u[:,0]
 return {'radial_corr':float(np.corrcoef(rad,rv)[0,1]),'radial_slope_px_per_px':float(np.polyfit(rad,rv,1)[0]),'median_abs_radial_px':float(np.median(np.abs(rv))),'median_abs_tangential_px':float(np.median(np.abs(tv)))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--min-events',type=int,default=5);ap.add_argument('--max-loo-pp-shift-px',type=float,default=8.);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 paths=sorted(a.bank.glob('event_*_frames/f*.png'));c=Cache(paths);hubs=[];edges=[]
 for he,hf in HUBS:
  hub=a.bank/f'event_{he}_frames'/hf
  if hub not in c.d:continue
  by=defaultdict(list)
  for p in paths:
   if eid(p)==he:continue
   z=oriented(c,p,hub);rec={'hub_event':he,'hub_frame':hf,'source_event':eid(p),'source_frame':p.name,**z};edges.append(rec)
   if z.get('pass'):
    qa=z['accepted_audit'];score=(qa['withheld_error']['p90_px'],qa['withheld_error']['median_px'],-qa['training_inliers']);by[eid(p)].append((score,rec))
  best=[]
  for e,rr in by.items():rr.sort(key=lambda z:z[0]);best.append(rr[0][1])
  best.sort(key=lambda z:z['source_event']);hubs.append({'hub_event':he,'hub_frame':hf,'independent_passing_event_count':len(best),'best_edges':best})
 hubs.sort(key=lambda z:z['independent_passing_event_count'],reverse=True);bh=hubs[0] if hubs else {'best_edges':[],'independent_passing_event_count':0}
 report={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','method':'existing v1 one-direction transfer gate; reverse pass may be inverted; multi-hub rotational self-calibration','guardrail':'No transfer threshold relaxed. Passing authorizes at most a PP/focal prior; no metric camera/render.','hubs':hubs,'all_edges':edges,'selected_hub':{'event':bh.get('hub_event'),'frame':bh.get('hub_frame')},'independent_passing_event_count':bh.get('independent_passing_event_count',0)}
 rows=bh.get('best_edges',[])
 if len(rows)>=a.min_events:
  x=solve(rows);pp=x[:2];ft=math.exp(float(x[2]));loo=[]
  for e in [r['source_event'] for r in rows]:
   sub=[r for r in rows if r['source_event']!=e];y=solve(sub,seed=(float(pp[0]),float(pp[1]),ft));loo.append({'held_out_event':e,'principal_point_px':y[:2].tolist(),'shift_px':float(np.linalg.norm(y[:2]-pp))})
  ml=max(z['shift_px'] for z in loo);rd=[{'event_id':r['source_event'],**radial(r,pp,ft,math.exp(float(x[3+i])))} for i,r in enumerate(rows)];sl=np.asarray([z['radial_slope_px_per_px'] for z in rd]);same=float(max(np.mean(sl>0),np.mean(sl<0)));coh=bool(same>=.75 and np.median(np.abs(sl))>=.002);g={'independent_events_at_least_5':len(rows)>=a.min_events,'leave_one_whole_event_out_pp_shift_at_most_8px':ml<=a.max_loo_pp_shift_px};passed=bool(all(g.values()));report.update({'status':'PASS_RIGHT_SLASH_INTRINSICS_PRIOR_V103' if passed else 'FAIL_RIGHT_SLASH_INTRINSICS_V103','shared_principal_point_px':pp.tolist(),'hub_focal_px':ft,'source_states':[{'event_id':r['source_event'],'frame':r['source_frame'],'focal_px':float(math.exp(x[3+i])),'accepted_direction':r['accepted_direction']} for i,r in enumerate(rows)],'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':ml,'radial_residual_diagnostic':rd,'radial_slope_same_sign_fraction':same,'coherent_radial_distortion_pattern_detected':coh,'gates':g,'principal_point_prior_allowed':passed})
 else:report.update({'status':'FAIL_RIGHT_SLASH_ORIENTED_PAIR_GRAPH_V103','gates':{'independent_events_at_least_5':False},'principal_point_prior_allowed':False})
 report['metric_event_camera_allowed']=False;report['replay_render_allowed']=False
 def cvt(o):
  if isinstance(o,np.generic):return o.item()
  if isinstance(o,np.ndarray):return o.tolist()
  raise TypeError(type(o).__name__)
 (a.out/'right_slash_oriented_pair_graph_v103.json').write_text(json.dumps(report,indent=2,default=cvt)+'\n');print(json.dumps({'status':report['status'],'hub':report['selected_hub'],'events':report['independent_passing_event_count'],'pp':report.get('shared_principal_point_px'),'max_loo':report.get('max_leave_one_event_out_pp_shift_px'),'radial':report.get('coherent_radial_distortion_pattern_detected'),'hub_counts':[(h['hub_event'],h['independent_passing_event_count']) for h in hubs]},indent=2,default=cvt),flush=True)
if __name__=='__main__':main()
