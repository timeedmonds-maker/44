from __future__ import annotations
"""Right Slash v110: expanded 45-event connected-graph intrinsic proof.

Failure-mode repair after v106/v107/v109 exhausted the single 540/f02 star: only
25/f03 and 621/f00 can supply the missing star leverage and both hit the focal ceiling.
This proof keeps the same shared PP + one Brown k1 model and the same exact transfer
thresholds, but removes the single-hub topology. Cheap ORB screening only chooses
which event/frame pairs deserve the expensive exact test; screening can never create
an accepted edge.
"""
import argparse,json,math,itertools,re
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from scipy.optimize import least_squares
from freeze_spin import diagnose_right_slash_allpair_radial_v104 as base
W,H=960,540
KNOWN_FRAME_EDGES=[
 (15,'f03.png',540,'f02.png'),(25,'f03.png',540,'f02.png'),
 (40,'f04.png',275,'f04.png'),(40,'f04.png',405,'f06.png'),
 (300,'f03.png',540,'f01.png'),(405,'f06.png',540,'f00.png'),
 (415,'f05.png',540,'f03.png'),(540,'f02.png',690,'f03.png'),
 (685,'f05.png',690,'f03.png'),(9,'f03.png',540,'f02.png'),
 (205,'f03.png',540,'f02.png'),(621,'f00.png',540,'f02.png')]
TEMPORAL_CLUSTERS=[[7,9,11,13],[105,107,109,113],[205,207,209,211],
                   [307,309,319,323],[405,407,409,411],[505,509,511,517],
                   [607,617,619,621]]
EVENT_RE=re.compile(r'event_(\d+)_frames$')
def eid(p):
 m=EVENT_RE.search(p.parent.name);return int(m.group(1)) if m else -1
def serial_edge(e):return {k:v for k,v in e.items() if k not in {'p','q','train_idx','held_idx'}}
def representative(d:Path):
 for f in ('f03.png','f02.png','f04.png','f01.png','f05.png','f00.png','f06.png'):
  if (d/f).exists():return f
 return sorted(d.glob('f*.png'))[0].name

def orb_frame_candidates(feat,pa,pb,topn=6):
 bf=cv2.BFMatcher(cv2.NORM_HAMMING);rows=[]
 for a in pa:
  ka,da=feat.orb.get(a,(None,None))
  if da is None:continue
  for b in pb:
   kb,db=feat.orb.get(b,(None,None))
   if db is None:continue
   raw=bf.knnMatch(da,db,k=2);good=base.one_to_one_ratio(raw,.78)
   if len(good)<12:continue
   p=np.float32([ka[m.queryIdx].pt for m in good]);q=np.float32([kb[m.trainIdx].pt for m in good])
   M,mask=cv2.findHomography(p,q,cv2.RANSAC,3.0,maxIters=5000,confidence=.995)
   if M is None or mask is None:continue
   nin=int(mask.sum());rows.append((nin,nin/max(1,len(good)),len(good),a,b))
 rows.sort(key=lambda z:(-z[0],-z[1],-z[2]));return rows[:topn]

def biconnected_components(nodes,edges):
 adj={n:set() for n in nodes}
 for e in edges:
  a,b=e['a_event'],e['b_event'];adj[a].add(b);adj[b].add(a)
 disc={};low={};parent={};stack=[];out=[];time=[0]
 def dfs(u):
  time[0]+=1;disc[u]=low[u]=time[0]
  for v in sorted(adj[u]):
   if v not in disc:
    parent[v]=u;stack.append((u,v));dfs(v);low[u]=min(low[u],low[v])
    if low[v]>=disc[u]:
     comp=set()
     while stack:
      x,y=stack.pop();comp.update((x,y))
      if (x,y)==(u,v):break
     if len(comp)>=2:out.append(sorted(comp))
   elif parent.get(u)!=v and disc[v]<disc[u]:
    low[u]=min(low[u],disc[v]);stack.append((u,v))
 for n in sorted(nodes):
  if n not in disc:
   dfs(n)
   if stack:
    c=set()
    while stack:
     x,y=stack.pop();c.update((x,y))
    if len(c)>=2:out.append(sorted(c))
 # Deduplicate and return largest first.
 z=[];seen=set()
 for c in out:
  k=tuple(c)
  if k not in seen:seen.add(k);z.append(c)
 return sorted(z,key=lambda c:(len(c),c),reverse=True),adj

def fit_physical(edges,seed_shared=None,jitter=None,warm_only=False):
 frames=[]
 for e in edges:
  for k in ((e['a_event'],e['a_frame']),(e['b_event'],e['b_frame'])):
   if k not in frames:frames.append(k)
 fi={k:i for i,k in enumerate(frames)};N=len(frames);E=len(edges)
 def unpack(x):return float(x[0]),float(x[1]),float(x[2]),np.exp(x[3:3+N]),x[3+N:].reshape(E,3)
 def residual(x):
  cx,cy,k1,fs,rv=unpack(x);out=[]
  for j,e in enumerate(edges):
   idx=e['train_idx']
   if len(idx)>70:idx=idx[np.linspace(0,len(idx)-1,70).astype(int)]
   p=e['p'][idx].copy();q=e['q'][idx].copy()
   if jitter is not None:
    p=p+jitter[j][0][:len(idx)];q=q+jitter[j][1][:len(idx)]
   fa=fs[fi[(e['a_event'],e['a_frame'])]];fb=fs[fi[(e['b_event'],e['b_frame'])]]
   rays=base.undistort_pix(p,fa,cx,cy,k1);R,_=cv2.Rodrigues(rv[j]);pred=base.project_ray((R@rays.T).T,fb,cx,cy,k1)
   out.append((pred-q).ravel()/math.sqrt(max(1,len(idx))/40.0))
  out.append(np.array([(cx-W/2)/350.0,(cy-H/2)/350.0,k1/0.5]))
  out.append((np.log(fs)-math.log(1400.0))/2.0)
  return np.concatenate(out)
 K=np.array([[1400.,0,480.],[0,1400.,270.],[0,0,1.]])
 r0=[]
 for e in edges:
  M=np.linalg.inv(K)@np.asarray(e['H'],float)@K;d=np.linalg.det(M);M=M/np.cbrt(abs(d));M=-M if d<0 else M;U,_,Vt=np.linalg.svd(M);R=U@Vt
  if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
  r,_=cv2.Rodrigues(R);r0.append(r.ravel())
 seeds=[]
 if seed_shared is not None:seeds.append(seed_shared)
 if not warm_only:seeds += [(480.,270.,0.),(480.,270.,.25),(480.,270.,-.25),(500.,280.,.35)]
 if not seeds:seeds=[(480.,270.,0.)]
 lo=np.r_[0.,0.,-.8,np.repeat(math.log(150.),N),np.repeat(-math.pi,E*3)];hi=np.r_[960.,540.,.8,np.repeat(math.log(4000.),N),np.repeat(math.pi,E*3)]
 best=None;bs=float('inf')
 for sx,sy,sk in seeds:
  x0=np.r_[sx,sy,sk,np.repeat(math.log(1400.),N),np.concatenate(r0)]
  try:z=least_squares(residual,x0,bounds=(lo,hi),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=12000 if not warm_only else 5000)
  except Exception:continue
  sc=float(np.mean(residual(z.x)**2))
  if np.isfinite(sc) and sc<bs:bs,best=sc,z.x
 if best is None:raise RuntimeError('physical fit failed')
 cx,cy,k1,fs,rv=unpack(best);held=[]
 for j,e in enumerate(edges):
  idx=e['held_idx'];fa=fs[fi[(e['a_event'],e['a_frame'])]];fb=fs[fi[(e['b_event'],e['b_frame'])]]
  rays=base.undistort_pix(e['p'][idx],fa,cx,cy,k1);R,_=cv2.Rodrigues(rv[j]);pred=base.project_ray((R@rays.T).T,fb,cx,cy,k1);er=np.linalg.norm(pred-e['q'][idx],axis=1)
  held.append({'a_event':e['a_event'],'a_frame':e['a_frame'],'b_event':e['b_event'],'b_frame':e['b_frame'],'error':base.estats(er),'pass':bool(len(er)>=10 and np.median(er)<=2.5 and np.percentile(er,90)<=4.)})
 fmap=[{'event_id':k[0],'frame':k[1],'focal_px':float(fs[i])} for i,k in enumerate(frames)];at=any(v['focal_px']<=155 or v['focal_px']>=3950 for v in fmap)
 return {'pp':[cx,cy],'k1':k1,'focals':fmap,'heldout':held,'all_heldout_pass':all(x['pass'] for x in held),'focal_at_bound':at,'score':bs}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--top-event-pairs',type=int,default=140);ap.add_argument('--top-frame-pairs',type=int,default=6);ap.add_argument('--perturbation-trials',type=int,default=12);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);cv2.setRNGSeed(0);cv2.setNumThreads(1)
 dirs=sorted(a.bank.glob('event_*_frames'));rep={int(EVENT_RE.search(d.name).group(1)):representative(d) for d in dirs};base.REP=rep
 paths=sorted(a.bank.glob('event_*_frames/f*.png'));feat=base.Features(paths);rank=base.orb_rank(feat,a.bank)
 pairs={tuple(sorted((x,y))) for _,_,x,y in rank[:a.top_event_pairs]}
 for cl in TEMPORAL_CLUSTERS:
  pairs.update(tuple(sorted(x)) for x in itertools.combinations(cl,2) if all(e in rep for e in x))
 pairs.update(tuple(sorted((x[0],x[2]))) for x in KNOWN_FRAME_EDGES)
 forced=defaultdict(list)
 for ea,fa,eb,fb in KNOWN_FRAME_EDGES:
  pa=a.bank/f'event_{ea}_frames'/fa;pb=a.bank/f'event_{eb}_frames'/fb
  if pa.exists() and pb.exists():forced[tuple(sorted((ea,eb)))].append((pa,pb) if ea<eb else (pb,pa))
 accepted=[];audit=[]
 for n,(x,y) in enumerate(sorted(pairs),1):
  pa=sorted((a.bank/f'event_{x}_frames').glob('f*.png'));pb=sorted((a.bank/f'event_{y}_frames').glob('f*.png'));cand=orb_frame_candidates(feat,pa,pb,a.top_frame_pairs)
  tests=[(z[3],z[4]) for z in cand]
  tests.extend(forced.get((x,y),[]));tests=list(dict.fromkeys(tests));best=[]
  for p,q in tests:
   z=base.exact_edge(feat,p,q)
   if z is not None:best.append(z)
  if best:
   best.sort(key=lambda z:(z['withheld_error']['p90_px'],z['withheld_error']['median_px'],-z['training_inliers']));accepted.append(best[0]);audit.append({'events':[x,y],'screened_frame_pairs':len(tests),'pass':True,'best':serial_edge(best[0])})
  else:audit.append({'events':[x,y],'screened_frame_pairs':len(tests),'pass':False})
  print('PAIR',n,'/',len(pairs),x,y,'PASS' if best else 'FAIL',flush=True)
 nodes=sorted(set([z['a_event'] for z in accepted]+[z['b_event'] for z in accepted]));bics,adj=biconnected_components(nodes,accepted) if nodes else ([],{})
 eligible=[c for c in bics if 415 in c and len(c)>=6];core=max(eligible,key=len) if eligible else []
 report={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','method':'expanded 45-event multi-hub graph; ORB screening only; exact mutual SIFT .72 + topology consistency + unchanged 24/1.5/10/2.5/4 transfer gate; shared PP + one Brown k1','accepted_edges':[serial_edge(z) for z in accepted],'pair_audit':audit,'biconnected_components':bics,'robust_core_events':core,'metric_event_camera_allowed':False,'replay_render_allowed':False}
 prior=False
 if core:
  ce=[z for z in accepted if z['a_event'] in core and z['b_event'] in core];full=fit_physical(ce);loo=[]
  for hold in core:
   sub=[z for z in ce if hold not in (z['a_event'],z['b_event'])];y=fit_physical(sub,seed_shared=(full['pp'][0],full['pp'][1],full['k1']))
   loo.append({'held_out_event':hold,'principal_point_px':y['pp'],'pp_shift_px':float(np.linalg.norm(np.asarray(y['pp'])-np.asarray(full['pp']))),'k1':y['k1'],'k1_abs_shift':abs(y['k1']-full['k1']),'k1_same_sign':bool(np.sign(y['k1'])==np.sign(full['k1']) and abs(y['k1'])>1e-8),'all_heldout_pass':y['all_heldout_pass'],'focal_at_bound':y['focal_at_bound']})
  maxpp=max(x['pp_shift_px'] for x in loo);maxdk=max(x['k1_abs_shift'] for x in loo);mags=[abs(full['k1'])]+[abs(x['k1']) for x in loo];ratio=max(mags)/max(min(mags),1e-8)
  rng=np.random.default_rng(110);pert=[];mpp=0.;psign=True;pat=False
  for t in range(a.perturbation_trials):
   jit=[]
   for e in ce:
    n=min(len(e['train_idx']),70);jit.append((rng.uniform(-.5,.5,(n,2)),rng.uniform(-.5,.5,(n,2))))
   y=fit_physical(ce,seed_shared=(full['pp'][0],full['pp'][1],full['k1']),jitter=jit,warm_only=True);ds=float(np.linalg.norm(np.asarray(y['pp'])-np.asarray(full['pp'])));same=bool(np.sign(y['k1'])==np.sign(full['k1']) and abs(y['k1'])>1e-8);mpp=max(mpp,ds);psign &= same;pat |= y['focal_at_bound'];pert.append({'trial':t,'pp_shift_px':ds,'k1':y['k1'],'k1_same_sign':same,'focal_at_bound':y['focal_at_bound']})
  gates={'robust_biconnected_core_at_least_6_events':len(core)>=6,'target_event_415_in_core':415 in core,'physical_model_all_heldout_edges_pass':full['all_heldout_pass'],'no_full_fit_focal_at_bound':not full['focal_at_bound'],'whole_event_loo_pp_shift_at_most_8px':maxpp<=8.,'whole_event_loo_k1_same_sign':all(x['k1_same_sign'] for x in loo),'whole_event_loo_k1_magnitude_ratio_at_most_2x':ratio<=2.,'whole_event_loo_k1_abs_shift_at_most_0_05':maxdk<=.05,'loo_fits_no_focal_at_bound':all(not x['focal_at_bound'] for x in loo),'half_pixel_pp_at_most_5px':mpp<=5.,'half_pixel_k1_sign_stable':psign,'half_pixel_no_focal_at_bound':not pat};prior=bool(all(gates.values()));report.update({'physical_full_fit':full,'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':maxpp,'max_leave_one_event_out_k1_abs_shift':maxdk,'k1_loo_max_min_ratio':ratio,'perturbation_trials':pert,'max_half_pixel_pp_shift_px':mpp,'gates':gates})
 report['status']='PASS_RIGHT_SLASH_EXPANDED_CONNECTED_INTRINSICS_PRIOR_V110' if prior else ('FAIL_RIGHT_SLASH_EXPANDED_CONNECTED_INTRINSICS_V110' if core else 'FAIL_RIGHT_SLASH_EXPANDED_GRAPH_NO_ROBUST_CORE_V110');report['principal_point_prior_allowed']=prior
 def cvt(o):
  if isinstance(o,np.generic):return o.item()
  if isinstance(o,np.ndarray):return o.tolist()
  raise TypeError(type(o).__name__)
 (a.out/'right_slash_expanded_connected_intrinsics_v110.json').write_text(json.dumps(report,indent=2,default=cvt)+'\n');print(json.dumps({'status':report['status'],'accepted_edges':len(accepted),'core':core,'pp':report.get('physical_full_fit',{}).get('pp'),'k1':report.get('physical_full_fit',{}).get('k1'),'max_loo_pp':report.get('max_leave_one_event_out_pp_shift_px'),'max_half_pixel_pp':report.get('max_half_pixel_pp_shift_px'),'prior':prior},indent=2,default=cvt),flush=True)
if __name__=='__main__':main()
