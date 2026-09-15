from __future__ import annotations
"""Right Slash v108: exhaustive same-hub conditioning search.

Enumerates every topology-filtered exact-gate source state to 540/f02 in the merged
v100b+v105 bank. Holds the five v106/v107 stable core states fixed, rejects candidate
sets with any focal near the existing solver bounds, and evaluates the best single-
and two-candidate sets with the unchanged shared-PP + one-k1 model and locked LOO /
12-trial half-pixel gates. No threshold or camera model changes.
"""
import argparse,json,itertools,re
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from scipy.spatial import cKDTree
from freeze_spin.diagnose_right_slash_pair_graph_v102 import Cache,audit,action_core,stats
from freeze_spin.prove_right_slash_radial_intrinsics_v104 import payload,solve,unpack,qa,subwarm
W,H=960,540
HUB=(540,'f02.png')
CORE=[(9,'f03.png'),(15,'f03.png'),(205,'f03.png'),(415,'f06.png'),(690,'f02.png')]
EVENT_RE=re.compile(r'event_(\d+)_frames$')
def eid(p):
 m=EVENT_RE.search(p.parent.name); return int(m.group(1)) if m else -1
def topo(p,q,k=8,min_overlap=2):
 if len(p)<=k:return np.ones(len(p),bool)
 _,a=cKDTree(p).query(p,k=k+1);_,b=cKDTree(q).query(q,k=k+1)
 return np.asarray([len(set(a[i,1:])&set(b[i,1:]))>=min_overlap for i in range(len(p))],bool)
class SiftBank:
 def __init__(self,paths):
  s=cv2.SIFT_create(nfeatures=9000,contrastThreshold=.015);self.f={}
  for p in paths:
   im=cv2.imread(str(p),0)
   if im is None or im.shape!=(H,W):continue
   self.f[p]=s.detectAndCompute(im,None)
def exact_edge(c,a,b):
 ka,da=c.f.get(a,(None,None));kb,db=c.f.get(b,(None,None))
 if da is None or db is None:return None
 bf=cv2.BFMatcher(cv2.NORM_L2);ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2)
 gab={m.queryIdx:m for m,n in ab if m.distance<.72*n.distance};gba={m.queryIdx:m for m,n in ba if m.distance<.72*n.distance}
 ms=[m for qi,m in gab.items() if m.trainIdx in gba and gba[m.trainIdx].trainIdx==qi]
 if len(ms)<30:return None
 p=np.float64([ka[m.queryIdx].pt for m in ms]);q=np.float64([kb[m.trainIdx].pt for m in ms]);k=topo(p,q);p=p[k];q=q[k]
 if len(p)<30:return None
 xa,ya=p[:,0],p[:,1];xb,yb=q[:,0],q[:,1]
 tg=((ya<.46*H)|(xa<.14*W)|(xa>.86*W))&((yb<.46*H)|(xb<.14*W)|(xb>.86*W));tr=tg&~action_core(p)&~action_core(q);wh=~tr&~action_core(p)&~action_core(q)
 if int(tr.sum())<12:return None
 M,mask=cv2.findHomography(p[tr].astype(np.float32),q[tr].astype(np.float32),cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
 if M is None:return None
 ids=np.where(tr)[0][mask.ravel().astype(bool)];hi=np.where(wh)[0];pred=cv2.perspectiveTransform(p[:,None,:].astype(np.float32),M)[:,0];e=np.linalg.norm(pred-q,axis=1);ts,hs=stats(e[ids]),stats(e[hi])
 gates={'training_inliers_at_least_24':len(ids)>=24,'training_p95_at_most_1_5px':ts['p95_px']<=1.5,'withheld_matches_at_least_10':len(hi)>=10,'withheld_median_at_most_2_5px':hs['median_px']<=2.5,'withheld_p90_at_most_4px':hs['p90_px']<=4.}
 if not all(gates.values()):return None
 return {'source_event':eid(a),'source_frame':a.name,'source':str(a),'audit':{'pass':True,'training_inliers':len(ids),'training_error':ts,'withheld_error':hs,'gates':gates},'geom':{'fit_source_px':p[ids].astype(float),'fit_target_px':q[ids].astype(float),'held_source_px':p[hi].astype(float),'held_target_px':q[hi].astype(float),'H':np.asarray(M,float)},'admission':'topology_filter_then_unchanged_exact_gate'}
def full_eval(rows,trials=12):
 rows=sorted(rows,key=lambda r:(r['source_event'],r['source_frame']));x=solve(rows);pp,hf,k1,st=unpack(x,len(rows));phys,physok=qa(x,rows);foc=[hf]+[s[0] for s in st];fbound=all(155<f<3950 for f in foc)
 loo=[]
 for i,r in enumerate(rows):
  sub=[z for j,z in enumerate(rows) if j!=i];y=solve(sub,subwarm(x,rows,sub),full_multistart=True);yp,yf,yk,_=unpack(y,len(sub));loo.append({'held_out_event':r['source_event'],'pp_shift_px':float(np.linalg.norm(yp-pp)),'principal_point_px':yp.tolist(),'hub_focal_px':yf,'k1':yk,'k1_same_sign':bool(np.sign(yk)==np.sign(k1) and abs(yk)>1e-8)})
 ml=max(z['pp_shift_px'] for z in loo);sign=all(z['k1_same_sign'] for z in loo);mags=[abs(k1)]+[abs(z['k1']) for z in loo];ratio=max(mags)/max(min(mags),1e-8)
 rng=np.random.default_rng(104);pert=[];mpp=mf=0.;psign=True
 for t in range(trials):
  jit=[]
  for r in rows:
   n=len(r['geom']['fit_source_px']);jit.append((rng.uniform(-.5,.5,(n,2)),rng.uniform(-.5,.5,(n,2))))
  y=solve(rows,x,jit,full_multistart=False);yp,yf,yk,_=unpack(y,len(rows));ds=float(np.linalg.norm(yp-pp));df=abs(yf-hf)/hf;same=bool(np.sign(yk)==np.sign(k1) and abs(yk)>1e-8);mpp=max(mpp,ds);mf=max(mf,df);psign &= same;pert.append({'trial':t,'pp_shift_px':ds,'hub_focal_fraction_shift':df,'k1':yk,'k1_same_sign':same})
 gates={'independent_events_at_least_6':len(rows)>=6,'physical_model_pixel_gates':physok,'focals_not_on_bounds':fbound,'loo_pp_at_most_8px':ml<=8.,'loo_k1_sign_stable':sign,'loo_k1_magnitude_ratio_at_most_2x':ratio<=2.,'half_pixel_pp_at_most_5px':mpp<=5.,'half_pixel_hub_focal_at_most_5pct':mf<=.05,'half_pixel_k1_sign_stable':psign};passed=all(gates.values())
 return {'pass':bool(passed),'events':[{'event_id':r['source_event'],'frame':r['source_frame'],'admission':r.get('admission')} for r in rows],'pp':pp.tolist(),'k1':k1,'hub_focal_px':hf,'source_states':[{'event_id':r['source_event'],'frame':r['source_frame'],'focal_px':st[i][0]} for i,r in enumerate(rows)],'physical_model_qa':phys,'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':ml,'k1_loo_max_min_ratio':ratio,'perturbation_trials':pert,'max_half_pixel_pp_shift_px':mpp,'max_half_pixel_hub_focal_fraction':mf,'gates':gates}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--top-singles',type=int,default=6);ap.add_argument('--top-pair-pool',type=int,default=4);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);cv2.setRNGSeed(0);cv2.setNumThreads(1)
 h=a.bank/f'event_{HUB[0]}_frames'/HUB[1];corep=[a.bank/f'event_{e}_frames'/f for e,f in CORE];cc=Cache([h,*corep]);core=[]
 for (e,f),p in zip(CORE,corep):
  z=audit(cc,p,h)
  if not z.get('pass'):raise RuntimeError(f'core transfer failed {e}/{f}')
  core.append({'source_event':e,'source_frame':f,'source':str(p),'audit':z,'geom':payload(cc,p,h),'admission':'v102_mutual_exact_gate'})
 paths=sorted(a.bank.glob('event_*_frames/f*.png'));sb=SiftBank(paths);by=defaultdict(list)
 for p in paths:
  if eid(p) in {HUB[0],*[e for e,_ in CORE]}:continue
  z=exact_edge(sb,p,h)
  if z:
   q=z['audit'];score=(q['withheld_error']['p90_px'],q['withheld_error']['median_px'],-q['training_inliers']);by[eid(p)].append((score,z));print('CANDIDATE_GATE_PASS',eid(p),p.name,q['training_inliers'],q['withheld_error'],flush=True)
 candidates=[]
 for e,rr in by.items():
  rr.sort(key=lambda z:z[0])
  for _,z in rr[:2]:candidates.append(z)
 nominal=[]
 for z in candidates:
  try:
   rows=sorted(core+[z],key=lambda r:(r['source_event'],r['source_frame']));x=solve(rows);pp,hf,k1,st=unpack(x,len(rows));phys,physok=qa(x,rows);foc=[hf]+[s[0] for s in st];ci=next(i for i,r in enumerate(rows) if r['source_event']==z['source_event'] and r['source_frame']==z['source_frame']);rec={'event_id':z['source_event'],'frame':z['source_frame'],'transfer':z['audit'],'pp':pp.tolist(),'k1':k1,'hub_focal_px':hf,'candidate_focal_px':st[ci][0],'max_focal_px':max(foc),'focals_not_on_bounds':all(155<f<3950 for f in foc),'physical_model_pixel_gates':physok};nominal.append((rec,z));print('NOMINAL',json.dumps(rec),flush=True)
  except Exception as ex:print('NOMINAL_FAIL',z['source_event'],z['source_frame'],repr(ex),flush=True)
 survivors=[x for x in nominal if x[0]['focals_not_on_bounds'] and x[0]['physical_model_pixel_gates']]
 survivors.sort(key=lambda x:(x[0]['max_focal_px'],x[0]['transfer']['withheld_error']['p90_px'],-x[0]['transfer']['training_inliers']))
 tests=[]
 for rec,z in survivors[:a.top_singles]:
  ev=full_eval(core+[z]);ev['kind']='single';tests.append(ev);print('FULL_SINGLE',z['source_event'],z['source_frame'],ev['pass'],ev['max_half_pixel_pp_shift_px'],ev['max_leave_one_event_out_pp_shift_px'],flush=True)
 passes=[x for x in tests if x['pass']]
 if not passes:
  pool=survivors[:a.top_pair_pool]
  for (_,z1),(_,z2) in itertools.combinations(pool,2):
   if z1['source_event']==z2['source_event']:continue
   ev=full_eval(core+[z1,z2]);ev['kind']='pair';tests.append(ev);print('FULL_PAIR',z1['source_event'],z2['source_event'],ev['pass'],ev['max_half_pixel_pp_shift_px'],ev['max_leave_one_event_out_pp_shift_px'],flush=True)
   if ev['pass']:passes.append(ev);break
 best=min(passes,key=lambda x:(x['max_half_pixel_pp_shift_px'],x['max_leave_one_event_out_pp_shift_px'])) if passes else (min(tests,key=lambda x:(sum(not v for v in x['gates'].values()),x['max_half_pixel_pp_shift_px'])) if tests else None)
 report={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','status':'PASS_RIGHT_SLASH_CONDITIONED_INTRINSICS_PRIOR_V108' if best and best['pass'] else 'FAIL_RIGHT_SLASH_CONDITIONED_INTRINSICS_V108','method':'exhaustive same-540/f02 topology-filtered exact-gate census; unchanged shared PP + one Brown k1; candidate focal-margin screening; locked full LOO and 12 half-pixel trials','candidate_census':[r for r,_ in nominal],'full_tests':tests,'selected_solution':best,'principal_point_prior_allowed':bool(best and best['pass']),'metric_event_camera_allowed':False,'replay_render_allowed':False}
 def cvt(o):
  if isinstance(o,np.generic):return o.item()
  if isinstance(o,np.ndarray):return o.tolist()
  raise TypeError(type(o).__name__)
 (a.out/'right_slash_conditioned_intrinsics_v108.json').write_text(json.dumps(report,indent=2,default=cvt)+'\n');print(json.dumps({'status':report['status'],'candidate_count':len(nominal),'survivor_count':len(survivors),'selected':best},indent=2,default=cvt),flush=True)
if __name__=='__main__':main()
