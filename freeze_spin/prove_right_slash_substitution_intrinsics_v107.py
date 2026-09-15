from __future__ import annotations
"""Right Slash v107: targeted event-621 substitution for the v106 conditioning failure.

No threshold or camera model changes. Keep the v106 540/f02 hub and one shared Brown k1,
remove the focal-bound event 25/f03, and replace it with 621/f00. Event 621 is admitted
only through the existing all-pair topology-consistency filter followed by the unchanged
24/1.5/10/2.5/4 px transfer gates. Passing grants only an intrinsics prior.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from freeze_spin.diagnose_right_slash_pair_graph_v102 import Cache,audit
from freeze_spin.prove_right_slash_radial_intrinsics_v104 import payload,solve,unpack,qa,subwarm
from freeze_spin.diagnose_right_slash_allpair_radial_v104 import Features,exact_edge

RAW=[(9,'f03.png'),(15,'f03.png'),(205,'f03.png'),(415,'f06.png'),(690,'f02.png')]
SUB=(621,'f00.png')
HUB=(540,'f02.png')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--perturbation-trials',type=int,default=12);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 cv2.setRNGSeed(0);cv2.setNumThreads(1)
 h=a.bank/f'event_{HUB[0]}_frames'/HUB[1]
 raw_paths=[a.bank/f'event_{e}_frames'/f for e,f in RAW]
 for p in [h,*raw_paths,a.bank/f'event_{SUB[0]}_frames'/SUB[1]]:
  if not p.exists():raise FileNotFoundError(p)
 c=Cache([h,*raw_paths]);rows=[]
 for (e,f),p in zip(RAW,raw_paths):
  z=audit(c,p,h)
  if not z.get('pass'):raise RuntimeError(f'locked raw transfer failed {e}/{f}: {z}')
  rows.append({'source_event':e,'source_frame':f,'source':str(p),'audit':z,'geom':payload(c,p,h),'admission':'v102_mutual_exact_gate'})
 # Event 621: same final numerical transfer gate, with the already-authorized model-free
 # local-neighbour topology consistency filter used by the all-pair diagnostic.
 sp=a.bank/f'event_{SUB[0]}_frames'/SUB[1]
 ft=Features([sp,h]);ez=exact_edge(ft,sp,h)
 if ez is None:raise RuntimeError('621/f00 -> 540/f02 topology-filtered exact transfer did not pass')
 geom={'fit_source_px':ez['p'][ez['train_idx']].astype(float),'fit_target_px':ez['q'][ez['train_idx']].astype(float),'held_source_px':ez['p'][ez['held_idx']].astype(float),'held_target_px':ez['q'][ez['held_idx']].astype(float),'H':np.asarray(ez['H'],float)}
 rows.append({'source_event':SUB[0],'source_frame':SUB[1],'source':str(sp),'audit':{'pass':True,'training_inliers':ez['training_inliers'],'training_error':ez['training_error'],'withheld_error':ez['withheld_error'],'gates':ez['gates']},'geom':geom,'admission':'allpair_topology_filter_then_unchanged_exact_gate'})
 rows.sort(key=lambda r:r['source_event'])
 x=solve(rows);pp,hubf,k1,st=unpack(x,len(rows));phys,phys_ok=qa(x,rows);focals=[hubf]+[s[0] for s in st];fbound=all(155<f<3950 for f in focals)
 loo=[]
 for i,r in enumerate(rows):
  sub=[z for j,z in enumerate(rows) if j!=i];y=solve(sub,subwarm(x,rows,sub),full_multistart=True);yp,yf,yk,_=unpack(y,len(sub));loo.append({'held_out_event':r['source_event'],'principal_point_px':yp.tolist(),'pp_shift_px':float(np.linalg.norm(yp-pp)),'hub_focal_px':yf,'k1':yk,'k1_same_sign':bool(np.sign(yk)==np.sign(k1) and abs(yk)>1e-8)})
 maxloo=max(r['pp_shift_px'] for r in loo);sign=all(r['k1_same_sign'] for r in loo);mags=[abs(k1)]+[abs(r['k1']) for r in loo];ratio=max(mags)/max(min(mags),1e-8)
 rng=np.random.default_rng(104);pert=[];mpp=0.;mf=0.;psign=True
 for t in range(a.perturbation_trials):
  jit=[]
  for r in rows:
   n=len(r['geom']['fit_source_px']);jit.append((rng.uniform(-.5,.5,(n,2)),rng.uniform(-.5,.5,(n,2))))
  y=solve(rows,x,jit,full_multistart=False);yp,yf,yk,_=unpack(y,len(rows));ds=float(np.linalg.norm(yp-pp));df=abs(yf-hubf)/hubf;same=bool(np.sign(yk)==np.sign(k1) and abs(yk)>1e-8);mpp=max(mpp,ds);mf=max(mf,df);psign &= same;pert.append({'trial':t,'pp_shift_px':ds,'hub_focal_fraction_shift':df,'k1':yk,'k1_same_sign':same})
 gates={'independent_events_at_least_6':len(rows)>=6,'physical_model_pixel_gates':phys_ok,'focals_not_on_bounds':fbound,'loo_pp_at_most_8px':maxloo<=8.,'loo_k1_sign_stable':sign,'loo_k1_magnitude_ratio_at_most_2x':ratio<=2.,'half_pixel_pp_at_most_5px':mpp<=5.,'half_pixel_hub_focal_at_most_5pct':mf<=.05,'half_pixel_k1_sign_stable':psign};passed=bool(all(gates.values()))
 rep={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','status':'PASS_RIGHT_SLASH_SUBSTITUTION_INTRINSICS_PRIOR_V107' if passed else 'FAIL_RIGHT_SLASH_SUBSTITUTION_INTRINSICS_V107','model':'same v106 shared PP + one shared Brown k1; k2=p1=p2=0; source focal/rotation','selected_hub':{'event':HUB[0],'frame':HUB[1]},'source_events':[{'event_id':r['source_event'],'frame':r['source_frame'],'admission':r['admission']} for r in rows],'shared_principal_point_px':pp.tolist(),'shared_k1_normalized_brown':k1,'hub_focal_px':hubf,'source_states':[{'event_id':r['source_event'],'frame':r['source_frame'],'focal_px':st[i][0]} for i,r in enumerate(rows)],'physical_model_qa':phys,'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':maxloo,'k1_loo_max_min_ratio':ratio,'perturbation_trials':pert,'max_half_pixel_pp_shift_px':mpp,'max_half_pixel_hub_focal_fraction':mf,'gates':gates,'principal_point_prior_allowed':passed,'metric_event_camera_allowed':False,'replay_render_allowed':False}
 def cvt(o):
  if isinstance(o,np.generic):return o.item()
  if isinstance(o,np.ndarray):return o.tolist()
  raise TypeError(type(o).__name__)
 (a.out/'right_slash_substitution_intrinsics_v107.json').write_text(json.dumps(rep,indent=2,default=cvt)+'\n')
 print(json.dumps({'status':rep['status'],'events':rep['source_events'],'pp':rep['shared_principal_point_px'],'k1':rep['shared_k1_normalized_brown'],'focals':rep['source_states'],'max_loo_pp':maxloo,'k1_ratio':ratio,'max_half_pixel_pp':mpp,'max_half_pixel_hub_focal_fraction':mf,'gates':gates},indent=2,default=cvt),flush=True)
if __name__=='__main__':main()
