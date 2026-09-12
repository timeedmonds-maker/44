from __future__ import annotations
"""v33d: joint exact-state search across the locked LAR+RAR+Broadcast cameras.
Identity is selected inside each feed before geometry. One real native frame per
camera per triplet. Physical centres stay fixed. No fourth camera, no render.
"""
import argparse, copy, json, math
from pathlib import Path
import cv2, numpy as np
from rfdetr import RFDETRKeypointPreview
from freeze_spin import prepare_v33a_exact_rar_frame_camera as v33a
from freeze_spin import run_v33b_rar_exact_state_sweep as v33b
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import run_v32w_lar_all_candidate_epipolar_mhr as v32w
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
LAR,BCAST,RAR=v32v.LAR,v32v.BCAST,v32v.RAR
BODY,MIN_CONF=v32v.BODY,v32v.MIN_CONF; W,H=v32j.W,v32j.H; RELS=tuple(range(-6,7))
ANCH={
 LAR:(3,[467.1226806640625,179.22335815429688,504.54278564453125,286.54254150390625],.55,15.,'v32w/v33c verified Adams #12'),
 BCAST:(3,[499.87261962890625,141.63543701171875,539.3041381835938,278.45684814453125],.55,15.,'v32z/v33c verified Adams #12'),
 RAR:(-3,[410.2623901,172.4596100,607.7608643,317.2488403],.35,30.,'v33c source-local RAR Adams candidate')}

def ctr(b): b=np.asarray(b,float); return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
def area(b): b=np.asarray(b,float); return max(1.,max(0.,b[2]-b[0])*max(0.,b[3]-b[1]))
def iou(a,b):
 a=np.asarray(a,float); b=np.asarray(b,float); x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0.,x2-x1)*max(0.,y2-y1); return inter/max(area(a)+area(b)-inter,1e-9)
def mask():
 m=np.full((H,W),255,np.uint8); m[120:350,300:680]=0; m[0:180,0:350]=0; m[150:330,0:180]=0; m[360:540,500:720]=0; return m

def transfer(base,target,cam):
 if np.array_equal(base,target): return copy.deepcopy(cam),{'status':'PASS_IDENTITY_TRANSFER','center_delta_cm':0.}
 orb=cv2.ORB_create(nfeatures=6500,fastThreshold=10); kt,dt=orb.detectAndCompute(target,mask()); kb,db=orb.detectAndCompute(base,mask())
 if dt is None or db is None: return None,{'status':'FAIL_DESCRIPTORS'}
 good=[]
 for p in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dt,db,k=2):
  if len(p)==2 and p[0].distance<.70*p[1].distance: good.append(p[0])
 if len(good)<500:return None,{'status':'FAIL_MATCH_COUNT','matches':len(good)}
 pt=np.float32([kt[x.queryIdx].pt for x in good]); pb=np.float32([kb[x.trainIdx].pt for x in good]); M,inl=cv2.findHomography(pt,pb,cv2.RANSAC,1.25,maxIters=12000,confidence=.999)
 if M is None:return None,{'status':'FAIL_H'}
 keep=inl.ravel().astype(bool); pred=cv2.perspectiveTransform(pt[:,None,:],M).reshape(-1,2); e=np.linalg.norm(pred-pb,axis=1)[keep]
 reg={'matches':len(good),'inliers':int(keep.sum()),'fraction':float(keep.mean()),'median_px':float(np.median(e)),'p95_px':float(np.percentile(e,95)),'max_px':float(e.max())}
 if not(reg['inliers']>=500 and reg['fraction']>=.40 and reg['median_px']<=.80 and reg['p95_px']<=1.50 and reg['max_px']<=2.25): return None,{'status':'FAIL_STATIC_GATE','registration':reg}
 I=np.linalg.inv(M); I/=I[2,2]; K,R,C=v33a.decompose_projection(I@v33a.camera_projection(cam)); C0=np.asarray(cam['C_world_cm'],float); f0=max(cam['K_px'][0][0],cam['K_px'][1][1]); f=max(K[0,0],K[1,1]); cd=float(np.linalg.norm(C-C0))
 if not(cd<=1e-5 and .70<=f/f0<=1.30 and abs(K[0,1])<=.15*f and -300<=K[0,2]<=W+300 and -300<=K[1,2]<=H+300): return None,{'status':'FAIL_CAMERA','registration':reg,'center_delta_cm':cd,'K':K.tolist()}
 out=copy.deepcopy(cam); out['K_px']=K.tolist(); out['R_world_to_camera']=R.tolist(); out['C_world_cm']=C.tolist(); out['extrinsic_world_to_camera_3x4']=np.c_[R,-R@C].tolist(); return out,{'status':'PASS_STATIC_TRANSFER','registration':reg,'center_delta_cm':cd,'K':K.tolist()}

def frames(stage,label):
 o={}
 for r in RELS:
  p=v32k.burst_path(stage,label,r); im=cv2.imread(str(p));
  if im is None: raise RuntimeError(f'missing {p}')
  o[r]=(p,im,cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
 return o
def detrow(d,img):
 dark,torso,n=v32w.appearance(d,img); return {'i':None,'box':np.asarray(d['box'],float),'xy':np.asarray(d['xy'],float),'conf':np.asarray(d['conf'],float),'dc':float(d.get('det_conf',0.)),'dark':float(dark),'torso':float(torso),'n':int(n)}
def pool(model,img,ab):
 ac=ctr(ab); out=[]
 for i,d in enumerate(v32j.infer(model,img)):
  x=detrow(d,img); dist=float(np.linalg.norm(ctr(x['box'])-ac))
  if x['n']<6 or not(x['torso']>=.28 or x['dark']>=.36 or dist<=110): continue
  x.update(i=i,dist=dist,aiou=float(iou(x['box'],ab))); out.append(x)
 return out
def track(model,label,fs):
 ar,ab,miou,mdist,prov=ANCH[label]; ab=np.asarray(ab,float); ac=ctr(ab); aa=area(ab); pools={r:pool(model,fs[r][1],ab) for r in RELS}
 if any(not pools[r] for r in RELS): raise RuntimeError(f'{label} empty pools {[r for r in RELS if not pools[r]]}')
 hard={x['i'] for x in pools[ar] if x['aiou']>=miou and x['dist']<=mdist}
 if not hard: raise RuntimeError(f'{label} anchor lock failed')
 dp=[]; bk=[]
 for t,r in enumerate(RELS):
  cs=np.full(len(pools[r]),np.inf); bp=np.full(len(pools[r]),-1,int)
  for j,x in enumerate(pools[r]):
   u=.010*float(np.linalg.norm(ctr(x['box'])-ac))+.75*abs(math.log(area(x['box'])/aa))-.70*x['torso']-.35*x['dark']-.15*x['dc']
   if r==ar:
    if x['i'] not in hard:u+=1e5
    u+=8*(1-x['aiou'])+.08*x['dist']
   if t==0:cs[j]=u
   else:
    for k,p in enumerate(pools[RELS[t-1]]):
     v=dp[t-1][k]+.026*float(np.linalg.norm(ctr(x['box'])-ctr(p['box'])))+2*(1-iou(x['box'],p['box']))+.55*abs(math.log(area(x['box'])/area(p['box'])))+u
     if v<cs[j]:cs[j]=v; bp[j]=k
  dp.append(cs); bk.append(bp)
 j=int(np.argmin(dp[-1])); chosen={}
 for t in range(len(RELS)-1,-1,-1):
  r=RELS[t]; chosen[r]=pools[r][j]; j=int(bk[t][j]) if t else j
 aud={'anchor_rel':ar,'anchor_box':ab.tolist(),'provenance':prov,'path_cost':float(np.min(dp[-1])),'states':{str(r):{'i':chosen[r]['i'],'box':chosen[r]['box'].tolist(),'dark':chosen[r]['dark'],'torso':chosen[r]['torso'],'joints':chosen[r]['n']} for r in RELS}}
 return chosen,aud
def valid(x): return np.asarray(x['conf'])>=MIN_CONF
def pgate(e): return e['joint_count']>=6 and e['median_px']<=18 and e['p90_px']<=35

def repro(scene,l,r,b):
 cams={c:v32j.cam(scene,c) for c in (LAR,RAR,BCAST)}; rows={LAR:l,RAR:r,BCAST:b}; vals=[]; nj=0
 for j in BODY:
  obs={c:rows[c]['xy'][j] for c in rows if valid(rows[c])[j]}
  if len(obs)<2:continue
  X=v32j.triangulate_rays(cams,obs)
  if X is None or not(-350<=X[0]<=1250 and -750<=X[1]<=750 and -60<=X[2]<=450):continue
  nj+=1
  for c in obs:
   uv=v32j.project(cams[c],X)
   if uv is not None:vals.append(float(np.linalg.norm(uv-rows[c]['xy'][j])))
 a=np.asarray(vals,float); q={'joints':nj,'observations':len(vals),'median_px':float(np.median(a)) if len(a) else 999.,'p90_px':float(np.percentile(a,90)) if len(a) else 999.}; q['gate']=bool(nj>=7 and q['median_px']<=12 and q['p90_px']<=24); return q

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--b32-root',type=Path,required=True); ap.add_argument('--v73-frame0257',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 stage=a.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text())
 if scene.get('resolution')!=[W,H] or set(scene.get('cameras',{}))!={LAR,BCAST,RAR}:raise RuntimeError('three-camera lock/source mismatch')
 cert=cv2.imread(str(a.v73_frame0257),0)
 if cert is None or cert.shape!=(H,W):raise RuntimeError('missing v73 cert')
 fs={c:frames(stage,c) for c in (LAR,RAR,BCAST)}; cams={LAR:{},RAR:{},BCAST:{}}; ca={LAR:{},RAR:{},BCAST:{}}
 for c in (LAR,BCAST):
  for rel in RELS:
   cc,au=transfer(fs[c][0][2],fs[c][rel][2],scene['cameras'][c]);
   if cc is None:raise RuntimeError(f'{c} rel{rel:+d} transfer {au}')
   cams[c][rel]=cc; ca[c][str(rel)]=au
 for rel in RELS:
  cc,au=v33b.transfer_rar_camera(cert,fs[RAR][rel][2],scene['cameras'][RAR]); ca[RAR][str(rel)]=au
  if cc is not None:cams[RAR][rel]=cc
 model=RFDETRKeypointPreview(); tr={}; ia={}
 for c in (LAR,RAR,BCAST):tr[c],ia[c]=track(model,c,fs[c])
 rows=[]
 for lr in RELS:
  for rr in RELS:
   if rr not in cams[RAR]:continue
   for br in RELS:
    s=copy.deepcopy(scene); s['cameras'][LAR]=cams[LAR][lr]; s['cameras'][RAR]=cams[RAR][rr]; s['cameras'][BCAST]=cams[BCAST][br]; l,r,b=tr[LAR][lr],tr[RAR][rr],tr[BCAST][br]
    e1=v32v.epipolar_stats(s,LAR,RAR,l['xy'],valid(l),r['xy'],valid(r)); e2=v32v.epipolar_stats(s,LAR,BCAST,l['xy'],valid(l),b['xy'],valid(b)); e3=v32v.epipolar_stats(s,RAR,BCAST,r['xy'],valid(r),b['xy'],valid(b)); g=[pgate(e1),pgate(e2),pgate(e3)]; span=max(lr,rr,br)-min(lr,rr,br); score=sum(e['median_px']+.25*e['p90_px'] for e in (e1,e2,e3))+.3*span+12*sum(max(0,6-e['joint_count']) for e in (e1,e2,e3)); q=repro(s,l,r,b) if all(g) else None
    rows.append({'rels':{LAR:lr,RAR:rr,BCAST:br},'frames':{LAR:fs[LAR][lr][0].name,RAR:fs[RAR][rr][0].name,BCAST:fs[BCAST][br][0].name},'score':float(score),'span':span,'epi':{'LR':v33b.serialize_epi(e1),'LB':v33b.serialize_epi(e2),'RB':v33b.serialize_epi(e3)},'pair_gates':g,'all_pairwise':bool(all(g)),'reprojection':q,'exact_state':bool(q and q['gate'])})
 rows.sort(key=lambda x:(not x['exact_state'],not x['all_pairwise'],x['score'])); win=rows[0] if rows else None; exact=bool(win and win['exact_state'])
 qa={'version':'v33d_locked_three_camera_joint_state_search','status':'PASS_V33D_EXACT_THREE_VIEW_STATE' if exact else 'FAIL_CLOSED_V33D_NO_EXACT_THREE_VIEW_STATE','camera_lock':[LAR,RAR,BCAST],'camera_count':3,'camera_addition_or_substitution_used':False,'physical_center_refit_from_players':False,'native_resolution':[W,H],'generated_rgb':False,'upscaled':False,'novel_view_rendered':False,'search_rels':[-6,6],'identity_tracks':ia,'camera_transfers':ca,'tested_triplets':len(rows),'pairwise_pass_triplets':sum(x['all_pairwise'] for x in rows),'exact_pass_triplets':sum(x['exact_state'] for x in rows),'winner':win,'top_triplets':rows[:50],'freeview_render_unlocked':False,'next_if_pass':'three-view source-grounded body/ball fit then visual QA','next_if_fail':'keep same three cameras; widen/refine native temporal search; do not change cameras'}
 (a.out/'v33d_locked_three_camera_joint_state_search.json').write_text(json.dumps(qa,indent=2)); print(json.dumps({k:qa[k] for k in ('status','tested_triplets','pairwise_pass_triplets','exact_pass_triplets','winner')},indent=2),flush=True)
 if not exact:raise SystemExit(6)
if __name__=='__main__':main()
