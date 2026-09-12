from __future__ import annotations

"""v32j: RF-DETR exact-freeze three-camera articulated pose diagnostic.

Purpose: determine whether the three accepted cameras constrain Steven Adams'
actual articulated position before any player surface is rendered.

This stage uses RF-DETR Keypoint Preview as OBSERVATION ONLY.  It does not create
player pixels or body geometry.  The camera matrices remain the accepted v32
metric cameras.  Triangulation uses metric bearing rays rather than raw-pixel DLT
so Broadcast's much larger focal length cannot numerically dominate RAR.

The solver explicitly scores physical bone plausibility and leaves the Left view
unmatched if Adams is too occluded.  No capsule/blob/visual-hull render is made.
"""

import argparse, json, math
from pathlib import Path
import cv2
import numpy as np

from rfdetr import RFDETRKeypointPreview

CAMS=("Left Above Rim","Broadcast","Right Above Rim")
LAR,BCAST,RAR=CAMS
W,H=960,540
NAMES=["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
BONES=[
 (5,6,20,75),(5,7,15,60),(7,9,12,60),(6,8,15,60),(8,10,12,60),
 (5,11,25,105),(6,12,25,105),(11,12,12,70),(11,13,25,85),(13,15,20,85),(12,14,25,85),(14,16,20,85)
]
DRAW=[(a,b) for a,b,_,_ in BONES]+[(0,5),(0,6)]


def iou(a,b):
 ax1,ay1,ax2,ay2=map(float,a); bx1,by1,bx2,by2=map(float,b)
 ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
 inter=max(0,ix2-ix1)*max(0,iy2-iy1)
 aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
 return inter/max(1e-9,aa+bb-inter)


def cam(scene,label):
 d=scene['cameras'][label]
 return {'K':np.asarray(d['K_px'],float),'R':np.asarray(d['R_world_to_camera'],float),'C':np.asarray(d['C_world_cm'],float)}


def project(c,X):
 xc=c['R']@(np.asarray(X,float)-c['C'])
 if abs(float(xc[2]))<1e-7: return None
 q=c['K']@xc; return q[:2]/q[2]


def ray(c,uv):
 x=np.linalg.inv(c['K'])@np.array([uv[0],uv[1],1.0])
 d=c['R'].T@x; return d/max(1e-12,np.linalg.norm(d))


def triangulate_rays(cams,obs):
 A=np.zeros((3,3),float); b=np.zeros(3,float)
 for label,uv in obs.items():
  d=ray(cams[label],uv); M=np.eye(3)-np.outer(d,d); A+=M; b+=M@cams[label]['C']
 if np.linalg.cond(A)>1e10: return None
 X=np.linalg.solve(A,b)
 return X if np.all(np.isfinite(X)) else None


def fundamental(c1,c2):
 R1,C1,K1=c1['R'],c1['C'],c1['K']; R2,C2,K2=c2['R'],c2['C'],c2['K']
 R21=R2@R1.T; t21=R2@(C1-C2)
 tx=np.array([[0,-t21[2],t21[1]],[t21[2],0,-t21[0]],[-t21[1],t21[0],0]],float)
 F=np.linalg.inv(K2).T@tx@R21@np.linalg.inv(K1); return F/max(1e-12,np.linalg.norm(F))


def line_dist(p,l):
 a,b,c=l; return abs(a*p[0]+b*p[1]+c)/max(1e-9,math.hypot(a,b))


def epi(F,p1,p2):
 x1=np.r_[p1,1.]; x2=np.r_[p2,1.]
 return .5*(line_dist(p2,F@x1)+line_dist(p1,F.T@x2))


def infer(model,image):
 rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
 kp=model.predict(rgb,threshold=0.15)
 try: kp=kp.with_nms()
 except Exception: pass
 xy=np.asarray(kp.xy,float)
 conf=getattr(kp,'keypoint_confidence',None); conf=np.ones(xy.shape[:2]) if conf is None else np.asarray(conf,float)
 dc=getattr(kp,'detection_confidence',None); dc=np.ones(len(xy)) if dc is None else np.asarray(dc,float)
 boxes=np.asarray(kp.data.get('xyxy',np.empty((0,4))),float)
 return [{'xy':xy[i], 'conf':conf[i], 'det_conf':float(dc[i]), 'box':boxes[i]} for i in range(len(xy))]


def dark_fraction(img,box):
 x1,y1,x2,y2=np.rint(box).astype(int); x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
 if x2<=x1 or y2<=y1:return 0.0
 p=img[y1+int(.12*(y2-y1)):y1+int(.65*(y2-y1)),x1+int(.12*(x2-x1)):x1+int(.88*(x2-x1))]
 if not p.size:return 0.0
 return float(np.mean(cv2.cvtColor(p,cv2.COLOR_BGR2HSV)[...,2]<110))


def bone_stats(joints):
 rows={}; ok=0; n=0
 for a,b,lo,hi in BONES:
  if a in joints and b in joints:
   L=float(np.linalg.norm(joints[a]-joints[b])); good=lo<=L<=hi; n+=1; ok+=int(good)
   rows[f'{NAMES[a]}-{NAMES[b]}']={'length_cm':L,'range_cm':[lo,hi],'plausible':good}
 return rows,n,ok,(ok/n if n else 0.0)


def pair_eval(cams,F,b,r,confmin=.20):
 common=[j for j in range(17) if b['conf'][j]>=confmin and r['conf'][j]>=confmin]
 es=[epi(F,b['xy'][j],r['xy'][j]) for j in common]
 joints={}
 for j in common:
  X=triangulate_rays(cams,{BCAST:b['xy'][j],RAR:r['xy'][j]})
  if X is not None and -350<=X[0]<=1250 and -750<=X[1]<=750 and -60<=X[2]<=450: joints[j]=X
 bones,nb,nok,frac=bone_stats(joints)
 repro=[]
 for j,X in joints.items():
  for label,d in ((BCAST,b),(RAR,r)):
   uv=project(cams[label],X)
   if uv is not None: repro.append(float(np.linalg.norm(uv-d['xy'][j])))
 return {'common':len(common),'median_epi':float(np.median(es)) if es else 999.,'p75_epi':float(np.percentile(es,75)) if es else 999.,'joints':joints,'bones':bones,'measured_bones':nb,'plausible_bones':nok,'bone_fraction':frac,'median_reproj':float(np.median(repro)) if repro else 999.}


def serialize_det(d):
 return {'box_xyxy':d['box'].tolist(),'detection_confidence':d['det_conf'],'keypoints':{NAMES[j]:{'xy':d['xy'][j].tolist(),'confidence':float(d['conf'][j])} for j in range(17)}}


def draw(img,d,joints,cams,label):
 out=img.copy(); col=(0,255,255)
 if d is not None:
  for a,b in DRAW:
   if d['conf'][a]>=.20 and d['conf'][b]>=.20: cv2.line(out,tuple(np.rint(d['xy'][a]).astype(int)),tuple(np.rint(d['xy'][b]).astype(int)),col,2,cv2.LINE_AA)
  for j in range(17):
   if d['conf'][j]>=.20: cv2.circle(out,tuple(np.rint(d['xy'][j]).astype(int)),3,col,-1,cv2.LINE_AA)
 for a,b in DRAW:
  if a in joints and b in joints:
   ua=project(cams[label],joints[a]); ub=project(cams[label],joints[b])
   if ua is not None and ub is not None: cv2.line(out,tuple(np.rint(ua).astype(int)),tuple(np.rint(ub).astype(int)),(255,0,255),1,cv2.LINE_AA)
 for j,X in joints.items():
  u=project(cams[label],X)
  if u is not None: cv2.circle(out,tuple(np.rint(u).astype(int)),3,(255,0,255),1,cv2.LINE_AA)
 cv2.rectangle(out,(0,0),(W,27),(0,0,0),-1); cv2.putText(out,f'v32j {label} | yellow RF-DETR | magenta 3D reprojection',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
 return out


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--b32-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
 stage=args.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text()); q=json.loads((stage/'v32_quality_cluster.json').read_text())
 assert scene['resolution']==[W,H] and set(scene['cameras'])==set(CAMS)
 cams={c:cam(scene,c) for c in CAMS}; F=fundamental(cams[BCAST],cams[RAR])
 fn={BCAST:'v32_chosen_Broadcast_frame0276.png',LAR:'v32_chosen_Left_Above_Rim_frame0260.png',RAR:'v32_chosen_Right_Above_Rim_frame0256.png'}
 imgs={c:cv2.imread(str(stage/fn[c])) for c in CAMS}; assert all(x is not None for x in imgs.values())
 model=RFDETRKeypointPreview(); det={c:infer(model,imgs[c]) for c in CAMS}
 focal=q['focal_player']['observations']; bb=np.asarray(focal[BCAST]['bbox_xyxy'],float); rb=np.asarray(focal[RAR]['bbox_xyxy'],float)
 bs=[]
 for i,d in enumerate(det[BCAST]):
  s=4*iou(d['box'],bb)+.5*dark_fraction(imgs[BCAST],d['box'])
  bs.append((s,i))
 bs.sort(reverse=True); bi=bs[0][1]; b=det[BCAST][bi]
 candidates=[]
 for i,r in enumerate(det[RAR]):
  pe=pair_eval(cams,F,b,r)
  anchor=iou(r['box'],rb); dark=dark_fraction(imgs[RAR],r['box'])
  # anatomy is deliberately strong: a low-epi hybrid two-person pose must lose.
  cost=pe['median_epi'] + 35*(1-pe['bone_fraction']) + max(0,8-pe['common'])*5 - 8*anchor - 2*dark
  candidates.append({'index':i,'cost':float(cost),'anchor_iou':float(anchor),'dark_fraction':dark,**{k:v for k,v in pe.items() if k not in ('joints','bones')},'bones':pe['bones'],'_joints':pe['joints']})
 candidates.sort(key=lambda x:x['cost']); best=candidates[0]; ri=best['index']; r=det[RAR][ri]; joints=best.pop('_joints')
 for row in candidates[1:]: row.pop('_joints',None)
 # Left is validation only; never forced.
 lrows=[]
 for i,l in enumerate(det[LAR]):
  errs=[]
  for j,X in joints.items():
   if l['conf'][j]<.20: continue
   u=project(cams[LAR],X)
   if u is not None: errs.append(float(np.linalg.norm(u-l['xy'][j])))
  med=float(np.median(errs)) if len(errs)>=4 else 999.; p75=float(np.percentile(errs,75)) if len(errs)>=4 else 999.
  lrows.append({'index':i,'joint_count':len(errs),'median_px':med,'p75_px':p75,'cost':med+.2*p75})
 lrows.sort(key=lambda x:x['cost']); li=None
 if lrows and lrows[0]['joint_count']>=4 and lrows[0]['median_px']<=32 and lrows[0]['p75_px']<=50: li=lrows[0]['index']
 l=None if li is None else det[LAR][li]
 # If a valid measured Left pose exists, re-triangulate each available joint from all three rays.
 final={}
 for j,X0 in joints.items():
  obs={BCAST:b['xy'][j],RAR:r['xy'][j]}
  if l is not None and l['conf'][j]>=.20: obs[LAR]=l['xy'][j]
  X=triangulate_rays(cams,obs); final[j]=X if X is not None else X0
 bones,nb,nok,bfrac=bone_stats(final)
 residual=[]
 for j,X in final.items():
  for label,d in ((BCAST,b),(RAR,r),(LAR,l)):
   if d is None or d['conf'][j]<.20: continue
   u=project(cams[label],X)
   if u is not None: residual.append(float(np.linalg.norm(u-d['xy'][j])))
 medres=float(np.median(residual)) if residual else 999.
 leftq=lrows[0] if lrows else {'joint_count':0,'median_px':999.,'p75_px':999.}
 anatomy_ok=nb>=5 and bfrac>=.70; geometry_ok=len(final)>=8 and medres<=18.; left_ok=l is not None
 status='PASS_V32J_ARTICULATED_POSITION_OBSERVATION' if anatomy_ok and geometry_ok else 'FAIL_CLOSED_V32J_ARTICULATED_OBSERVATION'
 qa={'version':'v32j_rfdetr_freeze_pose','status':status,'native_resolution':[W,H],'generated_rgb':False,'mesh_rendered':False,'camera_geometry_modified':False,'model_role':'RF-DETR Keypoint Preview observation only','selected':{BCAST:bi,RAR:ri,LAR:li},'broadcast_detection':serialize_det(b),'rar_detection':serialize_det(r),'left_detection':None if l is None else serialize_det(l),'rar_candidates':candidates,'left_candidates':lrows,'triangulated_joint_count':len(final),'median_observation_reprojection_px':medres,'bone_plausible_fraction':bfrac,'measured_bone_count':nb,'bone_details':bones,'left_measured_pose_found':left_ok,'left_best_median_px':leftq['median_px'],'left_best_p75_px':leftq['p75_px'],'joints':{NAMES[j]:{'world_cm':final[j].tolist()} for j in final},'gate':{'at_least_8_joints':len(final)>=8,'median_reprojection_le_18px':medres<=18.,'bone_fraction_ge_0_70':anatomy_ok,'left_pose_is_validation_not_forced':True,'surface_initializer_geometry_supported':bool(anatomy_ok and geometry_ok)}}
 (args.out/'v32j_rfdetr_pose_qa.json').write_text(json.dumps(qa,indent=2))
 ovs=[]
 for label,d in ((LAR,l),(BCAST,b),(RAR,r)):
  ov=draw(imgs[label],d,final,cams,label); cv2.imwrite(str(args.out/f'v32j_{label.replace(" ","_")}.png'),ov); ovs.append(ov)
 cv2.imwrite(str(args.out/'v32j_three_camera_montage.png'),np.hstack(ovs))
 print(json.dumps({'status':status,'joints':len(final),'median_reproj':medres,'bone_fraction':bfrac,'left':leftq,'selected':qa['selected']},indent=2))
 if not (anatomy_ok and geometry_ok): raise SystemExit(3)

if __name__=='__main__': main()
