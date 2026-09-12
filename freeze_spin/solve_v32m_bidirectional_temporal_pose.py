from __future__ import annotations

"""v32m: bidirectional temporal pose recovery at the exact RAR freeze.

Past anchors t-03,-02,-01 are tracked forward to t+00.  Future anchors
t+04,+05,+06 are tracked backward to t+00.  All anchors are real synchronized
RAR source frames and RF-DETR is observation-only.  Sparse LK is used only as a
measurement operator.  A freeze joint is accepted from multi-anchor consensus;
no image pixels or virtual views are synthesized.
"""

import argparse,json
from pathlib import Path
import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

LAR,BCAST,RAR=v32j.LAR,v32j.BCAST,v32j.RAR
CAMS=v32j.CAMS; W,H=v32j.W,v32j.H; NAMES=v32j.NAMES; DRAW=v32j.DRAW
PAST=(-3,-2,-1); FUTURE=(4,5,6); ANCHORS=PAST+FUTURE


def read_burst(stage,label,rel):
    xs=sorted((stage/'burst'/label.replace(' ','_')).glob(f'rel{rel:+03d}_frame*.png'))
    if len(xs)!=1: raise RuntimeError((label,rel,xs))
    im=cv2.imread(str(xs[0]));
    if im is None: raise RuntimeError(xs[0])
    return im


def torso_dark(img,d):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); vals=[]
    for j in (5,6,11,12):
        if d['conf'][j]<.2: continue
        x,y=np.rint(d['xy'][j]).astype(int); r=8
        p=hsv[max(0,y-r):min(H,y+r+1),max(0,x-r):min(W,x+r+1),2]
        if p.size: vals.append(float(np.mean(p<115)))
    return float(np.mean(vals)) if vals else 0.


def choose(img,dets,target):
    rows=[]
    for i,d in enumerate(dets):
        c=np.array([(d['box'][0]+d['box'][2])/2,(d['box'][1]+d['box'][3])/2]); dist=float(np.linalg.norm(c-target))
        dark=v32j.dark_fraction(img,d['box']); td=torso_dark(img,d); n=int(np.sum(d['conf']>=.2))
        score=2.7*dark+2.3*td+.35*d['det_conf']+.035*n-.006*dist
        rows.append({'index':i,'score':float(score),'dark_fraction':float(dark),'torso_dark':float(td),'confident_joints':n,'center_distance_px':dist})
    rows.sort(key=lambda x:x['score'],reverse=True)
    return rows[0]['index'],rows


def lk_step(src,dst,p,valid):
    lk=dict(winSize=(41,41),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,50,.01),minEigThreshold=1e-6)
    g1=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY); g0=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    q,s1,e1=cv2.calcOpticalFlowPyrLK(g1,g0,p,None,**lk); back,s2,e2=cv2.calcOpticalFlowPyrLK(g0,g1,q,None,**lk)
    fb=np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1); st=s1.reshape(-1).astype(bool)&s2.reshape(-1).astype(bool)
    inside=(q[:,0,0]>=0)&(q[:,0,0]<W)&(q[:,0,1]>=0)&(q[:,0,1]<H)
    good=valid&st&inside&(fb<=3.5)
    return q,good,fb


def track_to_zero(frames,start,xy,conf):
    p=xy.astype(np.float32).reshape(-1,1,2).copy(); valid=(conf>=.2).astype(bool); steps=[]
    if start<0:
        seq=list(range(start,0))
        for rel in seq:
            p,valid,fb=lk_step(frames[rel],frames[rel+1],p,valid)
            steps.append({'from_rel':rel,'to_rel':rel+1,'valid':int(np.sum(valid)),'fb_median_px':float(np.median(fb[valid])) if np.any(valid) else 999.})
    else:
        for rel in range(start,0,-1):
            p,valid,fb=lk_step(frames[rel],frames[rel-1],p,valid)
            steps.append({'from_rel':rel,'to_rel':rel-1,'valid':int(np.sum(valid)),'fb_median_px':float(np.median(fb[valid])) if np.any(valid) else 999.})
    return p.reshape(-1,2).astype(float),valid,steps


def consensus(rows):
    pts={}; audit={}
    for j in range(17):
        samples=[]
        for r in rows:
            if r['valid'][j]: samples.append((r['rel'],r['xy0'][j]))
        if len(samples)<2:
            audit[j]={'accepted':False,'support':len(samples),'anchors':[x[0] for x in samples]}; continue
        a=np.asarray([x[1] for x in samples],float); med=np.median(a,axis=0); ds=np.linalg.norm(a-med,axis=1)
        past=sum(1 for rel,_ in samples if rel<0); future=sum(1 for rel,_ in samples if rel>0)
        temporal_support=(past>=2) or (past>=1 and future>=1)
        ok=temporal_support and float(np.median(ds))<=10. and float(np.percentile(ds,90))<=20.
        audit[j]={'accepted':bool(ok),'support':len(samples),'past_support':past,'future_support':future,'anchors':[x[0] for x in samples],
                  'xy':med.tolist(),'median_spread_px':float(np.median(ds)),'p90_spread_px':float(np.percentile(ds,90))}
        if ok: pts[j]=med
    return pts,audit


def ownership(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); raw=(hsv[...,2]<100).astype(np.uint8)*255; roi=np.zeros_like(raw); roi[120:360,360:700]=255
    m=cv2.bitwise_and(raw,roi); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,stats,cent=cv2.connectedComponentsWithStats(m,8); cand=[]
    for i in range(1,n):
        area=int(stats[i,4]); cx,cy=cent[i]
        if area>=250 and 470<=cx<=660 and 150<=cy<=330: cand.append((area,i))
    if not cand:return np.zeros_like(m)
    return (lab==max(cand)[1]).astype(np.uint8)*255


def distance(mask,p):
    d=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,5); x,y=np.rint(p).astype(int)
    return float(d[y,x]) if 0<=x<W and 0<=y<H else 999.


def draw(img,pts=None,mask=None,title=''):
    out=img.copy()
    if mask is not None:
        cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out,cs,-1,(0,255,255),2)
    if pts:
        for a,b in DRAW:
            if a in pts and b in pts: cv2.line(out,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),(255,255,0),2,cv2.LINE_AA)
        for j,p in pts.items(): cv2.circle(out,tuple(np.rint(p).astype(int)),4,(255,255,0),-1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,28),(0,0,0),-1); cv2.putText(out,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--b32-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    stage=args.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text()); q=json.loads((stage/'v32_quality_cluster.json').read_text())
    cams={c:v32j.cam(scene,c) for c in CAMS}; model=RFDETRKeypointPreview(); frames={r:read_burst(stage,RAR,r) for r in range(-3,7)}
    rb=np.asarray(q['focal_player']['observations'][RAR]['bbox_xyxy'],float); target=np.array([(rb[0]+rb[2])/2,(rb[1]+rb[3])/2])
    tracks=[]; anchors={}
    for rel in ANCHORS:
        dets=v32j.infer(model,frames[rel]); idx,rank=choose(frames[rel],dets,target); d=dets[idx]; xy0,val,steps=track_to_zero(frames,rel,d['xy'],d['conf'])
        tracks.append({'rel':rel,'xy0':xy0,'valid':val}); anchors[str(rel)]={'selected_index':idx,'ranked':rank,'valid_t0_joint_count':int(np.sum(val)),'steps':steps}
        cv2.imwrite(str(args.out/f'v32m_anchor_{rel:+03d}.png'),draw(frames[rel],{j:d['xy'][j] for j in range(17) if d['conf'][j]>=.2},title=f'v32m RAR t{rel:+d} RF-DETR anchor'))
    pts,audit=consensus(tracks); own=ownership(frames[0]); od={j:distance(own,p) for j,p in pts.items()}; torso=[j for j in (5,6,11,12) if j in pts and od[j]<=18]
    cv2.imwrite(str(args.out/'v32m_rar_t00_bidirectional.png'),draw(frames[0],pts,own,'v32m RAR t+00 | bidirectional temporal consensus + uniform ownership'))

    bimg=cv2.imread(str(stage/'v32_chosen_Broadcast_frame0276.png')); limg=cv2.imread(str(stage/'v32_chosen_Left_Above_Rim_frame0260.png'))
    bd=v32j.infer(model,bimg); bb=np.asarray(q['focal_player']['observations'][BCAST]['bbox_xyxy'],float); bi=max(range(len(bd)),key=lambda i:4*v32j.iou(bd[i]['box'],bb)+.5*v32j.dark_fraction(bimg,bd[i]['box'])); b=bd[bi]
    joints={}; res={}
    for j,rp in pts.items():
        if b['conf'][j]<.2:continue
        X=v32j.triangulate_rays(cams,{BCAST:b['xy'][j],RAR:rp})
        if X is None or not (-350<=X[0]<=1250 and -750<=X[1]<=750 and -60<=X[2]<=450):continue
        eb=float(np.linalg.norm(v32j.project(cams[BCAST],X)-b['xy'][j])); er=float(np.linalg.norm(v32j.project(cams[RAR],X)-rp)); joints[j]=X; res[j]={'broadcast_px':eb,'rar_px':er}
    bones,nb,nok,bfrac=v32j.bone_stats(joints); errs=[e for x in res.values() for e in x.values()]; med=float(np.median(errs)) if errs else 999.
    ld=v32j.infer(model,limg); lrows=[]
    for i,d in enumerate(ld):
        es=[]
        for j,X in joints.items():
            if d['conf'][j]<.2:continue
            uv=v32j.project(cams[LAR],X)
            if uv is not None:es.append(float(np.linalg.norm(uv-d['xy'][j])))
        lrows.append({'index':i,'joint_count':len(es),'median_px':float(np.median(es)) if len(es)>=4 else 999.,'p75_px':float(np.percentile(es,75)) if len(es)>=4 else 999.})
    lrows.sort(key=lambda x:x['median_px']+.2*x['p75_px']); left=lrows[0] if lrows else {'joint_count':0,'median_px':999.,'p75_px':999.}
    status='PASS_V32M_BIDIRECTIONAL_TEMPORAL_POSE' if (len(pts)>=6 and len(torso)>=2 and len(joints)>=6 and nb>=4 and bfrac>=.70 and med<=16.) else 'FAIL_CLOSED_V32M_BIDIRECTIONAL_TEMPORAL_POSE'
    qa={'version':'v32m_bidirectional_temporal_pose','status':status,'native_resolution':[W,H],'source_frames_only':True,'generated_rgb':False,'mesh_rendered':False,
        'anchors':list(ANCHORS),'anchor_audit':anchors,'consensus_joint_count':len(pts),'consensus':{NAMES[j]:audit[j] for j in range(17)},
        'ownership_torso_joint_count':len(torso),'ownership_torso_joints':[NAMES[j] for j in torso],'ownership_distance_px':{NAMES[j]:d for j,d in od.items()},
        'triangulated_joint_count':len(joints),'median_two_view_reprojection_px':med,'measured_bone_count':nb,'bone_plausible_fraction':bfrac,'bone_details':bones,'left_best_validation':left,
        'joints':{NAMES[j]:{'world_cm':X.tolist(),'rar_t00_xy':pts[j].tolist(),'residuals_px':res[j]} for j,X in joints.items()},
        'gate':{'consensus_ge_6':len(pts)>=6,'torso_owned_ge_2':len(torso)>=2,'triangulated_ge_6':len(joints)>=6,'bones_ge_4':nb>=4,'bone_fraction_ge_0_70':bfrac>=.70,'median_reprojection_le_16px':med<=16.,'surface_stage_unlocked':status.startswith('PASS_')}}
    (args.out/'v32m_bidirectional_temporal_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps({'status':status,'consensus':len(pts),'torso_owned':len(torso),'triangulated':len(joints),'bones':nb,'bone_fraction':bfrac,'median_reproj':med,'left':left},indent=2))
    if not status.startswith('PASS_'):raise SystemExit(5)

if __name__=='__main__':main()
