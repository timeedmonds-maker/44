from __future__ import annotations

"""v32l: temporally deblend Adams in RAR with exact-frame pixel ownership.

v32j proved that single-frame RF-DETR can merge Adams and the Utah defender.
v32k proved that t+04..t+06 contain clean Adams-only RF-DETR poses, but its LK
photometric-error threshold incorrectly rejected all tracks.  v32l keeps the
clean anchors, uses forward/backward LK consistency as the tracking gate, and
adds an independent exact-freeze dark-uniform connected-component ownership
mask.  The mask is source pixels only and is never used to generate RGB.
"""

import argparse, json
from pathlib import Path
import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

LAR,BCAST,RAR=v32j.LAR,v32j.BCAST,v32j.RAR
CAMS=v32j.CAMS; W,H=v32j.W,v32j.H; NAMES=v32j.NAMES; DRAW=v32j.DRAW
ANCHORS=(4,5,6)


def burst(stage,label,rel):
    xs=sorted((stage/'burst'/label.replace(' ','_')).glob(f'rel{rel:+03d}_frame*.png'))
    if len(xs)!=1: raise RuntimeError((label,rel,xs))
    return cv2.imread(str(xs[0]))


def dark_fraction(img,box): return v32j.dark_fraction(img,box)


def torso_dark(img,d):
    vals=[]
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for j in (5,6,11,12):
        if d['conf'][j]<.2: continue
        x,y=np.rint(d['xy'][j]).astype(int); r=8
        p=hsv[max(0,y-r):min(H,y+r+1),max(0,x-r):min(W,x+r+1),2]
        if p.size: vals.append(float(np.mean(p<115)))
    return float(np.mean(vals)) if vals else 0.0


def choose_adams(img,dets,target):
    rows=[]
    for i,d in enumerate(dets):
        c=np.array([(d['box'][0]+d['box'][2])/2,(d['box'][1]+d['box'][3])/2])
        dist=float(np.linalg.norm(c-target)); dark=dark_fraction(img,d['box']); td=torso_dark(img,d)
        n=int(np.sum(d['conf']>=.2))
        score=2.7*dark+2.3*td+.35*d['det_conf']+.035*n-.006*dist
        rows.append({'index':i,'score':float(score),'dark_fraction':float(dark),'torso_dark':float(td),'confident_joints':n,'center_distance_px':dist})
    rows.sort(key=lambda x:x['score'],reverse=True)
    return (rows[0]['index'] if rows else None),rows


def lk_back(frames,start,xy,conf):
    p=xy.astype(np.float32).reshape(-1,1,2).copy(); valid=(conf>=.2).astype(bool)
    lk=dict(winSize=(41,41),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,50,.01),minEigThreshold=1e-6)
    steps=[]
    for rel in range(start,0,-1):
        g1=cv2.cvtColor(frames[rel],cv2.COLOR_BGR2GRAY); g0=cv2.cvtColor(frames[rel-1],cv2.COLOR_BGR2GRAY)
        prev,s1,e1=cv2.calcOpticalFlowPyrLK(g1,g0,p,None,**lk)
        back,s2,e2=cv2.calcOpticalFlowPyrLK(g0,g1,prev,None,**lk)
        fb=np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1)
        st=s1.reshape(-1).astype(bool)&s2.reshape(-1).astype(bool)
        inside=(prev[:,0,0]>=0)&(prev[:,0,0]<W)&(prev[:,0,1]>=0)&(prev[:,0,1]<H)
        valid=valid&st&inside&(fb<=3.5)
        steps.append({'to_rel':rel-1,'valid':int(np.sum(valid)),'fb_median_px':float(np.median(fb[valid])) if np.any(valid) else 999.0})
        p=prev
    return p.reshape(-1,2).astype(float),valid,steps


def consensus(rows):
    pts={}; audit={}
    for j in range(17):
        a=np.asarray([r['xy0'][j] for r in rows if r['valid'][j]],float)
        anchors=[r['rel'] for r in rows if r['valid'][j]]
        if len(a)<2:
            audit[j]={'accepted':False,'support':len(a),'anchors':anchors}; continue
        med=np.median(a,axis=0); ds=np.linalg.norm(a-med,axis=1)
        ok=float(np.median(ds))<=8.0 and float(np.percentile(ds,90))<=16.0
        audit[j]={'accepted':bool(ok),'support':len(a),'anchors':anchors,'xy':med.tolist(),'median_spread_px':float(np.median(ds)),'p90_spread_px':float(np.percentile(ds,90))}
        if ok: pts[j]=med
    return pts,audit


def ownership_mask(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); raw=(hsv[...,2]<100).astype(np.uint8)*255
    roi=np.zeros_like(raw); roi[120:360,360:700]=255; m=cv2.bitwise_and(raw,roi)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,stats,cent=cv2.connectedComponentsWithStats(m,8)
    cand=[]
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA]); cx,cy=cent[i]
        # exact event action region: prefer large dark component on Adams/right side of rim.
        if area>=250 and 470<=cx<=660 and 150<=cy<=330: cand.append((area,i))
    if not cand: return np.zeros_like(m)
    i=max(cand)[1]; return (lab==i).astype(np.uint8)*255


def dist_to_mask(mask,xy):
    inv=(mask==0).astype(np.uint8); d=cv2.distanceTransform(inv,cv2.DIST_L2,5)
    x,y=np.rint(xy).astype(int)
    if not (0<=x<W and 0<=y<H): return 999.0
    return float(d[y,x])


def draw(img,pts=None,mask=None,title=''):
    out=img.copy()
    if mask is not None:
        cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out,cs,-1,(0,255,255),2)
    if pts:
        for a,b in DRAW:
            if a in pts and b in pts: cv2.line(out,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),(255,255,0),2,cv2.LINE_AA)
        for j,p in pts.items(): cv2.circle(out,tuple(np.rint(p).astype(int)),4,(255,255,0),-1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,28),(0,0,0),-1); cv2.putText(out,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--b32-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    stage=a.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text()); q=json.loads((stage/'v32_quality_cluster.json').read_text())
    cams={c:v32j.cam(scene,c) for c in CAMS}; model=RFDETRKeypointPreview()
    rf={r:burst(stage,RAR,r) for r in range(0,7)}; assert all(x is not None for x in rf.values())
    rb=np.asarray(q['focal_player']['observations'][RAR]['bbox_xyxy'],float); target=np.array([(rb[0]+rb[2])/2,(rb[1]+rb[3])/2])
    tracks=[]; anchor_audit={}
    for rel in ANCHORS:
        ds=v32j.infer(model,rf[rel]); idx,rank=choose_adams(rf[rel],ds,target); d=ds[idx]
        xy0,val,steps=lk_back(rf,rel,d['xy'],d['conf']); tracks.append({'rel':rel,'xy0':xy0,'valid':val})
        anchor_audit[str(rel)]={'selected_index':idx,'ranked':rank,'valid_t0_joint_count':int(np.sum(val)),'steps':steps}
        cv2.imwrite(str(a.out/f'v32l_anchor_{rel:+03d}.png'),draw(rf[rel],{j:d['xy'][j] for j in range(17) if d['conf'][j]>=.2},title=f'v32l RAR t+{rel} RF-DETR Adams anchor'))
    pts,audit=consensus(tracks); own=ownership_mask(rf[0])
    own_dist={j:dist_to_mask(own,p) for j,p in pts.items()}
    torso_owned=[j for j in (5,6,11,12) if j in pts and own_dist[j]<=18.0]
    cv2.imwrite(str(a.out/'v32l_rar_t00_consensus_ownership.png'),draw(rf[0],pts,own,'v32l RAR t+00 | cyan uniform ownership | yellow temporal consensus'))

    bimg=cv2.imread(str(stage/'v32_chosen_Broadcast_frame0276.png')); limg=cv2.imread(str(stage/'v32_chosen_Left_Above_Rim_frame0260.png'))
    bd=v32j.infer(model,bimg); bb=np.asarray(q['focal_player']['observations'][BCAST]['bbox_xyxy'],float)
    bi=max(range(len(bd)),key=lambda i:4*v32j.iou(bd[i]['box'],bb)+.5*v32j.dark_fraction(bimg,bd[i]['box'])); b=bd[bi]
    joints={}; residual={}
    for j,rp in pts.items():
        if b['conf'][j]<.2: continue
        X=v32j.triangulate_rays(cams,{BCAST:b['xy'][j],RAR:rp})
        if X is None or not (-350<=X[0]<=1250 and -750<=X[1]<=750 and -60<=X[2]<=450): continue
        eb=float(np.linalg.norm(v32j.project(cams[BCAST],X)-b['xy'][j])); er=float(np.linalg.norm(v32j.project(cams[RAR],X)-rp))
        joints[j]=X; residual[j]={'broadcast_px':eb,'rar_px':er}
    bones,nb,nok,bfrac=v32j.bone_stats(joints); errs=[e for x in residual.values() for e in x.values()]; med=float(np.median(errs)) if errs else 999.

    ld=v32j.infer(model,limg); lrows=[]
    for i,d in enumerate(ld):
        es=[]
        for j,X in joints.items():
            if d['conf'][j]<.2: continue
            uv=v32j.project(cams[LAR],X)
            if uv is not None: es.append(float(np.linalg.norm(uv-d['xy'][j])))
        lrows.append({'index':i,'joint_count':len(es),'median_px':float(np.median(es)) if len(es)>=4 else 999.,'p75_px':float(np.percentile(es,75)) if len(es)>=4 else 999.})
    lrows.sort(key=lambda x:x['median_px']+.2*x['p75_px']); left=lrows[0] if lrows else {'joint_count':0,'median_px':999.,'p75_px':999.}

    status='PASS_V32L_TEMPORAL_OWNERSHIP' if (len(pts)>=6 and len(torso_owned)>=2 and len(joints)>=6 and nb>=4 and bfrac>=.70 and med<=16.) else 'FAIL_CLOSED_V32L_TEMPORAL_OWNERSHIP'
    qa={'version':'v32l_rar_temporal_ownership','status':status,'native_resolution':[W,H],'generated_rgb':False,'mesh_rendered':False,
        'anchors':list(ANCHORS),'anchor_audit':anchor_audit,'consensus_joint_count':len(pts),'consensus':{NAMES[j]:audit[j] for j in range(17)},
        'ownership_mask_rule':'largest V<100 connected component in fixed action ROI; source-pixel identity gate only','ownership_torso_joint_count':len(torso_owned),'ownership_torso_joints':[NAMES[j] for j in torso_owned],
        'ownership_distance_px':{NAMES[j]:d for j,d in own_dist.items()},'triangulated_joint_count':len(joints),'median_two_view_reprojection_px':med,
        'measured_bone_count':nb,'bone_plausible_fraction':bfrac,'bone_details':bones,'left_best_validation':left,
        'joints':{NAMES[j]:{'world_cm':X.tolist(),'rar_t00_xy':pts[j].tolist(),'residuals_px':residual[j]} for j,X in joints.items()},
        'gate':{'consensus_ge_6':len(pts)>=6,'torso_owned_ge_2':len(torso_owned)>=2,'triangulated_ge_6':len(joints)>=6,'bones_ge_4':nb>=4,'bone_fraction_ge_0_70':bfrac>=.70,'median_reprojection_le_16px':med<=16.,'surface_stage_unlocked':status.startswith('PASS_')}}
    (a.out/'v32l_temporal_ownership_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps({'status':status,'consensus':len(pts),'torso_owned':len(torso_owned),'triangulated':len(joints),'bones':nb,'bone_fraction':bfrac,'median_reproj':med,'left_best':left},indent=2))
    if not status.startswith('PASS_'): raise SystemExit(4)

if __name__=='__main__': main()
