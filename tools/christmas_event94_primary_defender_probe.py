#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from adams_screen_prod_poc import fetch_native_clip

# Validated event-94 release-time track boxes from the deterministic V3 run.
# Exact source time is nearest tracked frame to 10.56 s: 10.5370667 s.
TRACKS={
  1:(633.38184,282.72302,704.29320,408.56380),
  2:(515.67510,239.90216,568.92990,365.75770),
  5:(305.89166,214.07037,357.57343,329.72990),
  9:(356.51020,224.99799,406.01047,420.87730),
 10:(364.68646,225.32680,412.41342,340.55496),
}
LAL={1,2,5,10}
DURANT_TRACK=9
H=np.array([[4.763075788199326,6.6499295809312695,-3809.6913744473422],
            [-2.978826787652546,12.935424948142028,-1282.041548444928],
            [-0.00044782946726342653,0.0036479999461021674,1.0]],dtype=np.float64)

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=max(1,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)
    return inter/ua

def court(p):
    v=H@np.array([p[0],p[1],1.0]); return v[:2]/v[2]

def anchor_from_pose(kxy,kcf,box):
    # COCO pose ankles 15,16. If only one ankle is reliable, use it.
    good=[]
    for idx in (15,16):
        if idx < len(kxy) and float(kcf[idx])>=0.20 and kxy[idx][0]>1 and kxy[idx][1]>1:
            good.append(kxy[idx])
    if good:
        p=np.mean(np.asarray(good,dtype=float),axis=0)
        return (float(p[0]),float(p[1])), 'ankles', len(good)
    # Knees 13,14 + person-box floor projection. This keeps the x position pose-led
    # while using box bottom only for y when ankles are fully occluded.
    knees=[]
    for idx in (13,14):
        if idx < len(kxy) and float(kcf[idx])>=0.20 and kxy[idx][0]>1 and kxy[idx][1]>1:
            knees.append(kxy[idx])
    if knees:
        x=float(np.mean(np.asarray(knees),axis=0)[0]); return (x,float(box[3])), 'knee_x_box_floor', len(knees)
    return ((float(box[0]+box[2])/2.0,float(box[3])),'box_floor',0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='yolo11x-pose.pt'); ap.add_argument('--out',type=Path,default=Path('artifacts/christmas_primary_defender'))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    clip=a.out/'source_native.mp4'; fetch_native_clip('0022500012',94,clip,960)
    cap=cv2.VideoCapture(str(clip)); cap.set(cv2.CAP_PROP_POS_MSEC,10.56*1000); ok,fr=cap.read(); cap.release()
    if not ok: raise RuntimeError('release frame unavailable')
    model=YOLO(a.model); res=model.predict(fr,verbose=False,device='cpu',conf=.20)[0]
    poses=[]
    if res.boxes is not None and res.keypoints is not None:
        boxes=res.boxes.xyxy.cpu().numpy(); kxy=res.keypoints.xy.cpu().numpy(); kcf=res.keypoints.conf.cpu().numpy()
        for i,b in enumerate(boxes):
            bb=tuple(map(float,b)); best_tid=None; best_iou=0.0
            for tid,tb in TRACKS.items():
                s=iou(bb,tb)
                if s>best_iou: best_tid,best_iou=tid,s
            anc,method,n=anchor_from_pose(kxy[i],kcf[i],bb)
            poses.append({'pose_i':i,'box':bb,'track_id':best_tid,'iou':best_iou,'anchor':anc,'anchor_method':method,'ankles_used':n})
    # One-to-one: take best pose match for each target track.
    bytrack={}
    for p in sorted(poses,key=lambda z:z['iou'],reverse=True):
        tid=p['track_id']
        if tid is not None and p['iou']>=.15 and tid not in bytrack: bytrack[tid]=p
    if DURANT_TRACK not in bytrack: raise RuntimeError('Durant release pose not resolved')
    danc=bytrack[DURANT_TRACK]['anchor']; dc=court(danc)
    candidates=[]
    for tid in sorted(LAL):
        if tid not in bytrack: continue
        anc=bytrack[tid]['anchor']; cc=court(anc); dist_cm=float(np.linalg.norm(dc-cc)); dist_ft=dist_cm/30.48
        candidates.append({'track_id':tid,'distance_ft':dist_ft,'anchor':anc,'court_cm':cc.tolist(),'pose_i':bytrack[tid]['pose_i'],'iou':bytrack[tid]['iou'],'anchor_method':bytrack[tid]['anchor_method']})
    candidates.sort(key=lambda z:z['distance_ft'])
    primary=candidates[0] if candidates else None
    out={'release_s':10.56,'durant':{'track_id':9,'anchor':danc,'court_cm':dc.tolist(),'anchor_method':bytrack[9]['anchor_method'],'iou':bytrack[9]['iou']},'lal_candidates':candidates,'primary_defender':primary,'all_pose_matches':poses}
    (a.out/'primary_defender.json').write_text(json.dumps(out,indent=2))
    dbg=fr.copy()
    for p in poses:
        x1,y1,x2,y2=map(int,p['box']); tid=p['track_id']; col=(0,255,0) if tid==9 else (255,255,0)
        cv2.rectangle(dbg,(x1,y1),(x2,y2),col,1); cv2.circle(dbg,tuple(map(lambda z:int(round(z)),p['anchor'])),5,col,-1,cv2.LINE_AA)
        cv2.putText(dbg,f"T{tid} {p['anchor_method']} {p['iou']:.2f}",(x1,max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,0,0),3,cv2.LINE_AA); cv2.putText(dbg,f"T{tid} {p['anchor_method']} {p['iou']:.2f}",(x1,max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
    if primary:
        p1=tuple(map(lambda z:int(round(z)),danc)); p2=tuple(map(lambda z:int(round(z)),primary['anchor'])); cv2.line(dbg,p1,p2,(0,255,255),2,cv2.LINE_AA)
        cv2.putText(dbg,f"{primary['distance_ft']:.1f} FT",((p1[0]+p2[0])//2,(p1[1]+p2[1])//2-8),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA); cv2.putText(dbg,f"{primary['distance_ft']:.1f} FT",((p1[0]+p2[0])//2,(p1[1]+p2[1])//2-8),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(a.out/'primary_defender_debug.jpg'),dbg,[cv2.IMWRITE_JPEG_QUALITY,97]); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
