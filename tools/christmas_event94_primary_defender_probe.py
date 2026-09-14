#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
import cv2, numpy as np, requests
from ultralytics import YOLO

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from fetch_official_nba_event_clip import parse as resolve_clip, UA
HEADERS={'User-Agent':UA,'Referer':'https://clips.nba.com/'}

def choose_hls_variant(url,target_width=960):
    r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status(); text=r.text
    if '#EXT-X-STREAM-INF' not in text: return url,None
    lines=[x.strip() for x in text.splitlines() if x.strip()]; variants=[]
    for i,line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF'): continue
        m=re.search(r'RESOLUTION=(\d+)x(\d+)',line); bw=re.search(r'BANDWIDTH=(\d+)',line); uri=None
        for j in range(i+1,min(i+4,len(lines))):
            if not lines[j].startswith('#'): uri=lines[j]; break
        if not uri: continue
        w=int(m.group(1)) if m else 10**9; h=int(m.group(2)) if m else None; full=urljoin(url,uri)
        if not urlsplit(full).query and urlsplit(url).query:
            q=urlsplit(url); f=urlsplit(full); full=urlunsplit((f.scheme,f.netloc,f.path,q.query,f.fragment))
        variants.append((w,h,int(bw.group(1)) if bw else None,full))
    if not variants: return url,None
    under=[v for v in variants if v[0]<=target_width]; ch=max(under,key=lambda x:x[0]) if under else min(variants,key=lambda x:x[0])
    return ch[3],{'width':ch[0],'height':ch[1],'bandwidth':ch[2]}

def fetch_native_clip(game,event,out,target_width=960):
    page,angles,chosen=resolve_clip(game,event); hls,variant=choose_hls_variant(chosen['url'],target_width)
    headers=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',headers,'-i',hls,'-c','copy',str(out)],check=True,timeout=180)
    return {'clip_page':page,'angle':chosen['label'],'angle_count':len(angles),'variant':variant}

TRACKS={1:(633.38184,282.72302,704.29320,408.56380),2:(515.67510,239.90216,568.92990,365.75770),5:(305.89166,214.07037,357.57343,329.72990),9:(356.51020,224.99799,406.01047,420.87730),10:(364.68646,225.32680,412.41342,340.55496)}
LAL={1,2,5,10}; DURANT_TRACK=9
H=np.array([[4.763075788199326,6.6499295809312695,-3809.6913744473422],[-2.978826787652546,12.935424948142028,-1282.041548444928],[-0.00044782946726342653,0.0036479999461021674,1.0]],dtype=np.float64)

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b; ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); ua=max(1,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter); return inter/ua

def court(p):
    v=H@np.array([p[0],p[1],1.0]); return v[:2]/v[2]

def anchor_from_pose(kxy,kcf,box):
    good=[]
    for idx in (15,16):
        if idx<len(kxy) and float(kcf[idx])>=.12 and kxy[idx][0]>1 and kxy[idx][1]>1: good.append(kxy[idx])
    if good:
        p=np.mean(np.asarray(good,dtype=float),axis=0); return (float(p[0]),float(p[1])),'ankles',len(good)
    knees=[]
    for idx in (13,14):
        if idx<len(kxy) and float(kcf[idx])>=.12 and kxy[idx][0]>1 and kxy[idx][1]>1: knees.append(kxy[idx])
    if knees:
        p=np.mean(np.asarray(knees,dtype=float),axis=0); # estimate ground from lower-leg vector when ankles are hidden
        h=max(20.0,float(box[3]-box[1])); return (float(p[0]),min(539.0,float(p[1]+0.24*h))),'knees_extrapolated',len(knees)
    return ((float(box[0]+box[2])/2.0,float(box[3])),'box_floor',0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='yolo11x-pose.pt'); ap.add_argument('--out',type=Path,default=Path('artifacts/christmas_primary_defender')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    clip=a.out/'source_native.mp4'; meta=fetch_native_clip('0022500012',94,clip,960)
    cap=cv2.VideoCapture(str(clip)); cap.set(cv2.CAP_PROP_POS_MSEC,10.56*1000); ok,fr=cap.read(); cap.release()
    if not ok: raise RuntimeError('release frame unavailable')
    model=YOLO(a.model); res=model.predict(fr,verbose=False,device='cpu',conf=.15,imgsz=960)[0]; poses=[]
    if res.boxes is not None and res.keypoints is not None:
        boxes=res.boxes.xyxy.cpu().numpy(); kxy=res.keypoints.xy.cpu().numpy(); kcf=res.keypoints.conf.cpu().numpy()
        for i,b in enumerate(boxes):
            bb=tuple(map(float,b)); best_tid=None; best_iou=0.0
            for tid,tb in TRACKS.items():
                s=iou(bb,tb)
                if s>best_iou: best_tid,best_iou=tid,s
            anc,method,n=anchor_from_pose(kxy[i],kcf[i],bb)
            poses.append({'pose_i':i,'box':bb,'track_id':best_tid,'iou':best_iou,'anchor':anc,'anchor_method':method,'support':n})
    bytrack={}
    for p in sorted(poses,key=lambda z:z['iou'],reverse=True):
        tid=p['track_id']
        if tid is not None and p['iou']>=.10 and tid not in bytrack: bytrack[tid]=p
    if DURANT_TRACK not in bytrack: raise RuntimeError('Durant release pose not resolved')
    danc=bytrack[DURANT_TRACK]['anchor']; dc=court(danc); candidates=[]
    for tid in sorted(LAL):
        if tid not in bytrack: continue
        anc=bytrack[tid]['anchor']; cc=court(anc); dist_ft=float(np.linalg.norm(dc-cc))/30.48
        candidates.append({'track_id':tid,'distance_ft':dist_ft,'anchor':anc,'court_cm':cc.tolist(),'pose_i':bytrack[tid]['pose_i'],'iou':bytrack[tid]['iou'],'anchor_method':bytrack[tid]['anchor_method']})
    candidates.sort(key=lambda z:z['distance_ft']); primary=candidates[0] if candidates else None
    out={'release_s':10.56,'source':meta,'durant':{'track_id':9,'anchor':danc,'court_cm':dc.tolist(),'anchor_method':bytrack[9]['anchor_method'],'iou':bytrack[9]['iou']},'lal_candidates':candidates,'primary_defender':primary,'all_pose_matches':poses}
    (a.out/'primary_defender.json').write_text(json.dumps(out,indent=2)); dbg=fr.copy()
    for p in poses:
        x1,y1,x2,y2=map(int,p['box']); tid=p['track_id']; col=(0,255,0) if tid==9 else (255,255,0); cv2.rectangle(dbg,(x1,y1),(x2,y2),col,1); cv2.circle(dbg,tuple(int(round(z)) for z in p['anchor']),5,col,-1,cv2.LINE_AA)
        txt=f"T{tid} {p['anchor_method']} {p['iou']:.2f}"; cv2.putText(dbg,txt,(x1,max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),3,cv2.LINE_AA); cv2.putText(dbg,txt,(x1,max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
    if primary:
        p1=tuple(int(round(z)) for z in danc); p2=tuple(int(round(z)) for z in primary['anchor']); cv2.line(dbg,p1,p2,(0,255,255),2,cv2.LINE_AA); txt=f"{primary['distance_ft']:.1f} FT"; m=((p1[0]+p2[0])//2,(p1[1]+p2[1])//2-8); cv2.putText(dbg,txt,m,cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA); cv2.putText(dbg,txt,m,cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(a.out/'primary_defender_debug.jpg'),dbg,[cv2.IMWRITE_JPEG_QUALITY,97]); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
