from __future__ import annotations

"""v43: rank Broadcast / Mobile Broadcast as second distinct metric camera candidates.

This stage does NOT promote a camera. It derives each target candidate's metric
floor homography by source-pixel floor transfer to the accepted Left Above Rim
v35 floor, selects same-physical-camera samples from four independent same-game
events using full-scene static homography evidence, and proposes local orange-rim
components near the metric basket location for visual QA.
"""

import argparse,json,re
from pathlib import Path
import cv2
import numpy as np

from freeze_spin.audit_game_camera_registry_preflight_v1 import audit_pair

W,H=960,540
RIM_FLOOR=np.array([[38.1,0.]],np.float32)


def safe(s:str)->str:return re.sub(r'[^A-Za-z0-9]+','_',s).strip('_')

def apply_h(Hm,p):return cv2.perspectiveTransform(np.asarray(p,np.float32)[:,None,:],np.asarray(Hm,float))[:,0]

def sift(a,b):
    s=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015)
    ka,da=s.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);kb,db=s.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
    raw=cv2.BFMatcher().knnMatch(da,db,k=2);used={}
    for m,n in raw:
        if m.distance<.72*n.distance and (m.trainIdx not in used or m.distance<used[m.trainIdx].distance):used[m.trainIdx]=m
    good=list(used.values())
    return np.float32([ka[m.queryIdx].pt for m in good]),np.float32([kb[m.trainIdx].pt for m in good])

def floor_transfer(src,target):
    p,q=sift(src,target)
    m=(p[:,1]>250)&(q[:,1]>250)&(p[:,0]>35)&(p[:,0]<925)&(q[:,0]>35)&(q[:,0]<925)
    if int(m.sum())<30:return None
    Hm,mask=cv2.findHomography(p[m],q[m],cv2.RANSAC,1.75,maxIters=40000,confidence=.999)
    if Hm is None or mask is None:return None
    ii=mask.ravel().astype(bool);pp,qq=p[m][ii],q[m][ii];e=np.linalg.norm(apply_h(Hm,pp)-qq,axis=1)
    if len(e)<24:return None
    return {'H':Hm,'inliers':int(len(e)),'p95_px':float(np.percentile(e,95)),'median_px':float(np.median(e))}

def target_for_label(root,label):
    rows=[]
    for p in sorted(root.glob('*.png')):
        m=re.match(r'^[A-L]_(.+?)_\d+\.\d+s_frame\d+$',p.stem)
        if not m:continue
        parsed=m.group(1).replace('_',' ')
        if parsed==label:rows.append(p)
    if len(rows)!=1:raise RuntimeError(f'Expected one exact target for {label}, got {len(rows)}: {[p.name for p in rows]}')
    return rows[0]
def selected_event_samples(root,label,target):
    token=safe(label);out={}
    for e in (40,220,440,620):
        rows=sorted(root.glob(f'{token}__event{e:04d}__s*.png'))
        aud=[]
        for p in rows:
            r=audit_pair(p,target);r['file']=p.name;aud.append(r)
        aud.sort(key=lambda r:(1 if r.get('pass') else 0,int(r.get('training_inliers',0)),-float((r.get('withheld_error') or {}).get('median_px') or 1e9)),reverse=True)
        out[e]=aud[0] if aud else {'pass':False,'status':'no_samples'}
    return out

def rim_components(im,Hworld):
    foot=apply_h(Hworld,RIM_FLOOR)[0];fx,fy=float(foot[0]),float(foot[1])
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([2,90,70],np.uint8),np.array([28,255,255],np.uint8))
    roi=np.zeros_like(mask);x0=max(0,int(fx-105));x1=min(W,int(fx+105));y0=max(0,int(fy-225));y1=min(H,int(fy-25));roi[y0:y1,x0:x1]=255;mask=cv2.bitwise_and(mask,roi)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(mask,8)
    rows=[]
    for i in range(1,n):
        x,y,w,h,area=[int(v) for v in st[i]];cx,cy=[float(v) for v in cen[i]]
        if area<5 or w<4 or w>90 or h>35:continue
        ratio=w/max(h,1);dy=fy-cy
        if dy<25 or dy>225:continue
        score=abs(cx-fx)*1.5+abs(dy-130.0)*0.35-max(ratio,1.0)*4.0-area*0.08
        rows.append({'component':i,'bbox':[x,y,w,h],'area':area,'centroid_px':[cx,cy],'width_height_ratio':ratio,'vertical_above_floor_px':dy,'score':float(score)})
    rows.sort(key=lambda r:r['score'])
    return foot,rows[:8],mask

def overlay(im,foot,cands,path,title):
    o=im.copy();cv2.circle(o,tuple(np.round(foot).astype(int)),7,(255,255,0),2)
    for j,r in enumerate(cands):
        x,y,w,h=r['bbox'];col=(0,255,0) if j==0 else (255,0,255);cv2.rectangle(o,(x,y),(x+w,y+h),col,2);cv2.putText(o,str(j+1),(x,max(15,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.55,col,2,cv2.LINE_AA)
    cv2.putText(o,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA);cv2.imwrite(str(path),o)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target-frames',type=Path,required=True);ap.add_argument('--lar-floor-proof',type=Path,required=True);ap.add_argument('--broadcast-samples',type=Path,required=True);ap.add_argument('--mobile-samples',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    floor=json.loads(a.lar_floor_proof.read_text());Hlar=np.asarray(floor['floor_homography_world_to_image'],float);lar=target_for_label(a.target_frames,'Left Above Rim');larim=cv2.imread(str(lar));rep={'schema_version':1,'status':'PREFLIGHT_ONLY_NO_CAMERA_PERMISSION','candidates':{}}
    for label,sroot in [('Broadcast',a.broadcast_samples),('Mobile Broadcast',a.mobile_samples)]:
        tgt=target_for_label(a.target_frames,label);tim=cv2.imread(str(tgt));tr=floor_transfer(tim,larim)
        if tr is None:
            rep['candidates'][label]={'target_floor_transfer_pass':False};continue
        Hcand=np.linalg.inv(tr['H'])@Hlar;Hcand/=Hcand[2,2]
        sel=selected_event_samples(sroot,label,tgt);entry={'target_floor_transfer_pass':tr['p95_px']<=2.0,'target_floor_transfer':{k:v for k,v in tr.items() if k!='H'},'target_floor_homography_world_to_image':Hcand.tolist(),'same_physical_event_selection':{},'rim_proposals':{}}
        foot,cands,_=rim_components(tim,Hcand);entry['rim_proposals']['target_event_489']={'floor_basket_px':foot.tolist(),'components':cands};overlay(tim,foot,cands,a.out/f'{safe(label)}_target_rim_candidates.png',f'{label} target event 489')
        passing=0
        for e,r in sel.items():
            entry['same_physical_event_selection'][str(e)]={k:v for k,v in r.items() if k!='H_source_to_target'}
            if not r.get('pass'):continue
            passing+=1;sp=sroot/r['file'];sim=cv2.imread(str(sp));fr=floor_transfer(sim,tim)
            if fr is None:continue
            Hs=np.linalg.inv(fr['H'])@Hcand;Hs/=Hs[2,2];ft,cc,_=rim_components(sim,Hs);entry['rim_proposals'][f'event_{e}']={'selected_file':r['file'],'source_floor_transfer':{k:v for k,v in fr.items() if k!='H'},'floor_homography_world_to_image':Hs.tolist(),'floor_basket_px':ft.tolist(),'components':cc};overlay(sim,ft,cc,a.out/f'{safe(label)}_event{e}_rim_candidates.png',f'{label} event {e} {r["file"]}')
        entry['same_physical_pass_count']=passing;entry['candidate_for_metric_followup']=bool(entry['target_floor_transfer_pass'] and passing>=3 and len(entry['rim_proposals'].get('target_event_489',{}).get('components',[]))>0)
        rep['candidates'][label]=entry
    (a.out/'second_metric_camera_preflight_v43.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
