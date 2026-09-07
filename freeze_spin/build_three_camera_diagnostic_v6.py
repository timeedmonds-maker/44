from __future__ import annotations

"""v6 visual diagnostic.

Keep v5's exact plane-locked court/backboard reconstruction, but tighten dynamic
ownership for this particular 0->25 degree orbit:
- retain only on-court instances in the event's near-basket play region;
- Left Above Rim owns all player/ball pixels because every virtual viewpoint on
  this arc is much closer to that real camera than to Broadcast or Right Above
  Rim;
- Broadcast and Right Above Rim remain active for exact plane texture recovery,
  but are not allowed to paste competing subject depth clouds into this arc.

This is still source-only and introduces no generated fill or synthetic camera.
"""

import cv2
import numpy as np

from freeze_spin import build_three_camera_diagnostic_v5 as v5
from freeze_spin import build_three_camera_diagnostic_v4 as v4

_orig_detect=v5.detect_oncourt


def detect_near_play(model,image,K,R,C):
    _dyn,instances,balls=_orig_detect(model,image,K,R,C)
    keep=[]
    for inst in instances:
        x,y,_=inst['foot_world_cm']
        if -250.0<=float(x)<=1150.0 and abs(float(y))<=520.0:
            keep.append(inst)
    dyn=np.zeros((_dyn.shape[0],_dyn.shape[1]),bool)
    for inst in keep:
        dyn|=inst['mask']
    dyn=cv2.dilate(dyn.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)>0
    for b in balls[:3]:
        x1,y1,x2,y2=[int(round(v)) for v in b['box']]
        x1=max(0,x1-5);y1=max(0,y1-5);x2=min(dyn.shape[1]-1,x2+5);y2=min(dyn.shape[0]-1,y2+5)
        dyn[y1:y2+1,x1:x2+1]=True
    return dyn,keep,balls


def no_secondary_subject_fill(base,bmask,cand,cmask):
    return 0


def annotate_v6(img,angle,cov,dyn_cov):
    out=img.copy();cv2.rectangle(out,(0,0),(570,58),(0,0,0),-1)
    cv2.putText(out,'3-CAMERA DIAGNOSTIC v6 | PRIMARY SUBJECT OWNERSHIP',(12,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f'orbit {angle:04.1f} deg  grounded {cov*100:05.1f}%  subject {dyn_cov*100:04.1f}%',(12,44),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA)
    return out


v5.detect_oncourt=detect_near_play
v4.hard_fill=no_secondary_subject_fill
v4.annotate=annotate_v6

if __name__=='__main__':
    v5.main()
