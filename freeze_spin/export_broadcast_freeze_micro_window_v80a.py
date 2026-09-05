from __future__ import annotations

"""v80a discovery-only export around immutable Broadcast Frame C.

Purpose: find an adjacent native frame from the exact official event clip where the
rim/backboard are less occluded while proving that the optical state remains a
near-homographic match to the immutable Frame C target.  This does not replace
Frame C and authorizes no metric camera or replay.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

W,H=960,540


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def feature_stats(target: np.ndarray, frame: np.ndarray) -> dict:
    sift=cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.018, edgeThreshold=12)
    g0=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY); g1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    k0,d0=sift.detectAndCompute(g0,None); k1,d1=sift.detectAndCompute(g1,None)
    if d0 is None or d1 is None: return {'status':'insufficient_features'}
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0,d1,k=2)
    good=[m for m,n in pairs if m.distance < .72*n.distance]
    if len(good)<12: return {'status':'insufficient_matches','ratio_matches':len(good)}
    p=np.float32([k0[m.queryIdx].pt for m in good]); q=np.float32([k1[m.trainIdx].pt for m in good])
    Hm,mask=cv2.findHomography(p,q,cv2.RANSAC,2.0,maxIters=8000,confidence=.999)
    if Hm is None or mask is None: return {'status':'homography_failed','ratio_matches':len(good)}
    keep=mask.ravel().astype(bool); n=int(keep.sum())
    pred=cv2.perspectiveTransform(p.reshape(-1,1,2),Hm).reshape(-1,2)
    err=np.linalg.norm(pred-q,axis=1)[keep]
    # Spatial support: 6x4 target-image grid.
    cells=set()
    for x,y in p[keep]: cells.add((min(5,int(x/(W/6))),min(3,int(y/(H/4)))))
    return {
        'status':'ok','ratio_matches':len(good),'homography_inliers':n,
        'homography_inlier_ratio':float(n/max(len(good),1)),
        'homography_median_error_px':float(np.median(err)) if len(err) else 999.,
        'homography_p95_error_px':float(np.percentile(err,95)) if len(err) else 999.,
        'spatial_grid_cells':len(cells),'H_target_to_frame':Hm.tolist(),
    }


def montage(paths: list[Path], labels: list[str], out: Path) -> None:
    tiles=[]
    for p,l in zip(paths,labels):
        im=cv2.imread(str(p)); tile=cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile,(0,0),(480,32),(0,0,0),-1)
        cv2.putText(tile,l,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
    cols=4; rows=(len(tiles)+cols-1)//cols
    blank=np.zeros_like(tiles[0])
    while len(tiles)<rows*cols: tiles.append(blank.copy())
    canvas=np.vstack([np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(rows)])
    cv2.imwrite(str(out),canvas)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clip',type=Path,required=True); ap.add_argument('--target-frame',type=Path,required=True); ap.add_argument('--target-sha256',required=True); ap.add_argument('--freeze-time',type=float,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--radius-seconds',type=float,default=.65); ap.add_argument('--step-frames',type=int,default=2)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); frames=a.out/'frames'; frames.mkdir(exist_ok=True)
    target=cv2.imread(str(a.target_frame));
    if target is None or target.shape[:2]!=(H,W): raise RuntimeError('immutable target must be native 960x540')
    if sha256(a.target_frame)!=a.target_sha256: raise RuntimeError('immutable target SHA mismatch')
    cap=cv2.VideoCapture(str(a.clip));
    if not cap.isOpened(): raise RuntimeError('cannot open Broadcast clip')
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if (w,h)!=(W,H) or fps<=0 or count<=0: raise RuntimeError(f'unexpected clip metadata {(w,h,fps,count)}')
    centre=int(round(a.freeze_time*fps)); rad=int(round(a.radius_seconds*fps)); idxs=list(range(max(0,centre-rad),min(count,centre+rad+1),max(1,a.step_frames)))
    rows=[]; selected=[]; labels=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
        if not ok or im is None: continue
        t=i/fps; rel=t-a.freeze_time; name=f'Broadcast_micro__f{i:04d}__rel{rel:+.4f}s.png'; p=frames/name; cv2.imwrite(str(p),im)
        stat=feature_stats(target,im)
        # Sharpness localized around basket/backboard region visible in this feed.
        roi=cv2.cvtColor(im[45:220,430:720],cv2.COLOR_BGR2GRAY); sharp=float(cv2.Laplacian(roi,cv2.CV_64F).var())
        row={'frame_index':i,'time_seconds':t,'relative_to_freeze_seconds':rel,'file':name,'basket_roi_laplacian_variance':sharp,'feature_geometry':stat}; rows.append(row)
    cap.release()
    credible=[r for r in rows if r['feature_geometry'].get('status')=='ok' and r['feature_geometry'].get('homography_inliers',0)>=80 and r['feature_geometry'].get('spatial_grid_cells',0)>=10 and r['feature_geometry'].get('homography_p95_error_px',99)<=1.5]
    credible.sort(key=lambda r:(abs(r['relative_to_freeze_seconds']),-r['basket_roi_laplacian_variance']))
    # Keep broad temporal support, not just the closest frame.
    if len(credible)>16:
        pick=np.linspace(0,len(credible)-1,16).round().astype(int); credible=[credible[int(i)] for i in pick]
    for r in credible:
        selected.append(frames/r['file']); g=r['feature_geometry']; labels.append(f"{r['relative_to_freeze_seconds']:+.3f}s | H {g['homography_inliers']} p95 {g['homography_p95_error_px']:.2f}")
    if selected: montage(selected,labels,a.out/'credible_micro_window_montage.png')
    report={'status':'DISCOVERY_ONLY_NO_PROMOTION','camera':'Broadcast','game_id':'0022500301','event_id':489,'immutable_target':{'file':a.target_frame.name,'sha256':a.target_sha256,'freeze_time_seconds':a.freeze_time},'source_clip':str(a.clip),'fps':fps,'frame_count':count,'radius_seconds':a.radius_seconds,'step_frames':a.step_frames,'frames':rows,'credible_frames':credible,'guardrail':'Adjacent frames may support visibility/optical-state QA only. Immutable Frame C remains the event camera target. No transported/adjacent observation may be represented as a direct Frame C observation.','permissions':{'physical_camera_center_allowed':False,'metric_event_camera_allowed':False,'replay_render_allowed':False}}
    (a.out/'broadcast_freeze_micro_window_v80a.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'decoded':len(rows),'credible':len(credible),'best':[(r['relative_to_freeze_seconds'],r['feature_geometry']['homography_inliers'],r['feature_geometry']['homography_p95_error_px'],r['basket_roi_laplacian_variance']) for r in credible[:8]]},indent=2))

if __name__=='__main__': main()
