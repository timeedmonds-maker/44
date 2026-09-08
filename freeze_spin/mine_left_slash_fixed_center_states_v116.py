from __future__ import annotations

"""Discovery-only same-game Left Slash PTZ state miner.

Uses the fixed-optical-centre invariant directly: for pure pan/tilt/zoom about one
camera centre, static scene points at all depths are related by a single image
homography. Dynamic players are treated as RANSAC outliers. This replaces the
v115 camera-specific cross-depth masks, which were invalid for Left Slash.
"""

import argparse, hashlib, json, math, shutil
from pathlib import Path
import cv2
import numpy as np
from freeze_spin.scan_same_game_camera_priors import discover_events, extract_frames, safe_label, w

W,H=960,540

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def stats(v):
    v=np.asarray(v,float)
    if not len(v): return {'n':0,'median_px':None,'p95_px':None}
    return {'n':int(len(v)),'median_px':float(np.median(v)),'p95_px':float(np.percentile(v,95))}

def grid_spread(pts,cols=6,rows=4):
    cells=set()
    for x,y in pts:
        cells.add((min(cols-1,max(0,int(x/W*cols))),min(rows-1,max(0,int(y/H*rows)))))
    return len(cells)

def homography_diversity(Hm):
    corners=np.float32([[[0,0],[W-1,0],[W-1,H-1],[0,H-1]],[[W/2,H/2]]])
    q=cv2.perspectiveTransform(corners,Hm)
    quad=q[0]
    center=q[1][0]
    area=abs(cv2.contourArea(quad))/float((W-1)*(H-1))
    shift=float(np.linalg.norm(center-np.array([W/2,H/2],np.float32)))
    zoom=math.sqrt(max(area,1e-9))
    diversity=abs(math.log(max(zoom,1e-9)))+shift/300.0
    return {'projected_frame_area_ratio':float(area),'approx_linear_zoom_ratio':float(zoom),'projected_center_shift_px':shift,'diversity_index':float(diversity),'projected_corners':quad.tolist()}

def analyze(target_kp,target_desc,p:Path,ratio=.75,ransac=3.0):
    g=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if g is None: return {'status':'bad_image'}
    sift=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015,edgeThreshold=12)
    kp,desc=sift.detectAndCompute(g,None)
    if desc is None: return {'status':'no_features'}
    knn=cv2.BFMatcher(cv2.NORM_L2).knnMatch(target_desc,desc,k=2)
    good=[m for m,n in knn if m.distance<ratio*n.distance]
    if len(good)<18: return {'status':'insufficient_matches','good_matches':len(good)}
    a=np.float32([target_kp[m.queryIdx].pt for m in good])
    b=np.float32([kp[m.trainIdx].pt for m in good])
    Hm,mask=cv2.findHomography(a,b,cv2.RANSAC,ransac,maxIters=20000,confidence=.999)
    if Hm is None or mask is None: return {'status':'homography_failed','good_matches':len(good)}
    keep=mask.ravel().astype(bool)
    pred=cv2.perspectiveTransform(a.reshape(-1,1,2),Hm).reshape(-1,2)
    err=np.linalg.norm(pred-b,axis=1)[keep]
    n=int(keep.sum()); spread=grid_spread(a[keep]); rs=stats(err); ratio_in=n/max(1,len(good))
    div=homography_diversity(Hm)
    p95=rs['p95_px'] if rs['p95_px'] is not None else 99.0
    quality=float(p95+18/math.sqrt(max(1,n))+max(0,7-spread)*1.5+max(0,.30-ratio_in)*10)
    credible=bool(n>=24 and spread>=5 and ratio_in>=.22 and p95<=2.5)
    independent=bool(div['projected_center_shift_px']>=24 or abs(math.log(max(div['approx_linear_zoom_ratio'],1e-9)))>=.08)
    return {'status':'ok','good_matches':len(good),'ransac_inliers':n,'inlier_ratio':float(ratio_in),'grid_cells_6x4':spread,'inlier_residual':rs,'homography':Hm.tolist(),'transform':div,'quality_score_lower_is_better':quality,'credible_fixed_center_state_candidate':credible,'materially_different_optical_state':independent}

def montage(target,top,out):
    ims=[]
    t=cv2.imread(str(target)); cv2.putText(t,'IMMUTABLE LEFT SLASH FRAME C',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA); ims.append(t)
    for r in top:
        im=cv2.imread(r['selected_frame']); a=r['analysis']; d=a.get('transform',{})
        txt=f"e{r['event_probe']} {r['sample_name']} in={a.get('ransac_inliers')} p95={a.get('inlier_residual',{}).get('p95_px')} shift={d.get('projected_center_shift_px',-1):.1f} zoom={d.get('approx_linear_zoom_ratio',-1):.3f} cred={a.get('credible_fixed_center_state_candidate')}"
        cv2.putText(im,txt[:150],(12,32),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),4,cv2.LINE_AA); cv2.putText(im,txt[:150],(12,32),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2,cv2.LINE_AA); ims.append(im)
    cols=3; rows=math.ceil(len(ims)/cols); can=np.full((rows*H,cols*W,3),255,np.uint8)
    for i,im in enumerate(ims):
        y,x=divmod(i,cols); can[y*H:(y+1)*H,x*W:(x+1)*W]=im
    cv2.imwrite(str(out),can)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--game-id',required=True); ap.add_argument('--camera-label',default='Left Slash'); ap.add_argument('--target-frame',type=Path,required=True); ap.add_argument('--target-sha256',required=True); ap.add_argument('--count',type=int,default=40); ap.add_argument('--event-start',type=int,default=5); ap.add_argument('--event-stop',type=int,default=1205); ap.add_argument('--event-step',type=int,default=10); ap.add_argument('--samples-per-clip',type=int,default=15); ap.add_argument('--keep-top',type=int,default=24); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); sel=args.out/'selected_frames'; sel.mkdir(exist_ok=True); work=args.out/'work'; work.mkdir(exist_ok=True)
    if sha256(args.target_frame)!=args.target_sha256: raise SystemExit('target SHA mismatch')
    tg=cv2.imread(str(args.target_frame),cv2.IMREAD_GRAYSCALE)
    sift=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015,edgeThreshold=12); tk,td=sift.detectAndCompute(tg,None)
    events=discover_events(args.game_id,args.camera_label,args.count,args.event_start,args.event_stop,args.event_step)
    slug=safe_label(args.camera_label); results=[]; candidates=[]
    for rank,d in enumerate(events,1):
        eid=int(d['event_id']); clip=work/f'e{eid}_{slug}.mp4'; fd=work/f'e{eid}_frames'; rec={'rank':rank,'event_probe':eid,'title':d['title']}
        try:
            w.download_hls_source(d['url'],clip); frames=extract_frames(clip,fd,n=args.samples_per_clip); samples=[]
            for p in frames:
                a=analyze(tk,td,p); samples.append({'sample_name':p.name,'analysis':a,'path':str(p)})
                if a.get('status')=='ok': candidates.append((eid,p,a,d['title']))
            rec.update(status='ok',samples=[{'sample_name':s['sample_name'],'analysis':s['analysis']} for s in samples])
        except Exception as e: rec.update(status='failed',error=repr(e))
        finally: clip.unlink(missing_ok=True)
        results.append(rec); print(f'[{rank}/{len(events)}] event {eid}: {rec["status"]}',flush=True)
    # Evidence quality first; among credible states reward PTZ diversity.
    def key(x):
        a=x[2]; cred=0 if a.get('credible_fixed_center_state_candidate') else 1; indep=0 if a.get('materially_different_optical_state') else 1
        return (cred,indep,a.get('quality_score_lower_is_better',1e9),-a.get('transform',{}).get('diversity_index',0))
    candidates.sort(key=key); top=[]
    for eid,p,a,title in candidates[:args.keep_top]:
        dst=sel/f'event_{eid:04d}_{slug}_{p.name}'; shutil.copy2(p,dst); top.append({'event_probe':eid,'title':title,'sample_name':p.name,'selected_frame':str(dst),'selected_frame_sha256':sha256(dst),'analysis':a})
    montage(args.target_frame,top,args.out/'top_fixed_center_state_candidates_v116.png')
    for r in top: r['selected_frame']=str(Path('selected_frames')/Path(r['selected_frame']).name)
    credible=[r for r in top if r['analysis'].get('credible_fixed_center_state_candidate')]
    independent=[r for r in credible if r['analysis'].get('materially_different_optical_state')]
    payload={'game_id':args.game_id,'camera_label':args.camera_label,'immutable_target':{'file':args.target_frame.name,'sha256':args.target_sha256},'method':'single full-scene RANSAC homography: fixed-centre PTZ invariant; dynamic players are outliers','events':results,'top_candidates':top,'credible_candidate_count':len(credible),'credible_materially_different_count':len(independent),'permissions':{'metric_camera_promotion_allowed':False,'replay_render_allowed':False},'status':'DISCOVERY_ONLY_NO_PROMOTION'}
    (args.out/'left_slash_fixed_center_state_mining_v116.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({'credible':len(credible),'credible_materially_different':len(independent),'top':[(r['event_probe'],r['sample_name'],r['analysis']['ransac_inliers'],r['analysis']['transform']['projected_center_shift_px'],r['analysis']['transform']['approx_linear_zoom_ratio'],r['analysis']['credible_fixed_center_state_candidate']) for r in top[:12]]},indent=2))
    shutil.rmtree(work,ignore_errors=True)

if __name__=='__main__': main()
