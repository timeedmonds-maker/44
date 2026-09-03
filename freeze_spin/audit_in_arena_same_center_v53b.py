from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import cv2, numpy as np

W,H=960,540

def sha256(p:Path):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def sift_matches(a,b,ratio=.70):
    s=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.012)
    ka,da=s.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None)
    kb,db=s.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
    good=[]
    for m,n in cv2.BFMatcher().knnMatch(da,db,k=2):
        if m.distance < ratio*n.distance: good.append(m)
    return np.float32([ka[m.queryIdx].pt for m in good]),np.float32([kb[m.trainIdx].pt for m in good])

def homerr(H,p,q):
    z=cv2.perspectiveTransform(p[:,None,:].astype(np.float32),H.astype(float))[:,0]
    return np.linalg.norm(z-q,axis=1)

def jacobian_summary(H):
    p=np.array([[480,270],[481,270],[480,271]],np.float32)
    q=cv2.perspectiveTransform(p[:,None,:],H.astype(float))[:,0]
    J=np.column_stack([q[1]-q[0],q[2]-q[0]])
    scale=float(np.sqrt(abs(np.linalg.det(J))))
    angle=float(math.degrees(math.atan2(J[1,0],J[0,0])))
    return scale,angle,q[0].tolist()

def audit(target,cand):
    p,q=sift_matches(target,cand)
    out={'good_matches':int(len(p))}
    if len(p)<40:return out
    Hall,ma=cv2.findHomography(p,q,cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
    if Hall is None:return out
    ia=ma.ravel().astype(bool); ea=homerr(Hall,p,q)[ia]; pi=p[ia]
    if len(pi):
        mn=pi.min(0);mx=pi.max(0);bbox=float((mx[0]-mn[0])*(mx[1]-mn[1])/(W*H))
        quadrants=len(set((int(x>=W/2),int(y>=H/2)) for x,y in pi))
        cells=len(set((min(3,int(x/(W/4))),min(2,int(y/(H/3)))) for x,y in pi))
    else:bbox=0.;quadrants=0;cells=0
    bg=(p[:,1]<250)&(q[:,1]<250)
    if int(bg.sum())<30:return out
    Hbg,mb=cv2.findHomography(p[bg],q[bg],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
    if Hbg is None:return out
    ib=mb.ravel().astype(bool); eb=homerr(Hbg,p[bg],q[bg])[ib]
    test=(p[:,1]>285)&(p[:,1]<500)&(q[:,1]>250)&(q[:,1]<520)
    et=homerr(Hbg,p[test],q[test]) if int(test.sum()) else np.asarray([],float)
    scale,angle,center=jacobian_summary(Hbg)
    out.update({
      'whole_scene_inliers':int(ia.sum()),'whole_scene_p95_px':float(np.percentile(ea,95)) if len(ea) else 999.,
      'whole_scene_bbox_fraction':bbox,'whole_scene_quadrants':quadrants,'whole_scene_grid_cells':cells,
      'background_matches':int(bg.sum()),'background_inliers':int(ib.sum()),'background_p95_px':float(np.percentile(eb,95)) if len(eb) else 999.,
      'heldout_lower_matches':int(test.sum()),'heldout_lower_median_px':float(np.median(et)) if len(et) else 999.,
      'heldout_lower_fraction_lt2px':float(np.mean(et<2)) if len(et) else 0.,
      'heldout_lower_fraction_lt4px':float(np.mean(et<4)) if len(et) else 0.,
      'relative_scale_at_center':scale,'relative_rotation_deg_at_center':angle,'target_center_maps_to_px':center,
      'H_target_to_state':(Hbg/Hbg[2,2]).tolist(),
    })
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',type=Path,required=True);ap.add_argument('--states',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    target=cv2.imread(str(a.target));
    if target is None or target.shape[:2]!=(H,W):raise RuntimeError('target must be native 960x540')
    rows=[]
    for pth in sorted(a.states.glob('event_*_frames/f*.png')):
        im=cv2.imread(str(pth));
        if im is None or im.shape[:2]!=(H,W):continue
        m=audit(target,im);m['file']=str(pth.relative_to(a.states))
        import re
        z=re.search(r'event_(\d+)_frames/(f\d+)\.png$',str(pth).replace('\\','/'))
        m['event_probe']=int(z.group(1));m['frame']=z.group(2);rows.append(m)
    def passes(r):
        return (
          r.get('whole_scene_inliers',0)>=100 and r.get('whole_scene_p95_px',999)<=1.5 and
          r.get('whole_scene_bbox_fraction',0)>=.30 and r.get('whole_scene_grid_cells',0)>=5 and
          r.get('background_inliers',0)>=60 and r.get('background_p95_px',999)<=1.5 and
          r.get('heldout_lower_matches',0)>=40 and r.get('heldout_lower_fraction_lt2px',0)>=.50
        )
    for r in rows:r['candidate_gate']=bool(passes(r))
    by={}
    for r in rows:
        if not r['candidate_gate']:continue
        score=(r['heldout_lower_fraction_lt2px'],r['whole_scene_inliers'],-r['whole_scene_p95_px'])
        if r['event_probe'] not in by or score>by[r['event_probe']][0]:by[r['event_probe']]=(score,r)
    selected=[v[1] for _,v in sorted(by.items())]
    scales=[r['relative_scale_at_center'] for r in selected]
    gates={
      'at_least_four_distinct_events':len(selected)>=4,
      'useful_zoom_span':(max(scales)/min(scales)>=1.25) if len(scales)>=2 else False,
      'all_candidates_broad_static_scene':all(r['whole_scene_grid_cells']>=5 and r['whole_scene_bbox_fraction']>=.30 for r in selected),
      'all_candidates_background_to_floor_holdout':all(r['heldout_lower_fraction_lt2px']>=.50 for r in selected),
    }
    payload={
      'status':'PASS_IN_ARENA_STATIC_SCENE_FAMILY_V53B' if all(gates.values()) else 'FAIL_IN_ARENA_STATIC_SCENE_FAMILY_V53B',
      'version':'v53b','game_id':'0022500301','target_event_id':489,
      'target_camera_source_label':'In Arena','target_sha256_png':sha256(a.target),
      'method':'SIFT static-scene projective audit. Fit homography only to upper/background matches; require held-out lower-court matches to transfer without refitting. Deduplicate by event.',
      'guardrail':'Passing identifies frames compatible with one fixed optical center under pan/tilt/zoom. It does NOT prove a metric camera center and does NOT trust the In Arena label as a physical-camera identity.',
      'thresholds':{'whole_scene_inliers_min':100,'whole_scene_p95_px_max':1.5,'whole_scene_bbox_fraction_min':.30,'whole_scene_grid_cells_min':5,'background_inliers_min':60,'background_p95_px_max':1.5,'heldout_lower_matches_min':40,'heldout_lower_fraction_lt2px_min':.50},
      'selected_candidates':selected,'all_states':rows,'gates':gates,
      'static_scene_family_allowed':bool(all(gates.values())),
      'metric_event_camera_allowed':False,'replay_render_allowed':False,
      'next_gate':'Use selected H_target_to_state transforms as supporting constraints in a joint physical-camera solve anchored by sealed v52 floor geometry and independent non-coplanar rim/backboard observations. Allow state-specific focal length/rotation and small crop/principal-point variation; require multistart/holdout/perturbation stability.'
    }
    (a.out/'in_arena_static_scene_family_v53b.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'status':payload['status'],'selected':[(r['event_probe'],r['frame'],r['whole_scene_inliers'],r['heldout_lower_fraction_lt2px'],r['relative_scale_at_center']) for r in selected],'gates':gates},indent=2))
    if not all(gates.values()):raise SystemExit(2)
if __name__=='__main__':main()
