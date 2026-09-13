from __future__ import annotations

"""v33h2: calibrated single-view metric basketball reconstruction on v33e states.

Why this exists:
Visual QA proved the ball is directly visible in Right Above Rim (RAR) while the
same physical ball is occluded by the rim/player cluster in LAR and Broadcast.
Forcing a second detector hit created the v33f/v33g false positives.

This solver uses only source-grounded facts:
  1. exact v33e source frame and accepted per-frame RAR camera;
  2. temporally validated exact-frame semantic sports-ball observation;
  3. a directly measured circular ball edge in that exact source image;
  4. the known physical basketball radius (12.0 cm nominal);
  5. locked-camera projection into LAR/Broadcast, where missing visibility must be
     explained by Adams/rim occlusion or a consistent source candidate.

The ball depth is therefore a CALIBRATED SPHERE-SIZE METRIC RECONSTRUCTION. It is
not claimed as multi-view observed triangulation. No player/ball residual refits
any camera. No generated RGB, no temporal frame mixing, no fourth camera, native
960x540 only.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview, RFDETRMedium

from freeze_spin import run_v33e_locked_three_camera_wide_flow_state_search as v33e
from freeze_spin import run_v33d_locked_three_camera_joint_state_search as v33d
from freeze_spin import run_v33b_rar_exact_state_sweep as v33b
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import run_v33f_source_grounded_body_ball_qa as v33f
from freeze_spin import run_v33g_temporal_ball_qa as v33g

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
CAMS = (LAR, RAR, BCAST)
W, H = v32j.W, v32j.H
RIM = np.array([38.1, 0.0, 304.8], float)
BALL_RADIUS_CM = 12.0
MIN_RADIUS_PX = 11.0
MAX_RADIUS_PX = 27.0
MAX_CIRCLE_CENTER_DELTA_PX = 8.0
MIN_DILATED_EDGE_SUPPORT = 0.34
MIN_INTERIOR_ORANGE = 0.42
MAX_RIM_DISTANCE_CM = 185.0
SOURCE_CONFIRM_PX = 30.0
OCCLUSION_RIM_PX = 70.0
OCCLUSION_PLAYER_PAD = 34.0


def camera_for_rel(scene, cam_states, cam, rel):
    s = copy.deepcopy(scene); s['cameras'][cam] = cam_states[cam][rel]
    return v32j.cam(s, cam)


def fit_source_ball_circle(image: np.ndarray, cand: dict):
    cx, cy = map(float, cand['center'])
    b = np.asarray(cand['bbox'], float)
    pad = 10
    x1=max(0,int(math.floor(b[0]))-pad); y1=max(0,int(math.floor(b[1]))-pad)
    x2=min(W,int(math.ceil(b[2]))+pad); y2=min(H,int(math.ceil(b[3]))+pad)
    roi=image[y1:y2,x1:x2]
    if roi.size == 0: return None
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),1.1)
    edges=cv2.Canny(blur,45,120)
    dil=cv2.dilate(edges,np.ones((3,3),np.uint8))
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    orange=(cv2.inRange(hsv,np.array([1,55,35],np.uint8),np.array([34,255,255],np.uint8))>0)
    rows=[]
    seen=set()
    for p2 in (16,15,14,13,12,11,10,9,8):
        circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1,param1=70,param2=p2,
                                 minRadius=int(MIN_RADIUS_PX),maxRadius=int(MAX_RADIUS_PX),minDist=8)
        if circles is None: continue
        for lx,ly,r in circles[0]:
            gcx,gcy=float(x1+lx),float(y1+ly); r=float(r)
            delta=float(math.hypot(gcx-cx,gcy-cy))
            if delta>MAX_CIRCLE_CENTER_DELTA_PX: continue
            key=(round(gcx,1),round(gcy,1),round(r,1))
            if key in seen: continue
            seen.add(key)
            th=np.linspace(0,2*math.pi,240,endpoint=False)
            xs=np.rint(lx+r*np.cos(th)).astype(int); ys=np.rint(ly+r*np.sin(th)).astype(int)
            valid=(xs>=0)&(xs<roi.shape[1])&(ys>=0)&(ys<roi.shape[0])
            if int(valid.sum())<120: continue
            edge=float(np.mean(dil[ys[valid],xs[valid]]>0))
            yy,xx=np.indices(roi.shape[:2]); disk=(xx-lx)**2+(yy-ly)**2 <= (0.88*r)**2
            interior=float(np.mean(orange[disk])) if np.any(disk) else 0.0
            # Do not prefer an oversized circle just because it touches the hand.
            score=2.0*edge+1.25*interior-0.045*delta-0.012*abs(r-18.5)
            rows.append({'center_px':[gcx,gcy],'radius_px':r,'semantic_center_delta_px':delta,
                         'dilated_edge_support':edge,'interior_orange_fraction':interior,
                         'hough_param2':p2,'score':float(score)})
    rows.sort(key=lambda x:-x['score'])
    good=[x for x in rows if x['dilated_edge_support']>=MIN_DILATED_EDGE_SUPPORT and
                              x['interior_orange_fraction']>=MIN_INTERIOR_ORANGE]
    return (good[0] if good else None), rows[:12]


def metric_sphere_center(cam: dict, circle: dict):
    K=np.asarray(cam['K'],float); R=np.asarray(cam['R'],float); C=np.asarray(cam['C'],float)
    uv=np.asarray(circle['center_px'],float); fx,fy=float(K[0,0]),float(K[1,1]); f=math.sqrt(abs(fx*fy))
    r=float(circle['radius_px'])
    zabs=f*BALL_RADIUS_CM/r
    xn=np.linalg.inv(K)@np.array([uv[0],uv[1],1.0],float)
    # Accepted camera conventions can face along +/- camera-z. Determine sign
    # from the known physical rim rather than assuming OpenCV +z.
    rim_z=float((R@(RIM-C))[2]); sign=1.0 if rim_z>=0 else -1.0
    X=C+R.T@(sign*zabs*xn)
    return X, {'effective_focal_px':f,'camera_axial_depth_abs_cm':float(zabs),'camera_forward_sign':sign}


def serialize_candidate(c):
    o={k:v for k,v in c.items() if k!='center'}; o['center']=np.asarray(c['center'],float).tolist(); return o


def sphere_hypotheses(image, cams, rows, candidates):
    hyps=[]
    for cand in candidates[RAR]:
        if cand.get('source')!='semantic' or not cand.get('appearance_gate') or not cand.get('temporal_gate'): continue
        fitted=fit_source_ball_circle(image,cand)
        if not fitted or fitted[0] is None: continue
        circle, alternatives=fitted
        X,metric=metric_sphere_center(cams[RAR],circle)
        if not np.all(np.isfinite(X)) or not (140.0<=float(X[2])<=365.0): continue
        rimdist=float(np.linalg.norm(X-RIM))
        if rimdist>MAX_RIM_DISTANCE_CM: continue
        proj={c:v32j.project(cams[c],X) for c in CAMS}
        if any(not v33f.inside_image(proj[c]) for c in CAMS): continue
        evidence={}; explained=0
        for c in CAMS:
            uv=np.asarray(proj[c],float)
            if c==RAR:
                d=float(np.linalg.norm(uv-np.asarray(circle['center_px'],float)))
                evidence[c]={'status':'OBSERVED_CALIBRATED_SPHERE','source_circle_reprojection_px':d}
                explained += int(d<=1.5)
                continue
            pool=candidates.get(c,[]); nearest=None; nd=999.0
            if pool:
                nearest=min(pool,key=lambda z:float(np.linalg.norm(np.asarray(z['center'],float)-uv)))
                nd=float(np.linalg.norm(np.asarray(nearest['center'],float)-uv))
            if nearest is not None and nd<=SOURCE_CONFIRM_PX:
                evidence[c]={'status':'SOURCE_CANDIDATE_CONSISTENT','distance_px':nd,'candidate_source':nearest.get('source')}
                explained+=1
            else:
                rimuv=v32j.project(cams[c],RIM)
                near_rim=rimuv is not None and float(np.linalg.norm(uv-np.asarray(rimuv,float)))<=OCCLUSION_RIM_PX
                player=v33f.box_contains(rows[c]['box'],uv,OCCLUSION_PLAYER_PAD)
                occ=bool(near_rim or player)
                evidence[c]={'status':'EXPLICIT_SOURCE_OCCLUSION' if occ else 'UNEXPLAINED_MISSING_BALL',
                             'near_rim':bool(near_rim),'inside_adams_box':bool(player),
                             'nearest_candidate_distance_px':None if nearest is None else nd}
                explained+=int(occ)
        gate=bool(explained==3)
        score=(0.05*rimdist-2.0*circle['dilated_edge_support']-1.5*circle['interior_orange_fraction']
               -0.25*float(cand.get('quality',0.0)))
        hyps.append({'gate':gate,'score':float(score),'method':'exact RAR semantic source ball + measured image-circle radius + calibrated known basketball radius metric depth',
                     'world_cm':X.tolist(),'ball_radius_cm_nominal':BALL_RADIUS_CM,'source_candidate':serialize_candidate(cand),
                     'fitted_circle':circle,'circle_alternatives':alternatives,'metric_depth':metric,
                     'distance_to_rim_center_cm':rimdist,'projected_centers':{c:np.asarray(proj[c],float).tolist() for c in CAMS},
                     'view_evidence':evidence})
    hyps.sort(key=lambda x:(not x['gate'],x['score']))
    return (hyps[0] if hyps else None),hyps[:20]


def draw(image,row,joints,cams,label,candidates,best):
    out=image.copy(); valid=np.asarray(row['conf'],float)>=v32v.MIN_CONF
    for a,b in v32j.DRAW:
        if valid[a] and valid[b]: cv2.line(out,tuple(np.rint(row['xy'][a]).astype(int)),tuple(np.rint(row['xy'][b]).astype(int)),(0,255,255),2,cv2.LINE_AA)
    for j in v32v.BODY:
        if valid[j]: cv2.circle(out,tuple(np.rint(row['xy'][j]).astype(int)),3,(0,255,255),-1,cv2.LINE_AA)
    if best:
        uv=np.asarray(best['projected_centers'][label],float); cv2.circle(out,tuple(np.rint(uv).astype(int)),12,(255,255,0),2,cv2.LINE_AA)
        ev=best['view_evidence'][label]['status']; cv2.putText(out,ev,(max(4,int(uv[0])+14),max(48,int(uv[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,0),1,cv2.LINE_AA)
        if label==RAR:
            cc=np.asarray(best['fitted_circle']['center_px'],float); rr=int(round(best['fitted_circle']['radius_px']))
            cv2.circle(out,tuple(np.rint(cc).astype(int)),rr,(0,255,0),2,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,34),(0,0,0),-1)
    cv2.putText(out,f'v33h2 {label} | green=measured source ball circle | cyan=metric projection',(7,22),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clips-dir',type=Path,required=True); ap.add_argument('--b32-root',type=Path,required=True); ap.add_argument('--v73-frame0257',type=Path,required=True); ap.add_argument('--v33e-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    accepted=json.loads(a.v33e_json.read_text())
    if accepted.get('status')!='PASS_V33E_EXACT_THREE_VIEW_STATE' or accepted.get('camera_lock')!=[LAR,RAR,BCAST]: raise RuntimeError('requires accepted v33e lock')
    exact=[x for x in accepted.get('top_triplets',[]) if x.get('exact_state')]
    rels=tuple(range(-20,21)); v33d.RELS=rels
    stage=a.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text()); centers={k:int(v) for k,v in scene['freeze']['chosen_frame_indices'].items()}
    wide=a.out/'wide_stage'; wide.mkdir(parents=True,exist_ok=True); source_audit=v33e.export_wide_burst(a.clips_dir,centers,wide,rels); fs={c:v33d.frames(wide,c) for c in CAMS}
    cert=cv2.imread(str(a.v73_frame0257),0)
    if cert is None: raise RuntimeError('missing v73 RAR cert')
    cam_states={c:{} for c in CAMS}
    for c in (LAR,BCAST):
        for r in rels:
            cc,_=v33d.transfer(fs[c][0][2],fs[c][r][2],scene['cameras'][c]);
            if cc is not None: cam_states[c][r]=cc
    for r in rels:
        cc,_=v33b.transfer_rar_camera(cert,fs[RAR][r][2],scene['cameras'][RAR]);
        if cc is not None: cam_states[RAR][r]=cc
    kp=RFDETRKeypointPreview(); identity={}; obs={}
    for c in CAMS: identity[c],_=v33d.track(kp,c,fs[c])
    for c in CAMS: obs[c],_=v33e.flow_observations(fs[c],identity[c],c,rels)
    needed={c:sorted({int(x['rels'][c]) for x in exact}) for c in CAMS}; temporal={}
    for c in CAMS:
        s=set()
        for r in needed[c]:
            for d in (0,)+v33g.TEMPORAL_OFFSETS:
                rr=r+d
                if rr in fs[c] and rr in cam_states[c]: s.add(rr)
        temporal[c]=sorted(s)
    detector=RFDETRMedium(); cache={c:{} for c in CAMS}
    for c in CAMS:
        for r in temporal[c]:
            cam=camera_for_rel(scene,cam_states,c,r); rimuv=v32j.project(cam,RIM); cache[c][r]=[] if rimuv is None else v33g.build_frame_candidates(detector,fs[c][r][1],rimuv)
    val={c:{} for c in CAMS}
    for c in CAMS:
        for r in needed[c]:
            pool=[v33g.add_temporal_evidence(cache[c],r,q) for q in cache[c].get(r,[])]; pool=[q for q in pool if q.get('temporal_gate')]; pool.sort(key=lambda q:(-q.get('quality',0.0),-q.get('confidence',0.0))); val[c][r]=pool[:12]
    diagnostics=[]; passing=[]
    for rank,x in enumerate(exact,start=1):
        rel={c:int(x['rels'][c]) for c in CAMS}
        if any(rel[c] not in cam_states[c] for c in CAMS): continue
        s,cams=v33f.camera_for_state(scene,cam_states,rel); rows={c:obs[c][rel[c]] for c in CAMS}; body=v33f.body_fit(s,rows); candidates={c:val[c].get(rel[c],[]) for c in CAMS}
        ball,hyps=sphere_hypotheses(fs[RAR][rel[RAR]][1],cams,rows,candidates) if body['gate'] else (None,[]); gate=bool(body['gate'] and ball and ball['gate'])
        diagnostics.append({'rank':rank,'rels':rel,'frames':x['frames'],'v33e_score':float(x['score']),'body_gate':bool(body['gate']),'body_reprojection':body['reprojection'],'candidate_counts':{c:len(candidates[c]) for c in CAMS},'best_ball':ball,'top_ball_hypotheses':hyps[:5],'gate':gate})
        if gate: passing.append((rank,x,body,ball,cams,rows,candidates))
    chosen=min(passing,key=lambda t:(t[0],t[3]['score'])) if passing else None; passed=chosen is not None
    chosen_json=None
    if passed:
        rank,x,body,ball,cams,rows,candidates=chosen; rel={c:int(x['rels'][c]) for c in CAMS}; ovs=[]
        for c in CAMS:
            src=fs[c][rel[c]][1]; cv2.imwrite(str(a.out/f'v33h2_source_{c.replace(" ","_")}.png'),src); ov=draw(src,rows[c],body['_joints'],cams,c,candidates[c],ball); cv2.imwrite(str(a.out/f'v33h2_{c.replace(" ","_")}_ball_qa.png'),ov); ovs.append(ov)
        cv2.imwrite(str(a.out/'v33h2_three_camera_ball_montage.png'),np.hstack(ovs))
        export={'camera_lock':[LAR,RAR,BCAST],'resolution':[W,H],'rels':rel,'frames':x['frames'],'cameras':{c:{'K_px':cams[c]['K'].tolist(),'R_world_to_camera':cams[c]['R'].tolist(),'C_world_cm':cams[c]['C'].tolist()} for c in CAMS},'body_joints_world_cm':{v32j.NAMES[j]:np.asarray(X,float).tolist() for j,X in body['_joints'].items()},'ball':ball}
        (a.out/'v33h2_render_state.json').write_text(json.dumps(export,indent=2)); bj={k:v for k,v in body.items() if not k.startswith('_')}; chosen_json={'rank':rank,'rels':rel,'frames':x['frames'],'v33e_score':float(x['score']),'body':bj,'ball':ball}
    qa={'version':'v33h2_calibrated_sphere_ball_qa','status':'PASS_V33H2_CALIBRATED_SPHERE_AWAITING_VISUAL_QA' if passed else 'FAIL_CLOSED_V33H2_NO_CALIBRATED_SPHERE_STATE','upstream_v33e_status':accepted.get('status'),'camera_lock':[LAR,RAR,BCAST],'camera_count':3,'camera_addition_or_substitution_used':False,'physical_center_refit_from_player_or_ball':False,'exact_state_reopened':False,'native_resolution':[W,H],'generated_rgb':False,'upscaled':False,'novel_view_rendered':False,'ball_method_classification':'single-view observed RAR ball center + measured apparent sphere radius + calibrated camera + known 12cm physical radius; other locked views occlusion-validated','ball_radius_cm_nominal':BALL_RADIUS_CM,'chosen_state':chosen_json,'state_diagnostics':diagnostics,'render_numerically_eligible':bool(passed),'freeview_render_unlocked':False,'unlock_policy':'visual QA of source circle and LAR/Broadcast occlusion projections required before render'}
    (a.out/'v33h2_calibrated_sphere_ball_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps({'status':qa['status'],'chosen_state':qa['chosen_state'],'render_numerically_eligible':qa['render_numerically_eligible']},indent=2),flush=True)
    if not passed: raise SystemExit(8)

if __name__=='__main__': main()
