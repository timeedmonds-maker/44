from __future__ import annotations

"""v33h2 fallback: one-view basketball + body/rim/occlusion constrained 3-D.

Use only if the full accepted-state v33h search proves that the basketball is not
simultaneously visible in two locked cameras. This does not weaken camera or body
geometry. It models the actual source condition: a dunk ball can be cleanly visible
in RAR while geometrically occluded by the rim/player cluster in LAR and Broadcast.

The ball centre is never guessed freely in 3-D. A real exact-frame RAR ball blob must
be temporally moving, ball-sized/circular, and its fixed-camera ray must pass near an
accepted triangulated Adams wrist. The 3-D point is the closest point on that source
ray to the accepted wrist; it must then be near rim height/rim centre and its projections
into both other locked cameras must be explicitly occluded by the rim or a real person
box. Numerical pass still requires external visual QA before render.
"""

import argparse, copy, json, math
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

LAR,BCAST,RAR=v32v.LAR,v32v.BCAST,v32v.RAR
CAMS=(LAR,RAR,BCAST); W,H=v32j.W,v32j.H; RELS=tuple(range(-20,21))
RIM=np.array([38.1,0.0,304.8],float)
BALL_DIAM_CM=24.2


def ray_from_pixel(cam,uv):
    C,R,K=cam; u=np.r_[np.asarray(uv,float),1.0]
    dcam=np.linalg.inv(K)@u
    s=float(v32j.forward_sign(R,C)) if hasattr(v32j,'forward_sign') else (1.0 if (R@(RIM-C))[2]>0 else -1.0)
    d=R.T@(s*dcam); d/=max(np.linalg.norm(d),1e-12)
    return np.asarray(C,float),d


def person_boxes(model,image):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); d=model.predict(rgb,threshold=0.12)
    boxes=np.asarray(getattr(d,'xyxy',np.empty((0,4))),float); cls=np.asarray(getattr(d,'class_id',np.empty((0,))),int)
    out=[]
    for b,cid in zip(boxes,cls):
        if v33f.coco_name(int(cid))!='person' or not np.all(np.isfinite(b)): continue
        x1,y1,x2,y2=map(float,b)
        if (x2-x1)*(y2-y1)>=300: out.append([x1,y1,x2,y2])
    return out


def contains(box,uv,pad=12):
    x1,y1,x2,y2=box; return x1-pad<=uv[0]<=x2+pad and y1-pad<=uv[1]<=y2+pad


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clips-dir',type=Path,required=True); ap.add_argument('--b32-root',type=Path,required=True)
    ap.add_argument('--v73-frame0257',type=Path,required=True); ap.add_argument('--v33e-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    accepted=json.loads(args.v33e_json.read_text())
    if accepted.get('status')!='PASS_V33E_EXACT_THREE_VIEW_STATE' or accepted.get('camera_lock')!=[LAR,RAR,BCAST]: raise RuntimeError('requires accepted v33e')
    top=accepted['top_triplets'][0]
    if not top.get('exact_state'): raise RuntimeError('v33e winner is not exact')
    relmap={c:int(top['rels'][c]) for c in CAMS}
    if relmap!={LAR:-1,RAR:-6,BCAST:18}: raise RuntimeError(f'unexpected accepted winner {relmap}')

    stage=args.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text()); centers={k:int(v) for k,v in scene['freeze']['chosen_frame_indices'].items()}
    v33d.RELS=RELS; wide=args.out/'wide_stage'; wide.mkdir(exist_ok=True)
    v33e.export_wide_burst(args.clips_dir,centers,wide,RELS); fs={c:v33d.frames(wide,c) for c in CAMS}
    cert=cv2.imread(str(args.v73_frame0257),cv2.IMREAD_GRAYSCALE); cam_states={c:{} for c in CAMS}
    for c in (LAR,BCAST):
        for r in RELS:
            cc,_=v33d.transfer(fs[c][0][2],fs[c][r][2],scene['cameras'][c]);
            if cc is not None: cam_states[c][r]=cc
    for r in RELS:
        cc,_=v33b.transfer_rar_camera(cert,fs[RAR][r][2],scene['cameras'][RAR]);
        if cc is not None: cam_states[RAR][r]=cc
    s,cams=v33f.camera_for_state(scene,cam_states,relmap)

    kp=RFDETRKeypointPreview(); identity={}; obs={}
    for c in CAMS: identity[c],_=v33d.track(kp,c,fs[c])
    for c in CAMS: obs[c],_=v33e.flow_observations(fs[c],identity[c],c,RELS)
    rows={c:obs[c][relmap[c]] for c in CAMS}; body=v33f.body_fit(s,rows)
    if not body['gate']: raise RuntimeError('accepted v33e winner unexpectedly fails body gate')
    wrists=[body['_joints'][j] for j in (9,10) if j in body['_joints']]
    if not wrists: raise RuntimeError('no accepted wrist joints')

    det=RFDETRMedium()
    # Build temporal RAR candidates exactly as v33g, but select by physical ball size + source ray/body/rim geometry.
    cache={}
    for rr in range(relmap[RAR]-3,relmap[RAR]+4):
        if rr not in fs[RAR] or rr not in cam_states[RAR]: continue
        cam=v33g._camera_for_rel(scene,cam_states,RAR,rr); rimuv=v32j.project(cam,RIM)
        cache[rr]=[] if rimuv is None else v33g.build_frame_candidates(det,fs[RAR][rr][1],rimuv)
    raw=[]
    for q in cache.get(relmap[RAR],[]):
        q=v33g.add_temporal_evidence(cache,relmap[RAR],q)
        if not q.get('temporal_gate') or not q.get('appearance_gate'): continue
        x1,y1,x2,y2=map(float,q['bbox']); w=x2-x1; h=y2-y1; dia=math.sqrt(max(w*h,1.0))
        if not (20.0<=dia<=58.0): continue
        if q['source']=='orange_track':
            sh=q.get('shape',{})
            if float(sh.get('largest_orange_circularity',0))<0.42 or float(sh.get('largest_orange_fill',0))<0.52: continue
        C,d=ray_from_pixel(cams[RAR],q['center'])
        best=None
        for wi,W0 in enumerate(wrists):
            t=float(np.dot(W0-C,d)); X=C+t*d; perp=float(np.linalg.norm(X-W0))
            if best is None or perp<best[0]: best=(perp,wi,t,X)
        perp,wi,t,X=best; rimdist=float(np.linalg.norm(X-RIM)); z=float(X[2])
        # apparent size consistency under the solved camera
        Xc=cams[RAR][1]@(X-cams[RAR][0]); depth=abs(float(Xc[2])); f=float(np.mean([cams[RAR][2][0,0],cams[RAR][2][1,1]])); expected=f*BALL_DIAM_CM/max(depth,1.0)
        size_ratio=dia/max(expected,1e-6)
        gate=bool(t>0 and perp<=68.0 and 235.0<=z<=355.0 and rimdist<=125.0 and 0.48<=size_ratio<=1.75)
        raw.append({'gate':gate,'candidate':q,'world_cm':X,'wrist_index':wi,'wrist_ray_perp_cm':perp,'rim_distance_cm':rimdist,'z_cm':z,
                    'observed_equiv_diameter_px':dia,'expected_ball_diameter_px':expected,'size_ratio':size_ratio})
    good=[x for x in raw if x['gate']]
    if not good:
        result={'status':'FAIL_CLOSED_V33H2_NO_PHYSICAL_RAR_BALL','camera_lock':[LAR,RAR,BCAST],'candidates':[{**{k:v for k,v in x.items() if k not in ('candidate','world_cm')},'candidate':v33g.serial_candidate(x['candidate']),'world_cm':x['world_cm'].tolist()} for x in raw]}
        (args.out/'v33h2_ball_qa.json').write_text(json.dumps(result,indent=2)); raise SystemExit(9)
    good.sort(key=lambda x:(x['wrist_ray_perp_cm']+0.18*x['rim_distance_cm']+18*abs(math.log(max(x['size_ratio'],1e-6))) - 3*x['candidate'].get('quality',0.0)))
    best=good[0]; X=best['world_cm']

    # Independent occlusion evidence in the two non-support views.
    occ={}; overlays=[]
    for c in CAMS:
        im=fs[c][relmap[c]][1].copy(); uv=v32j.project(cams[c],X)
        if uv is None: raise RuntimeError(f'ball projects invalid in {c}')
        boxes=person_boxes(det,fs[c][relmap[c]][1]) if c!=RAR else []
        rimuv=v32j.project(cams[c],RIM); near_rim=bool(rimuv is not None and np.linalg.norm(uv-rimuv)<=38.0)
        by_person=any(contains(b,uv,14) for b in boxes)
        observed=(c==RAR)
        hidden=bool((not observed) and (near_rim or by_person))
        occ[c]={'projected_px':[float(x) for x in uv],'observed':observed,'near_rim_occlusion':near_rim,'person_occlusion':by_person,'person_boxes':boxes,'occluded':hidden}
        p=tuple(np.rint(uv).astype(int)); cv2.circle(im,p,12,(0,255,255),2,cv2.LINE_AA)
        cv2.putText(im,'REAL BALL' if observed else ('OCCLUDED PROJECTION' if hidden else 'UNVERIFIED'),(max(2,p[0]+14),max(18,p[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(args.out/f'v33h2_{c.replace(" ","_")}_ball_qa.png'),im); overlays.append(im)
    cv2.imwrite(str(args.out/'v33h2_three_camera_ball_montage.png'),np.hstack(overlays))
    pass_gate=bool(occ[LAR]['occluded'] and occ[BCAST]['occluded'])
    result={'version':'v33h2_occlusion_aware_single_view_ball_qa','status':'PASS_V33H2_NUMERICAL_AWAITING_VISUAL_QA' if pass_gate else 'FAIL_CLOSED_V33H2_OTHER_VIEWS_NOT_OCCLUDED',
            'camera_lock':[LAR,RAR,BCAST],'camera_count':3,'exact_state':{'rels':relmap,'frames':top['frames'],'v33e_score':top['score']},
            'body':{k:v for k,v in body.items() if not k.startswith('_')},
            'ball':{'world_cm':[float(x) for x in X],'source_view':RAR,'source_candidate':v33g.serial_candidate(best['candidate']),
                    'nearest_wrist_index':int(best['wrist_index']),'wrist_ray_perp_cm':best['wrist_ray_perp_cm'],'rim_distance_cm':best['rim_distance_cm'],'z_cm':best['z_cm'],
                    'observed_equiv_diameter_px':best['observed_equiv_diameter_px'],'expected_ball_diameter_px':best['expected_ball_diameter_px'],'size_ratio':best['size_ratio']},
            'view_evidence':occ,'gate':{'body_pass':True,'real_rar_ball_ray_pass':True,'lar_occluded':occ[LAR]['occluded'],'broadcast_occluded':occ[BCAST]['occluded'],'pass':pass_gate},
            'method':'exact-frame real RAR ball blob + temporal motion + solved fixed-centre RAR ray + closest point to accepted triangulated Adams wrist + rim/height/apparent-size gates + explicit source occlusion in both other locked cameras',
            'generated_rgb':False,'upscaled':False,'novel_view_rendered':False,'external_visual_qa_required':True,'freeview_render_unlocked':False}
    (args.out/'v33h2_ball_qa.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
    if not pass_gate: raise SystemExit(10)

if __name__=='__main__': main()
