from __future__ import annotations

"""v33h: deadline-path source-ball solve using one visible ball view + exact 3-view body contact.

This does NOT reopen v33e cameras or exact-state geometry.  It addresses the
specific visual fact established after v33f/v33g: at useful accepted exact states
the basketball is visibly exposed in Right Above Rim while LAR/Broadcast can be
occluded by Adams/rim structure.

A ball candidate may therefore be reconstructed from ONE real RAR observation
only when all of the following are true:
  * the exact-frame source candidate is a temporally validated moving basketball
    candidate from v33g, with semantic sports-ball support preferred/required;
  * it is spatially attached to a visible Adams wrist in the exact RAR frame;
  * its RAR camera ray passes within a basketball-contact distance of the
    three-view reconstructed wrist in metric 3-D;
  * the resulting metric center is physically plausible near the rim;
  * projection into each non-observing locked view is either supported by a real
    source candidate or explicitly explainable as occluded by Adams/rim.

The 3-D ball depth is therefore CONTACT-CONSTRAINED RECONSTRUCTION, not claimed
as multi-view observed triangulation.  No RGB is generated and neighboring
frames are localization evidence only.
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
WRISTS = (9, 10)
MAX_WRIST_PIXEL_DIST = 82.0
MAX_RAY_TO_WRIST_CM = 34.0
MAX_RIM_DIST_CM = 135.0
OCCLUSION_PLAYER_PAD = 28.0
OCCLUSION_RIM_PX = 52.0
SOURCE_CONFIRM_PX = 26.0


def camera_for_rel(scene: dict, cam_states: dict, cam: str, rel: int):
    s = copy.deepcopy(scene)
    s['cameras'][cam] = cam_states[cam][rel]
    return v32j.cam(s, cam)


def serialize_candidate(c: dict) -> dict:
    out = {k: v for k, v in c.items() if k != 'center'}
    out['center'] = np.asarray(c['center'], float).tolist()
    return out


def ray_contact_point(cam: dict, uv: np.ndarray, wrist: np.ndarray):
    d = v32j.ray(cam, uv)
    C = np.asarray(cam['C'], float)
    t = float(np.dot(np.asarray(wrist, float) - C, d))
    if t <= 20.0:
        return None
    X = C + t * d
    return X, float(np.linalg.norm(X - wrist)), t


def candidate_contact_hypotheses(cams, rows, body, candidates):
    joints = body['_joints']
    rrow = rows[RAR]
    xy = np.asarray(rrow['xy'], float)
    cf = np.asarray(rrow['conf'], float)
    hyps = []
    for cand in candidates[RAR]:
        # Deadline solve is deliberately strict on identity: a low-confidence
        # semantic sports-ball is acceptable only because source appearance,
        # temporal motion and hand contact all independently constrain it.
        if cand.get('source') != 'semantic':
            continue
        if not cand.get('appearance_gate') or not cand.get('temporal_gate'):
            continue
        box = np.asarray(cand['bbox'], float)
        bw, bh = float(box[2]-box[0]), float(box[3]-box[1])
        if not (5.0 <= bw <= 55.0 and 5.0 <= bh <= 55.0 and 0.45 <= bw/max(bh,1e-6) <= 2.1):
            continue
        center = np.asarray(cand['center'], float)
        for j in WRISTS:
            if j not in joints or j >= len(cf) or cf[j] < v32v.MIN_CONF:
                continue
            wrist_px = xy[j]
            pixdist = float(np.linalg.norm(center - wrist_px))
            if pixdist > MAX_WRIST_PIXEL_DIST:
                continue
            rc = ray_contact_point(cams[RAR], center, joints[j])
            if rc is None:
                continue
            X, contact_cm, ray_depth = rc
            if contact_cm > MAX_RAY_TO_WRIST_CM:
                continue
            if not (235.0 <= float(X[2]) <= 365.0):
                continue
            # The ball can be beside/above the hand; reject a solution far below
            # the reconstructed wrist because that is the exact v33g failure mode.
            dz = float(X[2] - joints[j][2])
            if not (-10.0 <= dz <= 42.0):
                continue
            rimdist = float(np.linalg.norm(X - RIM))
            if rimdist > MAX_RIM_DIST_CM:
                continue

            projected = {c: v32j.project(cams[c], X) for c in CAMS}
            if any(not v33f.inside_image(projected[c]) for c in CAMS):
                continue
            view_evidence = {}
            other_consistent = True
            explicit_occlusions = 0
            for c in CAMS:
                uv = np.asarray(projected[c], float)
                if c == RAR:
                    repro = float(np.linalg.norm(uv - center))
                    view_evidence[c] = {'status':'OBSERVED_SOURCE_BALL', 'reprojection_px':repro,
                                        'candidate_center':center.tolist()}
                    other_consistent &= repro <= 1.5
                    continue
                pool = candidates.get(c, [])
                nearest = None
                nd = 999.0
                if pool:
                    nearest = min(pool, key=lambda z: float(np.linalg.norm(np.asarray(z['center'],float)-uv)))
                    nd = float(np.linalg.norm(np.asarray(nearest['center'],float)-uv))
                if nearest is not None and nd <= SOURCE_CONFIRM_PX:
                    view_evidence[c] = {'status':'SOURCE_CANDIDATE_CONSISTENT', 'distance_px':nd,
                                        'candidate_source':nearest.get('source'),
                                        'candidate_center':np.asarray(nearest['center'],float).tolist()}
                else:
                    rim_uv = v32j.project(cams[c], RIM)
                    near_rim = rim_uv is not None and float(np.linalg.norm(uv-np.asarray(rim_uv,float))) <= OCCLUSION_RIM_PX
                    player_occ = v33f.box_contains(rows[c]['box'], uv, OCCLUSION_PLAYER_PAD)
                    occ = bool(near_rim or player_occ)
                    if occ:
                        explicit_occlusions += 1
                        view_evidence[c] = {'status':'EXPLICIT_SOURCE_OCCLUSION',
                                            'near_rim':bool(near_rim),'inside_adams_box':bool(player_occ),
                                            'nearest_candidate_distance_px':None if nearest is None else nd}
                    else:
                        other_consistent = False
                        view_evidence[c] = {'status':'UNEXPLAINED_MISSING_BALL',
                                            'nearest_candidate_distance_px':None if nearest is None else nd}

            gate = bool(other_consistent and explicit_occlusions +
                        sum(v['status']=='SOURCE_CANDIDATE_CONSISTENT' for k,v in view_evidence.items() if k!=RAR) >= 2)
            score = (0.85*pixdist + 1.35*contact_cm + 0.08*rimdist
                     - 5.0*float(cand.get('appearance_strength',0.0))
                     - 0.06*float(cand.get('temporal',{}).get('trajectory_span_rim_compensated_px',0.0)))
            hyps.append({'gate':gate,'score':float(score),'method':'RAR exact-frame semantic source ball + exact 3-view wrist contact depth',
                         'world_cm':X.tolist(),'wrist_joint':int(j),'wrist_name':v32j.NAMES[j],
                         'wrist_world_cm':np.asarray(joints[j],float).tolist(),
                         'wrist_source_px':wrist_px.tolist(),'source_ball_center_px':center.tolist(),
                         'source_ball_bbox':box.tolist(),'source_ball_confidence':float(cand.get('confidence',0.0)),
                         'source_ball_appearance_strength':float(cand.get('appearance_strength',0.0)),
                         'source_ball_temporal':cand.get('temporal',{}),
                         'wrist_pixel_distance_px':pixdist,'ray_to_wrist_distance_cm':contact_cm,
                         'ray_depth_cm':ray_depth,'ball_minus_wrist_z_cm':dz,
                         'distance_to_rim_center_cm':rimdist,
                         'projected_centers':{c:np.asarray(projected[c],float).tolist() for c in CAMS},
                         'view_evidence':view_evidence,'explicit_occlusion_count':int(explicit_occlusions)})
    hyps.sort(key=lambda x:(not x['gate'],x['score']))
    return (hyps[0] if hyps else None), hyps[:20]


def draw_contact_overlay(image, row, joints, cams, label, candidates, best):
    out = image.copy()
    valid = np.asarray(row['conf'],float) >= v32v.MIN_CONF
    for a,b in v32j.DRAW:
        if valid[a] and valid[b]:
            cv2.line(out,tuple(np.rint(row['xy'][a]).astype(int)),tuple(np.rint(row['xy'][b]).astype(int)),(0,255,255),2,cv2.LINE_AA)
    for j in v32v.BODY:
        if valid[j]: cv2.circle(out,tuple(np.rint(row['xy'][j]).astype(int)),3,(0,255,255),-1,cv2.LINE_AA)
    for a,b in v32j.DRAW:
        if a in joints and b in joints:
            ua,ub=v32j.project(cams[label],joints[a]),v32j.project(cams[label],joints[b])
            if ua is not None and ub is not None:
                cv2.line(out,tuple(np.rint(ua).astype(int)),tuple(np.rint(ub).astype(int)),(255,0,255),1,cv2.LINE_AA)
    if best is not None:
        uv=np.asarray(best['projected_centers'][label],float)
        cv2.circle(out,tuple(np.rint(uv).astype(int)),12,(255,255,0),2,cv2.LINE_AA)
        ev=best['view_evidence'][label]['status']
        cv2.putText(out,ev,(max(4,int(uv[0])+14),max(48,int(uv[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,0),1,cv2.LINE_AA)
        if label==RAR:
            c=np.asarray(best['source_ball_center_px'],float)
            cv2.circle(out,tuple(np.rint(c).astype(int)),7,(0,255,0),2,cv2.LINE_AA)
            w=np.asarray(best['wrist_source_px'],float)
            cv2.line(out,tuple(np.rint(c).astype(int)),tuple(np.rint(w).astype(int)),(0,255,0),1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,34),(0,0,0),-1)
    cv2.putText(out,f'v33h {label} | exact-state body | cyan=contact ball projection | green=RAR source ball',
                (7,22),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clips-dir',type=Path,required=True)
    ap.add_argument('--b32-root',type=Path,required=True)
    ap.add_argument('--v73-frame0257',type=Path,required=True)
    ap.add_argument('--v33e-json',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    accepted=json.loads(args.v33e_json.read_text())
    if accepted.get('status')!='PASS_V33E_EXACT_THREE_VIEW_STATE' or accepted.get('camera_lock')!=[LAR,RAR,BCAST]:
        raise RuntimeError('v33h requires accepted v33e locked-three-camera evidence')
    exact_rows=[x for x in accepted.get('top_triplets',[]) if x.get('exact_state')]
    if not exact_rows: raise RuntimeError('no accepted exact states in v33e artifact')

    rels=tuple(range(-20,21)); v33d.RELS=rels
    stage=args.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text())
    if scene.get('resolution')!=[W,H] or set(scene.get('cameras',{}))!={LAR,BCAST,RAR}:
        raise RuntimeError('locked camera/source mismatch')
    centers={k:int(v) for k,v in scene['freeze']['chosen_frame_indices'].items()}
    wide=args.out/'wide_stage'; wide.mkdir(parents=True,exist_ok=True)
    source_audit=v33e.export_wide_burst(args.clips_dir,centers,wide,rels)
    fs={c:v33d.frames(wide,c) for c in CAMS}

    cert=cv2.imread(str(args.v73_frame0257),cv2.IMREAD_GRAYSCALE)
    if cert is None or cert.shape!=(H,W): raise RuntimeError('missing accepted RAR certificate')
    cam_states={c:{} for c in CAMS}
    for c in (LAR,BCAST):
        for rel in rels:
            cc,_=v33d.transfer(fs[c][0][2],fs[c][rel][2],scene['cameras'][c])
            if cc is not None: cam_states[c][rel]=cc
    for rel in rels:
        cc,_=v33b.transfer_rar_camera(cert,fs[RAR][rel][2],scene['cameras'][RAR])
        if cc is not None: cam_states[RAR][rel]=cc

    kp=RFDETRKeypointPreview(); identity={}
    for c in CAMS: identity[c],_=v33d.track(kp,c,fs[c])
    obs={}
    for c in CAMS: obs[c],_=v33e.flow_observations(fs[c],identity[c],c,rels)

    exact_needed={c:sorted({int(x['rels'][c]) for x in exact_rows}) for c in CAMS}
    temporal_needed={}
    for c in CAMS:
        s=set()
        for r in exact_needed[c]:
            for d in (0,)+v33g.TEMPORAL_OFFSETS:
                rr=r+d
                if rr in fs[c] and rr in cam_states[c]: s.add(rr)
        temporal_needed[c]=sorted(s)

    ball_model=RFDETRMedium(); cache={c:{} for c in CAMS}
    for c in CAMS:
        for rel in temporal_needed[c]:
            cam=camera_for_rel(scene,cam_states,c,rel); rimuv=v32j.project(cam,RIM)
            cache[c][rel]=[] if rimuv is None else v33g.build_frame_candidates(ball_model,fs[c][rel][1],rimuv)
    validated={c:{} for c in CAMS}
    for c in CAMS:
        for rel in exact_needed[c]:
            pool=[v33g.add_temporal_evidence(cache[c],rel,q) for q in cache[c].get(rel,[])]
            pool=[q for q in pool if q.get('temporal_gate')]
            pool.sort(key=lambda q:(-q.get('quality',0.0),-q.get('confidence',0.0)))
            validated[c][rel]=pool[:12]

    diagnostics=[]; passing=[]
    for rank,x in enumerate(exact_rows,start=1):
        relmap={c:int(x['rels'][c]) for c in CAMS}
        if any(relmap[c] not in cam_states[c] for c in CAMS): continue
        s,cams=v33f.camera_for_state(scene,cam_states,relmap)
        rows={c:obs[c][relmap[c]] for c in CAMS}
        body=v33f.body_fit(s,rows)
        candidates={c:validated[c].get(relmap[c],[]) for c in CAMS}
        ball,hyps=candidate_contact_hypotheses(cams,rows,body,candidates) if body['gate'] else (None,[])
        gate=bool(body['gate'] and ball and ball['gate'])
        diagnostics.append({'rank':rank,'rels':relmap,'frames':x['frames'],'v33e_score':float(x['score']),
                            'body_gate':bool(body['gate']),'body_reprojection':body['reprojection'],
                            'candidate_counts':{c:len(candidates[c]) for c in CAMS},
                            'rar_candidates':[serialize_candidate(z) for z in candidates[RAR][:8]],
                            'best_contact_ball':ball,'top_contact_hypotheses':hyps[:5],'gate':gate})
        if gate: passing.append((rank,x,body,ball,cams,rows,candidates))

    chosen=min(passing,key=lambda t:(t[0],t[3]['score'])) if passing else None
    passed=chosen is not None
    if passed:
        rank,x,body,ball,cams,rows,candidates=chosen; relmap={c:int(x['rels'][c]) for c in CAMS}
        overlays=[]
        for c in CAMS:
            src=fs[c][relmap[c]][1]
            cv2.imwrite(str(args.out/f'v33h_source_{c.replace(" ","_")}.png'),src)
            ov=draw_contact_overlay(src,rows[c],body['_joints'],cams,c,candidates[c],ball)
            cv2.imwrite(str(args.out/f'v33h_{c.replace(" ","_")}_contact_ball_qa.png'),ov); overlays.append(ov)
        cv2.imwrite(str(args.out/'v33h_three_camera_contact_ball_montage.png'),np.hstack(overlays))
        # Export exact-state cameras for the renderer. Physical centres are the
        # accepted centres; only per-frame static-state transfer is represented.
        cameras_json={c:{'K_px':cams[c]['K'].tolist(),'R_world_to_camera':cams[c]['R'].tolist(),
                         'C_world_cm':cams[c]['C'].tolist()} for c in CAMS}
        export={'camera_lock':[LAR,RAR,BCAST],'resolution':[W,H],'rels':relmap,'frames':x['frames'],
                'cameras':cameras_json,'body_joints_world_cm':{v32j.NAMES[j]:np.asarray(X,float).tolist() for j,X in body['_joints'].items()},
                'ball':ball}
        (args.out/'v33h_render_state.json').write_text(json.dumps(export,indent=2))
        body_json={k:v for k,v in body.items() if not k.startswith('_')}
        chosen_json={'rank':rank,'rels':relmap,'frames':x['frames'],'v33e_score':float(x['score']),'body':body_json,'ball':ball}
    else:
        chosen_json=None

    qa={'version':'v33h_contact_constrained_ball_qa','status':'PASS_V33H_CONTACT_BALL_AWAITING_VISUAL_QA' if passed else 'FAIL_CLOSED_V33H_NO_CONTACT_BALL_STATE',
        'upstream_v33e_status':accepted.get('status'),'camera_lock':[LAR,RAR,BCAST],'camera_count':3,
        'camera_addition_or_substitution_used':False,'physical_center_refit_from_player_or_ball':False,'exact_state_reopened':False,
        'native_resolution':[W,H],'generated_rgb':False,'upscaled':False,'novel_view_rendered':False,
        'ball_method_classification':'reconstructed depth from one exact source ball observation constrained by three-view reconstructed wrist contact; NOT multi-view observed ball triangulation',
        'gates':{'max_wrist_pixel_distance_px':MAX_WRIST_PIXEL_DIST,'max_camera_ray_to_3d_wrist_cm':MAX_RAY_TO_WRIST_CM,
                 'max_rim_distance_cm':MAX_RIM_DIST_CM,'nonobserving_view_requires_source_candidate_or_explicit_occlusion':True,
                 'semantic_source_ball_required_in_RAR':True},
        'chosen_state':chosen_json,'state_diagnostics':diagnostics,
        'render_numerically_eligible':bool(passed),'freeview_render_unlocked':False,
        'unlock_policy':'visual QA of v33h montage is mandatory before novel-view rendering'}
    (args.out/'v33h_contact_ball_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps({'status':qa['status'],'chosen_state':qa['chosen_state'],'render_numerically_eligible':qa['render_numerically_eligible']},indent=2),flush=True)
    if not passed: raise SystemExit(7)

if __name__=='__main__': main()
