from __future__ import annotations

"""v33h: recover a source-grounded basketball from the FULL accepted v33e exact-state population.

Hard locks:
- event 489 only
- exactly LAR + RAR + Broadcast
- accepted physical camera centres are never refit
- v33e +/-20 native window and exact-state gates are unchanged
- one real source frame per camera per evaluated state
- neighboring frames are ball-localization evidence only, never exact-state observations
- native 960x540 only; no render in this stage

v33f/v33g showed that low-confidence sports-ball / orange detections can form geometrically
consistent but visually false hypotheses. v33h therefore recomputes the same 629 accepted
v33e exact states (integrity-checked against the accepted artifact), then requires a ball to:
1) be temporally moving relative to the projected physical rim in each support view;
2) be source-visible in >=2 locked cameras, with >=1 exact-frame semantic sports-ball hit;
3) satisfy tighter epipolar/reprojection gates;
4) reconstruct near rim height and near one of Adams' triangulated wrists;
5) visually remain locked for external source-frame QA before any render.
"""

import argparse
import copy
import json
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
RIM_WORLD_CM = np.array([38.1, 0.0, 304.8], float)
RELS = tuple(range(-20, 21))
PAIR_EPI_MAX_PX = 12.0
MATCH_MAX_PX = 10.0
MEDIAN_REPROJ_MAX_PX = 6.0
MAX_REPROJ_MAX_PX = 10.0
RIM_DISTANCE_MAX_CM = 155.0
BALL_Z_MIN_CM = 225.0
BALL_Z_MAX_CM = 390.0
WRIST_DISTANCE_MAX_CM = 90.0
IMAGE_DY_MIN_PX = -245.0
IMAGE_DY_MAX_PX = 95.0


def serial_candidate(c: dict) -> dict:
    return v33g.serial_candidate(c)


def recompute_exact_states(scene, cam_states, obs, fs):
    valid = {c: tuple(r for r in RELS if r in cam_states[c] and r in obs[c]) for c in CAMS}
    lr = v33e.pair_table(scene, cam_states, obs, LAR, RAR, valid[LAR], valid[RAR])
    lb = v33e.pair_table(scene, cam_states, obs, LAR, BCAST, valid[LAR], valid[BCAST])
    rb = v33e.pair_table(scene, cam_states, obs, RAR, BCAST, valid[RAR], valid[BCAST])
    exact = []
    tested = pair_pass = 0
    for l in valid[LAR]:
        for r in valid[RAR]:
            xlr = lr[(l, r)]
            for b in valid[BCAST]:
                tested += 1
                xlb, xrb = lb[(l, b)], rb[(r, b)]
                if not (xlr['gate'] and xlb['gate'] and xrb['gate']):
                    continue
                pair_pass += 1
                s = copy.deepcopy(scene)
                s['cameras'][LAR] = cam_states[LAR][l]
                s['cameras'][RAR] = cam_states[RAR][r]
                s['cameras'][BCAST] = cam_states[BCAST][b]
                q = v33d.repro(s, obs[LAR][l], obs[RAR][r], obs[BCAST][b])
                if not q['gate']:
                    continue
                span = max(l, r, b) - min(l, r, b)
                score = float(xlr['score'] + xlb['score'] + xrb['score'] + 0.30 * span)
                exact.append({
                    'rels': {LAR: int(l), RAR: int(r), BCAST: int(b)},
                    'frames': {LAR: fs[LAR][l][0].name, RAR: fs[RAR][r][0].name, BCAST: fs[BCAST][b][0].name},
                    'score': score, 'span': int(span), 'reprojection': q,
                    'pair_epipolar': {
                        'LR': v33e.serial_pair(xlr), 'LB': v33e.serial_pair(xlb), 'RB': v33e.serial_pair(xrb)
                    }
                })
    exact.sort(key=lambda x: (x['score'], x['reprojection']['p90_px'], x['reprojection']['median_px']))
    return exact, {'tested_triplets': tested, 'pairwise_pass_triplets': pair_pass, 'exact_pass_triplets': len(exact),
                   'valid_static_camera_states': {c: len(valid[c]) for c in CAMS}}


def candidate_physical_image_gate(c: dict) -> bool:
    if not c.get('temporal_gate') or not c.get('appearance_gate'):
        return False
    dy = float(c['basket_relative_xy'][1])
    if not (IMAGE_DY_MIN_PX <= dy <= IMAGE_DY_MAX_PX):
        return False
    x1, y1, x2, y2 = map(float, c['bbox'])
    w, h = x2-x1, y2-y1
    if w < 4 or h < 4 or w > 82 or h > 82:
        return False
    if c['source'] == 'orange_track':
        sh = c.get('shape', {})
        if float(sh.get('largest_orange_circularity', 0.0)) < 0.24:
            return False
        if float(sh.get('largest_orange_fill', 0.0)) < 0.34:
            return False
    return True


def occluded_in_view(cam, row, X) -> bool:
    uv = v32j.project(cam, X)
    if uv is None or not v33f.inside_image(uv):
        return False
    rim_uv = v32j.project(cam, RIM_WORLD_CM)
    near_rim = rim_uv is not None and float(np.linalg.norm(uv-rim_uv)) <= 32.0
    by_player = v33f.box_contains(row['box'], uv, 18.0)
    return bool(near_rim or by_player)


def ball_hypothesis(cams, rows, body_fit, candidates):
    wrists = [body_fit['_joints'][j] for j in (9, 10) if j in body_fit['_joints']]
    if not wrists:
        return None, []
    Fs = {(a,b): v32j.fundamental(cams[a], cams[b]) for a in CAMS for b in CAMS if a != b}
    hyps = []
    for ia in range(len(CAMS)):
        for ib in range(ia+1, len(CAMS)):
            a, b = CAMS[ia], CAMS[ib]
            for ca in candidates[a]:
                for cb in candidates[b]:
                    epi = float(v32j.epi(Fs[(a,b)], ca['center'], cb['center']))
                    if epi > PAIR_EPI_MAX_PX:
                        continue
                    X = v32j.triangulate_rays(cams, {a:ca['center'], b:cb['center']})
                    if X is None or not np.all(np.isfinite(X)):
                        continue
                    z = float(X[2]); rim_dist = float(np.linalg.norm(X-RIM_WORLD_CM))
                    wrist_dist = float(min(np.linalg.norm(X-w) for w in wrists))
                    if not (BALL_Z_MIN_CM <= z <= BALL_Z_MAX_CM):
                        continue
                    if rim_dist > RIM_DISTANCE_MAX_CM or wrist_dist > WRIST_DISTANCE_MAX_CM:
                        continue
                    projected = {c:v32j.project(cams[c], X) for c in CAMS}
                    if not all(v33f.inside_image(projected[c]) for c in CAMS):
                        continue
                    matched = {}
                    for c in CAMS:
                        if not candidates[c]:
                            continue
                        n = min(candidates[c], key=lambda q: float(np.linalg.norm(q['center']-projected[c])))
                        d = float(np.linalg.norm(n['center']-projected[c]))
                        if d <= MATCH_MAX_PX:
                            matched[c] = (n,d)
                    if len(matched) == 3:
                        X2 = v32j.triangulate_rays(cams, {c:matched[c][0]['center'] for c in CAMS})
                        if X2 is not None and np.all(np.isfinite(X2)):
                            X = X2
                            projected = {c:v32j.project(cams[c], X) for c in CAMS}
                            rem = {}
                            for c in CAMS:
                                n = min(candidates[c], key=lambda q: float(np.linalg.norm(q['center']-projected[c])))
                                d = float(np.linalg.norm(n['center']-projected[c]))
                                if d <= MATCH_MAX_PX:
                                    rem[c]=(n,d)
                            matched = rem
                            z=float(X[2]); rim_dist=float(np.linalg.norm(X-RIM_WORLD_CM)); wrist_dist=float(min(np.linalg.norm(X-w) for w in wrists))
                    support=len(matched)
                    if support < 2:
                        continue
                    semantic=sum(v[0]['source']=='semantic' for v in matched.values())
                    hidden=[c for c in CAMS if c not in matched and occluded_in_view(cams[c], rows[c], X)]
                    residual=[v[1] for v in matched.values()]
                    med=float(np.median(residual)); mx=float(max(residual))
                    pass_gate=bool(semantic>=1 and med<=MEDIAN_REPROJ_MAX_PX and mx<=MAX_REPROJ_MAX_PX and
                                   BALL_Z_MIN_CM<=z<=BALL_Z_MAX_CM and rim_dist<=RIM_DISTANCE_MAX_CM and
                                   wrist_dist<=WRIST_DISTANCE_MAX_CM and
                                   (support==3 or (support==2 and len(hidden)==1)))
                    if not pass_gate:
                        continue
                    quality=float(sum(v[0].get('quality',0.0) for v in matched.values()))
                    score=float(2.5*epi + 2.0*med + mx + 0.05*wrist_dist + 0.02*rim_dist - 1.5*quality - 4.0*(support-2))
                    hyps.append({
                        'gate': True, 'score':score, 'triangulation_pair':[a,b], 'pair_epipolar_px':epi,
                        'world_cm':X.tolist(), 'ball_height_cm':z, 'distance_to_rim_center_cm':rim_dist,
                        'distance_to_nearest_wrist_cm':wrist_dist, 'support_views':list(matched.keys()),
                        'support_count':support, 'semantic_support_count':semantic, 'occluded_views':hidden,
                        'support_sources':{c:matched[c][0]['source'] for c in matched},
                        'matched':{c:{'center':matched[c][0]['center'].tolist(), 'bbox':matched[c][0]['bbox'],
                                      'source':matched[c][0]['source'], 'confidence':float(matched[c][0]['confidence']),
                                      'appearance_strength':float(matched[c][0].get('appearance_strength',0.0)),
                                      'basket_relative_xy':matched[c][0]['basket_relative_xy'],
                                      'temporal':matched[c][0].get('temporal',{}), 'reprojection_error_px':matched[c][1]}
                                   for c in matched},
                        'projected_centers':{c:projected[c].tolist() for c in CAMS},
                        'median_supported_reprojection_px':med, 'max_supported_reprojection_px':mx
                    })
    hyps.sort(key=lambda h:(-h['support_count'], -h['semantic_support_count'], h['score']))
    return (hyps[0] if hyps else None), hyps[:20]


def draw_strip(fs, rel, selected, label, out_path):
    panels=[]
    for d in (-3,-2,-1,0,1,2,3):
        rr=rel+d
        if rr not in fs:
            continue
        im=fs[rr][1].copy()
        cv2.putText(im, f'{label} rel {rr:+d} {"EXACT" if d==0 else "neighbor"}', (10,24),
                    cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        if d==0 and selected is not None:
            p=np.rint(np.asarray(selected['center'],float)).astype(int)
            cv2.circle(im, tuple(p), 13, (0,255,255), 2, cv2.LINE_AA)
        panels.append(im)
    if panels:
        cv2.imwrite(str(out_path), np.hstack(panels))


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
        raise RuntimeError('v33h requires accepted v33e locked three-camera evidence')
    v33d.RELS=RELS
    stage=args.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text())
    if scene.get('resolution')!=[W,H] or set(scene.get('cameras',{}))!={LAR,BCAST,RAR}:
        raise RuntimeError('locked camera/source mismatch')
    centers={k:int(v) for k,v in scene['freeze']['chosen_frame_indices'].items()}
    wide=args.out/'wide_stage'; wide.mkdir(parents=True,exist_ok=True)
    source_audit=v33e.export_wide_burst(args.clips_dir,centers,wide,RELS)
    fs={c:v33d.frames(wide,c) for c in CAMS}

    cert=cv2.imread(str(args.v73_frame0257),cv2.IMREAD_GRAYSCALE)
    if cert is None or cert.shape!=(H,W): raise RuntimeError('missing accepted v73 RAR certificate frame')
    cam_states={c:{} for c in CAMS}; transfer={c:{} for c in CAMS}
    for c in (LAR,BCAST):
        for rel in RELS:
            cc,qa=v33d.transfer(fs[c][0][2],fs[c][rel][2],scene['cameras'][c]); transfer[c][str(rel)]=qa
            if cc is not None: cam_states[c][rel]=cc
    for rel in RELS:
        cc,qa=v33b.transfer_rar_camera(cert,fs[RAR][rel][2],scene['cameras'][RAR]); transfer[RAR][str(rel)]=qa
        if cc is not None: cam_states[RAR][rel]=cc

    kp=RFDETRKeypointPreview(); identity={}; identity_audit={}; obs={}; obs_audit={}
    for c in CAMS: identity[c],identity_audit[c]=v33d.track(kp,c,fs[c])
    for c in CAMS: obs[c],obs_audit[c]=v33e.flow_observations(fs[c],identity[c],c,RELS)

    exact_rows,recompute=recompute_exact_states(scene,cam_states,obs,fs)
    expected=int(accepted.get('exact_pass_triplets',-1))
    if recompute['exact_pass_triplets']!=expected:
        raise RuntimeError(f'v33e exact-state integrity mismatch: recomputed {recompute["exact_pass_triplets"]} expected {expected}')

    needed={c:sorted({int(x['rels'][c]) for x in exact_rows}) for c in CAMS}
    temporal_needed={}
    for c in CAMS:
        s=set()
        for r in needed[c]:
            for d in (0,)+tuple(v33g.TEMPORAL_OFFSETS):
                rr=r+d
                if rr in fs[c] and rr in cam_states[c]: s.add(rr)
        temporal_needed[c]=sorted(s)

    ball_model=RFDETRMedium(); cache={c:{} for c in CAMS}
    for c in CAMS:
        for rel in temporal_needed[c]:
            cam=v33g._camera_for_rel(scene,cam_states,c,rel); rim=v32j.project(cam,RIM_WORLD_CM)
            cache[c][rel]=[] if rim is None else v33g.build_frame_candidates(ball_model,fs[c][rel][1],rim)
    validated={c:{} for c in CAMS}
    for c in CAMS:
        for rel in needed[c]:
            pool=[v33g.add_temporal_evidence(cache[c],rel,q) for q in cache[c].get(rel,[])]
            pool=[q for q in pool if candidate_physical_image_gate(q)]
            pool.sort(key=lambda q:(q['source']!='semantic',-q.get('quality',0.0),-q.get('confidence',0.0)))
            validated[c][rel]=pool[:10]

    state_diag=[]; passing=[]
    for rank,x in enumerate(exact_rows,start=1):
        relmap={c:int(x['rels'][c]) for c in CAMS}
        s,cams=v33f.camera_for_state(scene,cam_states,relmap)
        rows={c:obs[c][relmap[c]] for c in CAMS}
        bfit=v33f.body_fit(s,rows)
        if not bfit['gate']:
            continue
        cand={c:validated[c].get(relmap[c],[]) for c in CAMS}
        bb,hyps=ball_hypothesis(cams,rows,bfit,cand)
        diag={'rank':rank,'rels':relmap,'frames':x['frames'],'v33e_score':float(x['score']),
              'v33e_reprojection':x['reprojection'], 'body_gate':bool(bfit['gate']),
              'candidate_counts':{c:len(cand[c]) for c in CAMS},
              'candidates':{c:[serial_candidate(z) for z in cand[c][:4]] for c in CAMS},
              'best_ball':bb,'top_ball_hypotheses':hyps[:5],'gate':bool(bb)}
        state_diag.append(diag)
        if bb is not None: passing.append((rank,x,bfit,bb,cams,rows,cand))

    chosen_tuple=min(passing,key=lambda t:(-t[3]['support_count'],-t[3]['semantic_support_count'],t[3]['score'],t[0])) if passing else None
    passed=chosen_tuple is not None
    chosen=chosen_body=chosen_ball=chosen_cams=chosen_rows=chosen_cand=None
    if passed:
        _,chosen,chosen_body,chosen_ball,chosen_cams,chosen_rows,chosen_cand=chosen_tuple
        relmap={c:int(chosen['rels'][c]) for c in CAMS}; montage=[]
        for c in CAMS:
            src=fs[c][relmap[c]][1]
            ov=v33g.draw_overlay(src,chosen_rows[c],chosen_body['_joints'],chosen_cams,c,chosen_cand[c],chosen_ball)
            cv2.imwrite(str(args.out/f'v33h_source_{c.replace(" ","_")}.png'),src)
            cv2.imwrite(str(args.out/f'v33h_{c.replace(" ","_")}_body_ball_qa.png'),ov); montage.append(ov)
            selected=None
            if c in chosen_ball['matched']:
                cc=chosen_ball['matched'][c]['center']
                selected=min(chosen_cand[c],key=lambda z:float(np.linalg.norm(z['center']-np.asarray(cc,float))))
            draw_strip(fs[c],relmap[c],selected,c,args.out/f'v33h_{c.replace(" ","_")}_7frame_strip.png')
        cv2.imwrite(str(args.out/'v33h_three_camera_body_ball_montage.png'),np.hstack(montage))

    body_json=None if chosen_body is None else {k:v for k,v in chosen_body.items() if not k.startswith('_')}
    qa={
        'version':'v33h_full_accepted_exact_state_ball_visibility_qa',
        'status':'PASS_V33H_NUMERICAL_AWAITING_VISUAL_QA' if passed else 'FAIL_CLOSED_V33H_NO_REAL_TWO_VIEW_BALL_STATE',
        'upstream_v33e_status':accepted.get('status'),'camera_lock':[LAR,RAR,BCAST],'camera_count':3,
        'camera_addition_or_substitution_used':False,'physical_center_refit_from_player_or_ball':False,
        'exact_state_reopened':False,'native_resolution':[W,H],'generated_rgb':False,'upscaled':False,'novel_view_rendered':False,
        'v33e_integrity':{'accepted_exact_state_count':expected,'recomputed_exact_state_count':recompute['exact_pass_triplets'],
                          'count_match':recompute['exact_pass_triplets']==expected,**recompute},
        'source_audit':source_audit,
        'ball_gate_policy':{'full_v33e_exact_population':True,'min_source_support_views':2,'min_semantic_support_views':1,
                            'pair_epipolar_px_max':PAIR_EPI_MAX_PX,'matched_center_px_max':MATCH_MAX_PX,
                            'median_reprojection_px_max':MEDIAN_REPROJ_MAX_PX,'max_reprojection_px_max':MAX_REPROJ_MAX_PX,
                            'ball_height_cm':[BALL_Z_MIN_CM,BALL_Z_MAX_CM],'rim_distance_cm_max':RIM_DISTANCE_MAX_CM,
                            'nearest_wrist_distance_cm_max':WRIST_DISTANCE_MAX_CM,
                            'image_dy_relative_rim_px':[IMAGE_DY_MIN_PX,IMAGE_DY_MAX_PX],
                            'neighbor_frames_localization_evidence_only':True,'no_neighbor_exact_state_mixing':True},
        'body_gate_policy':'unchanged v33f/v33e body reprojection + anatomy + leave-one-camera-out gates',
        'chosen_state':None if not passed else {'rank_in_full_exact_population':int(chosen_tuple[0]),
             'rels':{c:int(chosen['rels'][c]) for c in CAMS},'frames':chosen['frames'],'v33e_score':float(chosen['score']),
             'body':body_json,'ball':chosen_ball},
        'passing_state_count':len(passing),'state_diagnostics':state_diag,
        'numerical_body_ball_gate_passed':bool(passed),'external_visual_qa_required':True,
        'freeview_render_unlocked':False,
        'render_lock_reason':'await source/overlay visual QA of v33h selected ball' if passed else 'no hardened real two-view ball state passed'
    }
    (args.out/'v33h_body_ball_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps({'status':qa['status'],'v33e_integrity':qa['v33e_integrity'],'passing_state_count':len(passing),
                      'chosen_state':qa['chosen_state'],'freeview_render_unlocked':False},indent=2),flush=True)
    if not passed: raise SystemExit(8)

if __name__=='__main__': main()
