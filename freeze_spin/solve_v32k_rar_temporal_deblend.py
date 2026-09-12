from __future__ import annotations

"""v32k: temporal deblend of Steven Adams in the Right-Above-Rim camera.

The exact freeze (t+00) has a severe player overlap.  Single-frame keypoint
models merge Adams and the Utah defender.  This diagnostic uses ONLY real source
frames from the synchronized B32 RAR burst:

1. RF-DETR detects Adams independently in clearer t+03..t+06 frames.
2. Each accepted source pose is tracked backward through the real RAR frames
   with sparse pyramidal Lucas-Kanade and a forward/backward consistency check.
3. A t+00 joint is accepted only when >=2 independent temporal anchors agree.
4. Broadcast t+00 remains a direct RF-DETR observation.
5. Broadcast + temporally reconstructed RAR joints are triangulated with the
   accepted metric cameras; Left-Above-Rim is validation only.

No image synthesis, frame interpolation, player rendering, inpainting, camera
motion or generated RGB occurs here.  Temporal optical flow is used only as a
measurement/tracking operator on real source pixels.  Occluded/unstable joints
are dropped rather than invented.
"""

import argparse, json, math
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview

from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

LAR, BCAST, RAR = v32j.LAR, v32j.BCAST, v32j.RAR
CAMS = v32j.CAMS
W, H = v32j.W, v32j.H
NAMES = v32j.NAMES
DRAW = v32j.DRAW
BONES = v32j.BONES
ANCHOR_RELS = (3, 4, 5, 6)


def burst_path(stage: Path, label: str, rel: int) -> Path:
    xs = sorted((stage / 'burst' / label.replace(' ', '_')).glob(f'rel{rel:+03d}_frame*.png'))
    if len(xs) != 1:
        raise RuntimeError(f'expected one {label} rel{rel:+d} frame, found {xs}')
    return xs[0]


def patch_darkness(img: np.ndarray, xy, r: int = 7) -> float:
    x, y = np.rint(xy).astype(int)
    x1, x2 = max(0, x-r), min(img.shape[1], x+r+1)
    y1, y2 = max(0, y-r), min(img.shape[0], y+r+1)
    p = img[y1:y2, x1:x2]
    if not p.size:
        return 0.0
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[...,2] < 115))


def torso_dark_score(img: np.ndarray, det: dict) -> float:
    # shoulders + hips are the most useful jersey/shorts identity points.
    js = [5, 6, 11, 12]
    vals = [patch_darkness(img, det['xy'][j], 8) for j in js if det['conf'][j] >= .20]
    return float(np.mean(vals)) if vals else 0.0


def select_adams_detection(img: np.ndarray, detections, target_center: np.ndarray):
    rows = []
    for i, d in enumerate(detections):
        c = np.array([(d['box'][0]+d['box'][2])/2, (d['box'][1]+d['box'][3])/2], float)
        dist = float(np.linalg.norm(c-target_center))
        dark = v32j.dark_fraction(img, d['box'])
        td = torso_dark_score(img, d)
        confident = int(np.sum(d['conf'] >= .20))
        # strong identity preference for dark Houston uniform near the known action region.
        score = 2.6*dark + 2.2*td + .30*d['det_conf'] + .035*confident - .0060*dist
        rows.append({'index': i, 'score': float(score), 'dark_fraction': float(dark),
                     'torso_dark_score': float(td), 'confident_joints': confident,
                     'center_distance_px': dist})
    rows.sort(key=lambda x: x['score'], reverse=True)
    return (rows[0]['index'] if rows else None), rows


def track_back(frames: dict[int, np.ndarray], start_rel: int, xy: np.ndarray, conf: np.ndarray):
    pts = xy.astype(np.float32).reshape(-1,1,2).copy()
    valid = (conf >= .20).astype(bool)
    trace = {start_rel: {'xy': xy.tolist(), 'valid': valid.astype(int).tolist()}}
    lk = dict(winSize=(31,31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 40, 0.01),
              minEigThreshold=1e-5)
    for rel in range(start_rel, 0, -1):
        g1 = cv2.cvtColor(frames[rel], cv2.COLOR_BGR2GRAY)
        g0 = cv2.cvtColor(frames[rel-1], cv2.COLOR_BGR2GRAY)
        prev, st1, err1 = cv2.calcOpticalFlowPyrLK(g1, g0, pts, None, **lk)
        back, st2, err2 = cv2.calcOpticalFlowPyrLK(g0, g1, prev, None, **lk)
        fb = np.linalg.norm(back.reshape(-1,2)-pts.reshape(-1,2), axis=1)
        st = st1.reshape(-1).astype(bool) & st2.reshape(-1).astype(bool)
        inside = (prev[:,0,0] >= 0) & (prev[:,0,0] < W) & (prev[:,0,1] >= 0) & (prev[:,0,1] < H)
        good = valid & st & inside & (fb <= 2.5) & (err1.reshape(-1) <= 35.0)
        pts = prev
        valid = good
        trace[rel-1] = {'xy': pts.reshape(-1,2).tolist(), 'valid': valid.astype(int).tolist(),
                        'fb_px': fb.tolist(), 'lk_err': err1.reshape(-1).tolist()}
    return pts.reshape(-1,2).astype(float), valid, trace


def consensus_tracks(track_rows):
    out = {}
    audit = {}
    for j in range(17):
        pts=[]; anchors=[]
        for row in track_rows:
            if row['valid'][j]:
                pts.append(row['t0_xy'][j]); anchors.append(row['rel'])
        if len(pts) < 2:
            audit[j]={'accepted':False,'support':len(pts),'anchors':anchors}
            continue
        a=np.asarray(pts,float)
        med=np.median(a,axis=0)
        ds=np.linalg.norm(a-med,axis=1)
        medspread=float(np.median(ds)); p90=float(np.percentile(ds,90))
        accepted=bool(medspread <= 7.0 and p90 <= 14.0)
        audit[j]={'accepted':accepted,'support':len(pts),'anchors':anchors,
                  'median_xy':med.tolist(),'median_spread_px':medspread,'p90_spread_px':p90,
                  'samples':[p.tolist() for p in a]}
        if accepted: out[j]=med
    return out,audit


def draw_pose(img, det=None, pts=None, valid=None, title=''):
    out=img.copy()
    if det is not None:
        for a,b in DRAW:
            if det['conf'][a]>=.20 and det['conf'][b]>=.20:
                cv2.line(out, tuple(np.rint(det['xy'][a]).astype(int)), tuple(np.rint(det['xy'][b]).astype(int)), (0,255,255), 2, cv2.LINE_AA)
        for j in range(17):
            if det['conf'][j]>=.20: cv2.circle(out, tuple(np.rint(det['xy'][j]).astype(int)), 3, (0,255,255), -1, cv2.LINE_AA)
    if pts is not None:
        for a,b in DRAW:
            if a in pts and b in pts:
                cv2.line(out, tuple(np.rint(pts[a]).astype(int)), tuple(np.rint(pts[b]).astype(int)), (255,255,0), 2, cv2.LINE_AA)
        for j,p in pts.items(): cv2.circle(out, tuple(np.rint(p).astype(int)), 4, (255,255,0), -1, cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,28),(0,0,0),-1)
    cv2.putText(out,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--b32-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    stage=args.b32_root/'stage_a'
    scene=json.loads((stage/'v32_scene_manifest.json').read_text()); q=json.loads((stage/'v32_quality_cluster.json').read_text())
    cams={c:v32j.cam(scene,c) for c in CAMS}
    model=RFDETRKeypointPreview()

    # RAR real frames t+00..t+06.
    rar_frames={rel:cv2.imread(str(burst_path(stage,RAR,rel))) for rel in range(0,7)}
    assert all(x is not None for x in rar_frames.values())
    rb=np.asarray(q['focal_player']['observations'][RAR]['bbox_xyxy'],float)
    target_center=np.array([(rb[0]+rb[2])/2,(rb[1]+rb[3])/2],float)

    tracks=[]; anchor_audit={}
    for rel in ANCHOR_RELS:
        dets=v32j.infer(model,rar_frames[rel])
        idx,rows=select_adams_detection(rar_frames[rel],dets,target_center)
        if idx is None: continue
        d=dets[idx]
        t0,valid,trace=track_back(rar_frames,rel,d['xy'],d['conf'])
        tracks.append({'rel':rel,'index':idx,'det':d,'t0_xy':t0,'valid':valid,'trace':trace})
        anchor_audit[str(rel)]={'selected_index':idx,'ranked_candidates':rows,'valid_t0_joint_count':int(np.sum(valid))}
        cv2.imwrite(str(args.out/f'v32k_rar_anchor_rel{rel:+03d}.png'),draw_pose(rar_frames[rel],det=d,title=f'v32k RAR anchor t{rel:+d} | yellow RF-DETR'))

    consensus,audit=consensus_tracks(tracks)
    cv2.imwrite(str(args.out/'v32k_rar_t00_temporal_consensus.png'),draw_pose(rar_frames[0],pts=consensus,title='v32k RAR t+00 | cyan temporal consensus from real +3..+6 frames'))

    # Broadcast direct observation, anchored by known B32 focal box.
    bimg=cv2.imread(str(stage/'v32_chosen_Broadcast_frame0276.png')); limg=cv2.imread(str(stage/'v32_chosen_Left_Above_Rim_frame0260.png'))
    assert bimg is not None and limg is not None
    bdets=v32j.infer(model,bimg); bb=np.asarray(q['focal_player']['observations'][BCAST]['bbox_xyxy'],float)
    brows=[]
    for i,d in enumerate(bdets): brows.append((4*v32j.iou(d['box'],bb)+.5*v32j.dark_fraction(bimg,d['box']),i))
    brows.sort(reverse=True); bi=brows[0][1]; b=bdets[bi]

    # Metric triangulation uses only temporally supported RAR joints + direct Broadcast joints.
    joints={}; per_joint={}
    for j,ruv in consensus.items():
        if b['conf'][j] < .20: continue
        X=v32j.triangulate_rays(cams,{BCAST:b['xy'][j],RAR:ruv})
        if X is None or not (-350<=X[0]<=1250 and -750<=X[1]<=750 and -60<=X[2]<=450): continue
        eb=np.linalg.norm(v32j.project(cams[BCAST],X)-b['xy'][j])
        er=np.linalg.norm(v32j.project(cams[RAR],X)-ruv)
        joints[j]=X; per_joint[j]={'broadcast_px':float(eb),'rar_px':float(er)}

    bones,nb,nok,bfrac=v32j.bone_stats(joints)
    repro=[e for r in per_joint.values() for e in (r['broadcast_px'],r['rar_px'])]
    medres=float(np.median(repro)) if repro else 999.

    # Left remains independent validation only: best measured pose by projected-joint residual.
    ldets=v32j.infer(model,limg); lrows=[]
    for i,l in enumerate(ldets):
        es=[]
        for j,X in joints.items():
            if l['conf'][j] < .20: continue
            uv=v32j.project(cams[LAR],X)
            if uv is not None: es.append(float(np.linalg.norm(uv-l['xy'][j])))
        med=float(np.median(es)) if len(es)>=4 else 999.; p75=float(np.percentile(es,75)) if len(es)>=4 else 999.
        lrows.append({'index':i,'joint_count':len(es),'median_px':med,'p75_px':p75,'cost':med+.2*p75})
    lrows.sort(key=lambda x:x['cost']); li=None
    if lrows and lrows[0]['joint_count']>=4 and lrows[0]['median_px']<=32 and lrows[0]['p75_px']<=50: li=lrows[0]['index']
    l=None if li is None else ldets[li]

    # Draw t0 views with reconstructed metric joints in magenta.
    def draw_metric(img, label, measured=None):
        out=img.copy()
        if measured is not None:
            for a,bn in DRAW:
                if measured['conf'][a]>=.20 and measured['conf'][bn]>=.20:
                    cv2.line(out,tuple(np.rint(measured['xy'][a]).astype(int)),tuple(np.rint(measured['xy'][bn]).astype(int)),(0,255,255),2,cv2.LINE_AA)
        for a,bn in DRAW:
            if a in joints and bn in joints:
                ua=v32j.project(cams[label],joints[a]); ub=v32j.project(cams[label],joints[bn])
                if ua is not None and ub is not None: cv2.line(out,tuple(np.rint(ua).astype(int)),tuple(np.rint(ub).astype(int)),(255,0,255),2,cv2.LINE_AA)
        for j,X in joints.items():
            uv=v32j.project(cams[label],X)
            if uv is not None: cv2.circle(out,tuple(np.rint(uv).astype(int)),4,(255,0,255),-1,cv2.LINE_AA)
        cv2.rectangle(out,(0,0),(W,28),(0,0,0),-1); cv2.putText(out,f'v32k {label} | yellow measured | magenta metric pose',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
        return out

    ovs=[draw_metric(limg,LAR,l),draw_metric(bimg,BCAST,b),draw_metric(rar_frames[0],RAR,None)]
    for label,ov in zip(CAMS,ovs): cv2.imwrite(str(args.out/f'v32k_{label.replace(" ","_")}.png'),ov)
    cv2.imwrite(str(args.out/'v32k_three_camera_t00_montage.png'),np.hstack(ovs))

    accepted_consensus=len(consensus)
    support_counts=[row.get('support',0) for row in audit.values() if row.get('accepted')]
    left_best=lrows[0] if lrows else {'joint_count':0,'median_px':999.,'p75_px':999.}
    status='PASS_V32K_TEMPORAL_DEBLEND_OBSERVATION' if (accepted_consensus>=6 and len(joints)>=6 and nb>=4 and bfrac>=.70 and medres<=16.) else 'FAIL_CLOSED_V32K_TEMPORAL_DEBLEND'
    qa={
      'version':'v32k_rar_temporal_deblend','status':status,'native_resolution':[W,H],
      'source_frames_only':True,'generated_rgb':False,'mesh_rendered':False,'camera_geometry_modified':False,
      'temporal_method':'RF-DETR anchors at real RAR t+03..t+06; sparse LK backward tracking; forward/backward consistency; multi-anchor consensus',
      'optical_flow_role':'measurement/tracking only; never used to synthesize output pixels or intermediate views',
      'anchor_audit':anchor_audit,'consensus_joint_count':accepted_consensus,
      'consensus_joint_audit':{NAMES[j]:row for j,row in audit.items()},
      'broadcast_selected_index':bi,'left_selected_index':li,'left_candidates':lrows,
      'triangulated_joint_count':len(joints),'median_two_view_reprojection_px':medres,
      'measured_bone_count':nb,'bone_plausible_fraction':bfrac,'bone_details':bones,
      'left_best_validation':left_best,
      'joints':{NAMES[j]:{'world_cm':X.tolist(),'rar_t00_xy':consensus[j].tolist(),'residuals_px':per_joint[j]} for j,X in joints.items()},
      'gate':{'rar_consensus_ge_6':accepted_consensus>=6,'triangulated_ge_6':len(joints)>=6,'measured_bones_ge_4':nb>=4,
              'bone_fraction_ge_0_70':bfrac>=.70,'median_reprojection_le_16px':medres<=16.,
              'left_is_validation_only':True,'surface_stage_unlocked':status.startswith('PASS_')}
    }
    (args.out/'v32k_temporal_deblend_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps({'status':status,'consensus_joints':accepted_consensus,'triangulated_joints':len(joints),'median_reproj':medres,'bone_fraction':bfrac,'left_best':left_best},indent=2))
    if not status.startswith('PASS_'): raise SystemExit(3)


if __name__=='__main__': main()
