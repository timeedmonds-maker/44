#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kd_double_team_ballhandler_onnx import YoloOnnx, extract_frames, players_on_courtish
from adams_screen_temporal_onnx import associate, cluster_tracks, by_tid, norm_dist
from adams_screen_temporal_courtgate import raw_court_polygons, polygon_gate, keep_two_player_clusters

ROLE_COLORS = {
    'ballhandler': (255, 255, 255),
    'screener': (0, 255, 255),
    'screened_defender': (255, 0, 255),
    'screener_defender': (0, 165, 255),
}


def nearest_opponent(boxes, labels, subject_tid, own_label):
    if subject_tid not in boxes:
        return None
    s = boxes[subject_tid]
    h = max(1.0, s[3] - s[1])
    best = None
    for tid, box in boxes.items():
        if tid == subject_tid:
            continue
        lab = labels.get(tid)
        if lab is None or lab == own_label:
            continue
        d = norm_dist(s, box, h)
        if best is None or d < best[0]:
            best = (d, tid)
    return None if best is None else int(best[1])


def draw_frame(frame, boxes, roles, hull=None, title=''):
    out = frame.copy()
    if hull is not None:
        cv2.polylines(out, [hull.astype(np.int32)], True, (80, 220, 80), 2, cv2.LINE_AA)
    for role, tid in roles.items():
        if tid is None or tid not in boxes:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[tid][:4]]
        col = ROLE_COLORS[role]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
        cv2.putText(out, f'{role} T{tid}', (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2, cv2.LINE_AA)
    if title:
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .60, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def pad_to(im, w, h):
    canvas = np.full((h, w, 3), 15, np.uint8)
    ih, iw = im.shape[:2]
    s = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw*s))), max(1, int(round(ih*s)))
    rs = cv2.resize(im, (nw, nh))
    x = (w - nw) // 2; y = (h - nh) // 2
    canvas[y:y+nh, x:x+nw] = rs
    return canvas


def make_strip(frames, out_path):
    if not frames:
        return
    target_h = 360
    tiles = []
    for im in frames:
        h, w = im.shape[:2]
        tw = int(round(target_h * w / max(h, 1)))
        tiles.append(pad_to(im, tw, target_h))
    strip = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(out_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 94])


def crop_body(frame, box, expand=.10):
    x1, y1, x2, y2 = map(float, box[:4])
    w = x2-x1; h = y2-y1
    x1 -= expand*w; x2 += expand*w; y1 -= .05*h; y2 += .03*h
    ih, iw = frame.shape[:2]
    xa=max(0,int(x1)); ya=max(0,int(y1)); xb=min(iw,int(x2)); yb=min(ih,int(y2))
    c=frame[ya:yb,xa:xb]
    return c if c.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequences', required=True)
    ap.add_argument('--possessions', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--onnx-model', required=True)
    ap.add_argument('--court-model', required=True)
    ap.add_argument('--nbacv-src', required=True)
    ap.add_argument('--sample-fps', type=float, default=6.0)
    ap.add_argument('--max-seconds', type=float, default=16.0)
    ap.add_argument('--target-hls-width', type=int, default=960)
    ap.add_argument('--person-conf', type=float, default=.14)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    strips = args.out/'strips'; bodies=args.out/'bodies'; frames_out=args.out/'frames'
    strips.mkdir(exist_ok=True); bodies.mkdir(exist_ok=True); frames_out.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(args.nbacv_src)))

    from ultralytics import YOLO
    court_model = YOLO(args.court_model)
    person_model = YoloOnnx(args.onnx_model)

    seq = pd.read_csv(args.sequences, dtype={'game_id':str})
    poss = pd.read_csv(args.possessions, dtype={'game_id':str})
    for d in (seq, poss): d['game_id'] = d.game_id.str.zfill(10)
    seq = seq.merge(poss[['possession_uid','period','start_time','end_time','pts_poss','type_end','lineup_team','lineup_opp']],
                    on='possession_uid', how='left', suffixes=('','_poss'))
    seq['candidate_id'] = [f"S{i+1:03d}" for i in range(len(seq))]
    manifest=[]

    with tempfile.TemporaryDirectory(prefix='adams_screen_audit_') as td:
        root=Path(td)
        for ev, grp in seq.groupby('event_num', sort=False):
            ev=int(ev); evdir=root/str(ev)
            try:
                meta=extract_frames('0022500001', ev, evdir, args.sample_fps,
                                    args.max_seconds, args.target_hls_width)
                paths=meta['frames']; people_pf=[]
                for pth in paths:
                    fr=cv2.imread(str(pth))
                    people_pf.append([] if fr is None else players_on_courtish(
                        person_model.detect_class(fr,0,args.person_conf), fr.shape[0], max_players=18))
                hulls, raw_cov, bridged_cov = raw_court_polygons(paths,court_model)
                gated,_stats=polygon_gate(people_pf,hulls,paths)
                tracked=associate(gated)
                labels,_centers=cluster_tracks(paths,tracked)
                tracked,keep_labs,cluster_mass=keep_two_player_clusters(tracked,labels)
                boxes_pf=[by_tid(ds) for ds in tracked]

                for _, r in grp.iterrows():
                    cid=str(r.candidate_id)
                    start=int(r.start_sample); end=int(r.end_sample)
                    bh=int(r.ballhandler_tid); sc=int(r.screener_tid); sd=int(r.defender_tid)
                    pre=max(0,start-max(2,int(round(.75*args.sample_fps))))
                    pre_boxes=boxes_pf[min(pre,len(boxes_pf)-1)] if boxes_pf else {}
                    own_lab=labels.get(sc,labels.get(bh))
                    scdef=nearest_opponent(pre_boxes,labels,sc,own_lab) if own_lab is not None else None
                    roles={'ballhandler':bh,'screener':sc,'screened_defender':sd,'screener_defender':scdef}
                    picks=[]
                    for f in [pre,start,(start+end)//2,end,min(len(paths)-1,end+max(2,int(round(.75*args.sample_fps))))]:
                        if f not in picks and 0 <= f < len(paths): picks.append(f)
                    drawn=[]
                    for f in picks:
                        fr=cv2.imread(str(paths[f]));
                        if fr is None: continue
                        title=f'{cid}  event {ev}  t={f/args.sample_fps:.2f}s'
                        ann=draw_frame(fr,boxes_pf[f],roles,hulls[f] if f<len(hulls) else None,title)
                        cv2.imwrite(str(frames_out/f'{cid}_f{f:03d}.jpg'),ann,[cv2.IMWRITE_JPEG_QUALITY,94])
                        drawn.append(ann)
                    make_strip(drawn,strips/f'{cid}.jpg')

                    # Save full-body context crops for each role from best available picked frame.
                    crop_counts={}
                    for role,tid in roles.items():
                        n=0
                        if tid is not None:
                            od=bodies/cid/role; od.mkdir(parents=True,exist_ok=True)
                            for f in picks:
                                if f>=len(paths) or tid not in boxes_pf[f]: continue
                                fr=cv2.imread(str(paths[f]));
                                if fr is None: continue
                                c=crop_body(fr,boxes_pf[f][tid])
                                if c is not None:
                                    cv2.imwrite(str(od/f'f{f:03d}.jpg'),c,[cv2.IMWRITE_JPEG_QUALITY,96]); n+=1
                        crop_counts[f'{role}_body_crops']=n
                    manifest.append({
                        'candidate_id':cid,'possession_uid':r.possession_uid,'period':r.period,
                        'start_time':r.start_time,'end_time':r.end_time,'pts_poss':r.pts_poss,'type_end':r.type_end,
                        'event_num':ev,'screen_start_s':round(start/args.sample_fps,3),'screen_end_s':round(end/args.sample_fps,3),
                        'ballhandler_tid':bh,'screener_tid':sc,'screened_defender_tid':sd,'screener_defender_tid':scdef,
                        'lineup_team':r.lineup_team,'lineup_opp':r.lineup_opp,
                        'angle':meta.get('angle'),'raw_polygon_coverage':round(raw_cov,3),'bridged_polygon_coverage':round(bridged_cov,3),
                        'player_cluster_labels':'|'.join(map(str,sorted(keep_labs))),
                        'cluster_mass':json.dumps(cluster_mass,sort_keys=True),
                        'strip_path':str(strips/f'{cid}.jpg'),**crop_counts
                    })
            except Exception as e:
                for _,r in grp.iterrows():
                    manifest.append({'candidate_id':str(r.candidate_id),'event_num':ev,
                                     'error':f'{type(e).__name__}: {e}'})
            finally:
                shutil.rmtree(evdir,ignore_errors=True)

    m=pd.DataFrame(manifest)
    m.to_csv(args.out/'visual_audit_manifest.csv',index=False)
    qa={'candidates':int(len(seq)),'manifest_rows':int(len(m)),
        'error_rows':int(m.get('error',pd.Series('',index=m.index)).fillna('').astype(str).ne('').sum()),
        'note':'Full-context visual audit. Role boxes are CV-derived; player names remain unassigned until visual/jersey/lineup verification.'}
    (args.out/'qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps(qa,indent=2))

if __name__=='__main__': main()
