#!/usr/bin/env python3
"""Production Adams-screen POC v3.

Changes from v2:
- operates on an interaction window instead of the entire ~20 s event clip;
- conservative offline tracklet stitching after BoT-SORT;
- separate low-threshold RF-DETR pass for jersey-number regions;
- multi-scale (640/960/1280) court-keypoint inference with temporal pooling;
- preserves the repo-standard deterministic UHD presentation render.

All role names/metric court outputs remain abstaining unless supported by
explicit overrides / accepted calibration. No screen/defender identity is
fabricated from PBP alone.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import adams_screen_prod_poc as base
from trackers import BoTSORTTracker as _BoTSORTTracker


def center_bottom(box):
    return np.array([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)


def box_h(box):
    return max(1.0, float(box[3] - box[1]))


class TimeConsistentBoTSORT(_BoTSORTTracker):
    search_space = {}

    def __init__(self, analysis_fps: float, **kwargs):
        self._analysis_fps = float(analysis_fps)
        self._sample_index = 0
        super().__init__(frame_rate=self._analysis_fps, **kwargs)

    def update(self, detections, frame=None, timestamp=None):
        if timestamp is None:
            timestamp = self._sample_index / self._analysis_fps
        self._sample_index += 1
        return super().update(detections, frame=frame, timestamp=float(timestamp))


def conservative_stitch(track_rows, track_feats, team_map, fps):
    """Join only very high-confidence non-overlapping track fragments.

    The aim is not to force ten identities; it is to remove obvious short
    re-entry fragments while preserving abstention when association is unclear.
    """
    by = collections.defaultdict(list)
    for r in track_rows:
        tid = int(r['track_id'])
        if tid >= 0:
            by[tid].append(r)
    info = {}
    for tid, rows in by.items():
        rows = sorted(rows, key=lambda r: int(r['frame']))
        feats = track_feats.get(tid, [])
        info[tid] = {
            'start': int(rows[0]['frame']), 'end': int(rows[-1]['frame']),
            'first': tuple(float(rows[0][k]) for k in ('x1','y1','x2','y2')),
            'last': tuple(float(rows[-1][k]) for k in ('x1','y1','x2','y2')),
            'height': float(np.median([float(r['y2'])-float(r['y1']) for r in rows])),
            'feat': (np.median(np.asarray(feats), axis=0) if feats else None),
            'team': team_map.get(tid),
        }

    candidates = []
    tids = sorted(info)
    for a in tids:
        A = info[a]
        scores = []
        for b in tids:
            if a == b:
                continue
            B = info[b]
            gap_s = (B['start'] - A['end']) / float(fps)
            if not (0.0 < gap_s <= 0.85):
                continue
            if A['team'] is not None and B['team'] is not None and A['team'] != B['team']:
                continue
            h = max(20.0, 0.5 * (A['height'] + B['height']))
            ratio = B['height'] / max(A['height'], 1.0)
            if not (0.62 <= ratio <= 1.62):
                continue
            nd = float(np.linalg.norm(center_bottom(A['last']) - center_bottom(B['first'])) / h)
            if nd > 1.35:
                continue
            cd = 0.0
            if A['feat'] is not None and B['feat'] is not None:
                cd = float(np.linalg.norm(A['feat'] - B['feat']))
                if cd > 28.0:
                    continue
            score = nd + 0.025 * cd + 0.45 * gap_s
            scores.append((score, b))
        scores.sort()
        if scores and scores[0][0] < 1.55:
            # Require a clear best continuation when another candidate exists.
            if len(scores) == 1 or scores[0][0] <= 0.78 * scores[1][0]:
                candidates.append((scores[0][0], a, scores[0][1]))

    parent = {t:t for t in tids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    used_succ = set()
    links = []
    for score,a,b in sorted(candidates):
        if b in used_succ:
            continue
        # Prevent linking into a group that overlaps in time.
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        union(a,b); used_succ.add(b); links.append({'from':a,'to':b,'score':round(score,3)})

    groups = collections.defaultdict(list)
    for t in tids:
        groups[find(t)].append(t)
    # Stable compact IDs ordered by earliest appearance.
    ordered = sorted(groups.values(), key=lambda g: min(info[t]['start'] for t in g))
    remap = {}
    for gid, group in enumerate(ordered):
        for t in group:
            remap[t] = gid
    return remap, links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default='0022500001')
    ap.add_argument('--event', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--detector-onnx', type=Path, required=True)
    ap.add_argument('--object-eval-src', type=Path, required=True)
    ap.add_argument('--court-model', type=Path, required=True)
    ap.add_argument('--nbacv-src', type=Path, required=True)
    ap.add_argument('--overrides', type=Path)
    ap.add_argument('--xfg-row-json', type=Path)
    ap.add_argument('--analysis-fps', type=float, default=10.0)
    ap.add_argument('--court-fps', type=float, default=2.0)
    ap.add_argument('--number-fps', type=float, default=3.0)
    ap.add_argument('--start-s', type=float, default=0.0)
    ap.add_argument('--end-s', type=float)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(a.object_eval_src)); sys.path.insert(0, str(a.nbacv_src))
    from object_detection_eval.inference.detectors.rfdetr import RFDETRDetector
    from ultralytics import YOLO
    from nbacv.court import _court_infer, calibrate_video, project_point

    source = a.out/'source_native.mp4'
    meta = base.fetch_native_clip(a.game, a.event, source, 960)
    ctx = base.load_context(a.game, a.event, a.out, a.xfg_row_json)
    (a.out/'context.json').write_text(json.dumps(ctx, indent=2, default=str))

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); duration = n_all / fps if fps else 0
    start_s = max(0.0, float(a.start_s)); end_s = min(duration, float(a.end_s)) if a.end_s is not None else duration
    if end_s <= start_s:
        raise SystemExit('invalid interaction window')
    sf = max(0, int(round(start_s * fps))); ef = min(n_all, int(round(end_s * fps)))
    n = max(0, ef - sf)

    player_detector = RFDETRDetector(a.detector_onnx, base.LABEL_MAP, confidence_threshold=.20,
                                     num_select=300, input_height=640, input_width=640)
    number_detector = RFDETRDetector(a.detector_onnx, base.LABEL_MAP, confidence_threshold=.045,
                                     num_select=300, input_height=640, input_width=640)
    tracker = TimeConsistentBoTSORT(
        analysis_fps=max(.25,a.analysis_fps), enable_cmc=True, cmc_method='sparseOptFlow',
        track_activation_threshold=.30, high_conf_det_threshold=.25)
    court_model = YOLO(str(a.court_model))

    every = max(1, int(round(fps/max(a.analysis_fps,.25))))
    number_every = max(every, int(round(fps/max(a.number_fps,.25))))
    court_every = max(every, int(round(fps/max(a.court_fps,.25))))

    tracked_frames=[]; ball_frames=[]; track_rows=[]; number_rows=[]
    track_feats=collections.defaultdict(list); court_kp={}; court_scale_rows=[]
    good_court_size=640

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    rel=0
    while rel < n:
        ok, fr = cap.read()
        if not ok: break
        if rel % every != 0:
            rel += 1; continue
        dets = player_detector.predict(fr)
        players = base.collapse_players(base.to_sv(dets,W,H,base.PLAYER_CLASSES))
        tracked = tracker.update(players, fr, timestamp=rel/fps)
        td={}
        if len(tracked):
            for box,tid,cf in zip(tracked.xyxy,tracked.tracker_id,tracked.confidence):
                tid=int(tid)
                if tid < 0:
                    continue
                td[tid]=tuple(map(float,box))
                feat=base.jersey_feature(fr,box)
                if feat is not None: track_feats[tid].append(feat)
                track_rows.append({'frame':rel,'time_s':start_s+rel/fps,'track_id':tid,
                                   'x1':box[0],'y1':box[1],'x2':box[2],'y2':box[3],'conf':float(cf)})
        tracked_frames.append(td)
        balls=base.to_sv(dets,W,H,base.BALL_CLASSES); bxs=[]
        if len(balls):
            order=np.argsort(-(balls.confidence if balls.confidence is not None else np.ones(len(balls))))
            for j in order[:2]: bxs.append(tuple(map(float,balls.xyxy[j])))
        ball_frames.append(bxs)

        if rel % number_every == 0:
            ndets=number_detector.predict(fr)
            nums=base.to_sv(ndets,W,H,{base.NUMBER_CLASS})
            if len(nums):
                for box,cf in zip(nums.xyxy,nums.confidence):
                    best=None
                    for tid,pbox in td.items():
                        s=base.ios_number_player(box,pbox)
                        if best is None or s>best[0]: best=(s,tid,pbox)
                    if best and best[0] >= .55:
                        x1,y1,x2,y2=map(int,box)
                        crop=fr[max(0,y1-4):min(H,y2+4),max(0,x1-4):min(W,x2+4)]
                        cpath=a.out/f'num_f{rel:05d}_t{best[1]}.jpg'
                        if crop.size: cv2.imwrite(str(cpath),crop)
                        number_rows.append({'frame':rel,'time_s':start_s+rel/fps,'track_id':best[1],
                                            'ios':best[0],'conf':float(cf),'crop':cpath.name})

        if rel % court_every == 0:
            kp=None; attempted=[]
            for sz in [good_court_size] + [s for s in (640,960,1280) if s != good_court_size]:
                attempted.append(sz)
                cand=_court_infer(court_model,fr,sz,'cpu')
                if cand is not None and int((cand[1] >= .5).sum()) >= 6:
                    kp=cand; good_court_size=sz; break
                if kp is None: kp=cand
            court_kp[rel]=kp
            court_scale_rows.append({'frame':rel,'time_s':start_s+rel/fps,'chosen_or_last_size':good_court_size,
                                     'attempted':'|'.join(map(str,attempted)),
                                     'visible_kp':0 if kp is None else int((kp[1]>=.5).sum())})
        rel += 1
    cap.release()

    raw_team=base.cluster_teams(track_feats)
    remap, stitch_links=conservative_stitch(track_rows,track_feats,raw_team,fps)
    if remap:
        for r in track_rows:
            if int(r['track_id']) in remap: r['raw_track_id']=int(r['track_id']); r['track_id']=remap[int(r['track_id'])]
        for r in number_rows:
            if int(r['track_id']) in remap: r['raw_track_id']=int(r['track_id']); r['track_id']=remap[int(r['track_id'])]
        new_feats=collections.defaultdict(list)
        for raw,fs in track_feats.items():
            gid=remap.get(raw)
            if gid is not None: new_feats[gid].extend(fs)
        track_feats=new_feats
    team_map=base.cluster_teams(track_feats)

    # Rebuild sampled-frame dictionaries after stitching.
    sample_frames=sorted(set(int(r['frame']) for r in track_rows))
    td_by_frame=collections.defaultdict(dict)
    for r in track_rows:
        td_by_frame[int(r['frame'])][int(r['track_id'])]=(float(r['x1']),float(r['y1']),float(r['x2']),float(r['y2']))
    tracked_frames=[td_by_frame[f] for f in sample_frames]

    overrides=json.loads(a.overrides.read_text()) if a.overrides and a.overrides.exists() else {}
    # ball_frames correspond to analysis samples in chronological order.
    roles=base.resolve_roles(tracked_frames,ball_frames[:len(tracked_frames)],team_map,overrides)
    (a.out/'roles.json').write_text(json.dumps({'roles':roles,'team_map':team_map,'overrides':overrides,
                                                'stitch_links':stitch_links,'raw_to_stitched':remap},indent=2))

    calib=calibrate_video(court_kp, window=court_every*2, frame_hw=(H,W)) if court_kp else {}
    court_frames=[]
    for f in sorted(court_kp):
        rec=calib.get(f, {'H':None,'reason':'missing_calibration'})
        court_frames.append({'frame':f, **rec})
    court_by_frame={int(x['frame']):x['H'] for x in court_frames if x.get('H') is not None}

    pd.DataFrame(track_rows).to_csv(a.out/'tracks.csv',index=False)
    pd.DataFrame(number_rows).to_csv(a.out/'number_evidence.csv',index=False)
    pd.DataFrame(court_scale_rows).to_csv(a.out/'court_scale_attempts.csv',index=False)
    (a.out/'court_frames.json').write_text(json.dumps(court_frames,indent=2))

    # Number evidence contact sheet.
    crops=[]
    for r in number_rows[:72]:
        im=cv2.imread(str(a.out/r['crop']))
        if im is None: continue
        im=cv2.resize(im,(120,90),interpolation=cv2.INTER_CUBIC)
        cv2.putText(im,f"T{r['track_id']} {r['conf']:.2f}",(3,15),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
        crops.append(im)
    if crops:
        cols=6; rows=math.ceil(len(crops)/cols); sheet=np.zeros((rows*90,cols*120,3),np.uint8)
        for i,im in enumerate(crops): sheet[(i//cols)*90:(i//cols+1)*90,(i%cols)*120:(i%cols+1)*120]=im
        cv2.imwrite(str(a.out/'number_contact_sheet.jpg'),sheet)

    by_tid=collections.defaultdict(dict)
    for r in track_rows: by_tid[int(r['track_id'])][int(r['frame'])]=(r['x1'],r['y1'],r['x2'],r['y2'])
    def interp_box(tid, frame_i):
        d=by_tid.get(tid,{})
        if frame_i in d: return d[frame_i]
        ks=sorted(d)
        if not ks: return None
        lo=max([k for k in ks if k<=frame_i],default=None); hi=min([k for k in ks if k>=frame_i],default=None)
        if lo is None or hi is None or hi-lo > every*3: return None
        if lo==hi: return d[lo]
        t=(frame_i-lo)/(hi-lo); return tuple((1-t)*np.asarray(d[lo])+t*np.asarray(d[hi]))

    # Stitched track contact sheet across the interaction window.
    cap=cv2.VideoCapture(str(source)); cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    slots=set(np.linspace(0,max(n-1,0),12,dtype=int).tolist()); panels=[]; rel=0
    while rel<n:
        ok,fr=cap.read()
        if not ok: break
        if rel in slots:
            for tid in sorted(by_tid):
                b=interp_box(tid,rel)
                if b is None: continue
                col=base.color_for(team_map.get(tid),'other')
                cv2.rectangle(fr,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),col,2,cv2.LINE_AA)
                base.label(fr,(b[0],b[1]-3),f'T{tid}',col,.42)
            thumb=cv2.resize(fr,(480,270),interpolation=cv2.INTER_AREA)
            cv2.putText(thumb,f'{start_s+rel/fps:.2f}s  rel {rel/fps:.2f}s',(8,260),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            panels.append(thumb)
        rel+=1
    cap.release()
    if panels:
        cols=3; rows=math.ceil(len(panels)/cols); sheet=np.zeros((rows*270,cols*480,3),np.uint8)
        for i,im in enumerate(panels): sheet[(i//cols)*270:(i//cols+1)*270,(i%cols)*480:(i%cols+1)*480]=im
        cv2.imwrite(str(a.out/'track_contact_sheet.jpg'),sheet)

    def nearest_H(frame_i):
        if not court_by_frame: return None
        k=min(court_by_frame,key=lambda x:abs(x-frame_i))
        return court_by_frame[k] if abs(k-frame_i)<=court_every*2 else None

    out_native=a.out/'adams_screen_overlay_native.mp4'
    cap=cv2.VideoCapture(str(source)); cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    wr=cv2.VideoWriter(str(out_native),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
    traces=collections.defaultdict(lambda:collections.deque(maxlen=45)); rel=0
    initial_sep=None; max_sep=0
    while rel<n:
        ok,fr=cap.read()
        if not ok: break
        current={tid:interp_box(tid,rel) for tid in by_tid}; current={k:v for k,v in current.items() if v is not None}
        court_xy={}; Hm=nearest_H(rel)
        if Hm is not None:
            Hmat=np.asarray(Hm,np.float64)
            for tid,b in current.items():
                try:
                    p=project_point(Hmat,(b[0]+b[2])/2,b[3]); court_xy[tid]=(float(p[0]),float(p[1]))
                except Exception: pass
        metrics={'coverage':'identifying','bh_def_ft':'--','adams_def_ft':'--','advantage':'--'}
        bh=roles.get('ballhandler'); sd=roles.get('screened_defender'); ad=roles.get('adams'); afd=roles.get('adams_defender')
        def dist_ft(aid,bid):
            if aid in court_xy and bid in court_xy: return math.dist(court_xy[aid],court_xy[bid])/30.48
            return None
        d1=dist_ft(bh,sd) if bh is not None and sd is not None else None
        d2=dist_ft(ad,afd) if ad is not None and afd is not None else None
        if d1 is not None:
            metrics['bh_def_ft']=f'{d1:.1f}'; initial_sep=d1 if initial_sep is None else initial_sep
            max_sep=max(max_sep,d1); metrics['advantage']=f'+{max(0,max_sep-initial_sep):.1f} ft'
        if d2 is not None: metrics['adams_def_ft']=f'{d2:.1f}'
        if None not in (bh,sd,ad,afd): metrics['coverage']='tracking'
        for tid,b in current.items():
            role=next((r for r,t in roles.items() if t==tid),'other')
            col=base.color_for(team_map.get(tid),role); thick=4 if role!='other' else 2
            base.ring(fr,b,col,thick)
            cx=(b[0]+b[2])/2; cy=b[3]; traces[tid].append((int(cx),int(cy)))
            pts=list(traces[tid])
            for j in range(1,len(pts)): cv2.line(fr,pts[j-1],pts[j],col,1,cv2.LINE_AA)
            nm=overrides.get('names',{}).get(str(tid)) or overrides.get('names',{}).get(tid)
            if not nm: nm='ADAMS' if role=='adams' else ('BALLHANDLER' if role=='ballhandler' else f'T{tid}')
            base.label(fr,(b[0],b[1]-4),nm,col)
        if bh in current and sd in current:
            p1=(int((current[bh][0]+current[bh][2])/2),int(current[bh][3])); p2=(int((current[sd][0]+current[sd][2])/2),int(current[sd][3]))
            cv2.line(fr,p1,p2,(255,255,255),2,cv2.LINE_AA)
            if d1 is not None: base.label(fr,((p1[0]+p2[0])//2,(p1[1]+p2[1])//2),f'{d1:.1f} ft',(255,255,255),.42)
        if ad in current and afd in current:
            p1=(int((current[ad][0]+current[ad][2])/2),int(current[ad][3])); p2=(int((current[afd][0]+current[afd][2])/2),int(current[afd][3]))
            cv2.line(fr,p1,p2,(100,255,100),2,cv2.LINE_AA)
            if d2 is not None: base.label(fr,((p1[0]+p2[0])//2,(p1[1]+p2[1])//2),f'{d2:.1f} ft',(100,255,100),.42)
        base.draw_panel(fr,ctx,metrics)
        if court_xy: base.mini_court(fr,court_xy,roles,origin=(W-302,H-167))
        wr.write(fr); rel+=1
    cap.release(); wr.release()

    final=a.out/'adams_screen_overlay_h264.mp4'
    duration_out=n/fps
    base.run(['ffmpeg','-y','-v','error','-i',str(out_native),'-ss',f'{start_s:.3f}','-t',f'{duration_out:.3f}','-i',str(source),
              '-map','0:v:0','-map','1:a?','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',
              '-c:a','aac','-shortest',str(final)],timeout=240)
    uhd=a.out/'adams_screen_overlay_2160p.mp4'; base.render_presentation(final,uhd,'uhd')

    qa={'game_id':a.game,'event_num':a.event,'source':meta,'source_width':W,'source_height':H,'fps':fps,
        'source_frames':n_all,'window_start_s':start_s,'window_end_s':end_s,'window_frames':n,
        'analysis_every_frames':every,'number_every_frames':number_every,'court_every_frames':court_every,
        'raw_track_ids':len(set(int(r.get('raw_track_id',r['track_id'])) for r in track_rows)),
        'stitched_track_ids':len(by_tid),'stitch_links':stitch_links,'roles':roles,
        'number_crops':len(number_rows),'court_samples':len(court_kp),'court_calibrated_samples':len(court_by_frame),
        'native_output':str(final),'uhd_output':str(uhd),
        'note':'UHD uses the repo-standard deterministic render. Metric distances appear only when court calibration passes.'}
    (a.out/'qa.json').write_text(json.dumps(qa,indent=2,default=str)); print(json.dumps(qa,indent=2,default=str))


if __name__=='__main__':
    main()
