#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import cv2
import numpy as np
import pandas as pd
import requests
import supervision as sv
from trackers import BoTSORTTracker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_official_nba_event_clip import parse as resolve_clip, UA

PLAYER_CLASSES = {3, 4, 5, 6, 7}
BALL_CLASSES = {0, 1}
NUMBER_CLASS = 2
LABEL_MAP = {
    0: 'ball', 1: 'ball-in-basket', 2: 'number', 3: 'player',
    4: 'player-in-possession', 5: 'player-jump-shot',
    6: 'player-layup-dunk', 7: 'player-shot-block',
    8: 'referee', 9: 'rim',
}
HEADERS = {'User-Agent': UA, 'Referer': 'https://clips.nba.com/'}


def run(cmd: list[str], timeout=240):
    subprocess.run(cmd, check=True, timeout=timeout)


def choose_hls_variant(url: str, target_width=960):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = r.text
    if '#EXT-X-STREAM-INF' not in text:
        return url, None
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    variants = []
    for i, line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF'):
            continue
        m = re.search(r'RESOLUTION=(\d+)x(\d+)', line)
        bw = re.search(r'BANDWIDTH=(\d+)', line)
        uri = None
        for j in range(i + 1, min(i + 4, len(lines))):
            if not lines[j].startswith('#'):
                uri = lines[j]
                break
        if not uri:
            continue
        w = int(m.group(1)) if m else 10**9
        h = int(m.group(2)) if m else None
        full = urljoin(url, uri)
        if not urlsplit(full).query and urlsplit(url).query:
            q = urlsplit(url)
            f = urlsplit(full)
            full = urlunsplit((f.scheme, f.netloc, f.path, q.query, f.fragment))
        variants.append((w, h, int(bw.group(1)) if bw else None, full))
    if not variants:
        return url, None
    under = [v for v in variants if v[0] <= target_width]
    ch = max(under, key=lambda x: x[0]) if under else min(variants, key=lambda x: x[0])
    return ch[3], {'width': ch[0], 'height': ch[1], 'bandwidth': ch[2]}


def fetch_native_clip(game: str, event: int, out: Path, target_width=960):
    page, angles, chosen = resolve_clip(game, event)
    hls, variant = choose_hls_variant(chosen['url'], target_width)
    headers = f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    run([
        'ffmpeg', '-y', '-v', 'error', '-rw_timeout', '30000000', '-headers', headers,
        '-i', hls, '-c', 'copy', str(out)
    ], timeout=180)
    return {'clip_page': page, 'angle': chosen['label'], 'angle_count': len(angles), 'variant': variant}


def download_if_missing(url: str, path: Path, sha256: str | None = None):
    if path.exists() and path.stat().st_size > 1000:
        if not sha256:
            return
        if hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
            return
        path.unlink()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    if sha256:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha256:
            raise RuntimeError(f'checksum mismatch {got} != {sha256}')


def to_sv(dets, w, h, keep_classes):
    boxes, conf, cls = [], [], []
    for d in dets:
        if d.class_id not in keep_classes:
            continue
        b = d.bbox
        boxes.append([b.x*w, b.y*h, (b.x+b.w)*w, (b.y+b.h)*h])
        conf.append(d.confidence)
        cls.append(d.class_id)
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(boxes, np.float32),
        confidence=np.asarray(conf, np.float32),
        class_id=np.asarray(cls, np.int32),
    )


def collapse_players(d: sv.Detections):
    if len(d) == 0:
        return d
    out = sv.Detections(
        xyxy=d.xyxy.copy(),
        confidence=d.confidence.copy() if d.confidence is not None else None,
        class_id=np.zeros(len(d), dtype=np.int32),
    )
    return out.with_nms(threshold=0.50, class_agnostic=True)


def ios_number_player(number_box, player_box):
    x1 = max(number_box[0], player_box[0]); y1 = max(number_box[1], player_box[1])
    x2 = min(number_box[2], player_box[2]); y2 = min(number_box[3], player_box[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    na = max(1, (number_box[2]-number_box[0])*(number_box[3]-number_box[1]))
    return inter / na


def jersey_feature(frame, box):
    x1,y1,x2,y2 = map(int, box)
    hh=max(1,y2-y1); ww=max(1,x2-x1)
    a=max(0,y1+int(.18*hh)); b=min(frame.shape[0],y1+int(.58*hh))
    c=max(0,x1+int(.22*ww)); d=min(frame.shape[1],x2-int(.22*ww))
    if b-a<4 or d-c<4:
        return None
    crop=frame[a:b,c:d]
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).reshape(-1,3)
    if len(lab)<16:
        return None
    return np.median(lab,axis=0).astype(np.float32)


def cluster_teams(track_feats: dict[int, list[np.ndarray]]):
    tids=[]; X=[]
    for tid, feats in track_feats.items():
        if len(feats) >= 2:
            tids.append(tid); X.append(np.median(np.asarray(feats), axis=0))
    if len(X) < 4:
        return {}
    X=np.asarray(X,np.float32)
    cv2.setRNGSeed(17)
    _, lab, _ = cv2.kmeans(X, 2, None,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,.5), 8, cv2.KMEANS_PP_CENTERS)
    lab=lab.reshape(-1)
    return {tid:int(lab[i]) for i,tid in enumerate(tids)}


def load_context(game: str, event: int, outdir: Path, xfg_row_json: Path | None = None):
    # Exact PBP is public and mandatory. xFG is private project data, so public
    # repo44 runners accept a single exact-row handoff instead of requiring
    # cross-repo credentials.
    url='https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds'
    import pyreadr
    p=outdir/'pbp.rds'; download_if_missing(url,p)
    pbp=next(iter(pyreadr.read_r(str(p)).values()))
    gid=pbp['game_id'].astype(str).str.replace('.0','',regex=False).str.zfill(10)
    ev=pd.to_numeric(pbp['event_num'],errors='coerce')
    q=pbp[(gid==game)&(ev==event)].copy()
    row=q.iloc[0].to_dict() if len(q) else {}
    xrow={}
    if xfg_row_json and xfg_row_json.exists():
        obj=json.loads(xfg_row_json.read_text())
        xrow=obj.get('xfg', obj) if isinstance(obj,dict) else {}
    return {'pbp': row, 'xfg': xrow, 'xfg_status': 'provided' if xrow else 'not_provided'}


def color_for(team, role='other'):
    if role=='adams': return (0,215,255)
    if role=='ballhandler': return (255,220,50)
    if role=='screened_defender': return (255,100,255)
    if role=='adams_defender': return (80,255,80)
    return (235,235,235) if team is None else ((40,70,255) if team==0 else (255,170,30))


def ring(frame, box, color, thick=3):
    x1,y1,x2,y2=box
    cx=int((x1+x2)/2); cy=int(y2)
    bw=max(18,int((x2-x1)*.70)); bh=max(8,int(bw*.28))
    cv2.ellipse(frame,(cx,cy),(bw//2,bh//2),0,0,360,color,thick,cv2.LINE_AA)


def label(frame, xy, text, color, scale=.46):
    x,y=map(int,xy); font=cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_=cv2.getTextSize(text,font,scale,1)
    x=max(2,min(frame.shape[1]-tw-8,x)); y=max(th+5,min(frame.shape[0]-4,y))
    cv2.rectangle(frame,(x-3,y-th-4),(x+tw+4,y+3),(15,15,15),-1)
    cv2.putText(frame,text,(x,y),font,scale,color,1,cv2.LINE_AA)


def mini_court(frame, court_xy, roles, origin, size=(290,155)):
    ox,oy=origin; cw,ch=size
    overlay=frame.copy(); cv2.rectangle(overlay,(ox,oy),(ox+cw,oy+ch),(10,10,10),-1)
    cv2.addWeighted(overlay,.78,frame,.22,0,frame)
    cv2.rectangle(frame,(ox+4,oy+4),(ox+cw-4,oy+ch-4),(210,210,210),1)
    cv2.line(frame,(ox+cw//2,oy+4),(ox+cw//2,oy+ch-4),(150,150,150),1)
    cv2.circle(frame,(ox+cw//2,oy+ch//2),int(ch*.12),(150,150,150),1)
    for tid,(xcm,ycm) in court_xy.items():
        px=ox+4+int(np.clip(xcm/2800,0,1)*(cw-8)); py=oy+4+int(np.clip(ycm/1500,0,1)*(ch-8))
        role=next((r for r,t in roles.items() if t==tid), 'other')
        cv2.circle(frame,(px,py),5,color_for(None,role),-1,cv2.LINE_AA)


def draw_panel(frame, ctx, metrics):
    x0,y0,w,h=12,12,330,132
    ov=frame.copy(); cv2.rectangle(ov,(x0,y0),(x0+w,y0+h),(8,8,8),-1)
    cv2.addWeighted(ov,.80,frame,.20,0,frame)
    desc=str((ctx.get('pbp') or {}).get('description',''))
    xfg=ctx.get('xfg') or {}
    pct=xfg.get('xfg_xfg_pct','--')
    lines=[
        'ADAMS SCREEN ANALYTICS',
        desc[:47],
        f"xFG: {pct}%   shot: {xfg.get('xfg_shot_type','')}",
        f"BH-def: {metrics.get('bh_def_ft','--')} ft   Adams-def: {metrics.get('adams_def_ft','--')} ft",
        f"coverage: {metrics.get('coverage','identifying')}  advantage: {metrics.get('advantage','--')}",
    ]
    for i,t in enumerate(lines):
        cv2.putText(frame,t,(x0+8,y0+22+i*23),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1,cv2.LINE_AA)


def resolve_roles(tracked_frames, balls, team_map, overrides):
    roles={'adams':None,'ballhandler':None,'screened_defender':None,'adams_defender':None}
    for k in roles:
        if k in overrides and overrides[k] is not None:
            roles[k]=int(overrides[k])
    if roles['adams'] is None:
        return roles
    adams_team=team_map.get(roles['adams'])
    counts=collections.Counter()
    for fi,td in enumerate(tracked_frames):
        bb=balls[fi] if fi<len(balls) else []
        if not bb: continue
        bx=(bb[0][0]+bb[0][2])/2; by=(bb[0][1]+bb[0][3])/2
        cand=[]
        for tid,box in td.items():
            if team_map.get(tid)!=adams_team: continue
            cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
            cand.append((math.hypot(cx-bx,cy-by),tid))
        if cand: counts[min(cand)[1]]+=1
    if roles['ballhandler'] is None and counts:
        roles['ballhandler']=counts.most_common(1)[0][0]
    if roles['ballhandler'] is not None:
        mid=len(tracked_frames)//2; td=tracked_frames[mid]
        if roles['ballhandler'] in td:
            bh=td[roles['ballhandler']]
            opp=[tid for tid in td if team_map.get(tid) is not None and team_map.get(tid)!=adams_team]
            def d_bh(tid):
                b=td[tid]
                return math.hypot((b[0]+b[2]-bh[0]-bh[2])/2,b[3]-bh[3])
            if roles['screened_defender'] is None and opp:
                roles['screened_defender']=min(opp,key=d_bh)
            if roles['adams_defender'] is None and roles['adams'] in td:
                A=td[roles['adams']]
                others=[tid for tid in opp if tid!=roles['screened_defender']]
                if others:
                    roles['adams_defender']=min(others,key=lambda tid: math.hypot((td[tid][0]+td[tid][2]-A[0]-A[2])/2,td[tid][3]-A[3]))
    return roles


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--game',default='0022500001')
    ap.add_argument('--event',type=int,default=646)
    ap.add_argument('--out',type=Path,default=Path('artifacts/adams_screen_prod'))
    ap.add_argument('--detector-onnx',type=Path,required=True)
    ap.add_argument('--object-eval-src',type=Path,required=True)
    ap.add_argument('--court-model',type=Path,required=True)
    ap.add_argument('--nbacv-src',type=Path,required=True)
    ap.add_argument('--overrides',type=Path)
    ap.add_argument('--xfg-row-json',type=Path)
    ap.add_argument('--analysis-fps',type=float,default=10.0)
    ap.add_argument('--court-fps',type=float,default=2.0)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.object_eval_src)); sys.path.insert(0,str(a.nbacv_src))
    from object_detection_eval.inference.detectors.rfdetr import RFDETRDetector
    from ultralytics import YOLO
    from nbacv.court import _court_infer, fit_homography, project_point

    clip=a.out/'source_native.mp4'
    meta=fetch_native_clip(a.game,a.event,clip,960)
    ctx=load_context(a.game,a.event,a.out,a.xfg_row_json)
    (a.out/'context.json').write_text(json.dumps(ctx,indent=2,default=str))

    cap=cv2.VideoCapture(str(clip)); fps=cap.get(cv2.CAP_PROP_FPS) or 29.97
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector=RFDETRDetector(a.detector_onnx,LABEL_MAP,confidence_threshold=.20,num_select=300,input_height=640,input_width=640)
    tracker=BoTSORTTracker(frame_rate=max(1,int(round(fps))),enable_cmc=True,cmc_method='sparseOptFlow',track_activation_threshold=.30,high_conf_det_threshold=.25)
    court_model=YOLO(str(a.court_model))

    tracked_frames=[]; ball_frames=[]; number_rows=[]; track_feats=collections.defaultdict(list)
    court_frames=[]; track_rows=[]
    every=max(1,int(round(fps/a.analysis_fps)))
    court_every=max(every,int(round(fps/max(a.court_fps,0.25))))
    fi=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        if fi%every!=0:
            fi+=1; continue
        dets=detector.predict(fr)
        players=collapse_players(to_sv(dets,W,H,PLAYER_CLASSES))
        tracked=tracker.update(players,fr)
        td={}
        if len(tracked):
            for box,tid,cf in zip(tracked.xyxy,tracked.tracker_id,tracked.confidence):
                tid=int(tid); td[tid]=tuple(map(float,box)); feat=jersey_feature(fr,box)
                if feat is not None: track_feats[tid].append(feat)
                track_rows.append({'frame':fi,'time_s':fi/fps,'track_id':tid,'x1':box[0],'y1':box[1],'x2':box[2],'y2':box[3],'conf':float(cf)})
        tracked_frames.append(td)
        balls=to_sv(dets,W,H,BALL_CLASSES)
        bxs=[]
        if len(balls):
            order=np.argsort(-(balls.confidence if balls.confidence is not None else np.ones(len(balls))))
            for j in order[:2]: bxs.append(tuple(map(float,balls.xyxy[j])))
        ball_frames.append(bxs)
        nums=to_sv(dets,W,H,{NUMBER_CLASS})
        if len(nums):
            for box,cf in zip(nums.xyxy,nums.confidence):
                best=None
                for tid,pbox in td.items():
                    s=ios_number_player(box,pbox)
                    if best is None or s>best[0]: best=(s,tid,pbox)
                if best and best[0]>=.70:
                    x1,y1,x2,y2=map(int,box)
                    crop=fr[max(0,y1-3):min(H,y2+3),max(0,x1-3):min(W,x2+3)]
                    cpath=a.out/f'num_f{fi:05d}_t{best[1]}.jpg'
                    if crop.size: cv2.imwrite(str(cpath),crop)
                    number_rows.append({'frame':fi,'time_s':fi/fps,'track_id':best[1],'ios':best[0],'conf':float(cf),'crop':cpath.name})
        Hm=None; info={'reason':'not_sampled'}
        if fi % court_every == 0:
            kps=_court_infer(court_model,fr,640,'cpu')
            info={'reason':'no_keypoints'}
            if kps is not None:
                Hm,info=fit_homography(kps[0],kps[1],frame_hw=fr.shape[:2])
        court_frames.append({'frame':fi,'H':None if Hm is None else Hm.tolist(),'info':info})
        fi+=1
    cap.release()

    team_map=cluster_teams(track_feats)
    overrides=json.loads(a.overrides.read_text()) if a.overrides and a.overrides.exists() else {}
    roles=resolve_roles(tracked_frames,ball_frames,team_map,overrides)
    (a.out/'roles.json').write_text(json.dumps({'roles':roles,'team_map':team_map,'overrides':overrides},indent=2))
    pd.DataFrame(track_rows).to_csv(a.out/'tracks.csv',index=False)
    pd.DataFrame(number_rows).to_csv(a.out/'number_evidence.csv',index=False)
    (a.out/'court_frames.json').write_text(json.dumps(court_frames,indent=2))

    crops=[]
    for r in number_rows[:60]:
        im=cv2.imread(str(a.out/r['crop']))
        if im is None: continue
        im=cv2.resize(im,(100,80),interpolation=cv2.INTER_CUBIC)
        cv2.putText(im,f"T{r['track_id']}",(3,14),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
        crops.append(im)
    if crops:
        cols=6; rows=math.ceil(len(crops)/cols); sheet=np.zeros((rows*80,cols*100,3),np.uint8)
        for i,im in enumerate(crops): sheet[(i//cols)*80:(i//cols+1)*80,(i%cols)*100:(i%cols+1)*100]=im
        cv2.imwrite(str(a.out/'number_contact_sheet.jpg'),sheet)

    by_tid=collections.defaultdict(dict)
    for r in track_rows: by_tid[int(r['track_id'])][int(r['frame'])]=(r['x1'],r['y1'],r['x2'],r['y2'])
    def interp_box(tid,frame_i):
        d=by_tid.get(tid,{})
        if frame_i in d:return d[frame_i]
        ks=sorted(d)
        if not ks:return None
        lo=max([k for k in ks if k<=frame_i],default=None); hi=min([k for k in ks if k>=frame_i],default=None)
        if lo is None or hi is None or hi-lo>every*3:return None
        if lo==hi:return d[lo]
        t=(frame_i-lo)/(hi-lo); return tuple((1-t)*np.asarray(d[lo])+t*np.asarray(d[hi]))

    cap=cv2.VideoCapture(str(clip))
    sample_slots=np.linspace(0,max(n-1,0),12,dtype=int).tolist() if n else []
    panels=[]; slotset=set(sample_slots); frame_i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        if frame_i not in slotset:
            frame_i+=1; continue
        for tid in sorted(by_tid):
            b=interp_box(tid,frame_i)
            if b is None: continue
            col=color_for(team_map.get(tid),'other')
            cv2.rectangle(fr,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),col,2,cv2.LINE_AA)
            label(fr,(b[0],b[1]-3),f'T{tid}',col,.42)
        thumb=cv2.resize(fr,(480,270),interpolation=cv2.INTER_AREA)
        cv2.putText(thumb,f'frame {frame_i}  {frame_i/fps:.2f}s',(8,260),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        panels.append(thumb); frame_i+=1
    cap.release()
    if panels:
        cols=3; rows=math.ceil(len(panels)/cols); sheet=np.zeros((rows*270,cols*480,3),np.uint8)
        for i,im in enumerate(panels): sheet[(i//cols)*270:(i//cols+1)*270,(i%cols)*480:(i%cols+1)*480]=im
        cv2.imwrite(str(a.out/'track_contact_sheet.jpg'),sheet)

    court_by_frame={int(x['frame']):x['H'] for x in court_frames if x['H'] is not None}
    def nearest_H(frame_i):
        if not court_by_frame:return None
        k=min(court_by_frame,key=lambda x:abs(x-frame_i))
        return court_by_frame[k] if abs(k-frame_i)<=court_every*2 else None

    out_native=a.out/'adams_screen_overlay_native.mp4'
    cap=cv2.VideoCapture(str(clip)); fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    wr=cv2.VideoWriter(str(out_native),fourcc,fps,(W,H))
    traces=collections.defaultdict(lambda:collections.deque(maxlen=45)); frame_i=0
    initial_sep=None; max_sep=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        current={tid:interp_box(tid,frame_i) for tid in by_tid}; current={k:v for k,v in current.items() if v is not None}
        court_xy={}; Hm=nearest_H(frame_i)
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
        d1=dist_ft(bh,sd) if bh and sd else None; d2=dist_ft(ad,afd) if ad and afd else None
        if d1 is not None:
            metrics['bh_def_ft']=f'{d1:.1f}'; initial_sep=d1 if initial_sep is None else initial_sep
            max_sep=max(max_sep,d1); metrics['advantage']=f'+{max(0,max_sep-initial_sep):.1f} ft'
        if d2 is not None: metrics['adams_def_ft']=f'{d2:.1f}'
        if bh and sd and ad and afd: metrics['coverage']='tracking'
        for tid,b in current.items():
            role=next((r for r,t in roles.items() if t==tid),'other')
            col=color_for(team_map.get(tid),role); thick=4 if role!='other' else 2
            ring(fr,b,col,thick)
            cx=(b[0]+b[2])/2; cy=b[3]; traces[tid].append((int(cx),int(cy)))
            pts=list(traces[tid])
            for j in range(1,len(pts)): cv2.line(fr,pts[j-1],pts[j],col,1,cv2.LINE_AA)
            nm=overrides.get('names',{}).get(str(tid)) or overrides.get('names',{}).get(tid)
            if not nm: nm='ADAMS' if role=='adams' else ('BALLHANDLER' if role=='ballhandler' else f'T{tid}')
            label(fr,(b[0],b[1]-4),nm,col)
        if bh in current and sd in current:
            p1=(int((current[bh][0]+current[bh][2])/2),int(current[bh][3])); p2=(int((current[sd][0]+current[sd][2])/2),int(current[sd][3]))
            cv2.line(fr,p1,p2,(255,255,255),2,cv2.LINE_AA)
            if d1 is not None: label(fr,((p1[0]+p2[0])//2,(p1[1]+p2[1])//2),f'{d1:.1f} ft',(255,255,255),.42)
        if ad in current and afd in current:
            p1=(int((current[ad][0]+current[ad][2])/2),int(current[ad][3])); p2=(int((current[afd][0]+current[afd][2])/2),int(current[afd][3]))
            cv2.line(fr,p1,p2,(100,255,100),2,cv2.LINE_AA)
            if d2 is not None: label(fr,((p1[0]+p2[0])//2,(p1[1]+p2[1])//2),f'{d2:.1f} ft',(100,255,100),.42)
        draw_panel(fr,ctx,metrics)
        if court_xy: mini_court(fr,court_xy,roles,origin=(W-302,H-167))
        wr.write(fr); frame_i+=1
    cap.release(); wr.release()

    final=a.out/'adams_screen_overlay_h264.mp4'
    run(['ffmpeg','-y','-v','error','-i',str(out_native),'-i',str(clip),'-map','0:v:0','-map','1:a?','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(final)],timeout=240)
    uhd=a.out/'adams_screen_overlay_2160p.mp4'
    run(['ffmpeg','-y','-v','error','-i',str(final),'-vf','scale=3840:2160:flags=lanczos','-c:v','libx264','-crf','18','-preset','slow','-pix_fmt','yuv420p','-c:a','copy',str(uhd)],timeout=300)
    qa={'game_id':a.game,'event_num':a.event,'source':meta,'width':W,'height':H,'fps':fps,'frames':n,
        'analysis_every_frames':every,'court_every_frames':court_every,'tracks':len(by_tid),'team_map':team_map,
        'roles':roles,'number_crops':len(number_rows),'court_calibrated_samples':len(court_by_frame),
        'native_output':str(final),'uhd_output':str(uhd),
        'note':'2160p is deterministic Lanczos presentation upscale; source imagery remains official native NBA clip.'}
    (a.out/'qa.json').write_text(json.dumps(qa,indent=2,default=str)); print(json.dumps(qa,indent=2,default=str))

if __name__=='__main__':
    main()
