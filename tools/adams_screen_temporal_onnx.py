#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, shutil, sys, tempfile, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from kd_double_team_ballhandler_onnx import YoloOnnx, extract_frames, players_on_courtish, point_rect_distance
from kd_double_team_teamcolor_onnx import jersey_feature
from adams_screen_candidate_onnx import screen_events, foot, norm_dist


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    if inter<=0: return 0.
    aa=max(1.,(a[2]-a[0])*(a[3]-a[1])); bb=max(1.,(b[2]-b[0])*(b[3]-b[1]))
    return inter/(aa+bb-inter)

def center(b): return np.array([(b[0]+b[2])*0.5,(b[1]+b[3])*0.5],np.float32)

def associate(frames_people,max_gap=2):
    """Greedy short-clip tracker. Output detections with persistent tid."""
    next_tid=1; active={}; out=[]
    for fi,people in enumerate(frames_people):
        candidates=[]
        for tid,rec in active.items():
            if fi-rec['last']>max_gap: continue
            prev=rec['box']; ph=max(1.,prev[3]-prev[1]); pc=center(prev)
            for j,b in enumerate(people):
                dc=float(np.linalg.norm(center(b)-pc)/ph)
                if dc<=1.25:
                    cost=dc-0.45*iou(prev,b)
                    candidates.append((cost,tid,j))
        candidates.sort(); used_t=set(); used_j=set(); assigns={}
        for cost,tid,j in candidates:
            if tid in used_t or j in used_j: continue
            used_t.add(tid); used_j.add(j); assigns[j]=tid
        dets=[]
        for j,b in enumerate(people):
            tid=assigns.get(j)
            if tid is None:
                tid=next_tid; next_tid+=1
            active[tid]={'box':b,'last':fi}
            dets.append({'tid':tid,'box':b})
        active={tid:r for tid,r in active.items() if fi-r['last']<=max_gap}
        out.append(dets)
    return out

def ballhandler_tid(dets,balls):
    best=None
    for ball in balls:
        bx=(ball[0]+ball[2])*0.5; by=(ball[1]+ball[3])*0.5
        for d in dets:
            h=max(1.,d['box'][3]-d['box'][1]); z=point_rect_distance(bx,by,d['box'])/h
            if best is None or z<best[0]: best=(z,d['tid'])
    return None if best is None or best[0]>0.95 else best[1]

def propagate_ballhandler_tids(tracked, observed, fps, bridge_s=1.25, edge_s=0.50):
    """Conservatively carry a ballhandler track through short ball-missing gaps.

    The generic COCO sports-ball detector is sparse on broadcast footage.  A
    player track, however, is usually continuous through the screen action.
    We therefore fill only frames where the SAME observed ballhandler track
    remains present, bridge only short gaps between same-track observations,
    and stop at any conflicting observed ballhandler.  This is an inference,
    not a new ball observation.
    """
    out=list(observed)
    present=[{d['tid'] for d in ds} for ds in tracked]
    max_bridge=max(1,int(round(bridge_s*fps)))
    edge=max(1,int(round(edge_s*fps)))

    obs_by_tid=defaultdict(list)
    for i,tid in enumerate(observed):
        if tid is not None:
            obs_by_tid[int(tid)].append(i)

    # Bridge two observations of the same track when no different observed
    # ballhandler appears in between.
    for tid,idxs in obs_by_tid.items():
        for a,b in zip(idxs,idxs[1:]):
            if b-a>max_bridge:
                continue
            if any(observed[k] not in (None,tid) for k in range(a+1,b)):
                continue
            for k in range(a+1,b):
                if out[k] is None and tid in present[k]:
                    out[k]=tid

    # Short edge extension around each direct observation. A conflicting
    # direct observation is a hard boundary.
    for i,tid0 in enumerate(observed):
        if tid0 is None:
            continue
        tid=int(tid0)
        for step in (-1,1):
            for n in range(1,edge+1):
                k=i+step*n
                if k<0 or k>=len(out): break
                if observed[k] is not None and observed[k]!=tid: break
                if tid not in present[k]: break
                if out[k] is None: out[k]=tid
    return out

def cluster_tracks(frame_paths,tracked):
    feats=defaultdict(list)
    for fi,(pth,dets) in enumerate(zip(frame_paths,tracked)):
        fr=cv2.imread(str(pth))
        if fr is None: continue
        for d in dets:
            f=jersey_feature(fr,d['box'])
            if f is not None: feats[d['tid']].append(f)
    tids=[]; X=[]
    for tid,fs in feats.items():
        if fs:
            tids.append(tid); X.append(np.median(np.asarray(fs),axis=0))
    if len(X)<6: return {},{}
    X=np.asarray(X,np.float32); k=3 if len(X)>=8 else 2
    cv2.setRNGSeed(19)
    _c,lab,centers=cv2.kmeans(X,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,0.5),8,cv2.KMEANS_PP_CENTERS)
    lab=lab.reshape(-1)
    return {tid:int(lab[i]) for i,tid in enumerate(tids)},{i:centers[i] for i in range(k)}

def by_tid(dets): return {d['tid']:d['box'] for d in dets}

def track_motion(track_boxes,tid,a,b,ref_h):
    pts=[]
    for fi in range(max(0,a),min(len(track_boxes),b+1)):
        box=track_boxes[fi].get(tid)
        if box is not None: pts.append(foot(box))
    if len(pts)<2: return None
    return float(np.linalg.norm(pts[-1]-pts[0])/max(1.,ref_h))

def signed_side(ball_box,screen_box):
    return float((center(ball_box)[0]-center(screen_box)[0])/max(1.,ball_box[3]-ball_box[1]))

def detect_sequences(frame_paths,tracked,balls_pf,labels,centers,fps,
                     same_radius=2.7,def_radius=3.0,contact_radius=1.65,
                     still_move=0.75,bh_move=0.75,min_color_delta=18.0):
    boxes=[by_tid(d) for d in tracked]; hits=[]
    observed=[ballhandler_tid(tracked[i],balls_pf[i]) for i in range(len(tracked))]
    bh_tids=propagate_ballhandler_tids(tracked,observed,fps)
    for fi,bh in enumerate(bh_tids):
        if bh is None or bh not in boxes[fi] or bh not in labels: continue
        B=boxes[fi][bh]; h=max(1.,B[3]-B[1]); bl=labels[bh]
        for s,S in boxes[fi].items():
            if s==bh or labels.get(s)!=bl: continue
            dbs=norm_dist(B,S,h)
            if dbs>same_radius: continue
            sh=max(1.,S[3]-S[1])
            w=max(1,int(round(0.5*fps)))
            sm=track_motion(boxes,s,fi-w,fi+w,sh)
            bm=track_motion(boxes,bh,fi-w,fi+w,h)
            if sm is None or bm is None or sm>still_move or bm<bh_move: continue
            before=max(0,fi-w); after=min(len(boxes)-1,fi+w)
            if bh not in boxes[before] or s not in boxes[before] or bh not in boxes[after] or s not in boxes[after]: continue
            side0=signed_side(boxes[before][bh],boxes[before][s]); side1=signed_side(boxes[after][bh],boxes[after][s])
            crossed=(side0*side1<0) or abs(side1-side0)>=0.65
            if not crossed: continue
            for d,D in boxes[fi].items():
                dl=labels.get(d)
                if d in (bh,s) or dl is None or dl==bl: continue
                delta=float(np.linalg.norm(centers[dl]-centers[bl])) if dl in centers else 0.
                if delta<min_color_delta: continue
                dbd=norm_dist(B,D,h); dsd=norm_dist(S,D,0.5*(h+sh))
                if dbd>def_radius or dsd>contact_radius: continue
                score=(same_radius-dbs)+(def_radius-dbd)+(contact_radius-dsd)+(still_move-sm)+(bm-bh_move)+min(1.0,abs(side1-side0))
                hits.append({'frame':fi,'ballhandler_tid':bh,'screener_tid':s,'defender_tid':d,
                             'bh_screener_norm':round(dbs,3),'bh_defender_norm':round(dbd,3),'screener_defender_norm':round(dsd,3),
                             'screener_motion_norm':round(sm,3),'ballhandler_motion_norm':round(bm,3),
                             'side_before':round(side0,3),'side_after':round(side1,3),'color_delta':round(delta,1),'score':round(float(score),3),
                             'ballhandler_source':'observed' if observed[fi]==bh else 'propagated'})
    seq=[]
    for hrec in sorted(hits,key=lambda x:x['frame']):
        if seq and hrec['frame']-seq[-1]['end_frame']<=max(1,int(round(0.5*fps))) and hrec['ballhandler_tid']==seq[-1]['ballhandler_tid'] and hrec['screener_tid']==seq[-1]['screener_tid']:
            seq[-1]['end_frame']=hrec['frame']; seq[-1]['hits'].append(hrec)
            if hrec['score']>seq[-1]['best']['score']: seq[-1]['best']=hrec
        else:
            seq.append({'start_frame':hrec['frame'],'end_frame':hrec['frame'],'ballhandler_tid':hrec['ballhandler_tid'],'screener_tid':hrec['screener_tid'],'hits':[hrec],'best':hrec})
    return [s for s in seq if len({h['frame'] for h in s['hits']})>=2],bh_tids

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--onnx-model',required=True)
    ap.add_argument('--sample-fps',type=float,default=6.0); ap.add_argument('--max-seconds',type=float,default=14.0); ap.add_argument('--target-hls-width',type=int,default=640)
    ap.add_argument('--person-conf',type=float,default=0.14); ap.add_argument('--ball-conf',type=float,default=0.03)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); model=YoloOnnx(a.onnx_model)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10)
    rows=[]; seq_rows=[]
    with tempfile.TemporaryDirectory(prefix='adams_temporal_') as td:
        root=Path(td)
        for _,r in df.iterrows():
            t0=time.perf_counter(); events=screen_events(r); nseq=0; best=None; errors=[]; sampled=bh_obs=0
            for ev in events:
                evdir=root/f'{r.game_id}_{ev}'
                try:
                    meta=extract_frames(str(r.game_id),int(ev),evdir,a.sample_fps,a.max_seconds,a.target_hls_width)
                    paths=meta['frames']; frames_people=[]; balls_pf=[]
                    for pth in paths:
                        fr=cv2.imread(str(pth));
                        if fr is None: frames_people.append([]); balls_pf.append([]); continue
                        frames_people.append(players_on_courtish(model.detect_class(fr,0,a.person_conf),fr.shape[0]))
                        balls_pf.append(model.detect_class(fr,32,a.ball_conf,0.35))
                    tracked=associate(frames_people); labels,centers=cluster_tracks(paths,tracked)
                    seqs,bhs=detect_sequences(paths,tracked,balls_pf,labels,centers,a.sample_fps)
                    sampled+=len(paths); bh_obs+=sum(x is not None for x in bhs); nseq+=len(seqs)
                    for si,s in enumerate(seqs):
                        rec={'game_id':str(r.game_id),'possession_uid':r.possession_uid,'event_num':int(ev),'sequence_index':si,
                             'start_sample':s['start_frame'],'end_sample':s['end_frame'],'start_s':round(s['start_frame']/a.sample_fps,3),'end_s':round(s['end_frame']/a.sample_fps,3),
                             'ballhandler_tid':s['ballhandler_tid'],'screener_tid':s['screener_tid'],'n_hits':len({h['frame'] for h in s['hits']}),**{f'best_{k}':v for k,v in s['best'].items() if k not in ('frame','ballhandler_tid','screener_tid')}}
                        seq_rows.append(rec)
                        if best is None or rec['best_score']>best['best_score']: best=rec
                except Exception as e: errors.append(f'event {ev}: {type(e).__name__}: {e}')
                finally: shutil.rmtree(evdir,ignore_errors=True)
            rows.append({'game_id':str(r.game_id),'period':r.period,'possession_uid':r.possession_uid,'start_time':r.start_time,'end_time':r.end_time,'pts_poss':r.pts_poss,'type_end':r.type_end,
                         'screen_event_nums':'|'.join(map(str,events)),'temporal_screen_sequences':nseq,'temporal_candidate':nseq>0,
                         'best_event_num':None if best is None else best['event_num'],'best_start_s':None if best is None else best['start_s'],'best_end_s':None if best is None else best['end_s'],
                         'best_screen_score':None if best is None else best['best_score'],'sampled_frames':sampled,'ballhandler_observed_frames':bh_obs,'errors':' || '.join(errors),'runtime_seconds':round(time.perf_counter()-t0,2)})
            print(json.dumps(rows[-1],default=str),flush=True)
    out=pd.DataFrame(rows); out.to_csv(a.out/'temporal_candidates.csv',index=False); pd.DataFrame(seq_rows).to_csv(a.out/'screen_sequences.csv',index=False)
    qa={'rows':len(out),'temporal_candidates':int(out.temporal_candidate.sum()),'retention_rate':float(out.temporal_candidate.mean()) if len(out) else None,'total_sequences':int(out.temporal_screen_sequences.sum()),'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,'error_rows':int(out.errors.fillna('').astype(str).ne('').sum())}
    (a.out/'qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
