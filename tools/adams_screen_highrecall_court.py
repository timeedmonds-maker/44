#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from kd_double_team_ballhandler_onnx import YoloOnnx, extract_frames, players_on_courtish
from adams_screen_candidate_onnx import screen_events, foot, norm_dist
from adams_screen_temporal_onnx import associate, cluster_tracks, by_tid, ballhandler_tid, propagate_ballhandler_tids


def sparse_court_hulls(frame_paths,court_model,device='cpu',stride=2,conf=.30,min_kp=5):
    """Image-space court hulls, inferred sparsely then nearest-filled.

    This is a participation gate, not metric tracking. At 4 fps with stride=2
    the court model runs ~2 fps, cutting cost substantially while preserving
    enough camera-motion updates for the generous floor margin.
    """
    from nbacv.court import _court_infer
    n=len(frame_paths); hulls=[None]*n; sampled=[]; good_size=640
    for i in range(0,n,max(1,stride)):
        fr=cv2.imread(str(frame_paths[i]))
        if fr is None: continue
        best=None
        for sz in [good_size,960] if good_size!=960 else [960,640]:
            cand=_court_infer(court_model,fr,sz,device)
            if cand is None: continue
            xy,cf=cand; sel=(cf>=conf)&(xy[:,0]>1)&(xy[:,1]>1)
            if int(sel.sum())>=min_kp:
                best=cv2.convexHull(xy[sel].astype(np.float32)); good_size=sz; break
        if best is not None:
            hulls[i]=best; sampled.append(i)
    if sampled:
        for i in range(n):
            if hulls[i] is None:
                j=min(sampled,key=lambda x:abs(x-i)); hulls[i]=hulls[j]
    return hulls, len(sampled)/max(1,math.ceil(n/max(1,stride))), sum(h is not None for h in hulls)/max(n,1)


def polygon_gate(people_pf,hulls,frame_paths,margin_frac=.095):
    out=[]; removed=tested=0
    for pth,people,hull in zip(frame_paths,people_pf,hulls):
        fr=cv2.imread(str(pth))
        if fr is None or hull is None:
            out.append(people); continue
        margin=max(28.,margin_frac*fr.shape[0]); row=[]
        for b in people:
            tested+=1; pt=(float((b[0]+b[2])/2),float(b[3]))
            if cv2.pointPolygonTest(hull,pt,True)>=-margin: row.append(b)
            else: removed+=1
        out.append(row)
    return out,{'tested':tested,'removed':removed}


def keep_player_clusters(tracked,labels):
    mass=Counter()
    for ds in tracked:
        for d in ds:
            if d['tid'] in labels: mass[labels[d['tid']]]+=1
    keep={lab for lab,_ in mass.most_common(2)}
    if len(keep)<2: return tracked,keep,dict(mass)
    return [[d for d in ds if labels.get(d['tid']) in keep] for ds in tracked],keep,dict(mass)


def track_motion(boxes,tid,a,b,ref_h):
    pts=[]
    for f in range(max(0,a),min(len(boxes),b+1)):
        if tid in boxes[f]: pts.append(foot(boxes[f][tid]))
    if len(pts)<2: return None
    # net displacement and path length; path protects against curved cuts.
    net=float(np.linalg.norm(pts[-1]-pts[0])/max(ref_h,1.))
    path=sum(float(np.linalg.norm(pts[i]-pts[i-1])) for i in range(1,len(pts)))/max(ref_h,1.)
    return net,path


def rel_angle(a,b):
    v=foot(a)-foot(b)
    return float(math.atan2(v[1],v[0]))


def angle_delta(a,b):
    d=(b-a+math.pi)%(2*math.pi)-math.pi
    return abs(float(d))


def detect_highrecall(frame_paths,tracked,labels,balls_pf,fps,
                      teammate_radius=3.1,opp_screen_radius=2.25,
                      opp_user_radius=3.3,screen_still=1.05,user_move=.48):
    """Ball-independent high-recall screen-like interaction detector.

    A candidate requires a relatively stable teammate (prospective screener),
    another same-team player moving around/past him, and an opponent entering
    either player's interaction zone. This intentionally over-includes handoffs,
    brush screens and close cuts; visual QA is the precision stage.
    """
    boxes=[by_tid(ds) for ds in tracked]; hits=[]; w=max(1,int(round(.55*fps)))
    observed_bh=[ballhandler_tid(tracked[i],balls_pf[i]) for i in range(len(tracked))]
    bh=propagate_ballhandler_tids(tracked,observed_bh,fps)

    for fi,BX in enumerate(boxes):
        tids=list(BX)
        for s in tids:
            sl=labels.get(s)
            if sl is None: continue
            S=BX[s]; sh=max(1.,S[3]-S[1])
            sm=track_motion(boxes,s,fi-w,fi+w,sh)
            if sm is None or sm[0]>screen_still: continue
            for u in tids:
                if u==s or labels.get(u)!=sl: continue
                U=BX[u]; uh=max(1.,U[3]-U[1])
                if norm_dist(S,U,.5*(sh+uh))>teammate_radius: continue
                um=track_motion(boxes,u,fi-w,fi+w,uh)
                if um is None or max(um)<user_move: continue
                before=max(0,fi-w); after=min(len(boxes)-1,fi+w)
                if any(t not in boxes[f] for t in (s,u) for f in (before,after)): continue
                a0=rel_angle(boxes[before][u],boxes[before][s]); a1=rel_angle(boxes[after][u],boxes[after][s])
                turn=angle_delta(a0,a1)
                # Require meaningful relative travel but not a strict side-cross.
                rel0=norm_dist(boxes[before][s],boxes[before][u],.5*(sh+uh))
                rel1=norm_dist(boxes[after][s],boxes[after][u],.5*(sh+uh))
                if turn<0.22 and abs(rel1-rel0)<0.35: continue
                bestd=None
                for d,D in BX.items():
                    dl=labels.get(d)
                    if d in (s,u) or dl is None or dl==sl: continue
                    dsd=norm_dist(S,D,sh); dud=norm_dist(U,D,uh)
                    if dsd<=opp_screen_radius or dud<=opp_user_radius:
                        score=(teammate_radius-norm_dist(S,U,.5*(sh+uh))) + max(0,opp_screen_radius-dsd) + max(0,opp_user_radius-dud) + max(0,screen_still-sm[0]) + max(um) + min(1.5,turn)
                        if bestd is None or score>bestd[0]: bestd=(score,d,dsd,dud)
                if bestd is None: continue
                hits.append({'frame':fi,'screener_tid':s,'user_tid':u,'defender_tid':bestd[1],
                             'ballhandler_tid':bh[fi], 'screen_user_is_ballhandler':bool(bh[fi]==u),
                             'screener_motion':round(sm[0],3),'user_motion':round(max(um),3),
                             'relative_turn_rad':round(turn,3),'screen_defender_norm':round(bestd[2],3),
                             'user_defender_norm':round(bestd[3],3),'score':round(float(bestd[0]),3),
                             'ballhandler_source':'observed' if observed_bh[fi] is not None and observed_bh[fi]==bh[fi] else ('propagated' if bh[fi] is not None else 'none')})

    # Merge same screener+user across short gaps; require >=2 distinct frames.
    seq=[]; maxgap=max(1,int(round(.75*fps)))
    for h in sorted(hits,key=lambda x:x['frame']):
        match=None
        for q in reversed(seq[-8:]):
            if h['frame']-q['end_frame']>maxgap: break
            if h['screener_tid']==q['screener_tid'] and h['user_tid']==q['user_tid']:
                match=q; break
        if match is None:
            seq.append({'start_frame':h['frame'],'end_frame':h['frame'],'screener_tid':h['screener_tid'],'user_tid':h['user_tid'],'hits':[h],'best':h})
        else:
            match['end_frame']=h['frame']; match['hits'].append(h)
            if h['score']>match['best']['score']: match['best']=h
    seq=[q for q in seq if len({h['frame'] for h in q['hits']})>=2]
    return seq,bh,observed_bh


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--onnx-model',required=True); ap.add_argument('--court-model',required=True); ap.add_argument('--nbacv-src',required=True)
    ap.add_argument('--sample-fps',type=float,default=4.0); ap.add_argument('--max-seconds',type=float,default=16.0); ap.add_argument('--target-hls-width',type=int,default=960)
    ap.add_argument('--court-stride',type=int,default=2); ap.add_argument('--person-conf',type=float,default=.14); ap.add_argument('--ball-conf',type=float,default=.03)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(Path(a.nbacv_src)))
    from ultralytics import YOLO
    court_model=YOLO(a.court_model); model=YoloOnnx(a.onnx_model)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10)
    rows=[]; seqrows=[]
    with tempfile.TemporaryDirectory(prefix='adams_hrscreen_') as td:
      root=Path(td)
      for _,r in df.iterrows():
        t0=time.perf_counter(); events=screen_events(r); nseq=0; sampled=bh_obs=bh_prop=0; removed=0; cover=[]; errors=[]
        for ev in events:
          evdir=root/f'{r.game_id}_{ev}'
          try:
            meta=extract_frames(str(r.game_id),int(ev),evdir,a.sample_fps,a.max_seconds,a.target_hls_width); paths=meta['frames']; people=[]; balls=[]
            for p in paths:
              fr=cv2.imread(str(p));
              if fr is None: people.append([]); balls.append([]); continue
              people.append(players_on_courtish(model.detect_class(fr,0,a.person_conf),fr.shape[0],max_players=18)); balls.append(model.detect_class(fr,32,a.ball_conf,.35))
            hulls,rawcov,cov=sparse_court_hulls(paths,court_model,stride=a.court_stride); cover.append(cov)
            people,g=polygon_gate(people,hulls,paths); removed+=g['removed']; tracked=associate(people); labels,centers=cluster_tracks(paths,tracked); tracked,keep,mass=keep_player_clusters(tracked,labels)
            seqs,bht,obs=detect_highrecall(paths,tracked,labels,balls,a.sample_fps); sampled+=len(paths); bh_obs+=sum(x is not None for x in obs); bh_prop+=sum(x is not None for x in bht); nseq+=len(seqs)
            for si,s in enumerate(seqs):
              b=s['best']; seqrows.append({'game_id':str(r.game_id),'possession_uid':r.possession_uid,'event_num':int(ev),'sequence_index':si,
                'start_sample':s['start_frame'],'end_sample':s['end_frame'],'start_s':round(s['start_frame']/a.sample_fps,3),'end_s':round(s['end_frame']/a.sample_fps,3),
                'screener_tid':s['screener_tid'],'user_tid':s['user_tid'],'n_hits':len({h['frame'] for h in s['hits']}),
                **{f'best_{k}':v for k,v in b.items() if k not in ('frame','screener_tid','user_tid')}})
          except Exception as e: errors.append(f'event {ev}: {type(e).__name__}: {e}')
          finally: shutil.rmtree(evdir,ignore_errors=True)
        rows.append({'game_id':str(r.game_id),'period':r.period,'possession_uid':r.possession_uid,'start_time':r.start_time,'end_time':r.end_time,'pts_poss':r.pts_poss,'type_end':r.type_end,
          'lineup_team':r.get('lineup_team'),'lineup_opp':r.get('lineup_opp'),'screen_event_nums':'|'.join(map(str,events)),'highrecall_sequences':nseq,'highrecall_candidate':nseq>0,
          'sampled_frames':sampled,'ballhandler_direct_frames':bh_obs,'ballhandler_available_frames':bh_prop,'mean_polygon_coverage':round(float(np.mean(cover)),3) if cover else None,
          'polygon_removed_detections':removed,'errors':' || '.join(errors),'runtime_seconds':round(time.perf_counter()-t0,2)})
        print(json.dumps(rows[-1],default=str),flush=True)
    out=pd.DataFrame(rows); seq=pd.DataFrame(seqrows); out.to_csv(a.out/'highrecall_candidates.csv',index=False); seq.to_csv(a.out/'highrecall_sequences.csv',index=False)
    qa={'rows':len(out),'candidate_possessions':int(out.highrecall_candidate.sum()),'total_sequences':int(out.highrecall_sequences.sum()),'candidate_rate':float(out.highrecall_candidate.mean()) if len(out) else None,
      'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,'mean_polygon_coverage':float(out.mean_polygon_coverage.dropna().mean()) if len(out) and out.mean_polygon_coverage.notna().any() else None,
      'direct_ballhandler_fraction':float(out.ballhandler_direct_frames.sum()/max(out.sampled_frames.sum(),1)),'available_ballhandler_fraction':float(out.ballhandler_available_frames.sum()/max(out.sampled_frames.sum(),1)),
      'polygon_removed_detections':int(out.polygon_removed_detections.sum()),'error_rows':int(out.errors.fillna('').astype(str).ne('').sum()),
      'semantics':'High-recall screen-like interactions; ball-independent detection. Candidates require visual/identity QA and are not confirmed screens.'}
    (a.out/'qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))

if __name__=='__main__': main()
