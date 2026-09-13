#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, sys, tempfile, time
from pathlib import Path
import cv2, numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from kd_double_team_ballhandler_onnx import (
    YoloOnnx, extract_frames, players_on_courtish, ballhandler
)
from kd_double_team_teamcolor_onnx import jersey_feature


def parse_nums(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for s in str(v).split('|'):
        s=s.strip()
        if s.isdigit() and int(s) not in out: out.append(int(s))
    return out


def screen_events(row):
    nums=parse_nums(row.get('window_event_nums'))
    anchor=row.get('anchor_event_num')
    try: anchor=int(float(anchor))
    except Exception: anchor=None
    dur=float(row.get('duration_s') or 0)
    if not nums: return [anchor] if anchor else []
    if dur <= 14:
        picks=[anchor or nums[-1]]
    elif dur <= 24:
        picks=[nums[0], anchor or nums[-1]]
    else:
        picks=[nums[0], nums[len(nums)//2], anchor or nums[-1]]
    out=[]
    for x in picks:
        if x is not None and x not in out: out.append(int(x))
    return out


def foot(b):
    return np.array([(b[0]+b[2])*0.5,b[3]],np.float32)

def norm_dist(a,b,ref_h):
    pa,pb=foot(a),foot(b)
    d=pa-pb; d[1]*=0.72
    return float(np.linalg.norm(d)/max(1.,ref_h))


def color_clusters(frame,players):
    feats=[]; valid=[]
    for i,b in enumerate(players):
        f=jersey_feature(frame,b)
        if f is not None: feats.append(f); valid.append(i)
    if len(feats)<6: return None
    X=np.asarray(feats,np.float32); k=3 if len(X)>=8 else 2
    cv2.setRNGSeed(13)
    _c,lab,centers=cv2.kmeans(X,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.5),5,cv2.KMEANS_PP_CENTERS)
    lab=lab.reshape(-1)
    return {idx:int(lab[j]) for j,idx in enumerate(valid)},centers,np.bincount(lab,minlength=k)


def frame_screen_test(frame,players,bh_i,
                      teammate_radius=2.35,defender_radius=2.65,
                      contact_radius=1.75,min_color_delta=20.0):
    cc=color_clusters(frame,players)
    if cc is None or bh_i not in cc[0]: return {'observable':False,'reason':'teamcolor_unavailable'}
    labels,centers,sizes=cc; bh_lab=labels[bh_i]
    if sizes[bh_lab] < 2: return {'observable':False,'reason':'ballhandler_cluster_unstable'}
    bh=players[bh_i]; h=max(1.,bh[3]-bh[1])
    same=[]; opp=[]
    for j,p in enumerate(players):
        if j==bh_i or j not in labels: continue
        d=norm_dist(bh,p,h)
        if labels[j]==bh_lab: same.append((d,j))
        else:
            delta=float(np.linalg.norm(centers[labels[j]]-centers[bh_lab]))
            if delta>=min_color_delta: opp.append((d,j,delta))
    same.sort(); opp.sort()
    best=None
    for sd,sj in same:
        if sd>teammate_radius: break
        sh=max(1.,players[sj][3]-players[sj][1])
        for od,oj,delta in opp:
            if od>defender_radius: break
            td=norm_dist(players[sj],players[oj],0.5*(h+sh))
            if td<=contact_radius:
                score=(teammate_radius-sd)+(defender_radius-od)+(contact_radius-td)
                rec={'candidate':True,'screener_index':sj,'defender_index':oj,
                     'bh_screener_norm':round(sd,3),'bh_defender_norm':round(od,3),
                     'screener_defender_norm':round(td,3),'color_delta':round(delta,1),'score':round(score,3)}
                if best is None or rec['score']>best['score']: best=rec
    if best: return {'observable':True,'reason':'screen_geometry',**best}
    return {'observable':True,'candidate':False,'reason':'no_screen_geometry'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--onnx-model',required=True)
    ap.add_argument('--sample-fps',type=float,default=4.0); ap.add_argument('--max-seconds',type=float,default=14.0); ap.add_argument('--target-hls-width',type=int,default=640)
    ap.add_argument('--person-conf',type=float,default=0.14); ap.add_argument('--ball-conf',type=float,default=0.03); ap.add_argument('--min-player-count',type=int,default=6)
    ap.add_argument('--min-assessed-frames',type=int,default=4); ap.add_argument('--min-candidate-frames',type=int,default=2)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10); model=YoloOnnx(a.onnx_model)
    rows=[]; details=[]
    with tempfile.TemporaryDirectory(prefix='adams_screen_stagea_') as td:
        root=Path(td)
        for _,r in df.iterrows():
            t0=time.perf_counter(); sampled=bh_frames=assessed=candidate_frames=0; errors=[]; angles=[]; best_score=0.; best_event=None; best_sample=None
            events=screen_events(r)
            for ev in events:
                evdir=root/f'{r.game_id}_{ev}'
                try:
                    meta=extract_frames(str(r.game_id),int(ev),evdir,a.sample_fps,a.max_seconds,a.target_hls_width); angles.append(meta['angle'])
                    for k,pth in enumerate(meta['frames']):
                        fr=cv2.imread(str(pth))
                        if fr is None: continue
                        sampled+=1
                        people=players_on_courtish(model.detect_class(fr,0,a.person_conf),fr.shape[0])
                        balls=model.detect_class(fr,32,a.ball_conf,0.35)
                        bh=ballhandler(people,balls) if len(people)>=a.min_player_count else None
                        test=None
                        if bh is not None:
                            bh_frames+=1
                            test=frame_screen_test(fr,people,bh['player_index'])
                            if test.get('observable'):
                                assessed+=1
                                if test.get('candidate'):
                                    candidate_frames+=1
                                    if float(test.get('score',0))>best_score:
                                        best_score=float(test['score']); best_event=int(ev); best_sample=int(k)
                        details.append({
                            'possession_uid':r.possession_uid,'event_num':int(ev),'sample_index':k,'players':len(people),'balls':len(balls),
                            'ballhandler_observed':bh is not None,'screen_observable':False if test is None else test.get('observable',False),
                            'candidate_frame':False if test is None else test.get('candidate',False),'score':None if test is None else test.get('score'),
                            'bh_screener_norm':None if test is None else test.get('bh_screener_norm'),'bh_defender_norm':None if test is None else test.get('bh_defender_norm'),
                            'screener_defender_norm':None if test is None else test.get('screener_defender_norm'),'reason':'' if test is None else test.get('reason','')
                        })
                except Exception as e:
                    errors.append(f'event {ev}: {type(e).__name__}: {e}')
                finally:
                    shutil.rmtree(evdir,ignore_errors=True)
            if assessed < a.min_assessed_frames:
                decision='candidate'; reason='insufficient_observability_keep_for_recall'
            elif candidate_frames >= a.min_candidate_frames:
                decision='candidate'; reason='repeated_ballhandler_teammate_defender_screen_geometry'
            else:
                decision='screen_negative'; reason='observable_without_repeated_screen_geometry'
            rows.append({
                'game_id':str(r.game_id),'period':r.period,'possession_uid':r.possession_uid,'start_time':r.start_time,'end_time':r.end_time,'duration_s':r.duration_s,
                'pts_poss':r.pts_poss,'type_end':r.type_end,'anchor_event_num':r.anchor_event_num,'screen_event_nums':'|'.join(map(str,events)),
                'stage_a_decision':decision,'stage_a_reason':reason,'sampled_frames':sampled,'ballhandler_frames':bh_frames,'assessed_frames':assessed,
                'candidate_frames':candidate_frames,'best_screen_score':round(best_score,3),'best_event_num':best_event,'best_sample_index':best_sample,
                'resolved_angles':'|'.join(sorted(set(angles))),'errors':' || '.join(errors),'runtime_seconds':round(time.perf_counter()-t0,2),
                'semantics':'high-recall ballhandler/team-color screen candidate sieve; identity not yet assigned'
            })
            print(json.dumps(rows[-1],default=str),flush=True)
    out=pd.DataFrame(rows); out.to_csv(a.out/'screen_candidates.csv',index=False); pd.DataFrame(details).to_csv(a.out/'frame_detail.csv',index=False)
    qa={'rows':int(len(out)),'candidate':int((out.stage_a_decision=='candidate').sum()),'screen_negative':int((out.stage_a_decision=='screen_negative').sum()),
        'retention_rate':float((out.stage_a_decision=='candidate').mean()) if len(out) else None,'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,
        'mean_assessed_frames':float(out.assessed_frames.mean()) if len(out) else None,'error_rows':int(out.errors.fillna('').astype(str).ne('').sum())}
    (a.out/'qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
