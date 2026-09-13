#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, sys, tempfile, time
from pathlib import Path
import cv2, numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from kd_double_team_ballhandler_onnx import (
    YoloOnnx, nums, extract_frames, players_on_courtish, ballhandler, neighbor_distances
)


def jersey_feature(frame,b):
    x1,y1,x2,y2,_=b; w=max(1,int(x2-x1)); h=max(1,int(y2-y1))
    xa=max(0,int(x1+0.22*w)); xb=min(frame.shape[1],int(x1+0.78*w))
    ya=max(0,int(y1+0.24*h)); yb=min(frame.shape[0],int(y1+0.58*h))
    if xb-xa<3 or yb-ya<3: return None
    patch=frame[ya:yb,xa:xb]
    lab=cv2.cvtColor(patch,cv2.COLOR_BGR2LAB).reshape(-1,3)
    # Median is robust to jersey numbers, skin and small occlusions.
    return np.median(lab,axis=0).astype(np.float32)


def color_team_test(frame,players,target_i,radius_norm=4.25,min_center_delta=22.0):
    feats=[]; valid=[]
    for i,b in enumerate(players):
        f=jersey_feature(frame,b)
        if f is not None: feats.append(f); valid.append(i)
    if target_i not in valid or len(feats)<6: return {'observable':False,'reason':'insufficient_jersey_features'}
    X=np.asarray(feats,np.float32); k=3 if len(X)>=8 else 2
    cv2.setRNGSeed(7)
    _compact,labels,centers=cv2.kmeans(X,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.5),5,cv2.KMEANS_PP_CENTERS)
    labels=labels.reshape(-1); idx_to_label={idx:int(labels[j]) for j,idx in enumerate(valid)}
    tl=idx_to_label[target_i]; sizes=np.bincount(labels,minlength=k)
    # Need at least one same-uniform corroborator for the ballhandler cluster; otherwise abstain.
    if sizes[tl]<2: return {'observable':False,'reason':'target_color_cluster_unstable'}
    center_delta={j:float(np.linalg.norm(centers[j]-centers[tl])) for j in range(k)}
    ds=neighbor_distances(players,target_i)
    grouped={j:[] for j in range(k) if j!=tl and center_delta[j]>=min_center_delta}
    for d,j in ds:
        if d>radius_norm: continue
        lab=idx_to_label.get(j)
        if lab in grouped: grouped[lab].append(float(d))
    best=[]; best_label=None
    for lab,vals in grouped.items():
        vals=sorted(vals)
        if len(vals)>len(best): best=vals; best_label=lab
    return {
        'observable':True,'candidate':len(best)>=2,'target_cluster':tl,'target_cluster_size':int(sizes[tl]),
        'defender_cluster':best_label,'defender_cluster_count':len(best),'defender_neighbor_dists':[round(x,3) for x in best[:4]],
        'center_delta':None if best_label is None else round(center_delta[best_label],2),'reason':'ok'
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--onnx-model',required=True)
    ap.add_argument('--sample-fps',type=float,default=2.0); ap.add_argument('--max-seconds',type=float,default=12.0); ap.add_argument('--target-hls-width',type=int,default=640)
    ap.add_argument('--defender-radius-body-heights',type=float,default=4.25); ap.add_argument('--min-color-delta',type=float,default=22.0)
    ap.add_argument('--min-assessed-frames',type=int,default=3); ap.add_argument('--min-player-count',type=int,default=6)
    ap.add_argument('--person-conf',type=float,default=0.14); ap.add_argument('--ball-conf',type=float,default=0.035)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10); model=YoloOnnx(a.onnx_model)
    rows=[]; detail=[]
    with tempfile.TemporaryDirectory(prefix='kd_teamcolor_') as td:
        root=Path(td)
        for _,r in df.iterrows():
            t0=time.perf_counter(); game=str(r.game_id).zfill(10); sampled=bh_frames=assessed=candidate_frames=0; errors=[]; angles=[]; max_players=0
            for event in nums(r.screen_event_nums):
                evdir=root/f'{game}_{event}'
                try:
                    meta=extract_frames(game,event,evdir,a.sample_fps,a.max_seconds,a.target_hls_width); angles.append(meta['angle'])
                    for k,pth in enumerate(meta['frames']):
                        fr=cv2.imread(str(pth));
                        if fr is None: continue
                        sampled+=1; people=players_on_courtish(model.detect_class(fr,0,a.person_conf),fr.shape[0]); balls=model.detect_class(fr,32,a.ball_conf,0.35); max_players=max(max_players,len(people))
                        bh=ballhandler(people,balls) if len(people)>=a.min_player_count else None; tc=None
                        if bh is not None:
                            bh_frames+=1; tc=color_team_test(fr,people,bh['player_index'],a.defender_radius_body_heights,a.min_color_delta)
                            if tc.get('observable'):
                                assessed+=1; candidate_frames+=int(tc.get('candidate',False))
                        detail.append({'possession_uid':r.possession_uid,'game_id':game,'event_num':event,'sample_index':k,'players':len(people),'balls':len(balls),
                                       'ballhandler_observed':bh is not None,'teamcolor_observed':False if tc is None else tc.get('observable',False),
                                       'candidate_frame':False if tc is None else tc.get('candidate',False),'defender_cluster_count':None if tc is None else tc.get('defender_cluster_count'),
                                       'defender_neighbor_dists':'' if tc is None else '|'.join(map(str,tc.get('defender_neighbor_dists',[]))),
                                       'target_cluster_size':None if tc is None else tc.get('target_cluster_size'),'color_reason':'' if tc is None else tc.get('reason','')})
                except Exception as e: errors.append(f'event {event}: {type(e).__name__}: {e}')
                finally: shutil.rmtree(evdir,ignore_errors=True)
            if assessed<a.min_assessed_frames:
                decision='candidate'; reason='insufficient_ballhandler_teamcolor_observability'
            elif candidate_frames>0:
                decision='candidate'; reason='two_same_non_ballhandler_color_players_converge_near_ballhandler'
            else:
                decision='screen_negative'; reason='ballhandler_and_team_colors_observed_without_two_defender_color_convergence'
            row={'season':r.get('season','2025-26'),'game_id':game,'possession_uid':r.possession_uid,'candidate_priority':r.candidate_priority,'screen_event_nums':r.screen_event_nums,
                 'screen_decision':decision,'screen_reason':reason,'sampled_frames':sampled,'ballhandler_frames':bh_frames,'assessed_frames':assessed,'candidate_frames':candidate_frames,
                 'defender_radius_body_heights':a.defender_radius_body_heights,'min_color_delta':a.min_color_delta,'max_players_retained':max_players,
                 'resolved_angles':'|'.join(sorted(set(angles))),'errors':' || '.join(errors),'runtime_seconds':round(time.perf_counter()-t0,2),
                 'semantics':'team-color ballhandler upper-bound sieve; candidate is not a confirmed double'}
            rows.append(row); print(json.dumps(row,default=str),flush=True)
    out=pd.DataFrame(rows); out.to_csv(a.out/'screen_results.csv',index=False); pd.DataFrame(detail).to_csv(a.out/'frame_detail.csv',index=False)
    qa={'rows':int(len(out)),'candidate':int((out.screen_decision=='candidate').sum()),'screen_negative':int((out.screen_decision=='screen_negative').sum()),
        'candidate_retention_rate':float((out.screen_decision=='candidate').mean()) if len(out) else None,'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,
        'mean_ballhandler_frames':float(out.ballhandler_frames.mean()) if len(out) else None,'mean_assessed_frames':float(out.assessed_frames.mean()) if len(out) else None,
        'error_rows':int(out.errors.fillna('').astype(str).ne('').sum()),
        'note':'A candidate needs two nearby players sharing one non-ballhandler jersey-color cluster. Insufficient ball/team visibility always survives.'}
    (a.out/'screen_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
