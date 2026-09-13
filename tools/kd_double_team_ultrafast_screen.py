#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from fetch_official_nba_event_clip import parse as resolve_clip, UA


def nums(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for x in str(v).split('|'):
        x=x.strip()
        if x.isdigit() and int(x) not in out: out.append(int(x))
    return out


def extract_sparse_frames(game,event,outdir,fps=2.0,max_seconds=12.0):
    page,opts,ch=resolve_clip(game,event)
    outdir.mkdir(parents=True,exist_ok=True)
    headers=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    pat=str(outdir/'frame_%03d.jpg')
    subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',headers,'-i',ch['url'],
                    '-t',str(max_seconds),'-vf',f'fps={fps}','-q:v','4',pat],check=True,timeout=120)
    return {'page_url':page,'angle':ch['label'],'angle_count':len(opts),'frames':sorted(outdir.glob('frame_*.jpg'))}


def boxes(model,frame,device='cpu'):
    r=model.predict(frame,imgsz=640,device=device,verbose=False,conf=0.16,classes=[0])[0]
    if len(r.boxes)==0: return []
    xy=r.boxes.xyxy.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
    out=[]
    h,w=frame.shape[:2]
    for (x1,y1,x2,y2),q in zip(xy,cf):
        bh=float(y2-y1); bw=float(x2-x1)
        # Reject tiny audience detections and absurd aspect ratios, but keep a broad court-player envelope.
        if bh < 0.075*h or bh > 0.72*h: continue
        if bw <= 0 or bh/bw < 1.05: continue
        out.append((float(x1),float(y1),float(x2),float(y2),float(q)))
    return out


def crowding(bs, radius_body_heights):
    # Generic upper-bound screen: any detected person can be the target, any other detected person can be a helper.
    # Distances are footpoint distances normalized by target box height. This deliberately over-selects.
    n=len(bs)
    if n<3: return {'max_neighbors':0,'best_target':None,'neighbor_dists':[]}
    best=(0,None,[])
    for i,(x1,y1,x2,y2,q) in enumerate(bs):
        cx=(x1+x2)/2.0; fy=y2; h=max(1.0,y2-y1)
        ds=[]
        for j,(a,b,c,d,qq) in enumerate(bs):
            if i==j: continue
            px=(a+c)/2.0; py=d
            # compress vertical screen displacement a little because broadcast perspective magnifies depth motion.
            dn=float(np.hypot((px-cx)/h, 0.72*(py-fy)/h))
            ds.append((dn,j))
        ds.sort()
        cnt=sum(dn<=radius_body_heights for dn,_ in ds)
        if cnt>best[0]: best=(cnt,i,[round(dn,3) for dn,_ in ds[:4]])
    return {'max_neighbors':int(best[0]),'best_target':best[1],'neighbor_dists':best[2]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--person-model',default='yolo11n.pt'); ap.add_argument('--sample-fps',type=float,default=2.0)
    ap.add_argument('--max-seconds',type=float,default=12.0); ap.add_argument('--device',default='cpu')
    ap.add_argument('--radius-body-heights',type=float,default=2.5)
    ap.add_argument('--min-visible-persons',type=int,default=7)
    ap.add_argument('--min-observable-frames',type=int,default=8); ap.add_argument('--min-observable-fraction',type=float,default=0.55)
    ap.add_argument('--candidate-streak',type=int,default=2)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10)
    model=YOLO(a.person_model)
    rows=[]; detail=[]
    with tempfile.TemporaryDirectory(prefix='kd_ultrafast_') as td:
        root=Path(td)
        for _,r in df.iterrows():
            t0=time.perf_counter(); game=str(r.game_id).zfill(10); evs=nums(r.screen_event_nums)
            sampled=observable=0; cluster_frames=0; longest=cur=0; errors=[]; angles=[]; max_people=0
            for event in evs:
                evdir=root/f'{game}_{event}'
                try:
                    meta=extract_sparse_frames(game,event,evdir,a.sample_fps,a.max_seconds); angles.append(meta['angle'])
                    for k,p in enumerate(meta['frames']):
                        fr=cv2.imread(str(p));
                        if fr is None: continue
                        sampled+=1; bs=boxes(model,fr,a.device); max_people=max(max_people,len(bs))
                        obs=len(bs)>=a.min_visible_persons
                        if obs: observable+=1
                        c=crowding(bs,a.radius_body_heights) if obs else {'max_neighbors':0,'best_target':None,'neighbor_dists':[]}
                        hit=bool(obs and c['max_neighbors']>=2)
                        if hit:
                            cluster_frames+=1; cur+=1; longest=max(longest,cur)
                        else: cur=0
                        detail.append({'possession_uid':r.possession_uid,'game_id':game,'event_num':event,'sample_index':k,
                                       'people':len(bs),'observable':obs,'cluster_hit':hit,'max_neighbors':c['max_neighbors'],
                                       'best_target':c['best_target'],'nearest_norm_dists':'|'.join(map(str,c['neighbor_dists']))})
                except Exception as e: errors.append(f'event {event}: {type(e).__name__}: {e}')
                finally: shutil.rmtree(evdir,ignore_errors=True)
            frac=observable/sampled if sampled else 0.0
            enough=observable>=a.min_observable_frames and frac>=a.min_observable_fraction
            if not enough:
                decision='candidate'; reason='insufficient_visibility'
            elif longest>=a.candidate_streak:
                decision='candidate'; reason='persistent_generic_three_person_convergence'
            else:
                decision='screen_negative'; reason='well_observed_no_persistent_generic_three_person_convergence'
            row={'season':r.get('season','2025-26'),'game_id':game,'possession_uid':r.possession_uid,'candidate_priority':r.candidate_priority,
                 'screen_event_nums':r.screen_event_nums,'screen_decision':decision,'screen_reason':reason,'sampled_frames':sampled,
                 'observable_frames':observable,'observable_fraction':round(frac,4),'cluster_frames':cluster_frames,'longest_cluster_streak':longest,
                 'max_people_detected':max_people,'radius_body_heights':a.radius_body_heights,'resolved_angles':'|'.join(sorted(set(angles))),
                 'errors':' || '.join(errors),'runtime_seconds':round(time.perf_counter()-t0,2),
                 'semantics':'ultrafast upper-bound screen only; no positive is a confirmed double'}
            rows.append(row); print(json.dumps(row,default=str),flush=True)
    out=pd.DataFrame(rows); out.to_csv(a.out/'screen_results.csv',index=False); pd.DataFrame(detail).to_csv(a.out/'frame_detail.csv',index=False)
    qa={'rows':int(len(out)),'candidate':int((out.screen_decision=='candidate').sum()),'screen_negative':int((out.screen_decision=='screen_negative').sum()),
        'candidate_retention_rate':float((out.screen_decision=='candidate').mean()) if len(out) else None,
        'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,'total_runtime_seconds':float(out.runtime_seconds.sum()) if len(out) else 0,
        'mean_observable_fraction':float(out.observable_fraction.mean()) if len(out) else None,'radius_body_heights':a.radius_body_heights,
        'note':'Image-space upper-bound sieve. Team/KD/ball identity is intentionally deferred; poor visibility always survives.'}
    (a.out/'screen_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
