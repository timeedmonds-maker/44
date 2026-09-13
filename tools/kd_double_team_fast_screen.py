#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from fetch_official_nba_event_clip import parse as resolve_clip, UA

NBACV_SRC=Path(os.environ.get('NBACV_SRC','/tmp/nbacv/src'))
if NBACV_SRC.exists(): sys.path.insert(0,str(NBACV_SRC))
from nbacv.court import fit_homography, project_point, court_position_ok
from ultralytics import YOLO


def nums(v):
    out=[]
    if v is None or (isinstance(v,float) and pd.isna(v)): return out
    for x in str(v).split('|'):
        x=x.strip()
        if x.isdigit() and int(x) not in out: out.append(int(x))
    return out


def extract_sparse_frames(game,event,outdir,fps=2.0,max_seconds=12.0):
    page,opts,ch=resolve_clip(game,event)
    outdir.mkdir(parents=True,exist_ok=True)
    headers=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    pat=str(outdir/'frame_%03d.jpg')
    cmd=['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',headers,'-i',ch['url'],
         '-t',str(max_seconds),'-vf',f'fps={fps}','-q:v','3',pat]
    subprocess.run(cmd,check=True,timeout=120)
    frames=sorted(outdir.glob('frame_*.jpg'))
    return {'page_url':page,'angle':ch['label'],'angle_count':len(opts),'frames':frames}


def court_keypoints(model,frame,device='cpu'):
    best=None
    for imgsz in (640,960):
        r=model.predict(frame,imgsz=imgsz,device=device,verbose=False,conf=0.3)[0]
        if len(r.boxes)==0 or r.keypoints is None: continue
        bi=int(np.argmax(r.boxes.conf.cpu().numpy()))
        xy=r.keypoints.xy.cpu().numpy()[bi]
        cf=(r.keypoints.conf.cpu().numpy()[bi] if r.keypoints.conf is not None else np.ones(len(xy)))
        best=(xy,cf,imgsz)
        if int((cf>=0.5).sum())>=6: break
    return best


def person_boxes(model,frame,device='cpu'):
    r=model.predict(frame,imgsz=640,device=device,verbose=False,conf=0.18,classes=[0])[0]
    if len(r.boxes)==0: return []
    xyxy=r.boxes.xyxy.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
    return [(float(a),float(b),float(c),float(d),float(q)) for (a,b,c,d),q in zip(xyxy,cf)]


def max_neighbors(points,radius_ft):
    if len(points)<3: return 0
    a=np.asarray(points,dtype=float)
    d=np.sqrt(((a[:,None,:]-a[None,:,:])**2).sum(axis=2))
    np.fill_diagonal(d,np.inf)
    return int(max((d<=radius_ft).sum(axis=1)))


def process_frame(frame,person_model,court_model,device='cpu'):
    boxes=person_boxes(person_model,frame,device=device)
    rec={'person_detections':len(boxes),'calibrated':False,'positioned':0,
         'max_neighbors_10ft':0,'max_neighbors_12ft':0,'max_neighbors_14ft':0}
    # A negative is only possible from frames with broad enough player visibility.
    if len(boxes)<8:
        rec['reason']='fewer_than_8_person_detections'; return rec
    kp=court_keypoints(court_model,frame,device=device)
    if kp is None:
        rec['reason']='no_court_keypoints'; return rec
    xy,cf,imgsz=kp
    H,info=fit_homography(xy,cf,frame_hw=frame.shape[:2])
    rec['court_imgsz']=imgsz; rec['calibration_info']=info
    if H is None:
        rec['reason']='court_calibration_rejected'; return rec
    rec['calibrated']=True
    pts=[]
    for x1,y1,x2,y2,q in boxes:
        cm=project_point(H,(x1+x2)/2.0,y2)
        if court_position_ok(cm): pts.append((float(cm[0])/30.48,float(cm[1])/30.48))
    rec['positioned']=len(pts)
    if len(pts)<8:
        rec['reason']='fewer_than_8_court_positions'; return rec
    rec['observable']=True
    rec['max_neighbors_10ft']=max_neighbors(pts,10.0)
    rec['max_neighbors_12ft']=max_neighbors(pts,12.0)
    rec['max_neighbors_14ft']=max_neighbors(pts,14.0)
    rec['reason']='observable'
    return rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--person-model',default='yolo11n.pt')
    ap.add_argument('--court-model',default='/tmp/nbacv/models/court_yolo11m_pose.pt')
    ap.add_argument('--sample-fps',type=float,default=2.0)
    ap.add_argument('--max-seconds',type=float,default=12.0)
    ap.add_argument('--min-observable-frames',type=int,default=8)
    ap.add_argument('--min-observable-fraction',type=float,default=0.65)
    ap.add_argument('--device',default='cpu')
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10)
    person_model=YOLO(a.person_model); court_model=YOLO(a.court_model)
    results=[]; details=[]
    with tempfile.TemporaryDirectory(prefix='kd_fast_screen_') as td:
        root=Path(td)
        for ri,r in df.iterrows():
            t0=time.perf_counter(); game=str(r.game_id).zfill(10); evs=nums(r.screen_event_nums)
            sampled=observable=cluster10=cluster12=cluster14=calibrated=0; maxpos=0; errors=[]; angles=[]
            for event in evs:
                evdir=root/f'{game}_{event}'
                try:
                    meta=extract_sparse_frames(game,event,evdir,fps=a.sample_fps,max_seconds=a.max_seconds)
                    angles.append(meta['angle'])
                    for j,p in enumerate(meta['frames']):
                        frame=cv2.imread(str(p))
                        if frame is None: continue
                        sampled+=1
                        fr=process_frame(frame,person_model,court_model,device=a.device)
                        calibrated+=int(fr.get('calibrated',False)); observable+=int(fr.get('observable',False))
                        maxpos=max(maxpos,int(fr.get('positioned',0)))
                        cluster10+=int(fr.get('max_neighbors_10ft',0)>=2)
                        cluster12+=int(fr.get('max_neighbors_12ft',0)>=2)
                        cluster14+=int(fr.get('max_neighbors_14ft',0)>=2)
                        details.append({'possession_uid':r.possession_uid,'game_id':game,'event_num':event,
                                        'sample_index':j,'person_detections':fr.get('person_detections',0),
                                        'calibrated':fr.get('calibrated',False),'positioned':fr.get('positioned',0),
                                        'observable':fr.get('observable',False),'max_neighbors_10ft':fr.get('max_neighbors_10ft',0),
                                        'max_neighbors_12ft':fr.get('max_neighbors_12ft',0),'max_neighbors_14ft':fr.get('max_neighbors_14ft',0),
                                        'reason':fr.get('reason','')})
                except Exception as e:
                    errors.append(f'event {event}: {type(e).__name__}: {e}')
                finally:
                    shutil.rmtree(evdir,ignore_errors=True)
            obs_frac=(observable/sampled) if sampled else 0.0
            enough=observable>=a.min_observable_frames and obs_frac>=a.min_observable_fraction
            # Stage A is deliberately one-sided. Any plausible three-person cluster at 12 ft survives.
            # Poor visibility also survives. Only broad, well-observed, cluster-free possessions are screened negative.
            if not enough:
                decision='candidate'; reason='insufficient_observability'
            elif cluster12>0:
                decision='candidate'; reason='plausible_three_person_cluster_within_12ft'
            else:
                decision='screen_negative'; reason='well_observed_and_no_three_person_cluster_within_12ft'
            results.append({
                'season':r.get('season','2025-26'),'game_id':game,'possession_uid':r.possession_uid,
                'candidate_priority':r.candidate_priority,'screen_event_nums':r.screen_event_nums,
                'screen_decision':decision,'screen_reason':reason,'sampled_frames':sampled,'calibrated_frames':calibrated,
                'observable_frames':observable,'observable_fraction':round(obs_frac,4),'cluster_frames_10ft':cluster10,
                'cluster_frames_12ft':cluster12,'cluster_frames_14ft':cluster14,'max_positioned_tracks':maxpos,
                'resolved_angles':'|'.join(sorted(set(angles))),'errors':' || '.join(errors),
                'runtime_seconds':round(time.perf_counter()-t0,2),
                'semantics':'screen only; candidate is NOT a confirmed double and screen_negative is subject to validation audit'
            })
            print(json.dumps(results[-1],default=str),flush=True)
    out=pd.DataFrame(results); out.to_csv(a.out/'screen_results.csv',index=False)
    pd.DataFrame(details).to_csv(a.out/'frame_detail.csv',index=False)
    qa={
        'rows':int(len(out)),'screen_negative':int((out.screen_decision=='screen_negative').sum()),
        'candidate':int((out.screen_decision=='candidate').sum()),
        'candidate_retention_rate':float((out.screen_decision=='candidate').mean()) if len(out) else None,
        'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,
        'total_runtime_seconds':float(out.runtime_seconds.sum()) if len(out) else 0,
        'mean_observable_fraction':float(out.observable_fraction.mean()) if len(out) else None,
        'screen_radius_ft':12.0,'note':'12 ft is a conservative screening margin, not the Sloan double-team radius'
    }
    (a.out/'screen_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
