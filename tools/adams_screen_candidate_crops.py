#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, sys, tempfile
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from kd_double_team_ballhandler_onnx import YoloOnnx, extract_frames
from adams_screen_temporal_onnx import associate, cluster_tracks, by_tid, norm_dist


def torso_crop(frame, box):
    x1,y1,x2,y2=[int(v) for v in box]; w=x2-x1; h=y2-y1
    cx1=x1+int(.18*w); cx2=x2-int(.18*w); cy1=y1+int(.12*h); cy2=y1+int(.55*h)
    c=frame[max(0,cy1):max(0,cy2),max(0,cx1):max(0,cx2)]
    return c if c.size else None


def player_detections(model, frame, conf=.14):
    from kd_double_team_ballhandler_onnx import players_on_courtish
    return players_on_courtish(model.detect_class(frame,0,conf),frame.shape[0])


def nearest_opp(boxes, labels, subject_tid, own_lab):
    if subject_tid not in boxes: return None
    S=boxes[subject_tid]; h=max(1.,S[3]-S[1]); best=None
    for tid,B in boxes.items():
        if tid==subject_tid or labels.get(tid) is None or labels.get(tid)==own_lab: continue
        d=norm_dist(S,B,h)
        if best is None or d<best[0]: best=(d,tid)
    return None if best is None else best[1]


def collect_track_crops(paths, tracked, tid, outdir, max_crops=24):
    samples=[]
    for fi,(pth,dets) in enumerate(zip(paths,tracked)):
        box=next((d['box'] for d in dets if d['tid']==tid),None)
        if box is None: continue
        fr=cv2.imread(str(pth))
        if fr is None: continue
        c=torso_crop(fr,box)
        if c is not None and c.shape[0]>=20 and c.shape[1]>=10:
            samples.append((fi,c))
    if not samples: return 0
    idx=np.linspace(0,len(samples)-1,min(max_crops,len(samples))).astype(int)
    outdir.mkdir(parents=True,exist_ok=True)
    for k,i in enumerate(idx):
        fi,c=samples[i]; cv2.imwrite(str(outdir/f'{k:02d}_f{fi:03d}.jpg'),c)
    return len(idx)


def annotate(frame, boxes, tids, labels):
    out=frame.copy(); role_colors={'ballhandler':(255,255,255),'screener':(0,255,255),'screened_defender':(255,0,255),'screener_defender':(0,165,255)}
    for role,tid in tids.items():
        if tid is None or tid not in boxes: continue
        x1,y1,x2,y2=[int(v) for v in boxes[tid]]; color=role_colors[role]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f'{role} T{tid}',(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
    return out


def make_contact(candidate_id, crop_root, out_path):
    roles=['screener','ballhandler','screened_defender','screener_defender']; W=220; H=190
    canvas=np.full((H*len(roles),W*4,3),245,np.uint8)
    for r_i,role in enumerate(roles):
        cv2.putText(canvas,role.replace('_',' '),(6,r_i*H+22),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1,cv2.LINE_AA)
        files=sorted((crop_root/candidate_id/role).glob('*.jpg')) if (crop_root/candidate_id/role).exists() else []
        if files:
            picks=np.linspace(0,len(files)-1,min(3,len(files))).astype(int)
            for j,idx in enumerate(picks,1):
                im=cv2.imread(str(files[idx]));
                if im is None: continue
                h,w=im.shape[:2]; scale=min((W-12)/max(1,w),(H-38)/max(1,h)); nw=max(1,int(w*scale)); nh=max(1,int(h*scale))
                rs=cv2.resize(im,(nw,nh)); x=j*W+(W-nw)//2; y=r_i*H+30+(H-36-nh)//2
                canvas[y:y+nh,x:x+nw]=rs
    cv2.imwrite(str(out_path),canvas)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--onnx-model',required=True)
    ap.add_argument('--sample-fps',type=float,default=6.0); ap.add_argument('--max-seconds',type=float,default=14.0); ap.add_argument('--target-hls-width',type=int,default=640); ap.add_argument('--person-conf',type=float,default=.14)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); crop_root=a.out/'crops'; evidence=a.out/'evidence'; contacts=a.out/'contacts'; evidence.mkdir(exist_ok=True); contacts.mkdir(exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10); model=YoloOnnx(a.onnx_model); meta_rows=[]
    with tempfile.TemporaryDirectory(prefix='adams_crop_') as td:
        root=Path(td)
        for ev,grp in df.groupby('event_num',sort=False):
            ev=int(ev); evdir=root/str(ev)
            try:
                meta=extract_frames('0022500001',ev,evdir,a.sample_fps,a.max_seconds,a.target_hls_width); paths=meta['frames']; frames_people=[]
                for pth in paths:
                    fr=cv2.imread(str(pth)); frames_people.append([] if fr is None else player_detections(model,fr,a.person_conf))
                tracked=associate(frames_people); labels,_centers=cluster_tracks(paths,tracked); boxes_pf=[by_tid(x) for x in tracked]
                for _,r in grp.iterrows():
                    cid=str(r.candidate_id); bh=int(r.ballhandler_tid); sc=int(r.screener_tid); sd=int(r.best_defender_tid); sf=int(r.start_sample)
                    probe=max(0,min(len(boxes_pf)-1,sf-max(1,int(round(.5*a.sample_fps)))))
                    boxes=boxes_pf[probe]; own_lab=labels.get(sc,labels.get(bh)); screener_def=nearest_opp(boxes,labels,sc,own_lab) if own_lab is not None else None
                    bh_def=nearest_opp(boxes,labels,bh,own_lab) if own_lab is not None else None
                    # Prefer the temporal interaction defender for the screened defender, but retain nearest pre-screen defender too.
                    if sd not in {d['tid'] for ds in tracked for d in ds}: sd=bh_def
                    roles={'screener':sc,'ballhandler':bh,'screened_defender':sd,'screener_defender':screener_def}
                    counts={role:collect_track_crops(paths,tracked,tid,crop_root/cid/role) if tid is not None else 0 for role,tid in roles.items()}
                    afi=max(0,min(len(paths)-1,int(round((float(r.start_sample)+float(r.end_sample))/2))))
                    fr=cv2.imread(str(paths[afi]));
                    if fr is not None: cv2.imwrite(str(evidence/f'{cid}_event{ev}.jpg'),annotate(fr,boxes_pf[afi],roles,labels))
                    make_contact(cid,crop_root,contacts/f'{cid}.jpg')
                    meta_rows.append({'candidate_id':cid,'event_num':ev,'angle':meta.get('angle'),'probe_sample':probe,'evidence_sample':afi,
                                      'ballhandler_tid':bh,'screener_tid':sc,'screened_defender_tid':sd,'screener_defender_tid':screener_def,
                                      **{f'{k}_crops':v for k,v in counts.items()},'tracking_ok':sc in {d['tid'] for ds in tracked for d in ds}})
            except Exception as e:
                for _,r in grp.iterrows(): meta_rows.append({'candidate_id':str(r.candidate_id),'event_num':ev,'error':f'{type(e).__name__}: {e}'})
            finally: shutil.rmtree(evdir,ignore_errors=True)
    pd.DataFrame(meta_rows).to_csv(a.out/'crop_metadata.csv',index=False)
    (a.out/'qa.json').write_text(json.dumps({'candidates':len(df),'metadata_rows':len(meta_rows),'errors':sum(bool(x.get('error')) for x in meta_rows)},indent=2))
if __name__=='__main__': main()
