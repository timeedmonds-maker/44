#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
TOOLS=HERE.parent/'tools'
sys.path.insert(0,str(TOOLS))
from fetch_official_nba_event_clip import parse as resolve_clip, UA


def fetch_clip(game,event,out):
    page,opts,ch=resolve_clip(game,int(event))
    headers=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',headers,'-i',ch['url'],
                    '-map','0:v:0','-map','0:a:0?','-c','copy','-movflags','+faststart',str(out)],check=True,timeout=180)
    return {'page_url':page,'angle':ch['label'],'angle_count':len(opts)}


def contact_sheet(video,out_jpg,label,frames=20,width=420):
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx=np.linspace(0,max(n-1,0),frames).astype(int) if n else []
    ims=[]
    for f in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(f)); ok,im=cap.read()
        if not ok: continue
        h=max(1,int(round(H*width/max(W,1)))); im=cv2.resize(im,(width,h),interpolation=cv2.INTER_AREA)
        cv2.rectangle(im,(0,0),(width,28),(0,0,0),-1)
        cv2.putText(im,f'{label}  {f/fps:.2f}s',(6,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        ims.append(im)
    cap.release()
    if not ims: return False
    ph,pw=ims[0].shape[:2]; cols=4; rows=math.ceil(len(ims)/cols)
    sheet=np.zeros((rows*ph,cols*pw,3),np.uint8)
    for i,im in enumerate(ims): sheet[(i//cols)*ph:(i//cols+1)*ph,(i%cols)*pw:(i%cols+1)*pw]=im
    cv2.imwrite(str(out_jpg),sheet,[int(cv2.IMWRITE_JPEG_QUALITY),94]); return True


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidates',required=True); ap.add_argument('--out',required=True); ap.add_argument('--limit',type=int,default=12)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.candidates,dtype={'game_id':str}).head(a.limit); manifest=[]
    for _,r in df.iterrows():
        rank=int(r.get('rank',len(manifest)+1)); game=str(r.game_id).zfill(10); event=int(float(r.event_num_merge))
        stem=f'{rank:02d}_{game}_event{event}'; mp4=out/f'{stem}.mp4'; jpg=out/f'{stem}_contact_sheet.jpg'
        rec={'rank':rank,'game_id':game,'event_num':event,'description':str(r.get('event_description','')),'tier':str(r.get('candidate_tier','')),'score':float(r.get('pair_search_score',0))}
        try:
            rec.update(fetch_clip(game,event,mp4)); contact_sheet(mp4,jpg,f'R{rank} E{event}')
            rec['video']=mp4.name; rec['contact_sheet']=jpg.name
        except Exception as e:
            rec['error']=f'{type(e).__name__}: {e}'
        manifest.append(rec); print(json.dumps(rec),flush=True)
    (out/'preview_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
