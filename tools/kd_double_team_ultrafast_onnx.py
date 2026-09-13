#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import cv2
import numpy as np
import pandas as pd
import requests

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from fetch_official_nba_event_clip import parse as resolve_clip, UA

HEADERS={'User-Agent':UA,'Referer':'https://clips.nba.com/'}


def nums(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for x in str(v).split('|'):
        x=x.strip()
        if x.isdigit() and int(x) not in out: out.append(int(x))
    return out


def choose_hls_variant(url:str,target_width:int=640):
    """Prefer a smaller master-playlist rendition for screening only."""
    try:
        r=requests.get(url,headers=HEADERS,timeout=20); r.raise_for_status(); text=r.text
    except Exception:
        return url,None
    if '#EXT-X-STREAM-INF' not in text:
        return url,None
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    variants=[]
    for i,line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF'): continue
        m=re.search(r'RESOLUTION=(\d+)x(\d+)',line)
        bw=re.search(r'BANDWIDTH=(\d+)',line)
        uri=None
        for j in range(i+1,min(i+4,len(lines))):
            if not lines[j].startswith('#'):
                uri=lines[j]; break
        if not uri: continue
        w=int(m.group(1)) if m else 10**9; h=int(m.group(2)) if m else None
        b=int(bw.group(1)) if bw else None
        full=urljoin(url,uri)
        # Some masters rely on the master query for signed children.
        if not urlsplit(full).query and urlsplit(url).query:
            q=urlsplit(url); f=urlsplit(full)
            full=urlunsplit((f.scheme,f.netloc,f.path,q.query,f.fragment))
        variants.append((w,h,b,full))
    if not variants: return url,None
    under=[v for v in variants if v[0]<=target_width]
    chosen=max(under,key=lambda x:x[0]) if under else min(variants,key=lambda x:x[0])
    return chosen[3],{'width':chosen[0],'height':chosen[1],'bandwidth':chosen[2]}


def extract_sparse_frames(game,event,outdir,fps=2.0,max_seconds=12.0,target_width=640):
    page,opts,ch=resolve_clip(game,event)
    hls,variant=choose_hls_variant(ch['url'],target_width)
    outdir.mkdir(parents=True,exist_ok=True)
    headers=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    pat=str(outdir/'frame_%03d.jpg')
    subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',headers,'-i',hls,
                    '-t',str(max_seconds),'-vf',f'fps={fps},scale=640:-2','-q:v','5',pat],check=True,timeout=120)
    return {'page_url':page,'angle':ch['label'],'angle_count':len(opts),'variant':variant,
            'frames':sorted(outdir.glob('frame_*.jpg'))}


def letterbox(im,new=640):
    h,w=im.shape[:2]; scale=min(new/w,new/h); nw,nh=int(round(w*scale)),int(round(h*scale))
    resized=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    dw=new-nw; dh=new-nh; left=dw//2; top=dh//2
    canvas=np.full((new,new,3),114,dtype=np.uint8); canvas[top:top+nh,left:left+nw]=resized
    return canvas,scale,left,top


class YoloOnnxPerson:
    def __init__(self,path,conf=0.14,iou=0.50,input_size=640):
        self.net=cv2.dnn.readNetFromONNX(str(path)); self.conf=conf; self.iou=iou; self.input_size=input_size
    def predict(self,frame):
        inp,scale,left,top=letterbox(frame,self.input_size)
        blob=cv2.dnn.blobFromImage(inp,1/255.0,(self.input_size,self.input_size),swapRB=True,crop=False)
        self.net.setInput(blob); out=self.net.forward()
        arr=np.asarray(out)
        if arr.ndim==3: arr=arr[0]
        # Standard Ultralytics export is [84,N]; tolerate [N,84].
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 100: arr=arr.T
        if arr.shape[1] < 5: return []
        person=arr[:,4]
        keep=np.where(person>=self.conf)[0]
        if len(keep)==0: return []
        a=arr[keep]; scores=person[keep]
        xywh=a[:,:4]
        boxes=[]
        for x,y,w,h in xywh:
            x1=(x-w/2-left)/scale; y1=(y-h/2-top)/scale; x2=(x+w/2-left)/scale; y2=(y+h/2-top)/scale
            boxes.append([float(x1),float(y1),float(x2-x1),float(y2-y1)])
        idx=cv2.dnn.NMSBoxes(boxes,scores.tolist(),self.conf,self.iou)
        if len(idx)==0: return []
        idx=np.array(idx).reshape(-1)
        h0,w0=frame.shape[:2]; outb=[]
        for ii in idx:
            x,y,w,h=boxes[int(ii)]; x1=max(0,x); y1=max(0,y); x2=min(w0,x+w); y2=min(h0,y+h)
            bh=y2-y1; bw=x2-x1
            # broad court-player envelope; errs toward retaining false positives.
            if bh < 0.070*h0 or bh > 0.78*h0 or bw<=0 or bh/bw<1.0: continue
            outb.append((x1,y1,x2,y2,float(scores[int(ii)])))
        return outb


def crowding(bs,radius_body_heights):
    n=len(bs)
    if n<3: return {'max_neighbors':0,'best_target':None,'neighbor_dists':[]}
    best=(0,None,[])
    for i,(x1,y1,x2,y2,q) in enumerate(bs):
        cx=(x1+x2)/2.; fy=y2; h=max(1.,y2-y1); ds=[]
        for j,(a,b,c,d,qq) in enumerate(bs):
            if i==j: continue
            px=(a+c)/2.; py=d
            dn=float(np.hypot((px-cx)/h,0.72*(py-fy)/h)); ds.append((dn,j))
        ds.sort(); cnt=sum(d<=radius_body_heights for d,_ in ds)
        if cnt>best[0]: best=(cnt,i,[round(d,3) for d,_ in ds[:4]])
    return {'max_neighbors':int(best[0]),'best_target':best[1],'neighbor_dists':best[2]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--onnx-model',required=True)
    ap.add_argument('--sample-fps',type=float,default=2.0); ap.add_argument('--max-seconds',type=float,default=12.0)
    ap.add_argument('--radius-body-heights',type=float,default=2.5); ap.add_argument('--min-visible-persons',type=int,default=7)
    ap.add_argument('--min-observable-frames',type=int,default=8); ap.add_argument('--min-observable-fraction',type=float,default=0.55)
    ap.add_argument('--candidate-streak',type=int,default=2); ap.add_argument('--target-hls-width',type=int,default=640)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10)
    model=YoloOnnxPerson(a.onnx_model)
    rows=[]; detail=[]
    with tempfile.TemporaryDirectory(prefix='kd_ultrafast_onnx_') as td:
        root=Path(td)
        for _,r in df.iterrows():
            t0=time.perf_counter(); game=str(r.game_id).zfill(10); evs=nums(r.screen_event_nums)
            sampled=observable=cluster_frames=longest=cur=max_people=0; errors=[]; angles=[]; variants=[]
            for event in evs:
                evdir=root/f'{game}_{event}'
                try:
                    meta=extract_sparse_frames(game,event,evdir,a.sample_fps,a.max_seconds,a.target_hls_width)
                    angles.append(meta['angle']); variants.append(meta.get('variant'))
                    for k,p in enumerate(meta['frames']):
                        fr=cv2.imread(str(p));
                        if fr is None: continue
                        sampled+=1; bs=model.predict(fr); max_people=max(max_people,len(bs)); obs=len(bs)>=a.min_visible_persons
                        if obs: observable+=1
                        c=crowding(bs,a.radius_body_heights) if obs else {'max_neighbors':0,'best_target':None,'neighbor_dists':[]}
                        hit=bool(obs and c['max_neighbors']>=2)
                        if hit: cluster_frames+=1; cur+=1; longest=max(longest,cur)
                        else: cur=0
                        detail.append({'possession_uid':r.possession_uid,'game_id':game,'event_num':event,'sample_index':k,
                                       'people':len(bs),'observable':obs,'cluster_hit':hit,'max_neighbors':c['max_neighbors'],
                                       'best_target':c['best_target'],'nearest_norm_dists':'|'.join(map(str,c['neighbor_dists']))})
                except Exception as e: errors.append(f'event {event}: {type(e).__name__}: {e}')
                finally: shutil.rmtree(evdir,ignore_errors=True)
            frac=observable/sampled if sampled else 0.0; enough=observable>=a.min_observable_frames and frac>=a.min_observable_fraction
            if not enough: decision='candidate'; reason='insufficient_visibility'
            elif longest>=a.candidate_streak: decision='candidate'; reason='persistent_generic_three_person_convergence'
            else: decision='screen_negative'; reason='well_observed_no_persistent_generic_three_person_convergence'
            row={'season':r.get('season','2025-26'),'game_id':game,'possession_uid':r.possession_uid,'candidate_priority':r.candidate_priority,
                 'screen_event_nums':r.screen_event_nums,'screen_decision':decision,'screen_reason':reason,'sampled_frames':sampled,
                 'observable_frames':observable,'observable_fraction':round(frac,4),'cluster_frames':cluster_frames,'longest_cluster_streak':longest,
                 'max_people_detected':max_people,'radius_body_heights':a.radius_body_heights,'resolved_angles':'|'.join(sorted(set(angles))),
                 'hls_variants':json.dumps(variants,separators=(',',':')),'errors':' || '.join(errors),'runtime_seconds':round(time.perf_counter()-t0,2),
                 'semantics':'ONNX image-space upper-bound screen only; no positive is a confirmed double'}
            rows.append(row); print(json.dumps(row,default=str),flush=True)
    out=pd.DataFrame(rows); out.to_csv(a.out/'screen_results.csv',index=False); pd.DataFrame(detail).to_csv(a.out/'frame_detail.csv',index=False)
    qa={'rows':int(len(out)),'candidate':int((out.screen_decision=='candidate').sum()),'screen_negative':int((out.screen_decision=='screen_negative').sum()),
        'candidate_retention_rate':float((out.screen_decision=='candidate').mean()) if len(out) else None,'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,
        'total_runtime_seconds':float(out.runtime_seconds.sum()) if len(out) else 0,'mean_observable_fraction':float(out.observable_fraction.mean()) if len(out) else None,
        'radius_body_heights':a.radius_body_heights,'note':'No court homography, KD ID, team ID or ball ID at Stage A. Poor visibility always survives.'}
    (a.out/'screen_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
