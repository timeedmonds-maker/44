#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
import cv2, numpy as np, pandas as pd, requests

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


def choose_hls_variant(url,target_width=640):
    try:
        r=requests.get(url,headers=HEADERS,timeout=20); r.raise_for_status(); text=r.text
    except Exception: return url,None
    if '#EXT-X-STREAM-INF' not in text: return url,None
    lines=[x.strip() for x in text.splitlines() if x.strip()]; vs=[]
    for i,line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF'): continue
        m=re.search(r'RESOLUTION=(\d+)x(\d+)',line); bw=re.search(r'BANDWIDTH=(\d+)',line); uri=None
        for j in range(i+1,min(i+4,len(lines))):
            if not lines[j].startswith('#'): uri=lines[j]; break
        if not uri: continue
        w=int(m.group(1)) if m else 10**9; h=int(m.group(2)) if m else None; b=int(bw.group(1)) if bw else None
        full=urljoin(url,uri)
        if not urlsplit(full).query and urlsplit(url).query:
            q=urlsplit(url); f=urlsplit(full); full=urlunsplit((f.scheme,f.netloc,f.path,q.query,f.fragment))
        vs.append((w,h,b,full))
    if not vs: return url,None
    under=[v for v in vs if v[0]<=target_width]; ch=max(under,key=lambda x:x[0]) if under else min(vs,key=lambda x:x[0])
    return ch[3],{'width':ch[0],'height':ch[1],'bandwidth':ch[2]}


def extract_frames(game,event,outdir,fps=2,max_seconds=12,target_width=640):
    page,opts,ch=resolve_clip(game,event); hls,var=choose_hls_variant(ch['url'],target_width); outdir.mkdir(parents=True,exist_ok=True)
    headers=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'; pat=str(outdir/'f_%03d.jpg')
    subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',headers,'-i',hls,'-t',str(max_seconds),
                    '-vf',f'fps={fps},scale=640:-2','-q:v','5',pat],check=True,timeout=120)
    return {'frames':sorted(outdir.glob('f_*.jpg')),'angle':ch['label'],'variant':var,'page_url':page,'angle_count':len(opts)}


def letterbox(im,new=640):
    h,w=im.shape[:2]; s=min(new/w,new/h); nw,nh=int(round(w*s)),int(round(h*s)); r=cv2.resize(im,(nw,nh))
    dw,dh=new-nw,new-nh; l,t=dw//2,dh//2; c=np.full((new,new,3),114,np.uint8); c[t:t+nh,l:l+nw]=r
    return c,s,l,t


class YoloOnnx:
    def __init__(self,path,input_size=640): self.net=cv2.dnn.readNetFromONNX(str(path)); self.sz=input_size
    def raw(self,frame):
        inp,s,l,t=letterbox(frame,self.sz); blob=cv2.dnn.blobFromImage(inp,1/255.,(self.sz,self.sz),swapRB=True,crop=False)
        self.net.setInput(blob); a=np.asarray(self.net.forward()); a=a[0] if a.ndim==3 else a
        if a.shape[0]<a.shape[1] and a.shape[0]<=100: a=a.T
        return a,s,l,t
    def detect_class(self,frame,class_id,conf,iou=0.5):
        a,s,l,t=self.raw(frame); col=4+class_id
        if a.shape[1]<=col: return []
        scores=a[:,col]; ix=np.where(scores>=conf)[0]
        if not len(ix): return []
        boxes=[]
        for x,y,w,h in a[ix,:4]: boxes.append([(x-w/2-l)/s,(y-h/2-t)/s,w/s,h/s])
        keep=cv2.dnn.NMSBoxes(boxes,scores[ix].tolist(),conf,iou)
        if len(keep)==0: return []
        h0,w0=frame.shape[:2]; out=[]
        for k in np.asarray(keep).reshape(-1):
            x,y,w,h=boxes[int(k)]; x1=max(0.,float(x)); y1=max(0.,float(y)); x2=min(float(w0),float(x+w)); y2=min(float(h0),float(y+h))
            out.append((x1,y1,x2,y2,float(scores[ix[int(k)]])))
        return out


def players_on_courtish(dets,frame_h,max_players=13):
    out=[]
    for x1,y1,x2,y2,q in dets:
        bh=y2-y1; bw=x2-x1
        if bh<0.075*frame_h or bh>0.78*frame_h or bw<=0 or bh/bw<1.0: continue
        # Strongly down-rank distant bench/audience people while keeping refs.
        rank=bh*(0.55+0.45*min(1.0,y2/frame_h))*max(0.2,q)
        out.append((rank,(x1,y1,x2,y2,q)))
    out.sort(key=lambda x:x[0],reverse=True)
    return [b for _,b in out[:max_players]]


def point_rect_distance(px,py,b):
    x1,y1,x2,y2,_=b; dx=max(x1-px,0,px-x2); dy=max(y1-py,0,py-y2); return float(np.hypot(dx,dy))


def ballhandler(players,balls):
    best=None
    for ball in balls:
        bx=(ball[0]+ball[2])/2; by=(ball[1]+ball[3])/2
        for i,p in enumerate(players):
            h=max(1.,p[3]-p[1]); d=point_rect_distance(bx,by,p)/h
            # ball can be just outside a hand/arm box; keep generous.
            if best is None or d<best[0]: best=(d,i,ball)
    if best is None or best[0]>0.95: return None
    return {'player_index':best[1],'ball_box':best[2],'ball_to_player_norm':best[0]}


def neighbor_distances(players,target_i):
    t=players[target_i]; tx=(t[0]+t[2])/2; ty=t[3]; h=max(1.,t[3]-t[1]); ds=[]
    for j,p in enumerate(players):
        if j==target_i: continue
        px=(p[0]+p[2])/2; py=p[3]
        d=float(np.hypot((px-tx)/h,0.72*(py-ty)/h)); ds.append((d,j))
    ds.sort(); return ds


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--onnx-model',required=True)
    ap.add_argument('--sample-fps',type=float,default=2.0); ap.add_argument('--max-seconds',type=float,default=12.0); ap.add_argument('--target-hls-width',type=int,default=640)
    ap.add_argument('--second-neighbor-threshold',type=float,default=3.25); ap.add_argument('--min-ballhandler-frames',type=int,default=3)
    ap.add_argument('--min-player-count',type=int,default=6); ap.add_argument('--person-conf',type=float,default=0.14); ap.add_argument('--ball-conf',type=float,default=0.035)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10); model=YoloOnnx(a.onnx_model)
    rows=[]; detail=[]
    with tempfile.TemporaryDirectory(prefix='kd_bh_onnx_') as td:
        root=Path(td)
        for _,r in df.iterrows():
            t0=time.perf_counter(); game=str(r.game_id).zfill(10); sampled=bh_frames=candidate_frames=0; min_second=99.; errors=[]; angles=[]; max_players=0
            for event in nums(r.screen_event_nums):
                evdir=root/f'{game}_{event}'
                try:
                    meta=extract_frames(game,event,evdir,a.sample_fps,a.max_seconds,a.target_hls_width); angles.append(meta['angle'])
                    for k,pth in enumerate(meta['frames']):
                        fr=cv2.imread(str(pth));
                        if fr is None: continue
                        sampled+=1; people=players_on_courtish(model.detect_class(fr,0,a.person_conf),fr.shape[0]); balls=model.detect_class(fr,32,a.ball_conf,0.35); max_players=max(max_players,len(people))
                        bh=ballhandler(people,balls) if len(people)>=a.min_player_count else None
                        second=None; hit=False
                        if bh is not None:
                            ds=neighbor_distances(people,bh['player_index']); bh_frames+=1
                            if len(ds)>=2:
                                second=ds[1][0]; min_second=min(min_second,second); hit=second<=a.second_neighbor_threshold; candidate_frames+=int(hit)
                        detail.append({'possession_uid':r.possession_uid,'game_id':game,'event_num':event,'sample_index':k,'players':len(people),'balls':len(balls),
                                       'ballhandler_observed':bh is not None,'ball_to_player_norm':None if bh is None else round(bh['ball_to_player_norm'],3),
                                       'second_neighbor_norm':None if second is None else round(second,3),'candidate_frame':hit})
                except Exception as e: errors.append(f'event {event}: {type(e).__name__}: {e}')
                finally: shutil.rmtree(evdir,ignore_errors=True)
            if bh_frames<a.min_ballhandler_frames:
                decision='candidate'; reason='insufficient_ballhandler_observability'
            elif candidate_frames>0:
                decision='candidate'; reason='ballhandler_has_two_nearby_people_in_at_least_one_frame'
            else:
                decision='screen_negative'; reason='ballhandler_observed_and_second_nearest_person_never_entered_generous_radius'
            row={'season':r.get('season','2025-26'),'game_id':game,'possession_uid':r.possession_uid,'candidate_priority':r.candidate_priority,'screen_event_nums':r.screen_event_nums,
                 'screen_decision':decision,'screen_reason':reason,'sampled_frames':sampled,'ballhandler_frames':bh_frames,'candidate_frames':candidate_frames,
                 'min_second_neighbor_norm':None if min_second==99. else round(min_second,3),'max_players_retained':max_players,
                 'second_neighbor_threshold':a.second_neighbor_threshold,'resolved_angles':'|'.join(sorted(set(angles))),'errors':' || '.join(errors),
                 'runtime_seconds':round(time.perf_counter()-t0,2),'semantics':'ballhandler-centred upper-bound sieve; candidate is not a confirmed double'}
            rows.append(row); print(json.dumps(row,default=str),flush=True)
    out=pd.DataFrame(rows); out.to_csv(a.out/'screen_results.csv',index=False); pd.DataFrame(detail).to_csv(a.out/'frame_detail.csv',index=False)
    qa={'rows':int(len(out)),'candidate':int((out.screen_decision=='candidate').sum()),'screen_negative':int((out.screen_decision=='screen_negative').sum()),
        'candidate_retention_rate':float((out.screen_decision=='candidate').mean()) if len(out) else None,'mean_runtime_seconds':float(out.runtime_seconds.mean()) if len(out) else None,
        'mean_ballhandler_frames':float(out.ballhandler_frames.mean()) if len(out) else None,'rows_with_ballhandler_observability':int((out.ballhandler_frames>=a.min_ballhandler_frames).sum()),
        'error_rows':int(out.errors.fillna('').astype(str).ne('').sum()),'note':'Screen negative only when ballhandler is observed >= minimum frames and second-nearest person never enters a deliberately generous image-space radius.'}
    (a.out/'screen_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
