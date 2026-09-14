#!/usr/bin/env python3
"""Deterministic Christmas event 94 V9 multi-angle release montage.

Input is the validated V7 native broadcast overlay plus the rebuilt event-94
tracks and official clips.nba.com multi-angle downloads. This script:

1. applies the locked V8 presentation tweaks deterministically (official team
   ring colours and 17 FT shot-distance tag at release),
2. audio-synchronizes official alternate angles to Broadcast,
3. freezes the exact synchronized release state from two Above Rim cameras and
   one Handheld camera,
4. inserts a rapid deterministic cinema-style push/cross transition sequence,
5. returns to Broadcast and resumes the play to completion,
6. renders the final UHD presentation through render_deterministic_master.

No image generation, generative fill, synthesized player/ball frames or
appearance-based inference is used.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import correlate, correlation_lags

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_deterministic_master import render as render_presentation

PLAYERS = {9: ('DURANT','HOU'), 6: ('ADAMS','HOU'), 1: ('LARAVIA','LAL'), 11: ('AYTON','LAL')}
TEAM = {
    'HOU': {'rgb': (206,17,65),  'bgr': (65,17,206)},   # Rockets official main red
    'LAL': {'rgb': (253,185,39), 'bgr': (39,185,253)},  # Lakers official gold/yellow
}
START = 7.20
RELEASE = 10.56
BASE_FREEZE_S = 1.20
SHOT_DISTANCE_FT = 17.0
FONT_REG='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'


def font(sz,bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, sz)


def pil_text(frame, xy, text, size, fill=(255,255,255), bold=False, anchor=None):
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(im)
    d.text(xy,text,font=font(size,bold),fill=fill,anchor=anchor)
    return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)


def alpha_rect(frame,p1,p2,color,alpha):
    ov=frame.copy(); cv2.rectangle(ov,p1,p2,color,-1); cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)


def darker(bgr, scale=.58):
    return tuple(int(max(0,min(255,c*scale))) for c in bgr)


def lighter(bgr, bump=20):
    return tuple(int(max(0,min(255,c+bump))) for c in bgr)


def draw_ring(frame,cx,cy,team):
    """Locked equal-size ring geometry; no black pixels; official team main colour."""
    rx,ry=35,13; cx=int(round(cx)); cy=int(round(cy)); base=TEAM[team]['bgr']
    rear=darker(base); outer=lighter(base)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,188,253,rear,5,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,287,352,rear,5,cv2.LINE_AA)
    for a0,a1 in ((0,180),(165,200),(340,375)):
        cv2.ellipse(frame,(cx,cy),(rx,ry),0,a0,a1,outer,8,cv2.LINE_AA)
        cv2.ellipse(frame,(cx,cy),(rx,ry),0,a0,a1,base,5,cv2.LINE_AA)


def draw_distance_tag(frame,cx,name_top_y):
    text=f'{int(round(SHOT_DISTANCE_FT))} FT'; fs=12; f=font(fs,True)
    d=ImageDraw.Draw(Image.new('RGB',(1,1))); bb=d.textbbox((0,0),text,font=f); tw=bb[2]-bb[0]
    w=tw+18; h=20; x=int(round(cx-w/2)); y=int(round(name_top_y-h-6))
    x=max(2,min(frame.shape[1]-w-2,x)); y=max(2,min(frame.shape[0]-h-2,y))
    alpha_rect(frame,(x,y),(x+w,y+h),(72,72,74),.72); cv2.rectangle(frame,(x,y),(x+w,y+h),(112,112,116),1,cv2.LINE_AA)
    return pil_text(frame,(x+w/2,y+h/2),text,fs,(255,255,255),True,'mm')


def build_interpolators(df):
    out={}
    for tid in PLAYERS:
        g=df[df.track_id==tid].sort_values('time_s')
        if g.empty: continue
        t=g.time_s.to_numpy(float); vals=g[['x1','y1','x2','y2']].to_numpy(float)
        sm=vals.copy(); ker=np.array([1,2,3,2,1],float); ker/=ker.sum()
        for j in range(4):
            sm[:,j]=np.convolve(np.pad(vals[:,j],(2,2),mode='edge'),ker,mode='valid')
        out[tid]=(t,sm)
    return out


def box_at(interp,tid,t):
    if tid not in interp:return None
    tt,v=interp[tid]
    if t<tt[0]-.08 or t>tt[-1]+.08:return None
    return tuple(float(np.interp(t,tt,v[:,j])) for j in range(4))


def audio_feature(path:Path, sr=8000, hop=80):
    cmd=['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(sr),'-f','f32le','pipe:1']
    raw=subprocess.check_output(cmd)
    x=np.frombuffer(raw,dtype=np.float32)
    if len(x)<hop*10:return np.zeros(1,np.float32),sr/hop
    x=np.diff(x,prepend=x[:1]); x=np.abs(x)
    n=len(x)//hop; x=x[:n*hop].reshape(n,hop).mean(axis=1)
    x=np.log1p(20*x); x=x-np.median(x); s=np.std(x)
    if s>1e-6:x=x/s
    return x.astype(np.float32),sr/hop


def audio_offset(ref_path:Path, alt_path:Path, max_shift_s=4.0):
    ref,hz=audio_feature(ref_path); alt,hz2=audio_feature(alt_path)
    assert abs(hz-hz2)<1e-6
    c=correlate(alt,ref,mode='full',method='fft'); lags=correlation_lags(len(alt),len(ref),mode='full')
    keep=np.abs(lags)<=int(round(max_shift_s*hz)); ck=c[keep]; lk=lags[keep]
    i=int(np.argmax(ck)); lag=int(lk[i])
    # normalize only as a quality indicator; positive lag => same event later in alternate clip.
    denom=float(np.linalg.norm(ref)*np.linalg.norm(alt)+1e-9); score=float(ck[i]/denom)
    return lag/hz,score


def read_frame(path:Path,t:float):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC,max(0,t)*1000); ok,fr=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'Could not read {path} at {t:.3f}s')
    return fr


def fit_960(fr):
    h,w=fr.shape[:2]
    scale=max(960/w,540/h); nw,nh=int(round(w*scale)),int(round(h*scale))
    r=cv2.resize(fr,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
    x=max(0,(nw-960)//2); y=max(0,(nh-540)//2)
    return r[y:y+540,x:x+960].copy()


def select_angles(angle_json:Path):
    j=json.loads(angle_json.read_text()); rows=[r for r in j['angles'] if r.get('status')=='ok']
    def label(r):return (r.get('label') or '').lower().replace('-',' ').replace('_',' ')
    above=[r for r in rows if 'above' in label(r) and 'rim' in label(r)]
    hand=[r for r in rows if 'handheld' in label(r) or 'hand held' in label(r) or ('hand' in label(r) and 'held' in label(r))]
    if len(above)<2:
        # NBA naming occasionally shortens the second basket camera to Rim/Backboard.
        extras=[r for r in rows if ('rim' in label(r) or 'backboard' in label(r)) and r not in above]
        above.extend(extras)
    if not hand:
        hand=[r for r in rows if 'hand' in label(r)]
    if len(above)<2 or len(hand)<1:
        raise RuntimeError('Required official angles not resolved. Available labels: '+', '.join(r.get('label','') for r in rows))
    return [above[0],above[1],hand[0]],rows


def cinematic_transition(a,b,n):
    """Deterministic 3-5 frame push/cross transition, no synthesized scene content."""
    out=[]; H,W=a.shape[:2]
    for k in range(1,n+1):
        q=k/(n+1); shift=int(round(54*(1-q)))
        M1=np.float32([[1,0,-int(34*q)],[0,1,0]])
        M2=np.float32([[1,0,shift],[0,1,0]])
        aa=cv2.warpAffine(a,M1,(W,H),borderMode=cv2.BORDER_REFLECT)
        bb=cv2.warpAffine(b,M2,(W,H),borderMode=cv2.BORDER_REFLECT)
        fr=cv2.addWeighted(aa,1-q,bb,q,0)
        # one restrained exposure lift at transition midpoint gives a fast cinema shutter feel.
        lift=1.0 + 0.10*math.sin(math.pi*q)
        fr=np.clip(fr.astype(np.float32)*lift,0,255).astype(np.uint8)
        out.append(fr)
    return out


def build_v8_base(v7_native:Path,tracks:Path,out:Path,qa:dict):
    df=pd.read_csv(tracks); interp=build_interpolators(df)
    cap=cv2.VideoCapture(str(v7_native)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 29.97); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    release_out=RELEASE-START; freeze_end=release_out+BASE_FREEZE_S
    silent=out/'v8base_silent.mp4'; wr=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); i=0
    while True:
        ok,fr=cap.read()
        if not ok:break
        to=i/fps
        ts=START+to if to<=release_out else (RELEASE if to<=freeze_end else START+to-BASE_FREEZE_S)
        cur={}
        for tid,(name,tm) in PLAYERS.items():
            b=box_at(interp,tid,ts)
            if b is None:continue
            cur[tid]=b; draw_ring(fr,(b[0]+b[2])/2,b[3]-4,tm)
        if release_out-.01<=to<=freeze_end+.01 and 9 in cur:
            b=cur[9]; fr=draw_distance_tag(fr,(b[0]+b[2])/2,b[1]-31)
        wr.write(fr); i+=1
    cap.release(); wr.release()
    native=out/'v8base_native.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(silent),'-i',str(v7_native),'-map','0:v:0','-map','1:a:0?','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart','-shortest',str(native)],check=True)
    return native,fps,release_out,freeze_end


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v7-native',type=Path,required=True)
    ap.add_argument('--broadcast-source',type=Path,required=True)
    ap.add_argument('--tracks',type=Path,required=True)
    ap.add_argument('--base-qa',type=Path,required=True)
    ap.add_argument('--angles-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    qa=json.loads(a.base_qa.read_text())
    v8,fps,release_out,freeze_end=build_v8_base(a.v7_native,a.tracks,a.out,qa)
    selected,all_rows=select_angles(a.angles_dir/'angles.json')

    sync=[]; stills=[]
    for r in selected:
        p=a.angles_dir/f"angle_{int(r['index']):02d}.mp4"
        off,score=audio_offset(a.broadcast_source,p)
        at=RELEASE+off
        fr=fit_960(read_frame(p,at)); stills.append(fr)
        sync.append({'index':r['index'],'label':r['label'],'audio_offset_s':off,'audio_corr_score':score,'release_time_s':at})

    # Read the release frame from the corrected V8 base itself.
    broadcast_release=fit_960(read_frame(v8,release_out))
    # QA sheet: each chosen view and +/- one frame equivalent around synchronized release.
    qa_cells=[]
    for rec in sync:
        p=a.angles_dir/f"angle_{int(rec['index']):02d}.mp4"
        imgs=[]
        for dt in (-1/30,0,1/30):
            fr=fit_960(read_frame(p,rec['release_time_s']+dt)); cv2.putText(fr,f"{rec['label']} {dt:+.3f}s",(18,34),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA); imgs.append(cv2.resize(fr,(480,270)))
        qa_cells.append(np.hstack(imgs))
    cv2.imwrite(str(a.out/'release_sync_qa.jpg'),np.vstack(qa_cells),[cv2.IMWRITE_JPEG_QUALITY,96])

    cap=cv2.VideoCapture(str(v8)); basefps=float(cap.get(cv2.CAP_PROP_FPS) or fps); frames=[]
    while True:
        ok,fr=cap.read()
        if not ok:break
        frames.append(fr)
    cap.release(); rel_i=int(round(release_out*basefps)); post_i=int(round(freeze_end*basefps))+1
    rel_i=max(0,min(len(frames)-1,rel_i)); post_i=max(rel_i+1,min(len(frames),post_i))

    hold_b=int(round(.18*basefps)); hold_alt=int(round(.30*basefps)); trans_n=max(2,int(round(.10*basefps))); hold_return=int(round(.16*basefps))
    montage=[]; current=broadcast_release
    montage.extend([current.copy() for _ in range(hold_b)])
    for nxt in stills:
        montage.extend(cinematic_transition(current,nxt,trans_n)); montage.extend([nxt.copy() for _ in range(hold_alt)]); current=nxt
    montage.extend(cinematic_transition(current,broadcast_release,trans_n)); montage.extend([broadcast_release.copy() for _ in range(hold_return)])
    montage_s=len(montage)/basefps

    silent=a.out/'v9_silent.mp4'; wr=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*'mp4v'),basefps,(960,540))
    for fr in frames[:rel_i+1]:wr.write(fit_960(fr))
    for fr in montage:wr.write(fr)
    for fr in frames[post_i:]:wr.write(fit_960(fr))
    wr.release()

    # Keep real pre/post broadcast audio; montage interval is silent like the existing release freeze.
    native=a.out/'durant_adams_christmas_multiview_v9_native.mp4'
    fc=(f"[0:a]atrim=start=0:end={release_out:.6f},asetpts=PTS-STARTPTS[a0];"
        f"anullsrc=r=48000:cl=stereo,atrim=duration={montage_s:.6f}[sil];"
        f"[0:a]atrim=start={freeze_end:.6f},asetpts=PTS-STARTPTS[a1];"
        f"[a0][sil][a1]concat=n=3:v=0:a=1[a]")
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(v8),'-i',str(silent),'-filter_complex',fc,'-map','1:v:0','-map','[a]','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',str(native)],check=True)
    uhd=a.out/'durant_adams_christmas_multiview_v9_UHD.mp4'; pqa=render_presentation(native,uhd,'uhd',preset='medium')
    qa9={'deterministic_only':True,'game_id':'0022500012','event_num':94,'release_source_s':RELEASE,'release_output_s':release_out,'selected_angles':sync,'montage_duration_s':montage_s,'transition':'deterministic rapid push/cross + restrained exposure lift','team_ring_rgb':{'HOU':[206,17,65],'LAL':[253,185,39]},'shot_distance_ft':SHOT_DISTANCE_FT,'native':str(native),'uhd':str(uhd),'presentation_qa':pqa}
    (a.out/'qa_v9.json').write_text(json.dumps(qa9,indent=2)); print(json.dumps(qa9,indent=2))

if __name__=='__main__':main()
