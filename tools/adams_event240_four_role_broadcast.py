#!/usr/bin/env python3
"""Deterministic event-240 presentation render with ONLY the four screen participants.

No generated/altered basketball pixels. The underlying video is the official NBA
Broadcast HLS source produced by adams_screen_prod_v3.py. Graphics are OpenCV only.

Validated event-240 participant mapping (V3 track IDs from the locked interaction
window):
  T29 Reed Sheppard      ballhandler
  T34 Steven Adams       screener
  T35 Ajay Mitchell      screened defender
  T13 Isaiah Hartenstein screener defender
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from render_deterministic_master import render as render_presentation

PLAYERS = {
    29: {"label": "#15 SHEPPARD", "team": "HOU", "role": "ballhandler", "name": "Reed Sheppard"},
    34: {"label": "#12 ADAMS", "team": "HOU", "role": "screener", "name": "Steven Adams"},
    35: {"label": "#25 MITCHELL", "team": "OKC", "role": "screened defender", "name": "Ajay Mitchell"},
    13: {"label": "#55 HARTENSTEIN", "team": "OKC", "role": "screener defender", "name": "Isaiah Hartenstein"},
}
COLORS = {"HOU": (36, 54, 226), "OKC": (230, 133, 28)}
DARK = {"HOU": (18, 27, 105), "OKC": (90, 55, 10)}
OFFSETS = {29: (-62, -16), 34: (-82, -2), 35: (58, -16), 13: (74, -2)}


def rounded_rect(img, p1, p2, color, r=5):
    x1, y1 = p1; x2, y2 = p2
    r = max(1, min(r, (x2-x1)//2, (y2-y1)//2))
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1, cv2.LINE_AA)
    for x, y in ((x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)):
        cv2.circle(img, (x,y), r, color, -1, cv2.LINE_AA)


def interp_box(s: pd.DataFrame, t: float):
    ts = s.time_s.to_numpy(float)
    if len(ts) == 0 or t < ts[0]-0.05 or t > ts[-1]+0.05:
        return None
    j = int(np.searchsorted(ts, t))
    if j <= 0:
        r = s.iloc[0]; return [float(r[c]) for c in ("x1","y1","x2","y2")]
    if j >= len(s):
        r = s.iloc[-1]; return [float(r[c]) for c in ("x1","y1","x2","y2")]
    a, b = s.iloc[j-1], s.iloc[j]
    if float(b.time_s-a.time_s) > 0.35:
        return None
    q = (t-float(a.time_s)) / (float(b.time_s-a.time_s) + 1e-9)
    return [(1-q)*float(a[c])+q*float(b[c]) for c in ("x1","y1","x2","y2")]


def draw_ring(frame, box, color, dark):
    x1,y1,x2,y2 = box
    cx = int(round((x1+x2)/2)); cy = int(round(y2-1))
    rx = int(np.clip((x2-x1)*0.82, 26, 48)); ry = int(np.clip(rx*0.25, 7, 12))
    for c,thick,dy in ((dark,5,1),(color,3,0)):
        cv2.ellipse(frame,(cx,cy+dy),(rx,ry),0,28,152,c,thick,cv2.LINE_AA)
        cv2.ellipse(frame,(cx,cy+dy),(rx,ry),0,208,332,c,thick,cv2.LINE_AA)


def draw_label(frame, box, text, color, dark, offset):
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = box; cx = int(round((x1+x2)/2))
    font = cv2.FONT_HERSHEY_DUPLEX; scale = .55; thick = 1
    (tw,th),base = cv2.getTextSize(text,font,scale,thick)
    px,py = 8,5; W,H = tw+2*px, th+base+2*py
    ox,oy = offset
    x = int(round(cx-W/2+ox)); y = int(round(y1-H-8+oy))
    x = max(3,min(w-W-3,x)); y = max(3,min(h-H-3,y))
    rounded_rect(frame,(x+2,y+2),(x+W+2,y+H+2),(16,16,16),5)
    rounded_rect(frame,(x,y),(x+W,y+H),dark,5)
    rounded_rect(frame,(x+1,y+1),(x+W-1,y+H-1),color,4)
    cv2.putText(frame,text,(x+px,y+py+th),font,scale,(255,255,255),thick,cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--tracks', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start-s', type=float, default=9.0)
    ap.add_argument('--end-s', type=float, default=13.45)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.tracks)
    by = {tid: df[df.track_id==tid].sort_values('time_s') for tid in PLAYERS}
    missing = [tid for tid,s in by.items() if s.empty]
    if missing:
        raise SystemExit(f"Required validated participant tracks missing: {missing}")

    cap = cv2.VideoCapture(str(a.source)); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start_s*1000)
    silent = a.out/'event240_four_involved_silent.mp4'
    wr = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    preview_done = False; frames = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        if t < a.start_s-0.05: continue
        if t > a.end_s: break
        for tid,meta in PLAYERS.items():
            box = interp_box(by[tid], t)
            if box is None or box[3]-box[1] < 70: continue
            color,dark = COLORS[meta['team']],DARK[meta['team']]
            draw_ring(fr,box,color,dark)
            draw_label(fr,box,meta['label'],color,dark,OFFSETS[tid])
        wr.write(fr); frames += 1
        if not preview_done and 10.18 <= t <= 10.28:
            cv2.imwrite(str(a.out/'preview_native.jpg'), fr, [cv2.IMWRITE_JPEG_QUALITY,96]); preview_done=True
    wr.release(); cap.release()

    native = a.out/'event240_four_involved_native.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(silent),'-ss',str(a.start_s),'-to',str(a.end_s),'-i',str(a.source),
                    '-map','0:v:0','-map','1:a:0?','-c:v','libx264','-profile:v','high','-crf','16','-preset','medium','-pix_fmt','yuv420p',
                    '-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',str(native)], check=True)
    uhd = a.out/'event240_four_involved_UHD.mp4'
    uhd_qa = render_presentation(native, uhd, 'uhd', preset='veryfast')
    (a.out/'uhd_render_qa.json').write_text(json.dumps(uhd_qa, indent=2), encoding='utf-8')
    manifest = {
        'game_id':'0022500001','event_num':240,'gold_positive':True,
        'source':'official NBA Broadcast HLS, native 960x540','clip_window_s':[a.start_s,a.end_s],
        'annotated_players':[{'name':m['name'],'label':m['label'],'team':m['team'],'track_id':tid,'role':m['role']} for tid,m in PLAYERS.items()],
        'graphics':'deterministic OpenCV only; no AI-generated or altered basketball frames',
        'uhd':'deterministic presentation upscale only; not native 4K',
        'frames_rendered':frames
    }
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run(['zip','-j','-q',str(a.out/'event240_four_involved_package.zip'),str(uhd),str(native),str(a.out/'preview_native.jpg'),str(a.out/'manifest.json')],check=True)
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
