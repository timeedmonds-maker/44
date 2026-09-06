from __future__ import annotations
import argparse, json, math, shutil, subprocess
from pathlib import Path
import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 960, 540, 30
DEFAULT_ORDER = [
    "Left HandHeld", "Left Slash", "In Arena", "Mobile Broadcast",
    "Broadcast", "Right Slash", "Right HandHeld",
]

def load_frame(path: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"Could not extract {path} at {t}")
    if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    return frame

def rigid(image: np.ndarray, focus, zoom=1.0, dx=0.0, dy=0.0, border=cv2.BORDER_REFLECT_101):
    fx, fy = focus
    m = np.array([[zoom, 0, WIDTH/2 + dx - zoom*fx], [0, zoom, HEIGHT/2 + dy - zoom*fy]], np.float32)
    return cv2.warpAffine(image, m, (WIDTH, HEIGHT), flags=cv2.INTER_LANCZOS4, borderMode=border)

def alpha_warp(alpha: np.ndarray, focus, zoom=1.0, dx=0.0, dy=0.0):
    fx, fy = focus
    m = np.array([[zoom, 0, WIDTH/2 + dx - zoom*fx], [0, zoom, HEIGHT/2 + dy - zoom*fy]], np.float32)
    return cv2.warpAffine(alpha, m, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def make_action_alpha(focus):
    fx, fy = focus
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    dist = ((xx-fx)/(0.39*WIDTH))**2 + ((yy-fy)/(0.46*HEIGHT))**2
    return np.clip((1.10-dist)/0.20, 0, 1).astype(np.float32)

def compose(frame: np.ndarray, focus, alpha, zoom: float, drift: float):
    # Two rigid layers from the same real NBA frame. The action layer moves slightly
    # more than the background to create a shallow parallax cue. No player pixels bend.
    bg = rigid(frame, focus, zoom=zoom*0.985, dx=drift*20, dy=-abs(drift)*2)
    fg = rigid(frame, focus, zoom=zoom, dx=drift*31, dy=-abs(drift)*3)
    a = alpha_warp(alpha, focus, zoom=zoom, dx=drift*31, dy=-abs(drift)*3)[..., None]
    return np.clip(bg.astype(np.float32)*(1-a) + fg.astype(np.float32)*a, 0, 255).astype(np.uint8)

def hblur(frame: np.ndarray, strength: float):
    k = max(1, int(1 + 44*strength)); k += 1-k%2
    return cv2.GaussianBlur(frame, (k,1), 0)

def run(cmd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--contact-manifest', type=Path, required=True)
    ap.add_argument('--contact-dir', type=Path, required=True)
    ap.add_argument('--clips', type=Path, required=True)
    ap.add_argument('--impact-offset', type=float, default=0.88,
                    help='seconds into globally synchronized contact windows')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    cm=json.loads(args.contact_manifest.read_text())
    contacts={r['label']: args.contact_dir/r['file'] for r in cm['angles']}
    rows={r['label']:r for r in cfg['angles']}
    order=[x for x in DEFAULT_ORDER if x in contacts and 'focus' in rows[x]]
    if len(order)<5: raise RuntimeError(f'Insufficient focused views: {order}')
    frames={x:load_frame(contacts[x],args.impact_offset) for x in order}
    alphas={x:make_action_alpha(tuple(map(float,rows[x]['focus']))) for x in order}
    shutil.rmtree(args.work, ignore_errors=True); fd=args.work/'frames'; fd.mkdir(parents=True)
    rendered=[]
    for i,label in enumerate(order):
        focus=tuple(map(float,rows[label]['focus']))
        base_zoom=float(rows[label].get('zoom',1.0))
        zoom=max(1.04,min(1.15,base_zoom if base_zoom<1.16 else 1.13))
        for q in range(7):
            u=q/6; rendered.append(compose(frames[label],focus,alphas[label],zoom,(u-.5)*.18))
        if i==len(order)-1: continue
        nxt=order[i+1]
        focus_b=tuple(map(float,rows[nxt]['focus']))
        zoom_b=max(1.04,min(1.15,float(rows[nxt].get('zoom',1.08))))
        for q in range(8):
            t=(q+1)/9; ease=t*t*(3-2*t); blur=math.sin(math.pi*t)**0.65
            if t<0.5:
                fr=compose(frames[label],focus,alphas[label],zoom+0.015*ease,-0.15-0.55*ease)
            else:
                fr=compose(frames[nxt],focus_b,alphas[nxt],zoom_b-0.015*ease,0.55*(1-ease)+0.15)
            fr=hblur(fr,blur)
            fr=np.clip(fr.astype(np.float32)*(1-0.055*blur),0,255).astype(np.uint8)
            rendered.append(fr)
    for i,fr in enumerate(rendered): cv2.imwrite(str(fd/f'{i:05d}.png'),fr)
    orbit=args.work/'orbit.mp4'
    run(['ffmpeg','-nostdin','-y','-v','error','-framerate',str(FPS),'-i',str(fd/'%05d.png'),'-an','-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p',str(orbit)])
    broadcast=args.clips/rows['Broadcast']['file']
    corrected_freeze=float(rows['Broadcast']['freeze_time']) + (args.impact_offset-0.5)
    pre_start=max(0.0,corrected_freeze-2.4); slow_start=max(pre_start,corrected_freeze-0.50); post_end=corrected_freeze+2.2
    pre=args.work/'pre.mkv'; post=args.work/'post.mkv'
    run(['ffmpeg','-nostdin','-y','-v','error','-i',str(broadcast),'-filter_complex',
         f'[0:v]trim=start={pre_start}:end={slow_start},setpts=PTS-STARTPTS[v0];[0:v]trim=start={slow_start}:end={corrected_freeze},setpts=1.55*(PTS-STARTPTS)[v1];[v0][v1]concat=n=2:v=1:a=0,fps=30,format=yuv420p[v]',
         '-map','[v]','-an','-c:v','ffv1',str(pre)])
    run(['ffmpeg','-nostdin','-y','-v','error','-ss',f'{corrected_freeze:.5f}','-to',f'{post_end:.5f}','-i',str(broadcast),'-vf','fps=30,format=yuv420p','-an','-c:v','ffv1',str(post)])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    run(['ffmpeg','-nostdin','-y','-v','error','-i',str(pre),'-i',str(orbit),'-i',str(post),'-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]','-map','[v]','-an','-c:v','libx264','-preset','slow','-crf','14','-profile:v','high','-movflags','+faststart',str(args.out)])
    qa={
        'mode':'refined_nonwarping_level_c_2p5d_rigid_layers',
        'event':cfg['event'],
        'impact_offset_seconds_into_contact_window':args.impact_offset,
        'corrected_broadcast_freeze_time':corrected_freeze,
        'orbit_order':order,
        'policy':'native NBA frames only; no mesh morphing, no generative fill, no synthesized player/ball pixels; rigid transforms + action/background parallax + peak-blur real-view handoff'
    }
    args.out.with_suffix('.qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
