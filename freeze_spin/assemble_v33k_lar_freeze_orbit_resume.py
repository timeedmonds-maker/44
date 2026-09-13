from __future__ import annotations

"""v33k: assemble a seamless source-view freeze/orbit/resume proof after v33i visual approval.

The existing v33i static arc is anchored exactly at the Left Above Rim source frame.
Until a wider virtual path to the Broadcast optical centre is visually earned, the
honest seamless replay is therefore LAR source -> exact freeze -> v33i orbit out/back
-> same LAR source resumes. No crossfade, blur, frame interpolation, generated RGB,
upscale or UHD. Native 960x540 only.
"""

import argparse, json, subprocess
from pathlib import Path

import cv2
import numpy as np

LAR='Left Above Rim'


def run(cmd):
    print('RUN', ' '.join(map(str,cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def find_lar_clip(root:Path)->Path:
    xs=sorted(root.rglob('*_489_Left_Above_Rim_SOURCE.mp4'))
    if len(xs)!=1: raise RuntimeError(f'expected one event-489 LAR source clip; got {xs}')
    return xs[0]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clips-dir',type=Path,required=True); ap.add_argument('--arc-dir',type=Path,required=True)
    ap.add_argument('--v33i-qa',type=Path,required=True); ap.add_argument('--visual-approval',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fps',type=float,default=30.0); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    q=json.loads(args.v33i_qa.read_text()); a=json.loads(args.visual_approval.read_text())
    x=q['v33i_locked_state_adapter']
    if x['status']!='V33I_STATIC_ARC_RENDERED_AWAITING_VISUAL_QA': raise RuntimeError(x['status'])
    if a.get('approved') is not True or a.get('artifact_qa_status')!='V33I_VISUAL_PASS': raise RuntimeError('explicit v33i visual approval missing')
    if x['native_resolution']!=[960,540] or any(x[k] for k in ('generated_rgb','inpainting','crossfade','upscale','uhd')): raise RuntimeError('v33i source policy mismatch')
    center=int(x['absolute_frames'][LAR]); fps=float(args.fps); src=find_lar_clip(args.clips_dir)
    frames=[args.arc_dir/f'v12_{d:02d}deg.png' for d in (0,5,10,15,20,25)]
    for p in frames:
        im=cv2.imread(str(p));
        if im is None or im.shape[:2]!=(540,960): raise RuntimeError(f'bad arc frame {p}')
    cap=cv2.VideoCapture(str(src)); n=int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))); srcfps=float(cap.get(cv2.CAP_PROP_FPS) or fps); cap.release()
    if abs(srcfps-fps)>1.0: fps=srcfps
    pre0=max(0,center-int(round(2.0*fps))); post1=min(n-1,center+int(round(2.2*fps)))
    pre=args.out/'pre.mp4'; post=args.out/'post.mp4'; orbit=args.out/'orbit.mp4'
    # Frame-accurate video-only trims. End pre at the exact freeze frame; resume post at the following frame.
    run(['ffmpeg','-y','-v','error','-i',src,'-vf',f"select='between(n,{pre0},{center})',setpts=N/FRAME_RATE/TB,format=yuv420p",'-an','-r',str(fps),'-c:v','libx264','-crf','16','-preset','medium',pre])
    run(['ffmpeg','-y','-v','error','-i',src,'-vf',f"select='between(n,{center+1},{post1})',setpts=N/FRAME_RATE/TB,format=yuv420p",'-an','-r',str(fps),'-c:v','libx264','-crf','16','-preset','medium',post])
    # Stepped but fully rendered path: out 0->25, brief hold, back 20->0. No blended/interpolated frames.
    seq=[0,5,10,15,20,25,25,20,15,10,5,0]
    durations=[0.10,0.10,0.10,0.10,0.10,0.22,0.22,0.10,0.10,0.10,0.10,0.16]
    txt=args.out/'orbit_concat.txt'
    lines=[]
    for deg,dur in zip(seq,durations):
        lines += [f"file '{(args.arc_dir/f'v12_{deg:02d}deg.png').resolve()}'", f'duration {dur:.4f}']
    lines += [f"file '{(args.arc_dir/'v12_00deg.png').resolve()}'"]
    txt.write_text('\n'.join(lines)+'\n')
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',txt,'-vf',f'fps={fps},format=yuv420p','-an','-c:v','libx264','-crf','16','-preset','medium',orbit])
    concat=args.out/'final_concat.txt'; concat.write_text(f"file '{pre.resolve()}'\nfile '{orbit.resolve()}'\nfile '{post.resolve()}'\n")
    final=args.out/'adams_jazz_v33k_native_freeze_orbit_resume.mp4'
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',concat,'-an','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',final])
    qa={'version':'v33k_lar_freeze_orbit_resume','status':'V33K_NATIVE_SOURCE_VIEW_REPLAY_RENDERED','source_camera':LAR,'event':489,
        'freeze_frame_index':center,'pre_frame_range':[pre0,center],'post_frame_range':[center+1,post1],
        'orbit_degrees':seq,'orbit_frame_policy':'only v33i rendered source-grounded stills; repeated frames for timing only; no blended/interpolated RGB',
        'native_resolution':[960,540],'audio':False,'generated_rgb':False,'crossfade':False,'blur':False,'temporal_interpolation':False,'upscale':False,'uhd':False,
        'broadcast_transition_claimed':False,'reason_source_view_not_broadcast':'v33i static arc is exactly anchored to LAR; a seamless Broadcast-centred virtual path is not yet geometrically earned'}
    (args.out/'v33k_replay_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))

if __name__=='__main__': main()
