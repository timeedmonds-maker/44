#!/usr/bin/env python3
"""Build a deterministic 12-frame contact sheet from an already-fetched NBA event clip."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    p = subprocess.run([
        'ffprobe','-v','error','-show_entries','format=duration:stream=width,height,avg_frame_rate',
        '-of','json',str(path)
    ], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=12)
    a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    meta=probe(a.input)
    dur=float(meta.get('format',{}).get('duration') or 1.0)
    n=max(1,int(a.frames)); cols=4; rows=math.ceil(n/cols)
    # Sample from 5%-95% so title/end slates do not dominate the audit.
    span=max(0.1,dur*0.90); start=max(0.0,dur*0.05); fps=n/span
    vf=(f"trim=start={start:.4f}:duration={span:.4f},setpts=PTS-STARTPTS,"
        f"fps={fps:.8f},scale=480:270:force_original_aspect_ratio=decrease,"
        f"pad=480:270:(ow-iw)/2:(oh-ih)/2:black,tile={cols}x{rows}:nb_frames={n}:padding=2:margin=2")
    subprocess.run([
        'ffmpeg','-y','-v','error','-i',str(a.input),'-vf',vf,'-frames:v','1','-q:v','2',str(a.out)
    ],check=True)
    q={'input':str(a.input),'duration_s':dur,'frames_requested':n,'sample_start_s':start,
       'sample_span_s':span,'sheet':str(a.out),'probe':meta}
    a.out.with_suffix('.json').write_text(json.dumps(q,indent=2))
    print(json.dumps(q,indent=2))

if __name__=='__main__':
    main()
