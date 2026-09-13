#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

import adams_screen_candidate_crops as base
from fetch_official_nba_event_clip import parse as resolve_clip, UA
from kd_double_team_ballhandler_onnx import choose_hls_variant


def extract_frames_native(game, event, outdir, fps=6.0, max_seconds=14.0, target_width=960):
    page, opts, ch = resolve_clip(game, event)
    hls, var = choose_hls_variant(ch['url'], target_width)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    headers = f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    pat = str(outdir / 'f_%03d.jpg')
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-rw_timeout', '30000000',
        '-headers', headers, '-i', hls, '-t', str(max_seconds),
        '-vf', f'fps={fps},scale={int(target_width)}:-2', '-q:v', '3', pat
    ], check=True, timeout=120)
    return {
        'frames': sorted(outdir.glob('f_*.jpg')),
        'angle': ch['label'],
        'variant': var,
        'page_url': page,
        'angle_count': len(opts),
        'output_width': int(target_width),
    }


if __name__ == '__main__':
    # Monkey-patch the imported helper used by the validated crop pipeline.
    base.extract_frames = extract_frames_native
    base.main()
