#!/usr/bin/env python3
"""Deterministic Christmas event 94 broadcast V7.

V7 keeps the validated V6 video/tracking/geometry/timing and adds a reproducible
player-conditioned xFG line beneath league xFG. The metric is read from the
standard JSON emitted by tools/player_conditioned_xfg.py.

No generated imagery or generative video processing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import christmas_event94_broadcast_v6 as v6

base = v6.base


def load_metric() -> dict:
    p = os.environ.get('PLAYER_XFG_JSON', '').strip()
    if not p:
        raise RuntimeError('PLAYER_XFG_JSON is required for V7')
    j = json.loads(Path(p).read_text())
    if j.get('player_conditioned_xfg_pct') is None:
        raise RuntimeError(f'No player-conditioned xFG available: {j}')
    return j


METRIC = load_metric()
PLAYER_LABEL = os.environ.get('PLAYER_XFG_LABEL', '').strip() or (str(METRIC.get('player_name') or 'PLAYER').split()[-1].upper())
PLAYER_XFG = float(METRIC['player_conditioned_xfg_pct'])
SAMPLE_N = int(METRIC.get('sample_fga') or 0)


def draw_broadcast_tile(frame, x, y, w, h, kicker, value, accent=(190,190,190), value_size=28):
    if kicker != 'SHOT QUALITY':
        return v6.draw_broadcast_tile(frame, x, y, w, h, kicker, value, accent, value_size)

    # Taller ESPN-inspired shot-quality slab: league expectation first,
    # player-conditioned expectation immediately beneath it.
    w = max(w, 220)
    h = 88
    pts = [(x,y),(x+w-8,y),(x+w,y+8),(x+w,y+h),(x,y+h)]
    base.alpha_poly(frame, pts, (10,12,16), 0.91)
    cv2.polylines(frame, [np.array(pts,np.int32)], True, (68,72,78), 1, cv2.LINE_AA)
    cv2.line(frame, (x+8,y+7), (x+w-16,y+7), (42,48,220), 2, cv2.LINE_AA)
    frame = base.pil_text(frame, (x+10,y+13), 'SHOT QUALITY', 10, (188,193,199), True)
    frame = base.pil_text(frame, (x+10,y+28), f'xFG  {base.XFG:.1f}%', 23, (250,250,250), True)
    frame = base.pil_text(frame, (x+10,y+55), f'{PLAYER_LABEL} xFG  {PLAYER_XFG:.1f}%', 19, (250,250,250), True)
    return frame


# Preserve V6 presentation fixes and replace only the shot-quality tile.
base.draw_broadcast_tile = draw_broadcast_tile
base.draw_contact_strip = v6.draw_contact_strip
base.dotted_line = v6.dotted_line


def out_dir_from_argv() -> Path:
    try:
        i = sys.argv.index('--out')
        return Path(sys.argv[i+1])
    except Exception:
        raise RuntimeError('--out is required')


def finalize_v7_names(out: Path) -> None:
    old_native = out / 'durant_adams_christmas_broadcast_v5_native.mp4'
    old_uhd = out / 'durant_adams_christmas_broadcast_v5_UHD.mp4'
    new_native = out / 'durant_adams_christmas_broadcast_v7_native.mp4'
    new_uhd = out / 'durant_adams_christmas_broadcast_v7_UHD.mp4'
    if old_native.exists(): old_native.replace(new_native)
    if old_uhd.exists(): old_uhd.replace(new_uhd)

    qp = out / 'qa.json'
    qa = json.loads(qp.read_text())
    qa['native'] = str(new_native)
    qa['uhd'] = str(new_uhd)
    qa['player_conditioned_xfg'] = {
        'label': PLAYER_LABEL,
        'pct': PLAYER_XFG,
        'sample_fga': SAMPLE_N,
        'definition_version': METRIC.get('definition_version'),
        'comparison_rule': METRIC.get('comparison_rule'),
        'target': METRIC.get('target'),
    }
    pqa = qa.get('presentation_qa') or {}
    pqa['source'] = str(new_native)
    pqa['output'] = str(new_uhd)
    qa['presentation_qa'] = pqa
    qa['v7_note'] = 'On-screen player xFG is empirical FG% on same-player tracked season shots within +/-1 ft and +/-2.5pp league xFG, excluding target shot.'
    qp.write_text(json.dumps(qa, indent=2))


if __name__ == '__main__':
    out = out_dir_from_argv()
    base.main()
    finalize_v7_names(out)
    print(json.dumps({'v7_player_xfg_pct': PLAYER_XFG, 'sample_fga': SAMPLE_N, 'out': str(out)}, indent=2))
