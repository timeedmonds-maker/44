#!/usr/bin/env python3
"""Deterministic Christmas event 94 broadcast V7.

V7 keeps the validated video/tracking/geometry/timing and adds reproducible
player-conditioned xFG beneath league xFG.

Locked visual treatment:
- all analytics information panels use the durable neutral translucent-grey
  broadcast style from tools/broadcast_overlay_style.py
- no coloured accent bars/rules on those panels
- player/team labels and rings remain unchanged

No generated imagery or generative video processing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import christmas_event94_broadcast_v6 as v6
import broadcast_overlay_style as style

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
    # Accent is accepted for drop-in compatibility but deliberately ignored.
    if kicker != 'SHOT QUALITY':
        return style.draw_info_tile(base, frame, x, y, w, h, kicker, value, value_size)

    # Taller neutral shot-quality slab: league expectation first,
    # player-conditioned expectation directly beneath it.
    w = max(w, 220)
    h = 88
    pts = [(x, y), (x + w - 8, y), (x + w, y + 8), (x + w, y + h), (x, y + h)]
    style.gray_alpha_poly(frame, pts)
    style.panel_outline(frame, pts)
    frame = base.pil_text(frame, (x + 10, y + 13), 'SHOT QUALITY', 10, style.KICKER_TEXT, True)
    frame = base.pil_text(frame, (x + 10, y + 28), f'xFG  {base.XFG:.1f}%', 23, style.VALUE_TEXT, True)
    frame = base.pil_text(frame, (x + 10, y + 55), f'{PLAYER_LABEL} xFG  {PLAYER_XFG:.1f}%', 19, style.VALUE_TEXT, True)
    return frame


def draw_contact_strip(frame, x, y, w, elapsed):
    return style.draw_contact_strip(base, frame, x, y, w, elapsed)


# Install the durable neutral style for every analytics information panel.
# Overriding alpha_poly also converts the hard-coded release-distance tag in
# the validated V5 geometry path to the same translucent grey treatment.
base.alpha_poly = style.gray_alpha_poly
base.draw_broadcast_tile = draw_broadcast_tile
base.draw_contact_strip = draw_contact_strip
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
    if old_native.exists():
        old_native.replace(new_native)
    if old_uhd.exists():
        old_uhd.replace(new_uhd)

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
    qa['broadcast_panel_style'] = {
        'style_version': 'neutral_translucent_grey_v1',
        'fill_bgr': list(style.PANEL_FILL),
        'alpha': style.PANEL_ALPHA,
        'coloured_accent_bands': False,
    }
    pqa = qa.get('presentation_qa') or {}
    pqa['source'] = str(new_native)
    pqa['output'] = str(new_uhd)
    qa['presentation_qa'] = pqa
    qa['v7_note'] = (
        'On-screen player xFG is empirical FG% on same-player tracked season shots '
        'within +/-1 ft official SHOT_DISTANCE and +/-2.5pp league xFG, excluding '
        'the target shot. Analytics panels use neutral translucent grey with no '
        'coloured accent bands.'
    )
    qp.write_text(json.dumps(qa, indent=2))


if __name__ == '__main__':
    out = out_dir_from_argv()
    base.main()
    finalize_v7_names(out)
    print(json.dumps({
        'v7_player_xfg_pct': PLAYER_XFG,
        'sample_fga': SAMPLE_N,
        'panel_style': 'neutral_translucent_grey_v1',
        'out': str(out),
    }, indent=2))
