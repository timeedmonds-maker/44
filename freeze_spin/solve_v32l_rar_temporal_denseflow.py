from __future__ import annotations

"""v32l: replace sparse LK in v32k with dense bidirectional motion fields.

The semantic joint itself is often a poor corner, so sparse LK can reject every
joint even when the player motion is visually coherent.  v32l keeps all v32k
provenance/geometry/consensus gates but changes only the temporal measurement:
DIS dense optical flow is computed between consecutive real RAR source frames,
then each RF-DETR joint is advected backward through those real frames.  A joint
survives a step only if the backward and forward dense fields are mutually
consistent.  Dense flow is measurement only; no pixels are synthesized.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import solve_v32k_rar_temporal_deblend as v32k

W, H = v32k.W, v32k.H


def _sample(flow: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Robust local flow sample: median 3x3 neighborhood around a subpixel point."""
    x, y = float(p[0]), float(p[1])
    xi, yi = int(round(x)), int(round(y))
    x1, x2 = max(0, xi - 1), min(flow.shape[1], xi + 2)
    y1, y2 = max(0, yi - 1), min(flow.shape[0], yi + 2)
    patch = flow[y1:y2, x1:x2].reshape(-1, 2)
    if len(patch) == 0:
        return np.array([np.nan, np.nan], np.float32)
    return np.median(patch, axis=0).astype(np.float32)


def _dense_pair(g_from: np.ndarray, g_to: np.ndarray) -> np.ndarray:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    dis.setUseMeanNormalization(True)
    dis.setVariationalRefinementIterations(5)
    return dis.calc(g_from, g_to, None)


def dense_track_back(frames: dict[int, np.ndarray], start_rel: int, xy: np.ndarray, conf: np.ndarray):
    pts = np.asarray(xy, np.float32).copy()
    valid = np.asarray(conf >= .20, bool)
    trace = {start_rel: {'xy': pts.tolist(), 'valid': valid.astype(int).tolist()}}

    # Cache each adjacent bidirectional field once per anchor invocation.
    for rel in range(start_rel, 0, -1):
        g1 = cv2.cvtColor(frames[rel], cv2.COLOR_BGR2GRAY)
        g0 = cv2.cvtColor(frames[rel - 1], cv2.COLOR_BGR2GRAY)
        flow10 = _dense_pair(g1, g0)  # current -> previous
        flow01 = _dense_pair(g0, g1)  # previous -> current

        nxt = pts.copy()
        fb = np.full(len(pts), 999.0, np.float32)
        motion = np.full(len(pts), 999.0, np.float32)
        inside = np.zeros(len(pts), bool)
        for j, p in enumerate(pts):
            if not valid[j]:
                continue
            d10 = _sample(flow10, p)
            if not np.all(np.isfinite(d10)):
                continue
            pp = p + d10
            if not (1 <= pp[0] < W - 1 and 1 <= pp[1] < H - 1):
                continue
            d01 = _sample(flow01, pp)
            if not np.all(np.isfinite(d01)):
                continue
            cyc = pp + d01
            fb[j] = float(np.linalg.norm(cyc - p))
            motion[j] = float(np.linalg.norm(d10))
            inside[j] = True
            nxt[j] = pp

        # Consecutive NBA frames should not require huge point displacement.
        # Multi-anchor consensus is the stronger final gate, so this remains
        # permissive enough for hand/arm motion while rejecting wild fields.
        good = valid & inside & np.isfinite(fb) & (fb <= 5.0) & (motion <= 45.0)
        pts = nxt
        valid = good
        trace[rel - 1] = {
            'xy': pts.tolist(),
            'valid': valid.astype(int).tolist(),
            'fb_px': fb.tolist(),
            'motion_px': motion.tolist(),
            'valid_count': int(np.sum(valid)),
        }
    return pts.astype(float), valid, trace


def main():
    v32k.track_back = dense_track_back
    code = 0
    try:
        v32k.main()
    except SystemExit as e:
        code = int(e.code or 0)

    # Preserve v32k outputs for compatibility, but add an explicit v32l QA file
    # so no downstream stage can mistake which temporal tracker produced it.
    try:
        oi = sys.argv.index('--out')
        out = Path(sys.argv[oi + 1])
        p = out / 'v32k_temporal_deblend_qa.json'
        if p.exists():
            q = json.loads(p.read_text())
            q['version'] = 'v32l_rar_temporal_denseflow'
            q['status'] = q['status'].replace('V32K', 'V32L')
            q['temporal_method'] = ('RF-DETR anchors at real RAR t+03..t+06; DIS dense bidirectional '
                                    'backward tracking; forward/backward cycle consistency; multi-anchor consensus')
            q['dense_flow_role'] = 'measurement only; no source pixels synthesized or warped into output'
            (out / 'v32l_temporal_denseflow_qa.json').write_text(json.dumps(q, indent=2))
    except Exception as exc:
        print(f'V32L_QA_REWRITE_WARNING: {exc}', flush=True)

    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
