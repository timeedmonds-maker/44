from __future__ import annotations

"""v19 diagnostic: expose exact metric-floor source ownership.

The residual narrow purple/pink floor streak survived v16 semantic exclusion,
v17 cross-camera colour calibration and v18 RAR-before-Broadcast ownership. v19
changes no geometry or imagery. It writes a provenance image for every virtual
floor render showing whether each metric floor pixel came from Left Above Rim,
Right Above Rim or Broadcast, plus counts. This isolates the remaining artifact
before another renderer change.

Native 960x540 only. Diagnostic only; no generated pixels or upscale.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v17 as v17
from freeze_spin import build_three_camera_mesh_v18 as v18


_CALL = 0
_OUT = None
_PROV_QA = []


def sample_plane_with_provenance(P, support, sources, plane):
    global _CALL
    if plane != "floor":
        return v17._ORIGINAL_SAMPLE(P, support, sources, plane)

    transforms = v17._fit_photo_transforms(sources)
    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    owned = np.zeros(v12.H * v12.W, bool)
    prov = np.zeros(v12.H * v12.W, np.uint8)
    ids = np.where(support)[0]
    counts = {label: 0 for label in v12.CAMERAS}
    Pw = P[ids] if len(ids) else np.empty((0,3),np.float64)

    label_code = {v12.A: 1, v12.C: 2, v12.B: 3}
    for label in (v12.A, v12.C, v12.B):
        if not len(ids): break
        src = sources[label]
        C, R, K = src["C"], src["R"], src["K"]
        sgn = v12.v3.forward_sign(R, C)
        Xc = (R @ (Pw - C).T).T
        q = (K @ Xc.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        ok = np.isfinite(uv).all(axis=1) & (sgn * Xc[:, 2] > 20) & (u >= 0) & (u < v12.W) & (v >= 0) & (v < v12.H)
        loc = np.where(ok)[0]
        if len(loc): loc = loc[src["floor_vis"][v[loc], u[loc]].astype(bool)]
        if not len(loc): continue
        tgt = ids[loc]; free = ~owned[tgt]; loc = loc[free]; tgt = tgt[free]
        if not len(loc): continue
        cols = src["image"][v[loc], u[loc]]
        if label != v12.A: cols = v17._apply_photo(cols, transforms[label])
        out[tgt] = cols; owned[tgt] = True; prov[tgt] = label_code[label]
        counts[label] += int(len(tgt))

    call = _CALL; _CALL += 1
    _PROV_QA.append({"call": int(call), "counts": counts})
    if _OUT is not None:
        vis = np.zeros((v12.H*v12.W,3),np.uint8)
        # BGR: LAR green, RAR cyan, Broadcast red; unsupported black.
        vis[prov==1] = (0,200,0)
        vis[prov==2] = (220,220,0)
        vis[prov==3] = (0,0,230)
        cv2.imwrite(str(_OUT/f"v19_floor_provenance_call{call:02d}.png"),vis.reshape(v12.H,v12.W,3))
        for code,label in ((1,v12.A),(2,v12.C),(3,v12.B)):
            m=(prov==code).reshape(v12.H,v12.W).astype(np.uint8)*255
            cv2.imwrite(str(_OUT/f"v19_floor_{label.replace(' ','_')}_call{call:02d}.png"),m)
    return out.reshape(v12.H,v12.W,3), owned.reshape(v12.H,v12.W)


def main():
    global _CALL, _OUT
    _CALL=0; _PROV_QA.clear()
    oi=sys.argv.index('--out'); _OUT=Path(sys.argv[oi+1]); _OUT.mkdir(parents=True,exist_ok=True)
    v18.sample_plane_rar_first = sample_plane_with_provenance
    assert (int(v12.W),int(v12.H))==(960,540)
    v18.main()
    qp=_OUT/'three_camera_mesh_v12_qa.json'
    q=json.loads(qp.read_text())
    q['v19_diagnostic']={
        'resolution':[960,540],
        'floor_provenance_codes':{'1':v12.A,'2':v12.C,'3':v12.B},
        'calls':_PROV_QA,
        'render_change':False,
        'upscale':False,'uhd':False,
    }
    qp.write_text(json.dumps(q,indent=2))


if __name__=='__main__':
    main()
