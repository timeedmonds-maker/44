from __future__ import annotations

"""v20 diagnostic: remove Broadcast from metric-floor appearance ownership.

v19 provenance showed the residual purple/pink floor seam is concentrated in
secondary-camera ownership islands. This test keeps the solved three-camera
player/ball geometry untouched, but permits only Left Above Rim then Right Above
Rim to texture the metric floor. Broadcast remains available everywhere else it
is already used by the reconstruction; it is excluded only as a floor-appearance
source. Unsupported floor pixels remain black so the test cannot hide the seam
with interpolation or generated fill.

Native 960x540 only. No generated pixels, interpolation fill, upscale or UHD.
"""

import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v17 as v17


_CALLS = []


def sample_floor_no_broadcast(P, support, sources, plane):
    if plane != "floor":
        return v17._ORIGINAL_SAMPLE(P, support, sources, plane)

    transforms = v17._fit_photo_transforms(sources)
    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    owned = np.zeros(v12.H * v12.W, bool)
    ids = np.where(support)[0]
    counts = {label: 0 for label in v12.CAMERAS}
    if not len(ids):
        _CALLS.append(counts)
        return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)
    Pw = P[ids]

    for label in (v12.A, v12.C):
        src = sources[label]
        C, R, K = src["C"], src["R"], src["K"]
        sgn = v12.v3.forward_sign(R, C)
        Xc = (R @ (Pw - C).T).T
        q = (K @ Xc.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        ok = (
            np.isfinite(uv).all(axis=1) & (sgn * Xc[:, 2] > 20)
            & (u >= 0) & (u < v12.W) & (v >= 0) & (v < v12.H)
        )
        loc = np.where(ok)[0]
        if len(loc):
            loc = loc[src["floor_vis"][v[loc], u[loc]].astype(bool)]
        if not len(loc):
            continue
        tgt = ids[loc]
        free = ~owned[tgt]
        loc = loc[free]; tgt = tgt[free]
        if not len(loc):
            continue
        cols = src["image"][v[loc], u[loc]]
        if label != v12.A:
            cols = v17._apply_photo(cols, transforms[label])
        out[tgt] = cols
        owned[tgt] = True
        counts[label] += int(len(tgt))

    _CALLS.append(counts)
    return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)


def main():
    _CALLS.clear()
    v17.sample_plane_photometric = sample_floor_no_broadcast
    assert (int(v12.W), int(v12.H)) == (960, 540)
    v17.main()

    oi = sys.argv.index("--out")
    out = Path(sys.argv[oi + 1])
    qp = out / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    q["v20_diagnostic"] = {
        "resolution": [960, 540],
        "floor_sources": [v12.A, v12.C],
        "broadcast_floor_appearance_enabled": False,
        "floor_source_ownership_by_render_call": _CALLS,
        "unsupported_floor_policy": "black diagnostic hole; no interpolation or generated fill",
        "render_geometry_change": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))


if __name__ == "__main__":
    main()
