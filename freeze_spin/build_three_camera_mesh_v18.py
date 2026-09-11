from __future__ import annotations

"""v18: isolate the residual floor streak by changing secondary floor ownership.

v17 converted the bright white streak into a photometrically matched purple/pink
streak, proving the artifact is a secondary-camera metric-floor seam rather than
player RGB contamination. v18 keeps every v17 correction but changes only metric
floor source ownership from LAR -> Broadcast -> RAR to LAR -> RAR -> Broadcast.
Right Above Rim is the stronger basket-local metric anchor, so it gets first right
of refusal for floor pixels disoccluded from the Left Above Rim anchor.

Native 960x540 only. No generated pixels, interpolation fill, upscale or UHD.
"""

import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v17 as v17


_OWNERSHIP_CALLS = []


def sample_plane_rar_first(P, support, sources, plane):
    if plane != "floor":
        return v17._ORIGINAL_SAMPLE(P, support, sources, plane)

    transforms = v17._fit_photo_transforms(sources)
    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    owned = np.zeros(v12.H * v12.W, bool)
    ids = np.where(support)[0]
    counts = {label: 0 for label in v12.CAMERAS}
    if not len(ids):
        _OWNERSHIP_CALLS.append(counts)
        return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)
    Pw = P[ids]

    for label in (v12.A, v12.C, v12.B):
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
        if not len(loc):
            continue
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

    _OWNERSHIP_CALLS.append(counts)
    return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)


def main():
    _OWNERSHIP_CALLS.clear()
    v17.sample_plane_photometric = sample_plane_rar_first
    assert (int(v12.W), int(v12.H)) == (960, 540)
    v17.main()

    try:
        oi = sys.argv.index("--out")
        out = Path(sys.argv[oi + 1])
        qp = out / "three_camera_mesh_v12_qa.json"
        q = json.loads(qp.read_text())
        q["v18_renderer"] = {
            "resolution": [960, 540],
            "floor_source_priority": [v12.A, v12.C, v12.B],
            "floor_source_ownership_by_render_call": _OWNERSHIP_CALLS,
            "hypothesis": "residual purple/pink streak is Broadcast floor reprojection entering LAR disocclusion holes; prefer stronger RAR basket-local floor anchor first",
            "generated_texture": False,
            "upscale": False,
            "uhd": False,
        }
        qp.write_text(json.dumps(q, indent=2))
    except Exception as exc:
        print("V18_QA_APPEND_WARNING", repr(exc), flush=True)


if __name__ == "__main__":
    main()
