from __future__ import annotations

"""v31: preserve focal-player identity by changing only focal appearance ownership.

User visual QA on v30 correctly rejected the replay because the focal player is not
recognisable in the novel views.  The v23 geometry is retained, but its renderer
continued to choose source texture primarily by view-angle similarity, which
favoured the heavily occluded Left-Above-Rim frame.  The cleanest identity view of
the focal player is Right-Above-Rim, whose focal mask contains substantially more
source pixels.

v31 keeps v30 background/court/secondary-player/ball geometry unchanged and changes
only the source-image priority for the focal v22/v23 mesh:
  Right Above Rim -> Broadcast -> Left Above Rim.
Every RGB sample is still reprojected from the official native exact-state frames
and remains gated by that camera's identity mask.  No generated appearance,
inpainting, morphing, upscale or UHD is introduced.
"""

import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v30 as v30


_ORIGINAL_FACE_SOURCE = v13._face_source
_QA = {"focal_calls": 0, "focal_face_choices": {v12.A: 0, v12.B: 0, v12.C: 0}, "focal_untextured_faces": 0}


def focal_identity_source_priority(mesh, cams, Ct, faces):
    if not bool(mesh.get("v22_focal_subject")):
        return _ORIGINAL_FACE_SOURCE(mesh, cams, Ct, faces)

    _QA["focal_calls"] += 1
    best = np.full(len(faces), -1, np.int32)
    source_masks = mesh.get("source_masks", {})

    # Identity preservation outranks target-angle similarity for the focal player.
    # The RAR exact frame has the cleanest, largest source-grounded focal mask;
    # Broadcast is the second-best full-body view; LAR is the final fallback.
    for label in (v12.C, v12.B, v12.A):
        if label not in source_masks or label not in mesh.get("vis", {}):
            continue
        vis = np.asarray(mesh["vis"][label]).astype(bool)
        fv = np.all(vis[faces], axis=1)
        take = (best < 0) & fv
        best[take] = int(v12.CAMERAS.index(label))
        n = int(np.sum(take))
        _QA["focal_face_choices"][label] += n

    good = best >= 0
    _QA["focal_untextured_faces"] += int(np.sum(~good))
    safe = best.copy()
    safe[~good] = 0
    return safe, good


def main():
    _QA["focal_calls"] = 0
    _QA["focal_untextured_faces"] = 0
    for k in _QA["focal_face_choices"]:
        _QA["focal_face_choices"][k] = 0

    v13._face_source = focal_identity_source_priority
    assert (int(v12.W), int(v12.H)) == (960, 540)
    v30.main()

    oi = sys.argv.index("--out")
    out = Path(sys.argv[oi + 1])
    qp = out / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    q["v31_focal_identity_appearance"] = {
        "resolution": [960, 540],
        "reason": "v30 visual QA: focal player was not recognisable; prior renderer overused heavily occluded LAR appearance",
        "source_priority": [v12.C, v12.B, v12.A],
        "source_policy": "official exact-state RGB only, perspective-correct reprojection, identity-mask gated",
        "focal_calls": int(_QA["focal_calls"]),
        "focal_face_choices": {k: int(v) for k, v in _QA["focal_face_choices"].items()},
        "focal_untextured_faces": int(_QA["focal_untextured_faces"]),
        "geometry_change": False,
        "background_change": False,
        "floor_change": False,
        "ball_change": False,
        "generated_texture": False,
        "inpainting": False,
        "crossfade": False,
        "upscale": False,
        "uhd": False,
        "visual_pass_required": True,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q["v31_focal_identity_appearance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
