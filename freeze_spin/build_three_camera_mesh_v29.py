from __future__ import annotations

"""v29: source-grounded metric billboards for non-principal on-court people.

v28 combines the best validated static layers: v21 clean court + v26 accepted-
temporal arena coverage + v23 focal Adams + the three-view ball.  A remaining
visual failure is structural: the static background correctly excludes every
on-court person, while only the principal action tracks are reconstructed as 3-D
meshes.  Those deliberately removed secondary people therefore become black
human-shaped holes in novel views.

v29 fills that specific representation gap without inventing appearance.  Every
non-principal LAR person instance is placed on a vertical metric billboard through
its observed floor-contact point.  The billboard plane faces the physical LAR
camera, so its source image is exact at the anchor and acquires deterministic
parallax under the 0..25 degree virtual orbit.  Target pixels are ray/plane
intersections reprojected into the real LAR frame and are accepted only inside a
slightly padded real person mask.  Principal mesh identities are excluded by mask
overlap and continue to use the full v23/v14 3-D reconstruction.

Native 960x540 only. No generated texture, inpainting, optical-flow morph,
crossfade, UHD or upscale. Every billboard RGB value is sampled from the exact
official source frame.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v14 as v14
from freeze_spin import build_three_camera_mesh_v15 as v15
from freeze_spin import build_three_camera_mesh_v16 as v16
from freeze_spin import build_three_camera_mesh_v28 as v28


_ORIGINAL_RENDER = v14.render_triangles_mask_safe
_FULL_INSTANCES = {}
_SPRITE_QA = {"render_calls": [], "source_people": {}}


def _label_for_image(image):
    for label, im in v12.IMAGES.items():
        if image is im:
            return label
    return None


def detect_capture_full_oncourt(model, image, K, R, C):
    """Replicate v15's detector contract while retaining all broad on-court people."""
    full_dyn, instances, balls = v15._ORIGINAL_DETECT_ONCOURT(model, image, K, R, C)
    label = _label_for_image(image)
    if label is not None:
        _FULL_INSTANCES[label] = instances
        _SPRITE_QA["source_people"][label] = {
            "broad_oncourt_instances": int(len(instances)),
            "mask_pixels": [int(np.asarray(x.get("mask", False)).astype(bool).sum()) for x in instances],
            "foot_world_cm": [
                [float(v) for v in x.get("foot_world_cm", [np.nan, np.nan, np.nan])] for x in instances
            ],
        }

    keep = []
    for inst in instances:
        x, y, _ = inst["foot_world_cm"]
        if -250.0 <= float(x) <= 1150.0 and abs(float(y)) <= 520.0:
            keep.append(inst)

    dyn = full_dyn.astype(np.uint8)
    dyn = cv2.dilate(dyn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    for b in balls[:3]:
        x1, y1, x2, y2 = [int(round(v)) for v in b["box"]]
        x1 = max(0, x1 - 7); y1 = max(0, y1 - 7)
        x2 = min(v12.W - 1, x2 + 7); y2 = min(v12.H - 1, y2 + 7)
        dyn[y1:y2 + 1, x1:x2 + 1] = True
    return dyn, keep, balls


def _target_rays(K, R, C):
    yy, xx = np.indices((v12.H, v12.W), np.float64)
    hp = np.stack([xx.ravel(), yy.ravel(), np.ones(v12.H * v12.W)], axis=0)
    dc = np.linalg.inv(K) @ hp
    s = float(v12.v3.forward_sign(R, C))
    dw = (R.T @ (s * dc)).T
    dw /= np.maximum(np.linalg.norm(dw, axis=1, keepdims=True), 1e-12)
    return dw


def _mesh_union_lar(meshes):
    out = np.zeros((v12.H, v12.W), bool)
    for mesh in meshes:
        sm = mesh.get("source_masks", {}).get(v12.A)
        if sm is not None:
            out |= np.asarray(sm).astype(bool)
    return out


def _select_sprites(meshes):
    rows = []
    mesh_union = _mesh_union_lar(meshes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for ii, inst in enumerate(_FULL_INSTANCES.get(v12.A, [])):
        mask = np.asarray(inst.get("mask", False)).astype(bool)
        area = int(mask.sum())
        foot = np.asarray(inst.get("foot_world_cm", [np.nan, np.nan, np.nan]), np.float64)
        if area < 80 or not np.isfinite(foot).all():
            continue
        if not (-320.0 <= foot[0] <= 1650.0 and abs(float(foot[1])) <= 880.0):
            continue
        overlap = int(np.sum(mask & mesh_union))
        ratio = float(overlap / max(1, area))
        if ratio >= 0.16:
            rows.append({"instance": int(ii), "status": "principal_mesh_excluded", "mask_pixels": area, "mesh_overlap_fraction": ratio})
            continue
        padded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        rows.append({
            "instance": int(ii), "status": "secondary_billboard", "mask_pixels": area,
            "padded_mask_pixels": int(padded.sum()), "mesh_overlap_fraction": ratio,
            "foot_world_cm": [float(x) for x in foot], "mask": padded,
        })
    return rows


def _render_billboards(base_img, cams, Kt, Rt, Ct, meshes):
    if v13._is_anchor_pose(cams, Rt, Ct):
        return base_img, {"angle_anchor": True, "billboard_pixels": 0, "billboards": 0}

    selected = _select_sprites(meshes)
    sprites = [r for r in selected if r["status"] == "secondary_billboard"]
    if not sprites:
        return base_img, {"angle_anchor": False, "billboard_pixels": 0, "billboards": 0, "selection": selected}

    dw = _target_rays(Kt, Rt, Ct)
    Ct3 = np.asarray(Ct, np.float64).reshape(3)
    Cs, Rs, Ks = cams[v12.A]
    out = base_img.copy().reshape(-1, 3)
    zbuf = np.full(v12.H * v12.W, np.inf, np.float64)
    owner = np.full(v12.H * v12.W, -1, np.int16)
    source = v12.IMAGES[v12.A]

    for si, row in enumerate(sprites):
        foot = np.asarray(row["foot_world_cm"], np.float64)
        foot[2] = 0.0
        n = np.asarray([Cs[0] - foot[0], Cs[1] - foot[1], 0.0], np.float64)
        nn = float(np.linalg.norm(n))
        if nn < 1e-6:
            continue
        n /= nn
        den = dw @ n
        numer = float(np.dot(foot - Ct3, n))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = numer / den
        good = np.isfinite(t) & (np.abs(den) > 1e-8) & (t > 20.0) & (t < 12000.0)
        ids = np.where(good)[0]
        if not len(ids):
            continue
        P = Ct3[None, :] + t[ids, None] * dw[ids]
        zg = (P[:, 2] >= -25.0) & (P[:, 2] <= 330.0)
        ids = ids[zg]; P = P[zg]
        if not len(ids):
            continue

        uv, _depth, valid = v12.v8.project_metric(cams[v12.A], P)
        src_ok = (
            valid & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0) & (uv[:, 0] < v12.W - 1.0)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] < v12.H - 1.0)
        )
        jj = np.where(src_ok)[0]
        if not len(jj):
            continue
        ui = np.rint(uv[jj, 0]).astype(np.int32)
        vi = np.rint(uv[jj, 1]).astype(np.int32)
        inside = row["mask"][vi, ui]
        jj = jj[inside]
        if not len(jj):
            continue
        tgt = ids[jj]
        nearer = t[tgt] < zbuf[tgt]
        jj = jj[nearer]; tgt = tgt[nearer]
        if not len(jj):
            continue
        cols = v12.v8.bilinear_sample(source, uv[jj])
        out[tgt] = cols
        zbuf[tgt] = t[tgt]
        owner[tgt] = int(si)

    counts = []
    for si, row in enumerate(sprites):
        counts.append({"source_instance": int(row["instance"]), "rendered_pixels": int(np.sum(owner == si))})
    return out.reshape(v12.H, v12.W, 3), {
        "angle_anchor": False,
        "billboard_pixels": int(np.sum(owner >= 0)),
        "billboards": int(len(sprites)),
        "per_billboard": counts,
        "selection": [{k: v for k, v in r.items() if k != "mask"} for r in selected],
    }


def render_triangles_with_secondary_billboards(base_img, cams, Kt, Rt, Ct, meshes, ball_mesh=None, ball_color=None):
    with_people, q = _render_billboards(base_img, cams, Kt, Rt, Ct, meshes)
    _SPRITE_QA["render_calls"].append(q)
    return _ORIGINAL_RENDER(with_people, cams, Kt, Rt, Ct, meshes, ball_mesh, ball_color)


def main():
    _FULL_INSTANCES.clear(); _SPRITE_QA["render_calls"] = []; _SPRITE_QA["source_people"] = {}

    # v16's detector wrapper calls this stored broad detector. Capture all broad
    # people before returning the same near-play subset used by the reconstruction.
    v16._ORIGINAL_BROAD_DETECT = detect_capture_full_oncourt
    # v14.main will install its current render function into v13/v12 at runtime.
    v14.render_triangles_mask_safe = render_triangles_with_secondary_billboards

    assert (int(v12.W), int(v12.H)) == (960, 540)
    v28.main()

    oi = sys.argv.index("--out")
    out = Path(sys.argv[oi + 1])
    qp = out / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    calls = _SPRITE_QA["render_calls"]
    q["v29_secondary_people_billboards"] = {
        "resolution": [960, 540],
        "representation": "vertical metric billboards through observed LAR floor-contact points for broad on-court people not represented by accepted principal 3-D meshes",
        "source": "exact Left Above Rim official native frame only",
        "principal_exclusion": "LAR identity-mask overlap with accepted 3-D meshes >= 0.16",
        "render_calls": calls,
        "source_people": _SPRITE_QA["source_people"],
        "total_billboard_pixels": int(sum(int(x.get("billboard_pixels", 0)) for x in calls)),
        "generated_texture": False,
        "inpainting": False,
        "crossfade": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q["v29_secondary_people_billboards"], indent=2), flush=True)


if __name__ == "__main__":
    main()
