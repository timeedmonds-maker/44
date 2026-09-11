from __future__ import annotations

"""v14 native three-camera source-grounded surfel + mask-safe texture experiment.

v13 proved the exact 0-degree anchor and perspective-correct real-pixel player
texturing, but visual QA still exposed two renderer failures: unsupported black
arena bands as the virtual camera moved, and a long bright player-texture streak
caused by sampling outside the identity mask inside otherwise valid mesh faces.

v14 changes only image formation. Geometry, v11 exact-state frames, camera solves,
three-camera articulated track and ball triangulation remain unchanged.

- native 960x540 only; no upscale/UHD path;
- exact Left Above Rim frame remains bit-exact at 0 degrees;
- static non-court/non-board/non-player pixels are backprojected with MoGe depth
  aligned into the accepted metric world, rendered as source-coloured surfels,
  then only residual holes fall back to v13's coherent single-source rotation warp;
- every player texture sample must land inside the assigned real source identity
  mask (slightly eroded for bilinear safety), so court/crowd pixels cannot be
  stretched across a player mesh face;
- no generated texture, optical-flow morph, cross-fade or invented player pixels.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v12b as v12b


_ORIGINAL_MOGE_INFER = v12.moge_infer
_ORIGINAL_BUILD_MESH = v12.build_anatomical_mesh
_ORIGINAL_V13_BG = v13.far_background_coherent
_DEPTH_CACHE = {}
_STATIC_CLOUD_CACHE = {}
_TEXTURE_STATS = {"accepted_samples": 0, "rejected_outside_identity_mask": 0, "faces_rendered": 0}


def moge_infer_cached(model, image, tokens):
    out = _ORIGINAL_MOGE_INFER(model, image, tokens)
    _DEPTH_CACHE[id(image)] = (np.asarray(out[0]).copy(), np.asarray(out[2]).copy())
    return out


def build_anatomical_mesh_with_masks(cams, track, instances, voxel_cm=2.5):
    mesh, qa = _ORIGINAL_BUILD_MESH(cams, track, instances, voxel_cm=voxel_cm)
    if mesh is None:
        return mesh, qa
    source_masks = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for label in v12.CAMERAS:
        idx = track.get(label)
        if idx is None:
            continue
        m = instances[label][idx]["mask"].astype(np.uint8)
        er = cv2.erode(m, kernel, iterations=1) > 0
        if int(er.sum()) < 40:
            er = m > 0
        source_masks[label] = er
    mesh["source_masks"] = source_masks
    return mesh, qa


def _build_static_cloud(label, cams, images, dynamic_masks):
    if label in _STATIC_CLOUD_CACHE:
        return _STATIC_CLOUD_CACHE[label]
    image = images[label]
    key = id(image)
    if key not in _DEPTH_CACHE:
        raise RuntimeError(f"v14 missing cached MoGe depth for {label}")
    depth, valid = _DEPTH_CACHE[key]
    Cc, Rc, Kc = cams[label]
    align, _ = v12.v3.robust_depth_align(depth, valid, Kc, Rc, Cc)
    z = align[0] * depth.astype(np.float64) + align[1]

    excluded = v13._static_exclusion(label, cams)
    keep = (
        valid.astype(bool)
        & np.isfinite(z) & (z > 40.0) & (z < 12000.0)
        & (~dynamic_masks[label].astype(bool))
        & (~excluded)
    )

    z32 = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    gx = cv2.Sobel(z32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z32, cv2.CV_32F, 0, 1, ksize=3)
    rel = np.sqrt(gx * gx + gy * gy) / np.maximum(np.abs(z32), 80.0)
    keep &= rel < 0.42
    keep = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0

    ys, xs = np.where(keep)
    if len(xs) == 0:
        cloud = (np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8))
        _STATIC_CLOUD_CACHE[label] = cloud
        return cloud

    zz = z[ys, xs]
    xn = (xs.astype(np.float64) - Kc[0, 2]) / Kc[0, 0]
    yn = (ys.astype(np.float64) - Kc[1, 2]) / Kc[1, 1]
    sign = float(v12.v3.forward_sign(Rc, Cc))
    Xc = sign * np.column_stack([xn * zz, yn * zz, zz])
    Xw = (Rc.T @ Xc.T).T + Cc
    finite = np.isfinite(Xw).all(axis=1)
    Xw = Xw[finite].astype(np.float32)
    cols = image[ys[finite], xs[finite]].copy()
    cloud = (Xw, cols)
    _STATIC_CLOUD_CACHE[label] = cloud
    return cloud


def far_background_static_surfel(Kt, Rt, cams, images, dynamic_masks):
    if v13._is_anchor_pose(cams, Rt, v12.CURRENT_CT):
        return images[v12.A].copy(), np.ones((v12.H, v12.W), bool)

    out = np.zeros((v12.H, v12.W, 3), np.uint8)
    owned = np.zeros((v12.H, v12.W), bool)

    for label in (v12.A, v12.B, v12.C):
        cloud = _build_static_cloud(label, cams, images, dynamic_masks)
        if len(cloud[0]) == 0:
            continue
        ri, rm = v12.v4.raster(cloud, Kt, Rt, v12.CURRENT_CT, radius=1)
        mk = rm > 0
        take = mk & (~owned)
        out[take] = ri[take]
        owned[take] = True

    warp, wmask = _ORIGINAL_V13_BG(Kt, Rt, cams, images, dynamic_masks)
    take = wmask & (~owned)
    out[take] = warp[take]
    owned[take] = True
    return out, owned


def _raster_textured_face_masked(img, zbuf, p, zv, verts3, source_cam, source_img, source_mask):
    xmin = max(0, int(math.floor(float(np.min(p[:, 0])))))
    xmax = min(v12.W - 1, int(math.ceil(float(np.max(p[:, 0])))))
    ymin = max(0, int(math.floor(float(np.min(p[:, 1])))))
    ymax = min(v12.H - 1, int(math.ceil(float(np.max(p[:, 1])))))
    if xmax < xmin or ymax < ymin:
        return 0
    if (xmax - xmin + 1) * (ymax - ymin + 1) > 8500:
        return 0

    x0, y0 = p[0]; x1, y1 = p[1]; x2, y2 = p[2]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(den)) < 1e-7:
        return 0
    gy, gx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
    xx = gx.astype(np.float64) + 0.5
    yy = gy.astype(np.float64) + 0.5
    l0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
    l1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
    if not np.any(inside):
        return 0

    z0, z1, z2 = [max(1e-6, float(x)) for x in zv]
    invz = l0 / z0 + l1 / z1 + l2 / z2
    inside &= invz > 1e-9
    if not np.any(inside):
        return 0
    depth = 1.0 / np.maximum(invz, 1e-9)
    patch_z = zbuf[ymin:ymax + 1, xmin:xmax + 1]
    take = inside & (depth < patch_z)
    if not np.any(take):
        return 0

    w0 = (l0 / z0) / invz; w1 = (l1 / z1) / invz; w2 = (l2 / z2) / invz
    ids = np.where(take)
    X = (
        w0[ids][:, None] * verts3[0].reshape(1, 3)
        + w1[ids][:, None] * verts3[1].reshape(1, 3)
        + w2[ids][:, None] * verts3[2].reshape(1, 3)
    )
    suv, _, sok = v12.v8.project_metric(source_cam, X)
    good = (
        sok
        & np.isfinite(suv).all(axis=1)
        & (suv[:, 0] >= 0) & (suv[:, 0] < v12.W - 1)
        & (suv[:, 1] >= 0) & (suv[:, 1] < v12.H - 1)
    )
    if not np.any(good):
        return 0

    gi = np.where(good)[0]
    su = np.rint(suv[gi, 0]).astype(np.int32)
    sv = np.rint(suv[gi, 1]).astype(np.int32)
    in_identity = source_mask[sv, su]
    _TEXTURE_STATS["rejected_outside_identity_mask"] += int(np.sum(~in_identity))
    gi = gi[in_identity]
    if len(gi) == 0:
        return 0

    cols = v12.v8.bilinear_sample(source_img, suv[gi])
    py = ids[0][gi]; px = ids[1][gi]
    img[ymin + py, xmin + px] = cols
    patch_z[py, px] = depth[ids][gi]
    _TEXTURE_STATS["accepted_samples"] += int(len(gi))
    return int(len(gi))


def render_triangles_mask_safe(base_img, cams, Kt, Rt, Ct, meshes, ball_mesh=None, ball_color=None):
    if v13._is_anchor_pose(cams, Rt, Ct):
        return v12.IMAGES[v12.A].copy(), 0

    img = base_img.copy()
    zbuf = np.full((v12.H, v12.W), np.inf, np.float32)
    rendered_faces = 0

    for mesh in meshes:
        verts = mesh["verts"].astype(np.float64)
        faces = mesh["faces"]
        tuv, tdepth, tvalid = v12.v8.project_metric((Ct, Rt, Kt), verts)
        best_src, has_src = v13._face_source(mesh, cams, Ct, faces)
        face_ok = np.all(tvalid[faces], axis=1) & has_src
        order = np.where(face_ok)[0]
        order = order[np.argsort(np.mean(tdepth[faces[order]], axis=1))[::-1]]
        for fi in order:
            f = faces[fi]
            p = tuv[f]
            if np.max(p[:, 0]) < 0 or np.min(p[:, 0]) >= v12.W or np.max(p[:, 1]) < 0 or np.min(p[:, 1]) >= v12.H:
                continue
            area = abs(float(v12b.cross_compat(p[1] - p[0], p[2] - p[0])))
            if not np.isfinite(area) or area < 0.18 or area > 6000:
                continue
            label = v12.CAMERAS[int(best_src[fi])]
            sm = mesh.get("source_masks", {}).get(label)
            if sm is None:
                continue
            npx = _raster_textured_face_masked(img, zbuf, p, tdepth[f], verts[f], cams[label], v12.IMAGES[label], sm)
            if npx:
                rendered_faces += 1

    if ball_mesh is not None:
        bv, bf = ball_mesh
        uv, depth, valid = v12.v8.project_metric((Ct, Rt, Kt), bv)
        rows = []
        for f in bf:
            if not np.all(valid[f]):
                continue
            p = uv[f]
            if np.max(p[:, 0]) < 0 or np.min(p[:, 0]) >= v12.W or np.max(p[:, 1]) < 0 or np.min(p[:, 1]) >= v12.H:
                continue
            rows.append((float(np.mean(depth[f])), p))
        rows.sort(key=lambda x: x[0], reverse=True)
        for _, p in rows:
            cv2.fillConvexPoly(img, np.rint(p).astype(np.int32), tuple(int(x) for x in np.asarray(ball_color)), lineType=cv2.LINE_AA)

    _TEXTURE_STATS["faces_rendered"] += int(rendered_faces)
    return img, int(rendered_faces)


def main():
    _DEPTH_CACHE.clear(); _STATIC_CLOUD_CACHE.clear()
    for k in _TEXTURE_STATS:
        _TEXTURE_STATS[k] = 0

    v12.moge_infer = moge_infer_cached
    v12.build_anatomical_mesh = build_anatomical_mesh_with_masks
    v13.far_background_coherent = far_background_static_surfel
    v13.render_triangles_textured = render_triangles_mask_safe

    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v13.main()

    try:
        oi = sys.argv.index("--out")
        out = Path(sys.argv[oi + 1])
        qp = out / "three_camera_mesh_v12_qa.json"
        q = json.loads(qp.read_text())
        q["v14_renderer"] = {
            "resolution": [960, 540],
            "anchor_0deg": "exact Left Above Rim source frame",
            "static_world": "metric floor/backboard/rim + MoGe-aligned source-coloured static surfels from three solved cameras; coherent LAR rotation warp only fills residual holes",
            "player_texture": "perspective-correct real source RGB gated per sample by assigned eroded identity mask",
            "static_cloud_points": {label: int(len(_STATIC_CLOUD_CACHE.get(label, ([], []))[0])) for label in v12.CAMERAS},
            "texture_stats": dict(_TEXTURE_STATS),
            "generated_texture": False,
            "upscale": False,
            "uhd": False,
        }
        qp.write_text(json.dumps(q, indent=2))
    except Exception as exc:
        print("V14_QA_APPEND_WARNING", repr(exc), flush=True)


if __name__ == "__main__":
    main()
