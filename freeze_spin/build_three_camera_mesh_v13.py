from __future__ import annotations

"""v13 native three-camera render-quality experiment.

v12b proved that the exact v11 visual state can yield a real player mesh using
Left Above Rim + Broadcast + Right Above Rim simultaneously. v13 keeps that
geometry and changes only the image formation layer:

- native 960x540 only (hard assertion; no upscale/UHD path);
- 0 degrees is source-locked to the exact Left Above Rim frame;
- distant arena background comes from one coherent source warp rather than a
  mosaic of competing cameras, with metric floor/backboard/rim removed from the
  far-background layer before they are reconstructed;
- player faces are perspective-correct textured from the best visible real
  source view instead of being filled with a single mean triangle colour.

No generated texture, optical-flow morph, camera cross-fade or invented player
pixels are used. This is still a six-frame 0/5/10/15/20/25 degree static test.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v12b as v12b


_STATIC_EXCLUSION = {}
_ORIGINAL_DRAW_RIM = v12.draw_metric_rim


def _is_anchor_pose(cams, Rt, Ct) -> bool:
    C0, R0, _ = cams[v12.A]
    return bool(np.linalg.norm(np.asarray(Ct) - np.asarray(C0)) < 1e-4 and np.max(np.abs(np.asarray(Rt) - np.asarray(R0))) < 1e-6)


def _static_exclusion(label, cams):
    if label in _STATIC_EXCLUSION:
        return _STATIC_EXCLUSION[label]
    Cc, Rc, Kc = cams[label]
    ex = np.zeros((v12.H, v12.W), bool)

    tf, Pf = v12.v4.ray_plane_map(Kc, Rc, Cc, "floor")
    floor = (
        np.isfinite(tf) & (tf > 20) & (tf < 12000)
        & (Pf[:, :, 0] >= -300.0) & (Pf[:, :, 0] <= 1600.0)
        & (np.abs(Pf[:, :, 1]) <= 900.0)
    )
    ex |= floor

    tb, Pb = v12.v4.ray_plane_map(Kc, Rc, Cc, "board")
    board = (
        np.isfinite(tb) & (tb > 20) & (tb < 12000)
        & (np.abs(Pb[:, :, 1]) <= v12.v4.BOARD_Y + 10.0)
        & (Pb[:, :, 2] >= v12.v4.BOARD_Z0 - 10.0)
        & (Pb[:, :, 2] <= v12.v4.BOARD_Z1 + 10.0)
    )
    board = cv2.dilate(board.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=1) > 0
    ex |= board

    rim_uv, _, valid = v12.v8.project_metric(cams[label], v12.v8.rim_points())
    rim_mask = np.zeros((v12.H, v12.W), np.uint8)
    good = rim_uv[valid]
    if len(good) >= 2:
        pts = np.rint(good).astype(np.int32)
        cv2.polylines(rim_mask, [pts], False, 255, 18, cv2.LINE_AA)
        for p in pts[::max(1, len(pts)//16)]:
            cv2.circle(rim_mask, tuple(p), 10, 255, -1, cv2.LINE_AA)
    ex |= rim_mask > 0

    _STATIC_EXCLUSION[label] = ex
    return ex


def far_background_coherent(Kt, Rt, cams, images, dynamic_masks):
    """Single-source source-grounded distant warp; no multi-camera background mosaic."""
    if _is_anchor_pose(cams, Rt, v12.CURRENT_CT):
        return images[v12.A].copy(), np.ones((v12.H, v12.W), bool)

    yy, xx = np.indices((v12.H, v12.W), np.float64)
    hp = np.stack([xx.ravel(), yy.ravel(), np.ones(v12.H * v12.W)], axis=0)
    dcam = np.linalg.inv(Kt) @ hp
    dw = Rt.T @ dcam

    label = v12.A
    Cc, Rc, Kc = cams[label]
    ds = Rc @ dw
    q = Kc @ ds
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (q[:2] / q[2:3]).T
    sign = float(v12.v3.forward_sign(Rc, Cc))
    valid = (
        np.isfinite(uv).all(axis=1)
        & (sign * ds[2] > 1e-6)
        & (uv[:, 0] >= 0) & (uv[:, 0] < v12.W - 1)
        & (uv[:, 1] >= 0) & (uv[:, 1] < v12.H - 1)
    )
    cols = v12.v8.bilinear_sample(images[label], uv)
    ui = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, v12.W - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, v12.H - 1)
    ex = _static_exclusion(label, cams)
    ids = np.where(valid)[0]
    if len(ids):
        valid[ids] &= ~dynamic_masks[label][vi[ids], ui[ids]]
        valid[ids] &= ~ex[vi[ids], ui[ids]]
    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    out[valid] = cols[valid]
    return out.reshape(v12.H, v12.W, 3), valid.reshape(v12.H, v12.W)


def _face_source(mesh, cams, Ct, faces):
    verts = mesh["verts"].astype(np.float64)
    cent = verts[faces].mean(axis=1)
    tv = np.asarray(Ct, np.float64).reshape(1, 3) - cent
    tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    score = np.full((len(faces), len(v12.CAMERAS)), -999.0, np.float64)
    for si, label in enumerate(v12.CAMERAS):
        vis = mesh["vis"][label]
        face_vis = np.all(vis[faces], axis=1)
        sv = cams[label][0].reshape(1, 3) - cent
        sv /= np.maximum(np.linalg.norm(sv, axis=1, keepdims=True), 1e-9)
        s = np.sum(tv * sv, axis=1)
        score[face_vis, si] = s[face_vis]
    best = np.argmax(score, axis=1)
    best_score = score[np.arange(len(faces)), best]
    return best, best_score > -100.0


def _raster_textured_face(img, zbuf, p, zv, verts3, source_cam, source_img):
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
    cols = v12.v8.bilinear_sample(source_img, suv[good])
    py = ids[0][good]; px = ids[1][good]
    img[ymin + py, xmin + px] = cols
    patch_z[py, px] = depth[ids][good]
    return int(np.sum(good))


def render_triangles_textured(base_img, cams, Kt, Rt, Ct, meshes, ball_mesh=None, ball_color=None):
    if _is_anchor_pose(cams, Rt, Ct):
        return v12.IMAGES[v12.A].copy(), 0

    img = base_img.copy()
    zbuf = np.full((v12.H, v12.W), np.inf, np.float32)
    rendered_faces = 0
    rendered_pixels = 0

    for mesh in meshes:
        verts = mesh["verts"].astype(np.float64)
        faces = mesh["faces"]
        tuv, tdepth, tvalid = v12.v8.project_metric((Ct, Rt, Kt), verts)
        best_src, has_src = _face_source(mesh, cams, Ct, faces)
        face_ok = np.all(tvalid[faces], axis=1) & has_src
        order = np.where(face_ok)[0]
        order = order[np.argsort(np.mean(tdepth[faces[order]], axis=1))[::-1]]
        for fi in order:
            f = faces[fi]
            p = tuv[f]
            if np.max(p[:, 0]) < 0 or np.min(p[:, 0]) >= v12.W or np.max(p[:, 1]) < 0 or np.min(p[:, 1]) >= v12.H:
                continue
            area = abs(float(v12b.cross_compat(p[1] - p[0], p[2] - p[0])))
            if area < 0.18 or area > 6000:
                continue
            label = v12.CAMERAS[int(best_src[fi])]
            npx = _raster_textured_face(img, zbuf, p, tdepth[f], verts[f], cams[label], v12.IMAGES[label])
            if npx:
                rendered_faces += 1
                rendered_pixels += npx

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

    render_triangles_textured.last_pixels = int(rendered_pixels)
    return img, int(rendered_faces)


render_triangles_textured.last_pixels = 0
_LAST_CAMS = {}


def draw_metric_rim_anchor_safe(img, Kt, Rt, Ct, colour):
    C0, R0, _ = _LAST_CAMS[v12.A]
    if np.linalg.norm(np.asarray(Ct) - np.asarray(C0)) < 1e-4 and np.max(np.abs(np.asarray(Rt) - np.asarray(R0))) < 1e-6:
        return
    _ORIGINAL_DRAW_RIM(img, Kt, Rt, Ct, colour)


def main():
    global _LAST_CAMS
    _STATIC_EXCLUSION.clear()
    v12b.POSE_CACHE.clear()

    v12.attach_rfdetr_poses = v12b.attach_rfdetr_poses_fixed
    v12.attach_right_above_rim = v12b.attach_right_above_rim_fixed
    v12.v8.bilinear_sample = v12b.bilinear_sample_chunked
    v12.np.cross = v12b.cross_compat

    v12.far_background = far_background_coherent
    v12.render_triangles = render_triangles_textured
    v12.draw_metric_rim = draw_metric_rim_anchor_safe

    original_load = v12.base.load_cameras
    def load_cameras_capture(*args, **kwargs):
        cams = original_load(*args, **kwargs)
        _LAST_CAMS.clear(); _LAST_CAMS.update(cams)
        return cams
    v12.base.load_cameras = load_cameras_capture

    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v12.main()

    try:
        oi = sys.argv.index("--out")
        out = Path(sys.argv[oi + 1])
        qp = out / "three_camera_mesh_v12_qa.json"
        q = json.loads(qp.read_text())
        q["v13_renderer"] = {
            "resolution": [960, 540],
            "anchor_0deg": "exact Left Above Rim source frame",
            "background": "single-source coherent distant rotation warp with metric floor/backboard/rim exclusion",
            "players": "perspective-correct per-face reprojection of real source RGB onto anatomical 3-D meshes",
            "generated_texture": False,
            "upscale": False,
            "uhd": False,
        }
        qp.write_text(json.dumps(q, indent=2))
    except Exception as exc:
        print("V13_QA_APPEND_WARNING", repr(exc), flush=True)


if __name__ == "__main__":
    main()
