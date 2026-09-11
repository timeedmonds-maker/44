from __future__ import annotations

"""v27: finite-depth source-grounded arena shell renderer.

Visual QA of v26 showed that its temporal floor expansion adds useful coverage but
can admit moving-player contamination, while the distant/infinity arena model
still leaves large black disocclusion regions.  The foreground (v23 focal player),
three-view ball, exact-state frames, camera calibrations and v21 metric court atlas
are therefore left unchanged in v27.

The only experimental change is the static arena representation.  Instead of
assuming all crowd pixels live at infinity, v27 intersects each virtual-camera ray
with a coarse finite-depth arena shell made from three vertical planes behind the
basket and along the sidelines.  Those metric 3-D shell points are projected into
all three solved cameras.  Accepted same-camera temporal registrations from v26
may extend source coverage, but rejected temporal frames are never sampled.

The shell is deliberately conservative: it is an R&D proxy for distant arena
geometry, not an invented texture.  Every non-black RGB sample is copied from an
official native source frame.  The regulation floor/backboard/rim remain separate
metric layers.  0 degrees stays the untouched exact LAR source frame.

Native 960x540 only.  No generated texture, inpainting, optical flow, crossfade,
UHD or upscale.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v15 as v15
from freeze_spin import build_three_camera_mesh_v21 as v21
from freeze_spin import build_three_camera_mesh_v24 as v24
from freeze_spin import build_three_camera_mesh_v26 as v26


# Coarse basket-local arena shell.  Values are intentionally broad and are only
# used to account for finite camera translation when sampling real arena pixels.
# The shell is NOT used for players, ball, court, rim, or regulation backboard.
BASELINE_X = -1050.0
SIDE_Y = 1550.0
FAR_X = 3000.0
Z_MIN = 120.0
Z_MAX = 1500.0
X_MIN_SIDE = -1050.0
X_MAX_SIDE = 3000.0
Y_HALF_BASE = 1900.0

_BG_QA = []


def _pop_arg(name: str) -> str:
    i = sys.argv.index(name)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


def _target_rays(K, R, C):
    yy, xx = np.indices((v12.H, v12.W), np.float64)
    hp = np.stack([xx.ravel(), yy.ravel(), np.ones(v12.H * v12.W)], axis=0)
    dc = np.linalg.inv(K) @ hp
    # v12 virtual cameras come from the accepted LAR lineage (+Z forward).
    dw = (R.T @ dc).T
    n = np.linalg.norm(dw, axis=1, keepdims=True)
    dw /= np.maximum(n, 1e-12)
    return np.asarray(C, np.float64).reshape(1, 3), dw


def _arena_shell_points(K, R, C):
    C0, d = _target_rays(K, R, C)
    C0 = C0[0]
    N = len(d)
    best_t = np.full(N, np.inf, np.float64)
    best_P = np.zeros((N, 3), np.float64)
    best_plane = np.full(N, -1, np.int8)

    def offer(t, P, valid, pid):
        nonlocal best_t, best_P, best_plane
        take = valid & np.isfinite(t) & (t > 20.0) & (t < best_t)
        if np.any(take):
            best_t[take] = t[take]
            best_P[take] = P[take]
            best_plane[take] = int(pid)

    # Baseline stands behind the backboard.
    den = d[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (BASELINE_X - C0[0]) / den
    P = C0[None, :] + t[:, None] * d
    valid = (
        np.abs(den) > 1e-8
        & (np.abs(P[:, 1]) <= Y_HALF_BASE)
        & (P[:, 2] >= Z_MIN) & (P[:, 2] <= Z_MAX)
    )
    offer(t, P, valid, 0)

    # Left/right sideline stands.
    for pid, yplane in ((1, -SIDE_Y), (2, SIDE_Y)):
        den = d[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (yplane - C0[1]) / den
        P = C0[None, :] + t[:, None] * d
        valid = (
            np.abs(den) > 1e-8
            & (P[:, 0] >= X_MIN_SIDE) & (P[:, 0] <= X_MAX_SIDE)
            & (P[:, 2] >= Z_MIN) & (P[:, 2] <= Z_MAX)
        )
        offer(t, P, valid, pid)

    # Opposite-end arena wall, useful only for rays that miss the nearer planes.
    den = d[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (FAR_X - C0[0]) / den
    P = C0[None, :] + t[:, None] * d
    valid = (
        np.abs(den) > 1e-8
        & (np.abs(P[:, 1]) <= Y_HALF_BASE)
        & (P[:, 2] >= Z_MIN) & (P[:, 2] <= Z_MAX)
    )
    offer(t, P, valid, 3)

    return best_P, np.isfinite(best_t), best_plane, best_t


def _source_project(cam, P):
    uv, _depth, valid = v12.v8.project_metric(cam, P)
    good = (
        valid & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0) & (uv[:, 0] < v12.W - 1.0)
        & (uv[:, 1] >= 0.0) & (uv[:, 1] < v12.H - 1.0)
    )
    return uv, good


def _camera_samples(label, P, shell_valid, cams, images, dynamic_masks):
    uv, exact_geom = _source_project(cams[label], P)
    exact_ok = shell_valid & exact_geom
    ui = np.zeros(len(P), np.int32); vi = np.zeros(len(P), np.int32)
    ids = np.where(exact_geom)[0]
    if len(ids):
        ui[ids] = np.rint(uv[ids, 0]).astype(np.int32)
        vi[ids] = np.rint(uv[ids, 1]).astype(np.int32)
        ui[ids] = np.clip(ui[ids], 0, v12.W - 1)
        vi[ids] = np.clip(vi[ids], 0, v12.H - 1)
        # A shell sample must look like static non-court/non-board source content
        # in the exact frame whenever that exact correspondence is visible.
        ex = v13._static_exclusion(label, cams)
        exact_ok[ids] &= ~dynamic_masks[label][vi[ids], ui[ids]]
        exact_ok[ids] &= ~ex[vi[ids], ui[ids]]

    cols = []
    masks = []
    c0 = np.zeros((len(P), 3), np.uint8)
    if np.any(exact_ok):
        c0[exact_ok] = v12.v8.bilinear_sample(images[label], uv[exact_ok])
    cols.append(c0); masks.append(exact_ok)

    temporal = v26._ensure_temporal(label, cams, images, dynamic_masks)
    for row in temporal:
        # Accepted candidate->anchor H.  A metric shell point first projects to
        # the exact camera; inverse H maps that exact coordinate to the real
        # candidate frame.  This may legitimately recover pixels outside the
        # exact frame after a pan/zoom, but rejected H are never present here.
        uc = v26._perspective_points(row['Hinv'], uv)
        ok = (
            shell_valid & np.isfinite(uv).all(axis=1) & np.isfinite(uc).all(axis=1)
            & (uc[:, 0] >= 0.0) & (uc[:, 0] < v12.W - 1.0)
            & (uc[:, 1] >= 0.0) & (uc[:, 1] < v12.H - 1.0)
        )
        # If exact correspondence is visible but identified as court/player/
        # board, temporal frames are not allowed to reinterpret it as arena.
        ok[exact_geom & (~exact_ok)] = False
        c = np.zeros((len(P), 3), np.uint8)
        if np.any(ok):
            c[ok] = v12.v8.bilinear_sample(row['image'], uc[ok])
        cols.append(c); masks.append(ok)

    stack = np.stack(cols, axis=0).reshape(len(cols), v12.H, v12.W, 3)
    vm = np.stack(masks, axis=0).reshape(len(masks), v12.H, v12.W)
    medoid, valid, support = v26._actual_medoid(stack, vm, min_support=1)
    return medoid.reshape(-1, 3), valid.reshape(-1), support.reshape(-1), len(temporal)


def far_background_finite_shell(Kt, Rt, cams, images, dynamic_masks):
    # This also initializes accepted temporal registrations before v21 asks for
    # any source appearance.  Floor remains the clean v21 exact-state atlas.
    for label in v12.CAMERAS:
        v26._ensure_temporal(label, cams, images, dynamic_masks)

    if v13._is_anchor_pose(cams, Rt, v12.CURRENT_CT):
        return images[v12.A].copy(), np.ones((v12.H, v12.W), bool)

    P, shell_valid, plane_id, shell_t = _arena_shell_points(Kt, Rt, v12.CURRENT_CT)

    # Rigid known surfaces are rendered by their metric layers, not the shell.
    _, sf = v12.v4.virtual_plane_points(Kt, Rt, v12.CURRENT_CT, 'floor')
    _, sb = v12.v4.virtual_plane_points(Kt, Rt, v12.CURRENT_CT, 'board')
    shell_valid &= ~(sf | sb)

    # Build all camera candidate fields first, then use hard source ownership.
    camera_fields = {}
    for label in v12.CAMERAS:
        col, ok, support, nt = _camera_samples(label, P, shell_valid, cams, images, dynamic_masks)
        camera_fields[label] = (col, ok, support, nt)

    # Prefer the solved camera whose centre is closest to the target camera for
    # this finite shell proxy.  Other physical cameras fill only uncovered pixels.
    Ct = np.asarray(v12.CURRENT_CT, np.float64)
    pref = sorted(v12.CAMERAS, key=lambda l: float(np.linalg.norm(cams[l][0] - Ct)))
    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    owned = np.zeros(v12.H * v12.W, bool)
    qa_sources = {}
    for label in pref:
        col, ok, support, nt = camera_fields[label]
        before = int(np.sum(ok))
        take = ok & (~owned)
        out[take] = col[take]
        owned[take] = True
        qa_sources[label] = {
            'valid_union_pixels_before_ownership': before,
            'owned_pixels': int(take.sum()),
            'accepted_temporal_frames': int(nt),
            'support_ge_2_pixels': int(np.sum(support >= 2)),
        }

    q = {
        'owned_fraction': float(owned.mean()),
        'shell_valid_fraction': float(shell_valid.mean()),
        'source_order': pref,
        'source_counts': qa_sources,
        'plane_pixels': {
            'baseline': int(np.sum((plane_id == 0) & shell_valid)),
            'side_negative_y': int(np.sum((plane_id == 1) & shell_valid)),
            'side_positive_y': int(np.sum((plane_id == 2) & shell_valid)),
            'far_end': int(np.sum((plane_id == 3) & shell_valid)),
        },
        'median_shell_distance_cm': float(np.median(shell_t[shell_valid])) if np.any(shell_valid) else None,
    }
    _BG_QA.append(q)
    return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)


def main():
    _BG_QA.clear()
    v26._TEMPORAL.clear(); v26._REG_QA.clear(); v26._ATLAS_QA.clear(); v26._BG_QA.clear()
    v26._CLIPS_DIR = Path(_pop_arg('--clips-dir'))
    si = sys.argv.index('--sync-qa')
    v26._SYNC_QA_PATH = Path(sys.argv[si + 1])

    # Keep v21 court atlas; only replace v15's arena renderer.
    v15.far_background_multisource_clean = far_background_finite_shell
    assert (int(v12.W), int(v12.H)) == (960, 540)
    v24.main()

    oi = sys.argv.index('--out'); out = Path(sys.argv[oi + 1])
    qp = out / 'three_camera_mesh_v12_qa.json'
    q = json.loads(qp.read_text())
    q['v27_finite_arena_shell'] = {
        'resolution': [960, 540],
        'shell_geometry_cm': {
            'baseline_x': BASELINE_X, 'side_y_abs': SIDE_Y, 'far_x': FAR_X,
            'z_min': Z_MIN, 'z_max': Z_MAX,
        },
        'background_render_calls': _BG_QA,
        'temporal_registration': v26._REG_QA,
        'floor_representation': 'unchanged clean v21 exact-state registered metric court atlas',
        'foreground_geometry_change': False,
        'ball_geometry_change': False,
        'generated_texture': False,
        'inpainting': False,
        'crossfade': False,
        'upscale': False,
        'uhd': False,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q['v27_finite_arena_shell'], indent=2), flush=True)


if __name__ == '__main__':
    main()
