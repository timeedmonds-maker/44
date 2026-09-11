from __future__ import annotations

"""v30: extended temporal coverage + layered metric arena hole fill.

v29 addresses secondary on-court people with source-grounded metric billboards.
The remaining dominant failure is static disocclusion: as the virtual camera orbits,
large target rays leave the exact LAR field of view.  v26's accepted temporal LAR
registrations help but use only a narrow +/-120-frame window; v27's single coarse
arena shell regressed coverage because its planes were too distant and omitted the
low courtside volume.

v30 keeps all v29 foreground/court/ball work and changes only the background:
1) expand accepted same-camera temporal candidates across the available event clip;
2) render the existing v26 temporal/infinity background first;
3) fill only still-unowned background rays with a layered finite metric arena shell
   containing near baseline/sideline courtside planes plus farther stand planes;
4) every shell sample is reprojected into one of the three solved exact cameras or
   an accepted same-camera temporal frame. Rejected registrations are never used.

This is source-grounded deterministic image formation, not inpainting. Native
960x540 only; no generated texture, crossfade, UHD or upscale.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v26 as v26
from freeze_spin import build_three_camera_mesh_v29 as v29


_ORIGINAL_TEMPORAL_BG = v26.far_background_temporal_angular
_BG_QA = []

# Broader event-clip search. Decode failures are harmless; only robustly accepted
# registrations enter the source pool.
EXTENDED_OFFSETS = (
    -240, -210, -180, -150, -120, -90, -60, -45, -30, -15,
    15, 30, 45, 60, 90, 120, 150, 180, 210, 240,
)


def _target_rays(K, R, C):
    yy, xx = np.indices((v12.H, v12.W), np.float64)
    hp = np.stack([xx.ravel(), yy.ravel(), np.ones(v12.H * v12.W)], axis=0)
    dc = np.linalg.inv(K) @ hp
    s = float(v12.v3.forward_sign(R, C))
    dw = (R.T @ (s * dc)).T
    dw /= np.maximum(np.linalg.norm(dw, axis=1, keepdims=True), 1e-12)
    return np.asarray(C, np.float64).reshape(3), dw


def _layered_shell_points(K, R, C):
    C0, d = _target_rays(K, R, C)
    npx = len(d)
    best_t = np.full(npx, np.inf, np.float64)
    best_p = np.zeros((npx, 3), np.float64)
    best_id = np.full(npx, -1, np.int16)

    def offer(t, p, valid, pid):
        take = valid & np.isfinite(t) & (t > 20.0) & (t < best_t)
        if np.any(take):
            best_t[take] = t[take]
            best_p[take] = p[take]
            best_id[take] = int(pid)

    def xplane(x, yabs, z0, z1, pid):
        den = d[:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (float(x) - C0[0]) / den
        p = C0[None, :] + t[:, None] * d
        valid = ((np.abs(den) > 1e-8) & (np.abs(p[:, 1]) <= float(yabs)) &
                 (p[:, 2] >= float(z0)) & (p[:, 2] <= float(z1)))
        offer(t, p, valid, pid)

    def yplane(y, x0, x1, z0, z1, pid):
        den = d[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (float(y) - C0[1]) / den
        p = C0[None, :] + t[:, None] * d
        valid = ((np.abs(den) > 1e-8) & (p[:, 0] >= float(x0)) & (p[:, 0] <= float(x1)) &
                 (p[:, 2] >= float(z0)) & (p[:, 2] <= float(z1)))
        offer(t, p, valid, pid)

    # Near courtside volume: actual baseline/sideline seating is much closer and
    # lower than v27's first shell. These planes are reached first where relevant.
    xplane(-360.0, 1250.0, -40.0, 650.0, 0)
    yplane(-930.0, -500.0, 2600.0, -40.0, 520.0, 1)
    yplane( 930.0, -500.0, 2600.0, -40.0, 520.0, 2)

    # Farther bowl/stands catch rays above/behind the courtside layer.
    xplane(-1150.0, 2000.0, 0.0, 1700.0, 3)
    yplane(-1700.0, -1150.0, 3300.0, 0.0, 1700.0, 4)
    yplane( 1700.0, -1150.0, 3300.0, 0.0, 1700.0, 5)
    xplane(3400.0, 2000.0, 0.0, 1500.0, 6)
    return best_p, np.isfinite(best_t), best_id, best_t


def _sample_shell_camera(label, P, shell_valid, cams, images, dynamic_masks):
    uv, _depth, geom = v12.v8.project_metric(cams[label], P)
    base_geom = shell_valid & geom & np.isfinite(uv).all(axis=1)
    exact_inside = (base_geom & (uv[:, 0] >= 0.0) & (uv[:, 0] < v12.W) &
                    (uv[:, 1] >= 0.0) & (uv[:, 1] < v12.H))
    exact_ok = exact_inside.copy()
    ii = np.where(exact_inside)[0]
    if len(ii):
        ui = np.clip(np.rint(uv[ii, 0]).astype(np.int32), 0, v12.W - 1)
        vi = np.clip(np.rint(uv[ii, 1]).astype(np.int32), 0, v12.H - 1)
        ex = v13._static_exclusion(label, cams)
        exact_ok[ii] &= ~dynamic_masks[label][vi, ui]
        exact_ok[ii] &= ~ex[vi, ui]

    samples = []
    masks = []
    c0 = np.zeros((len(P), 3), np.uint8)
    exact_sample = exact_ok & (uv[:, 0] < v12.W - 1.0) & (uv[:, 1] < v12.H - 1.0)
    if np.any(exact_sample):
        c0[exact_sample] = v12.v8.bilinear_sample(images[label], uv[exact_sample])
    samples.append(c0.reshape(v12.H, v12.W, 3))
    masks.append(exact_sample.reshape(v12.H, v12.W))

    rows = v26._TEMPORAL.get(label, [])
    for row in rows:
        uc = v26._perspective_points(row['Hinv'], uv)
        ok = (shell_valid & np.isfinite(uv).all(axis=1) & np.isfinite(uc).all(axis=1) &
              (uc[:, 0] >= 0.0) & (uc[:, 0] < v12.W - 1.0) &
              (uc[:, 1] >= 0.0) & (uc[:, 1] < v12.H - 1.0))
        # If the exact camera sees this correspondence and classifies it as
        # player/court/board, a temporal frame cannot relabel it as arena.
        ok[exact_inside & (~exact_ok)] = False
        c = np.zeros((len(P), 3), np.uint8)
        if np.any(ok):
            c[ok] = v12.v8.bilinear_sample(row['image'], uc[ok])
        samples.append(c.reshape(v12.H, v12.W, 3)); masks.append(ok.reshape(v12.H, v12.W))

    st = np.stack(samples, axis=0); vm = np.stack(masks, axis=0)
    medoid, valid, support = v26._actual_medoid(st, vm, min_support=1)
    return medoid.reshape(-1, 3), valid.reshape(-1), support.reshape(-1)


def far_background_temporal_plus_layered_shell(Kt, Rt, cams, images, dynamic_masks):
    base, base_mask = _ORIGINAL_TEMPORAL_BG(Kt, Rt, cams, images, dynamic_masks)
    if v13._is_anchor_pose(cams, Rt, v12.CURRENT_CT):
        return base, base_mask

    P, shell_valid, plane_id, shell_t = _layered_shell_points(Kt, Rt, v12.CURRENT_CT)
    # Known rigid court/backboard layers own their target rays later in the stack.
    _, sf = v12.v4.virtual_plane_points(Kt, Rt, v12.CURRENT_CT, 'floor')
    _, sb = v12.v4.virtual_plane_points(Kt, Rt, v12.CURRENT_CT, 'board')
    shell_valid &= ~(sf | sb)

    out = base.reshape(-1, 3).copy()
    owned = base_mask.reshape(-1).copy()
    fills = {}
    # LAR remains the natural appearance anchor; Bcast/RAR fill only residuals.
    for label in (v12.A, v12.B, v12.C):
        col, ok, support = _sample_shell_camera(label, P, shell_valid, cams, images, dynamic_masks)
        take = ok & (~owned)
        out[take] = col[take]
        owned[take] = True
        fills[label] = {
            'candidate_pixels': int(np.sum(ok)),
            'filled_residual_pixels': int(np.sum(take)),
            'support_ge_2_pixels': int(np.sum(support >= 2)),
        }

    _BG_QA.append({
        'base_owned_fraction': float(base_mask.mean()),
        'final_owned_fraction': float(owned.mean()),
        'shell_valid_fraction': float(shell_valid.mean()),
        'fills': fills,
        'plane_pixels': {str(i): int(np.sum((plane_id == i) & shell_valid)) for i in range(7)},
        'median_shell_distance_cm': float(np.median(shell_t[shell_valid])) if np.any(shell_valid) else None,
    })
    return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)


def main():
    _BG_QA.clear()
    v26.OFFSETS = EXTENDED_OFFSETS
    v26.far_background_temporal_angular = far_background_temporal_plus_layered_shell

    assert (int(v12.W), int(v12.H)) == (960, 540)
    v29.main()

    oi = sys.argv.index('--out'); out = Path(sys.argv[oi + 1])
    qp = out / 'three_camera_mesh_v12_qa.json'; q = json.loads(qp.read_text())
    q['v30_layered_arena_fill'] = {
        'resolution': [960, 540],
        'extended_temporal_offsets': list(EXTENDED_OFFSETS),
        'background_calls': _BG_QA,
        'layer_policy': 'v26 accepted-temporal infinity background first; layered finite metric arena fills residual source-supported holes only',
        'near_planes_cm': {'baseline_x': -360.0, 'sideline_y_abs': 930.0, 'z_max': 650.0},
        'far_planes_cm': {'baseline_x': -1150.0, 'side_y_abs': 1700.0, 'far_x': 3400.0, 'z_max': 1700.0},
        'rejected_temporal_frames_sampled': False,
        'foreground_geometry_change': False,
        'floor_change': False,
        'ball_geometry_change': False,
        'generated_texture': False,
        'inpainting': False,
        'crossfade': False,
        'upscale': False,
        'uhd': False,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q['v30_layered_arena_fill'], indent=2), flush=True)


if __name__ == '__main__':
    main()
