from __future__ import annotations

"""v7 subject-surface diagnostic.

v6 isolated the remaining visual failure to free per-pixel monocular subject
depths: those points are correct enough at the source view but fan into trails
when the virtual camera moves.  v7 keeps all v6 camera/court rules, but replaces
each retained person cloud with one rigid, source-textured vertical surface.

For grounded instances the surface is anchored at the solved floor contact.  If
that contact would require a large depth correction, the surface is instead
anchored at the learned source depth at the lowest visible point, preserving an
airborne/occluded subject without forcing it to the floor.  No generated pixels,
interpolation between cameras, morphing, or synthetic appearance is introduced.
"""

import cv2
import numpy as np

from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v5 as v5
from freeze_spin import build_three_camera_diagnostic_v6 as v6
from freeze_spin import build_three_camera_diagnostic_v4 as v4

H, W = v5.H, v5.W


def _ray_world(xs, ys, K, R, C):
    s = v3.forward_sign(R, C)
    xn = (xs.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (ys.astype(np.float64) - K[1, 2]) / K[1, 1]
    dc = np.column_stack([xn, yn, np.ones_like(xn)])
    return (s * dc) @ R


def rigid_subject_surfaces(image, depth, valid, dynamic, instances, K, R, C, align):
    z_est = align[0] * depth.astype(np.float64) + align[1]
    points_all = []
    colours_all = []
    corrections = []

    for inst in instances:
        mm = inst['mask'].astype(bool)
        ys, xs = np.where(mm & valid & np.isfinite(z_est))
        if len(xs) < 50:
            corrections.append({
                'score': inst['score'], 'foot_px': inst['foot_px'],
                'foot_world_cm': inst['foot_world_cm'],
                'depth_offset_cm': None, 'surface_mode': 'rejected_too_few_pixels'
            })
            continue

        fx, fy = [int(v) for v in inst['foot_px']]
        y0 = max(0, fy - 8); y1 = min(H, fy + 1)
        x0 = max(0, fx - 10); x1 = min(W, fx + 11)
        local = mm[y0:y1, x0:x1] & valid[y0:y1, x0:x1] & np.isfinite(z_est[y0:y1, x0:x1])
        vals = z_est[y0:y1, x0:x1][local]
        raw_foot = float(np.median(vals)) if len(vals) >= 3 else float('nan')
        floor_t = float(inst['floor_camera_depth_cm'])
        delta = floor_t - raw_foot if np.isfinite(raw_foot) else float('nan')

        # Grounded: exact solved floor anchor. Airborne/occluded: retain source
        # depth at the lowest visible point instead of forcing a false contact.
        if np.isfinite(delta) and abs(delta) <= 70.0:
            anchor = np.asarray(inst['foot_world_cm'], np.float64)
            mode = 'floor_anchored_vertical_surface'
            applied = float(delta)
        else:
            dfoot = _ray_world(np.asarray([fx]), np.asarray([fy]), K, R, C)[0]
            if not np.isfinite(raw_foot) or raw_foot <= 20.0:
                # fallback to floor support only if learned foot depth is absent
                anchor = np.asarray(inst['foot_world_cm'], np.float64)
                mode = 'floor_anchor_fallback'
                applied = None
            else:
                anchor = C + raw_foot * dfoot
                mode = 'learned_depth_vertical_surface'
                applied = None

        # Vertical plane facing the source camera in the horizontal world plane.
        n = np.asarray([C[0] - anchor[0], C[1] - anchor[1], 0.0], np.float64)
        nn = float(np.linalg.norm(n))
        if nn < 1e-6:
            # Unlikely for LAR; use camera right-axis-derived horizontal normal.
            n = np.asarray([R[2, 0], R[2, 1], 0.0], np.float64)
            nn = float(np.linalg.norm(n))
        if nn < 1e-6:
            continue
        n /= nn

        dw = _ray_world(xs, ys, K, R, C)
        denom = dw @ n
        numer = float((anchor - C) @ n)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = numer / denom
        X = C.reshape(1, 3) + t[:, None] * dw

        # Keep only physically plausible person support around the source mask.
        ok = (
            np.isfinite(t) & (t > 20.0) & (t < 12000.0) &
            np.isfinite(X).all(axis=1) &
            (X[:, 2] >= -12.0) & (X[:, 2] <= 360.0)
        )
        X = X[ok]
        cols = image[ys[ok], xs[ok]].copy()
        if len(X):
            points_all.append(X.astype(np.float32))
            colours_all.append(cols)

        corrections.append({
            'score': inst['score'], 'foot_px': inst['foot_px'],
            'foot_world_cm': inst['foot_world_cm'],
            'depth_offset_cm': applied, 'surface_mode': mode,
            'surface_points': int(len(X))
        })

    if not points_all:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)), corrections
    return (np.concatenate(points_all, axis=0), np.concatenate(colours_all, axis=0)), corrections


def annotate_v7(img, angle, cov, dyn_cov):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (610, 58), (0, 0, 0), -1)
    cv2.putText(out, '3-CAMERA DIAGNOSTIC v7 | RIGID SUBJECT SURFACES', (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, f'orbit {angle:04.1f} deg  grounded {cov*100:05.1f}%  subject {dyn_cov*100:04.1f}%',
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, .50, (255,255,255), 1, cv2.LINE_AA)
    return out


# Preserve v6's near-play detector and no-secondary-subject ownership.
v5.detect_oncourt = v6.detect_near_play
v5.corrected_dynamic_cloud = rigid_subject_surfaces
v4.hard_fill = v6.no_secondary_subject_fill
v4.annotate = annotate_v7

if __name__ == '__main__':
    v5.main()
