from __future__ import annotations
"""v78: implementation-corrected Brown-Conrady In-Arena certification.

This wrapper preserves every v77 observation, parameter bound, model degree of
freedom and frozen v76 acceptance threshold.  It corrects one coordinate-metric
bug found after v77: training line/circle residuals were measured after
undistortion, while acceptance thresholds are defined in observed distorted
pixels; the floor summary also measured observations to finite line segments.

v78 uses the same v45 convention: evaluate each implicit primitive through the
inverse distortion map, then divide by its numerical gradient in the observed
(distorted) image.  This yields a first-order Euclidean residual in native
observed pixels.  No threshold or evidence point is changed.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from freeze_spin import prove_in_arena_brown_v77 as v77
from freeze_spin import solve_in_arena_direct_camera_v55a as d
from freeze_spin.prove_in_arena_noncoplanar_center_v54 import RIM_CENTER

EPS = 0.25


def _line_implicit(p: np.ndarray, k: np.ndarray, name: str, pixels_distorted: np.ndarray) -> np.ndarray:
    pred_u, _ = d.project(p, d.line_world(name))
    line = d.image_line(pred_u)
    pixels_u = v77.undistort(np.asarray(pixels_distorted, float), p[7:9], k)
    return pixels_u @ line[:2] + line[2]


def signed_line_pixel_residual(p: np.ndarray, k: np.ndarray, name: str, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, float)
    f = _line_implicit(p, k, name, pixels)
    gx = (_line_implicit(p, k, name, pixels + [EPS, 0.0]) - _line_implicit(p, k, name, pixels - [EPS, 0.0])) / (2.0 * EPS)
    gy = (_line_implicit(p, k, name, pixels + [0.0, EPS]) - _line_implicit(p, k, name, pixels - [0.0, EPS])) / (2.0 * EPS)
    return f / np.maximum(np.hypot(gx, gy), 1e-6)


def _floor_conic(p: np.ndarray, center_x: float, radius: float) -> np.ndarray:
    rvec = p[:3]
    center = p[3:6]
    focal = math.exp(float(p[6]))
    cx, cy = p[7:9]
    import cv2
    rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    translation = -rotation @ center
    intrinsic = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    floor_h = intrinsic @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    circle = np.asarray([[1.0, 0.0, -center_x], [0.0, 1.0, 0.0], [-center_x, 0.0, center_x**2 - radius**2]], dtype=np.float64)
    inv = np.linalg.inv(floor_h)
    return inv.T @ circle @ inv


def _circle_implicit(p: np.ndarray, k: np.ndarray, center_x: float, radius: float, pixels_distorted: np.ndarray) -> np.ndarray:
    conic = _floor_conic(p, center_x, radius)
    pixels_u = v77.undistort(np.asarray(pixels_distorted, float), p[7:9], k)
    h = np.column_stack([pixels_u, np.ones(len(pixels_u))])
    return np.einsum('ni,ij,nj->n', h, conic, h)


def signed_circle_pixel_residual(p: np.ndarray, k: np.ndarray, center_x: float, radius: float, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, float)
    f = _circle_implicit(p, k, center_x, radius, pixels)
    gx = (_circle_implicit(p, k, center_x, radius, pixels + [EPS, 0.0]) - _circle_implicit(p, k, center_x, radius, pixels - [EPS, 0.0])) / (2.0 * EPS)
    gy = (_circle_implicit(p, k, center_x, radius, pixels + [0.0, EPS]) - _circle_implicit(p, k, center_x, radius, pixels - [0.0, EPS])) / (2.0 * EPS)
    return f / np.maximum(np.hypot(gx, gy), 1e-6)


def corrected_line_train_residual(p, lines, k):
    return np.concatenate([signed_line_pixel_residual(p, k, name, obs) for name, obs in lines.items()])


def corrected_circle_train_residual(p, obs, cx, r, k):
    return signed_circle_pixel_residual(p, k, cx, r, obs)


_original_summarize = v77.summarize


def corrected_summarize(sol, keys, lines, restricted, ft, rims, ellipses, transfers, obj, obs):
    qa = _original_summarize(sol, keys, lines, restricted, ft, rims, ellipses, transfers, obj, obs)
    target = sol['states']['target']
    k = sol['distortion']
    signed = np.concatenate([signed_line_pixel_residual(target, k, name, px) for name, px in lines.items()])
    qa['line_p95_px'] = float(np.percentile(np.abs(signed), 95))
    qa['line_rms_px'] = float(np.sqrt(np.mean(signed**2)))
    qa['line_max_px'] = float(np.max(np.abs(signed)))
    qa['line_metric'] = 'observed-distorted-image first-order signed Euclidean distance to infinite projected regulation line; v45 Brown-Conrady convention'
    return qa


def _arg_path(flag: str) -> Path:
    i = sys.argv.index(flag)
    return Path(sys.argv[i + 1])


def main():
    v77.line_train_residual = corrected_line_train_residual
    v77.circle_train_residual = corrected_circle_train_residual
    v77.summarize = corrected_summarize
    v77.main()

    out = _arg_path('--out')
    src = out / 'in_arena_brown_v77.json'
    report = json.loads(src.read_text(encoding='utf-8'))
    passed = bool(all(report['gates'].values()))
    report['version'] = 'v78'
    report['status'] = 'PASS_IN_ARENA_BROWN_METRIC_CAMERA_V78' if passed else 'FAIL_IN_ARENA_BROWN_METRIC_CAMERA_V78'
    report['method'] = 'v77 immutable evidence/model with corrected observed-image Brown-Conrady primitive residual metric; all v76 thresholds and v77 distortion bounds unchanged'
    report['implementation_correction'] = {
        'v77_issue_1': 'line/circle training residuals were measured in undistorted-pixel metric rather than observed distorted-image pixels',
        'v77_issue_2': 'floor-line summary used nearest distance to finite projected world segments rather than the infinite regulation lines used by the solver',
        'v78_fix': 'v45-style implicit primitive residual evaluated through inverse distortion with numerical observed-image gradient; floor summary uses the same native-pixel infinite-line metric',
        'thresholds_changed': False,
        'observations_changed': False,
        'model_capacity_changed': False,
    }
    report['permissions']['physical_camera_center_allowed'] = passed
    report['permissions']['metric_event_camera_allowed'] = passed
    report['permissions']['replay_render_allowed'] = False
    dst = out / 'in_arena_brown_v78.json'
    dst.write_text(json.dumps(v77.safe(report), indent=2), encoding='utf-8')
    old_overlay = out / 'in_arena_brown_v77_overlay.png'
    new_overlay = out / 'in_arena_brown_v78_overlay.png'
    if old_overlay.exists():
        old_overlay.replace(new_overlay)
    print(json.dumps(v77.safe({'status': report['status'], 'camera_center_cm': report['best']['camera_center_cm'], 'floor_line_p95_px': report['best']['line_p95_px'], 'gates': report['gates'], 'permissions': report['permissions']}), indent=2))


if __name__ == '__main__':
    main()
