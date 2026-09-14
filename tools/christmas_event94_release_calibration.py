#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adams_screen_prod_poc import fetch_native_clip


def fit_raw(xy, cf, vertices, conf, ransac=10.0):
    sel = (cf >= conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
    n = int(sel.sum())
    if n < 4:
        return {"H": None, "n_kp": n, "reason": "too_few"}
    src = xy[sel].astype(np.float64)
    dst = vertices[sel].astype(np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    if H is None:
        return {"H": None, "n_kp": n, "reason": "fit_failed"}
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(n, bool)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)
    return {
        "H": H.tolist(),
        "n_kp": n,
        "inliers": int(inl.sum()),
        "median_reproj_cm_all": float(np.median(err)),
        "median_reproj_cm_inliers": float(np.median(err[inl])) if inl.any() else None,
        "selected_indices": np.flatnonzero(sel).astype(int).tolist(),
        "inlier_selected_indices": np.flatnonzero(sel)[inl].astype(int).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default='0022500012')
    ap.add_argument('--event', type=int, default=94)
    ap.add_argument('--release-s', type=float, default=10.56)
    ap.add_argument('--court-model', type=Path, required=True)
    ap.add_argument('--nbacv-src', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('artifacts/christmas_release_calibration'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.nbacv_src))
    from nbacv.court import _court_infer, fit_homography, homography_plausible
    from nbacv.court_config import BasketballCourtConfiguration

    clip = a.out / 'source_native.mp4'
    meta = fetch_native_clip(a.game, a.event, clip, 960)
    cap = cv2.VideoCapture(str(clip))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.release_s * 1000.0)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError('release frame unavailable')
    cv2.imwrite(str(a.out / 'release_frame.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])

    model = YOLO(str(a.court_model))
    cfg = BasketballCourtConfiguration()
    vertices = np.asarray(cfg.vertices, dtype=np.float64)
    result = {
        'game_id': a.game,
        'event_num': a.event,
        'release_s': a.release_s,
        'fps': fps,
        'source': meta,
        'frame_hw': [int(frame.shape[0]), int(frame.shape[1])],
        'attempts': [],
    }
    best = None
    best_score = (-1, 1e99)
    for sz in (640, 960, 1280, 1536):
        kp = _court_infer(model, frame, sz, 'cpu')
        if kp is None:
            result['attempts'].append({'imgsz': sz, 'detected': False})
            continue
        xy, cf = kp
        rec = {'imgsz': sz, 'detected': True,
               'xy': xy.tolist(), 'conf': cf.tolist(),
               'n_conf_03': int((cf >= .3).sum()),
               'n_conf_04': int((cf >= .4).sum()),
               'n_conf_05': int((cf >= .5).sum())}
        Hstd, info = fit_homography(xy, cf, frame_hw=frame.shape[:2])
        rec['standard'] = {'H': None if Hstd is None else Hstd.tolist(), **info}
        rec['raw_fits'] = {}
        for c in (.30, .35, .40, .45, .50, .55, .60):
            rf = fit_raw(xy, cf, vertices, c)
            if rf.get('H') is not None:
                okp, why = homography_plausible(np.asarray(rf['H']), frame.shape[:2])
                rf['plausible'] = bool(okp); rf['plausibility'] = why
                score = (rf.get('inliers', 0), rf.get('median_reproj_cm_inliers') or 1e99)
                if okp and (score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1])):
                    best_score = score; best = {'imgsz': sz, 'conf': c, **rf}
            rec['raw_fits'][f'{c:.2f}'] = rf
        result['attempts'].append(rec)

        ann = frame.copy()
        for i, ((x, y), conf) in enumerate(zip(xy, cf)):
            if conf < .25 or x <= 1 or y <= 1: continue
            col = (0, 255, 0) if conf >= .5 else (0, 180, 255)
            cv2.circle(ann, (int(round(x)), int(round(y))), 4, col, -1, cv2.LINE_AA)
            cv2.putText(ann, str(i), (int(x)+5, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(ann, str(i), (int(x)+5, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(str(a.out / f'keypoints_{sz}.jpg'), ann, [cv2.IMWRITE_JPEG_QUALITY, 97])

    result['best_plausible_raw_fit'] = best
    (a.out / 'release_calibration.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'best': best, 'attempt_summary': [
        {k:v for k,v in r.items() if k in ('imgsz','detected','n_conf_03','n_conf_04','n_conf_05','standard')}
        for r in result['attempts']
    ]}, indent=2))

if __name__ == '__main__':
    main()
