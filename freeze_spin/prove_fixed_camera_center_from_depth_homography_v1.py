from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_W = 960
IMAGE_H = 540


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def robust(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {'n': 0, 'median_px': None, 'p75_px': None, 'p90_px': None,
                'p95_px': None, 'fraction_under_3px': 0.0}
    return {
        'n': int(v.size),
        'median_px': float(np.median(v)),
        'p75_px': float(np.percentile(v, 75)),
        'p90_px': float(np.percentile(v, 90)),
        'p95_px': float(np.percentile(v, 95)),
        'fraction_under_3px': float(np.mean(v <= 3.0)),
    }


def spatial_cells(points: np.ndarray, cols: int = 6, rows: int = 4) -> int:
    cells = set()
    for x, y in np.asarray(points, dtype=float):
        gx = min(cols - 1, max(0, int(x / IMAGE_W * cols)))
        gy = min(rows - 1, max(0, int(y / IMAGE_H * rows)))
        cells.add((gx, gy))
    return len(cells)


def deterministic_fit_partition(point: np.ndarray) -> bool:
    x, y = map(float, point)
    return (int(x // 24.0) + 3 * int(y // 24.0)) % 2 == 0


def event_id_from_dir(path: Path) -> int | None:
    m = re.search(r'event_(\d+)_frames$', path.name)
    return int(m.group(1)) if m else None


def draw_evidence(target_bgr: np.ndarray, source_bgr: np.ndarray, A: np.ndarray, B: np.ndarray,
                  fit_idx: np.ndarray, hold_idx: np.ndarray, bg_idx: np.ndarray,
                  errors: np.ndarray, event_id: int, frame_name: str, out: Path) -> None:
    canvas = np.concatenate([target_bgr.copy(), source_bgr.copy()], axis=1)
    cv2.rectangle(canvas, (0, 190), (780, 440), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (959, 174), (0, 255, 255), 1, cv2.LINE_AA)

    def draw_group(indices, good_color, bad_color, limit):
        n = 0
        for i in indices:
            if n >= limit:
                break
            p = tuple(np.round(A[i]).astype(int))
            q0 = np.round(B[i]).astype(int)
            q = (int(q0[0] + IMAGE_W), int(q0[1]))
            c = good_color if errors[i] <= 3.0 else bad_color
            cv2.line(canvas, p, q, c, 1, cv2.LINE_AA)
            cv2.circle(canvas, p, 2, c, -1, cv2.LINE_AA)
            cv2.circle(canvas, q, 2, c, -1, cv2.LINE_AA)
            n += 1

    draw_group(bg_idx, (0, 255, 255), (0, 0, 255), 75)
    draw_group(hold_idx, (255, 255, 0), (0, 0, 255), 55)
    draw_group(fit_idx, (0, 255, 0), (0, 140, 0), 35)
    cv2.rectangle(canvas, (0, 0), (1919, 38), (0, 0, 0), -1)
    cv2.putText(canvas, f'Play by Play fixed-centre evidence | event {event_id} {frame_name} | green=fit cyan=held-out court yellow=non-coplanar/background red=>3px',
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def draw_warp(target_bgr: np.ndarray, source_bgr: np.ndarray, H: np.ndarray,
              event_id: int, frame_name: str, out: Path) -> None:
    warped = cv2.warpPerspective(target_bgr, H, (IMAGE_W, IMAGE_H))
    blend = cv2.addWeighted(source_bgr, 0.5, warped, 0.5, 0.0)
    panel = np.concatenate([source_bgr, warped, blend], axis=1)
    cv2.rectangle(panel, (0,0), (IMAGE_W*3-1, 38), (0,0,0), -1)
    cv2.putText(panel, f'event {event_id} {frame_name}: source | court-fit target warp | 50/50 blend',
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), panel)


def analyze(target_kp, target_desc, source_path: Path, gates: dict) -> dict | None:
    source_gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if source_gray is None or source_gray.shape != (IMAGE_H, IMAGE_W):
        return None
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=10)
    source_kp, source_desc = sift.detectAndCompute(source_gray, None)
    if source_desc is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(target_desc, source_desc, k=2)
    good = [m for m, n in knn if m.distance < 0.72 * n.distance]
    if len(good) < 60:
        return None
    A = np.float32([target_kp[m.queryIdx].pt for m in good])
    B = np.float32([source_kp[m.trainIdx].pt for m in good])

    H_all, mask_all = cv2.findHomography(A, B, cv2.RANSAC, 2.5, maxIters=10000, confidence=0.999)
    if H_all is None or mask_all is None:
        return None
    all_inliers = mask_all.ravel().astype(bool)
    full_inliers = int(np.sum(all_inliers))
    full_ratio = float(np.mean(all_inliers))
    full_spread = spatial_cells(A[all_inliers])
    pred_all = cv2.perspectiveTransform(A.reshape(-1,1,2), H_all).reshape(-1,2)
    full_errors = np.linalg.norm(pred_all - B, axis=1)

    floor_mask = (A[:,0] <= 780.0) & (A[:,1] >= 190.0) & (A[:,1] <= 440.0)
    floor_idx = np.where(floor_mask)[0]
    fit_select = np.asarray([deterministic_fit_partition(A[i]) for i in floor_idx], dtype=bool)
    fit_idx = floor_idx[fit_select]
    hold_idx = floor_idx[~fit_select]
    bg_idx = np.where(A[:,1] < 175.0)[0]
    if len(fit_idx) < 15 or len(hold_idx) < 20 or len(bg_idx) < 40:
        return None

    H_fit, fit_mask = cv2.findHomography(A[fit_idx], B[fit_idx], cv2.RANSAC, 2.5,
                                         maxIters=10000, confidence=0.999)
    if H_fit is None or fit_mask is None:
        return None
    fit_inliers = int(np.sum(fit_mask))
    pred = cv2.perspectiveTransform(A.reshape(-1,1,2), H_fit).reshape(-1,2)
    errors = np.linalg.norm(pred - B, axis=1)
    hold = robust(errors[hold_idx])
    bg = robust(errors[bg_idx])
    full = robust(full_errors[all_inliers])

    gate_values = {
        'full_scene_inliers': full_inliers >= gates['min_full_scene_inliers'],
        'full_scene_inlier_ratio': full_ratio >= gates['min_full_scene_inlier_ratio'],
        'full_scene_spatial_cells': full_spread >= gates['min_full_scene_spatial_cells'],
        'court_fit_inliers': fit_inliers >= gates['min_court_fit_inliers'],
        'heldout_court_count': hold['n'] >= gates['min_heldout_court_matches'],
        'heldout_court_median': hold['median_px'] <= gates['max_heldout_court_median_px'],
        'heldout_court_fraction_3px': hold['fraction_under_3px'] >= gates['min_heldout_court_fraction_under_3px'],
        'noncoplanar_count': bg['n'] >= gates['min_noncoplanar_matches'],
        'noncoplanar_median': bg['median_px'] <= gates['max_noncoplanar_median_px'],
        'noncoplanar_fraction_3px': bg['fraction_under_3px'] >= gates['min_noncoplanar_fraction_under_3px'],
    }
    passed = all(gate_values.values())
    score = (
        float(bg['median_px']) + float(hold['median_px'])
        + max(0.0, gates['min_noncoplanar_fraction_under_3px'] - bg['fraction_under_3px']) * 20.0
        + max(0.0, gates['min_heldout_court_fraction_under_3px'] - hold['fraction_under_3px']) * 20.0
    )
    return {
        'source_file': str(source_path),
        'good_matches': len(good),
        'full_scene': {
            'ransac_inliers': full_inliers,
            'inlier_ratio': full_ratio,
            'spatial_grid_cells_6x4': full_spread,
            'inlier_residual': full,
        },
        'court_fit': {
            'candidate_matches': int(len(fit_idx)),
            'ransac_inliers': fit_inliers,
        },
        'heldout_court': hold,
        'noncoplanar_background': bg,
        'gates': gate_values,
        'passed': bool(passed),
        'score_lower_is_better': float(score),
        '_H_fit': H_fit,
        '_A': A,
        '_B': B,
        '_fit_idx': fit_idx,
        '_hold_idx': hold_idx,
        '_bg_idx': bg_idx,
        '_errors': errors,
    }


def clean_for_json(row: dict) -> dict:
    return {k:v for k,v in row.items() if not k.startswith('_')}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--target-frame',type=Path,required=True)
    ap.add_argument('--target-sha256',required=True)
    ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--camera-label',required=True)
    ap.add_argument('--min-independent-events',type=int,default=4)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()

    gates = {
        'min_full_scene_inliers': 100,
        'min_full_scene_inlier_ratio': 0.60,
        'min_full_scene_spatial_cells': 10,
        'min_court_fit_inliers': 15,
        'min_heldout_court_matches': 20,
        'max_heldout_court_median_px': 2.0,
        'min_heldout_court_fraction_under_3px': 0.55,
        'min_noncoplanar_matches': 40,
        'max_noncoplanar_median_px': 2.0,
        'min_noncoplanar_fraction_under_3px': 0.70,
        'min_independent_events': int(args.min_independent_events),
    }

    actual_sha = sha256(args.target_frame)
    if actual_sha != args.target_sha256:
        raise SystemExit(f'Immutable target SHA mismatch: {actual_sha}')
    target_gray=cv2.imread(str(args.target_frame),cv2.IMREAD_GRAYSCALE)
    target_bgr=cv2.imread(str(args.target_frame),cv2.IMREAD_COLOR)
    if target_gray is None or target_gray.shape != (IMAGE_H, IMAGE_W):
        raise SystemExit('Immutable target must be native 960x540')
    sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=0.02,edgeThreshold=10)
    target_kp,target_desc=sift.detectAndCompute(target_gray,None)
    if target_desc is None:
        raise SystemExit('No target descriptors')

    args.out.mkdir(parents=True,exist_ok=True)
    evidence_dir=args.out/'evidence'; evidence_dir.mkdir(exist_ok=True)
    event_dirs=[d for d in sorted(args.samples.glob('event_*_frames')) if d.is_dir() and event_id_from_dir(d) is not None]
    if not event_dirs:
        raise SystemExit(f'No event_*_frames directories under {args.samples}')

    all_rows=[]; best_by_event=[]
    for d in event_dirs:
        eid=event_id_from_dir(d)
        rows=[]
        for p in sorted(d.glob('f*.png')):
            a=analyze(target_kp,target_desc,p,gates)
            if a is None:
                continue
            a['event_id']=eid; a['frame_name']=p.name
            rows.append(a); all_rows.append(a)
        passed=[r for r in rows if r['passed']]
        if passed:
            best=min(passed,key=lambda r:r['score_lower_is_better'])
            best_by_event.append(best)
            src=cv2.imread(best['source_file'])
            draw_evidence(target_bgr,src,best['_A'],best['_B'],best['_fit_idx'],best['_hold_idx'],best['_bg_idx'],best['_errors'],eid,best['frame_name'],evidence_dir/f'event_{eid}_correspondence_evidence.png')
            draw_warp(target_bgr,src,best['_H_fit'],eid,best['frame_name'],evidence_dir/f'event_{eid}_warp_blend.png')

    best_by_event.sort(key=lambda r:r['event_id'])
    independent_event_count=len(best_by_event)
    status='PASS_SHARED_OPTICAL_CENTER_PRIOR' if independent_event_count >= gates['min_independent_events'] else 'FAIL_SHARED_OPTICAL_CENTER_PRIOR'
    permissions={
        'shared_optical_center_prior_allowed': status.startswith('PASS_'),
        'metric_camera_promotion_allowed': False,
        'static_novel_view_allowed': False,
        'replay_render_allowed': False,
    }
    report={
        'status':status,
        'game_id':'0022500301',
        'camera_label':args.camera_label,
        'immutable_target':{'file':args.target_frame.name,'sha256':actual_sha,'geometry':'960x540 native NBA frame'},
        'proof_claim': 'Independent same-game images are consistent with a shared effective optical centre strongly enough to pool static geometry as a prior. This does not solve a metric camera.',
        'geometric_basis': 'For images acquired from a common optical centre, changes in camera rotation and intrinsics are related by a single image homography independent of scene depth. Here the homography is estimated only from one deterministic subset of court correspondences, then tested on unseen court correspondences and a separate non-coplanar basket/crowd/background band.',
        'guardrails': [
            'Full-image homography fit alone is never sufficient.',
            'The non-coplanar/background band is not used to estimate the court-fit homography.',
            'Dynamic player/ball pixels are not annotated or used as metric anchors; feature outliers are tolerated only through explicit robust fractions.',
            'Raw lens distortion is not modelled here; the later metric camera solve must independently validate exact Frame C geometry.',
            'This proof cannot promote a metric camera, novel view, or replay.'
        ],
        'gates':gates,
        'independent_passing_event_count':independent_event_count,
        'passing_events':[clean_for_json(r) for r in best_by_event],
        'all_analyzed_frames':[clean_for_json(r) for r in all_rows],
        'permissions':permissions,
    }
    (args.out/'fixed_camera_center_depth_homography_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':status,'independent_passing_event_count':independent_event_count,'passing_events':[(r['event_id'],r['frame_name'],round(r['heldout_court']['median_px'],3),round(r['noncoplanar_background']['median_px'],3)) for r in best_by_event],'permissions':permissions},indent=2),flush=True)
    if status.startswith('FAIL_'):
        raise SystemExit(status)

if __name__=='__main__':
    main()
