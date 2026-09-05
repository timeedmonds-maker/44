from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
IMAGE_W = 960
IMAGE_H = 540
INCH_CM = 2.54
FOOT_CM = 12.0 * INCH_CM
RIM_Z_CM = 10.0 * FOOT_CM
RIM_RADIUS_CM = 9.0 * INCH_CM
RIM_X_CM = 15.0 * INCH_CM
TARGET_OUTER_W_CM = 24.0 * INCH_CM
TARGET_OUTER_H_CM = 18.0 * INCH_CM
TARGET_STRIPE_CM = 2.0 * INCH_CM
TARGET_INNER_W_CM = TARGET_OUTER_W_CM - 2.0 * TARGET_STRIPE_CM
TARGET_INNER_H_CM = TARGET_OUTER_H_CM - 2.0 * TARGET_STRIPE_CM
BACKBOARD_W_CM = 6.0 * FOOT_CM
BACKBOARD_H_CM = 3.5 * FOOT_CM
PAINT_W_CM = 16.0 * FOOT_CM
FT_BOARD_DISTANCE_CM = 15.0 * FOOT_CM

def world_landmarks() -> dict[str, np.ndarray]:
    lane_half = PAINT_W_CM / 2.0
    inner_hw = TARGET_INNER_W_CM / 2.0
    return {'target_inner_top_left': np.array([0.0, -inner_hw, RIM_Z_CM + TARGET_OUTER_H_CM - TARGET_STRIPE_CM]), 'target_inner_top_right': np.array([0.0, +inner_hw, RIM_Z_CM + TARGET_OUTER_H_CM - TARGET_STRIPE_CM]), 'target_inner_bottom_right': np.array([0.0, +inner_hw, RIM_Z_CM + TARGET_STRIPE_CM]), 'target_inner_bottom_left': np.array([0.0, -inner_hw, RIM_Z_CM + TARGET_STRIPE_CM]), 'rim_left': np.array([RIM_X_CM, -RIM_RADIUS_CM, RIM_Z_CM]), 'rim_right': np.array([RIM_X_CM, +RIM_RADIUS_CM, RIM_Z_CM]), 'baseline_left_lane': np.array([-4.0 * FOOT_CM, -lane_half, 0.0]), 'baseline_right_lane': np.array([-4.0 * FOOT_CM, +lane_half, 0.0]), 'ft_left_lane': np.array([FT_BOARD_DISTANCE_CM, -lane_half, 0.0]), 'ft_right_lane': np.array([FT_BOARD_DISTANCE_CM, +lane_half, 0.0])}

def rim_circle(samples: int=240) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return np.column_stack([RIM_X_CM + RIM_RADIUS_CM * np.cos(t), RIM_RADIUS_CM * np.sin(t), np.full_like(t, RIM_Z_CM)]).astype(np.float64)
RING = rim_circle()

def board_corners(board_bottom_z_cm: float) -> np.ndarray:
    hw = BACKBOARD_W_CM / 2.0
    z0 = float(board_bottom_z_cm)
    z1 = z0 + BACKBOARD_H_CM
    return np.asarray([[0.0, -hw, z1], [0.0, +hw, z1], [0.0, +hw, z0], [0.0, -hw, z0]], dtype=np.float64)

def project(params: np.ndarray, obj: np.ndarray):
    rvec = params[:3]
    tvec = params[3:6]
    focal = float(np.exp(params[6]))
    cx, cy = (float(params[7]), float(params[8]))
    R, _ = cv2.Rodrigues(rvec)
    cam = (R @ obj.T).T + tvec
    uv = np.column_stack([focal * cam[:, 0] / cam[:, 2] + cx, focal * cam[:, 1] / cam[:, 2] + cy])
    return (uv, cam, focal, R)

def canonical_ellipse(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 5:
        raise ValueError('At least five points are required for ellipse fit')
    (cx, cy), (a, b), angle = cv2.fitEllipse(pts)
    if a >= b:
        major, minor, major_angle = (float(a), float(b), float(angle))
    else:
        major, minor, major_angle = (float(b), float(a), float((angle + 90.0) % 180.0))
    if major_angle >= 90.0:
        major_angle -= 180.0
    return np.array([float(cx), float(cy), major, minor, major_angle], dtype=np.float64)

def angle_diff_deg(a: float, b: float) -> float:
    return float((a - b + 90.0) % 180.0 - 90.0)

def view_points(view: dict, world: dict[str, np.ndarray]):
    names = list(view['landmarks'].keys())
    missing = [n for n in names if n not in world]
    if missing:
        raise KeyError(f"Unknown world landmarks for {view['label']}: {missing}")
    obj = np.asarray([world[n] for n in names], dtype=np.float64)
    obs = np.asarray([view['landmarks'][n] for n in names], dtype=np.float64)
    return (names, obj, obs)

def observed_rim(view: dict):
    samples = view.get('rim_curve_samples_px')
    if not samples:
        return (None, None)
    pts = np.asarray(samples, dtype=np.float64)
    return (canonical_ellipse(pts), pts)

def observed_board(view: dict):
    payload = view.get('backboard_outer_corners_px')
    if not payload:
        return None
    return np.asarray([payload[k] for k in ('top_left', 'top_right', 'bottom_right', 'bottom_left')], dtype=np.float64)

def seed_points(view: dict, world: dict[str, np.ndarray], obj: np.ndarray, obs: np.ndarray):
    seed_obj = [p for p in obj]
    seed_obs = [p for p in obs]
    seeds = view.get('rim_seed_points_px')
    if seeds:
        for name in ('rim_left', 'rim_right'):
            seed_obj.append(world[name])
            seed_obs.append(np.asarray(seeds[name], dtype=np.float64))
    return (np.asarray(seed_obj, dtype=np.float64), np.asarray(seed_obs, dtype=np.float64))

def solve_camera(view: dict, world: dict[str, np.ndarray], *, obs_override=None, rim_override=None, board_override=None, warm_params=None):
    names, obj, obs0 = view_points(view, world)
    obs = np.asarray(obs0 if obs_override is None else obs_override, dtype=np.float64)
    rim_desc0, rim_samples0 = observed_rim(view)
    rim_samples = rim_samples0 if rim_override is None else np.asarray(rim_override, dtype=np.float64)
    rim_desc = None if rim_samples is None else canonical_ellipse(rim_samples)
    board_obs0 = observed_board(view)
    board_obs = board_obs0 if board_override is None else np.asarray(board_override, dtype=np.float64)
    seed_obj, seed_obs = seed_points(view, world, obj, obs)
    pp_sigma = float(view.get('principal_point_prior_sigma_px', 80.0))
    pp_bound = float(view.get('principal_point_bound_px', 220.0))
    focal_prior = float(view.get('focal_prior_px', 900.0))
    focal_sigma_log = float(view.get('focal_prior_sigma_log', 1.8))
    rim_weight = float(view.get('rim_curve_weight', 1.0))
    board_weight = float(view.get('backboard_weight', 1.0))
    if warm_params is not None:
        starts = [np.asarray(warm_params, dtype=np.float64).copy()]
    else:
        starts = []
        for f0 in (400.0, 600.0, 800.0, 1000.0, 1200.0, 1600.0, 2200.0):
            K = np.asarray([[f0, 0.0, IMAGE_W / 2.0], [0.0, f0, IMAGE_H / 2.0], [0.0, 0.0, 1.0]])
            ok, rvec, tvec = cv2.solvePnP(seed_obj, seed_obs, K, None, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                continue
            base = np.r_[rvec.ravel(), tvec.ravel(), np.log(f0), IMAGE_W / 2.0, IMAGE_H / 2.0]
            if board_obs is not None:
                base = np.r_[base, float(view.get('backboard_bottom_initial_cm', 9.0 * FOOT_CM))]
            starts.append(base)
    candidates = []
    for p0 in starts:
        def residual(p):
            uv, cam, focal, _ = project(p, obj)
            out = [(uv - obs).ravel()]
            ring_cam = np.empty((0, 3), dtype=np.float64)
            if rim_desc is not None:
                pred_uv, ring_cam, _, _ = project(p, RING)
                pred = canonical_ellipse(pred_uv)
                angle_px = math.radians(angle_diff_deg(pred[4], rim_desc[4])) * (rim_desc[2] / 2.0)
                out.append(np.array([pred[0] - rim_desc[0], pred[1] - rim_desc[1], pred[2] - rim_desc[2], (pred[3] - rim_desc[3]) / 0.8, angle_px]) * rim_weight)
            board_cam = np.empty((0, 3), dtype=np.float64)
            if board_obs is not None:
                bobj = board_corners(float(p[9]))
                buv, board_cam, _, _ = project(p, bobj)
                out.append((buv - board_obs).ravel() * board_weight)
            out.append(np.array([(p[7] - IMAGE_W / 2.0) / pp_sigma, (p[8] - IMAGE_H / 2.0) / pp_sigma, (np.log(focal) - np.log(focal_prior)) / focal_sigma_log]))
            depth = cam[:, 2]
            if len(ring_cam):
                depth = np.r_[depth, ring_cam[:, 2]]
            if len(board_cam):
                depth = np.r_[depth, board_cam[:, 2]]
            out.append(np.minimum(depth - 20.0, 0.0) / 5.0)
            return np.concatenate(out)
        lower = np.r_[[-np.inf] * 6, np.log(150.0), IMAGE_W / 2.0 - pp_bound, IMAGE_H / 2.0 - pp_bound]
        upper = np.r_[[+np.inf] * 6, np.log(4000.0), IMAGE_W / 2.0 + pp_bound, IMAGE_H / 2.0 + pp_bound]
        if board_obs is not None:
            lower = np.r_[lower, 200.0]
            upper = np.r_[upper, 360.0]
        try:
            opt = least_squares(residual, p0, bounds=(lower, upper), loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=40000)
        except Exception:
            continue
        uv, cam, focal, R = project(opt.x, obj)
        rmse = float(np.sqrt(np.mean(np.sum((uv - obs) ** 2, axis=1))))
        center = -R.T @ opt.x[3:6]
        plausible = float(np.min(cam[:, 2])) > 20.0 and 150.0 <= focal <= 4000.0 and (-0.3 * IMAGE_W <= opt.x[7] <= 1.3 * IMAGE_W) and (-0.3 * IMAGE_H <= opt.x[8] <= 1.3 * IMAGE_H) and (-2000.0 <= center[2] <= 3000.0)
        rim_metrics = None
        if rim_desc is not None:
            pred_desc = canonical_ellipse(project(opt.x, RING)[0])
            rim_metrics = np.array([float(np.linalg.norm(pred_desc[:2] - rim_desc[:2])), abs(float(pred_desc[2] - rim_desc[2])), abs(float(pred_desc[3] - rim_desc[3])), abs(angle_diff_deg(pred_desc[4], rim_desc[4]))])
        board_rmse = None
        if board_obs is not None:
            buv = project(opt.x, board_corners(float(opt.x[9])))[0]
            board_rmse = float(np.sqrt(np.mean(np.sum((buv - board_obs) ** 2, axis=1))))
        extra = 0.0
        if rim_metrics is not None:
            extra += float(np.sum(rim_metrics[:3])) / 3.0 + float(rim_metrics[3]) / 5.0
        if board_rmse is not None:
            extra += board_rmse / 4.0
        score = float(opt.cost)
        candidates.append((not plausible, score, opt.x, rmse, center, focal, rim_metrics, board_rmse))
    if not candidates:
        raise RuntimeError(f"No camera solution converged for {view['label']}")
    candidates.sort(key=lambda x: (x[0], x[1], abs(np.log(x[5] / focal_prior))))
    return (names, obj, obs, rim_samples, board_obs, candidates[0])

def perturbation_sensitivity(view: dict, world: dict[str, np.ndarray], base_params: np.ndarray, base_center: np.ndarray, trials: int, seed: int):
    _, _, obs = view_points(view, world)
    _, rim = observed_rim(view)
    board = observed_board(view)
    rng = np.random.default_rng(seed)
    shifts = []
    for _ in range(trials):
        pobs = obs + rng.uniform(-0.5, 0.5, size=obs.shape)
        prim = None if rim is None else rim + rng.uniform(-0.5, 0.5, size=rim.shape)
        pboard = None if board is None else board + rng.uniform(-0.5, 0.5, size=board.shape)
        try:
            *_, solved = solve_camera(view, world, obs_override=pobs, rim_override=prim, board_override=pboard, warm_params=base_params)
        except Exception:
            return (float('inf'), [])
        if solved[0]:
            return (float('inf'), [])
        shifts.append(float(np.linalg.norm(solved[4] - base_center)))
    return (max(shifts) if shifts else float('inf'), shifts)

def draw_overlay(image: np.ndarray, params: np.ndarray, names: list[str], obj: np.ndarray, obs: np.ndarray, rim_samples: np.ndarray | None, board_obs: np.ndarray | None, out: Path):
    overlay = image.copy()
    uv = project(params, obj)[0]
    index = {name: i for i, name in enumerate(names)}
    order = ['target_inner_top_left', 'target_inner_top_right', 'target_inner_bottom_right', 'target_inner_bottom_left']
    if all((n in index for n in order)):
        rect = np.round(np.asarray([uv[index[n]] for n in order])).astype(int)
        cv2.polylines(overlay, [rect], True, (0, 255, 0), 2, cv2.LINE_AA)
    lane_order = ['baseline_left_lane', 'baseline_right_lane', 'ft_right_lane', 'ft_left_lane']
    if all((n in index for n in lane_order)):
        lane = np.round(np.asarray([uv[index[n]] for n in lane_order])).astype(int)
        cv2.polylines(overlay, [lane], True, (255, 180, 0), 2, cv2.LINE_AA)
    if rim_samples is not None:
        ring_uv = project(params, RING)[0]
        cv2.polylines(overlay, [np.round(ring_uv).astype(int)], True, (255, 0, 255), 2, cv2.LINE_AA)
        for p in np.round(rim_samples).astype(int):
            cv2.circle(overlay, tuple(p), 3, (0, 215, 255), -1, cv2.LINE_AA)
    if board_obs is not None:
        bproj = project(params, board_corners(float(params[9])))[0]
        cv2.polylines(overlay, [np.round(bproj).astype(int)], True, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.polylines(overlay, [np.round(board_obs).astype(int)], True, (0, 165, 255), 1, cv2.LINE_AA)
    for p in np.round(obs).astype(int):
        cv2.circle(overlay, tuple(p), 3, (0, 255, 255), -1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    payload = json.loads(args.landmarks.read_text(encoding='utf-8'))
    world = world_landmarks()
    args.out.mkdir(parents=True, exist_ok=True)
    max_rmse = float(payload.get('max_landmark_rmse_px', 3.0))
    max_rim_center = float(payload.get('max_rim_center_error_px', 3.0))
    max_rim_major = float(payload.get('max_rim_major_axis_error_px', 4.0))
    max_rim_minor = float(payload.get('max_rim_minor_axis_error_px', 2.0))
    max_rim_angle = float(payload.get('max_rim_angle_error_deg', 3.0))
    max_board_rmse = float(payload.get('max_backboard_corner_rmse_px', 12.5))
    max_sensitivity = float(payload.get('max_half_pixel_camera_center_shift_cm', 75.0))
    max_board_z_spread = float(payload.get('max_cross_view_backboard_bottom_spread_cm', 5.0))
    perturb_trials = int(payload.get('perturbation_trials', 24))
    required_views = int(payload.get('minimum_required_passed_views', 4))
    min_baseline_gate = float(payload.get('min_distinct_camera_baseline_cm', 50.0))
    rows = []
    for vi, view in enumerate(payload['views']):
        names, obj, obs, rim_samples, board_obs, solved = solve_camera(view, world)
        rejected, score, params, rmse, center, focal, rim_metrics_arr, board_rmse = solved
        image = cv2.imread(str(args.images / view['image']))
        if image is None:
            raise RuntimeError(f"Missing image {view['image']}")
        draw_overlay(image, params, names, obj, obs, rim_samples, board_obs, args.out / f"{view['index']:02d}_{view['label'].replace(' ', '_')}_overlay.png")
        sensitivity, perturb_shifts = perturbation_sensitivity(view, world, params, center, perturb_trials, 20260901 + vi * 97)
        rim_ok = True
        rim_metrics = None
        if rim_metrics_arr is not None:
            rim_metrics = {'center_error_px': round(float(rim_metrics_arr[0]), 4), 'major_axis_error_px': round(float(rim_metrics_arr[1]), 4), 'minor_axis_error_px': round(float(rim_metrics_arr[2]), 4), 'angle_error_deg': round(float(rim_metrics_arr[3]), 4)}
            rim_ok = rim_metrics['center_error_px'] <= max_rim_center and rim_metrics['major_axis_error_px'] <= max_rim_major and (rim_metrics['minor_axis_error_px'] <= max_rim_minor) and (rim_metrics['angle_error_deg'] <= max_rim_angle)
        board_ok = board_rmse is None or board_rmse <= max_board_rmse
        status = not rejected and rmse <= max_rmse and (sensitivity <= max_sensitivity) and rim_ok and board_ok
        rows.append({'index': int(view['index']), 'label': view['label'], 'landmark_names': names, 'landmark_rmse_px': round(float(rmse), 4), 'focal_px': round(float(focal), 3), 'principal_point_px': [round(float(params[7]), 3), round(float(params[8]), 3)], 'camera_center_basket_local_cm': [round(float(x), 3) for x in center], 'rim_conic': rim_metrics, 'backboard_corner_rmse_px': None if board_rmse is None else round(float(board_rmse), 4), 'estimated_backboard_bottom_z_cm': None if board_obs is None else round(float(params[9]), 4), 'half_pixel_independent_perturb_max_camera_center_shift_cm': round(float(sensitivity), 4), 'half_pixel_independent_perturb_p95_camera_center_shift_cm': None if not perturb_shifts else round(float(np.percentile(perturb_shifts, 95)), 4), 'status': 'pass' if status else 'reject'})
    board_zs = [r['estimated_backboard_bottom_z_cm'] for r in rows if r['estimated_backboard_bottom_z_cm'] is not None and r['status'] == 'pass']
    board_z_spread = max(board_zs) - min(board_zs) if len(board_zs) >= 2 else 0.0
    passed_rows = [r for r in rows if r['status'] == 'pass']
    centers = [np.asarray(r['camera_center_basket_local_cm'], dtype=float) for r in passed_rows]
    pairwise = []
    for i in range(len(passed_rows)):
        for j in range(i + 1, len(passed_rows)):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            pairwise.append({'a': passed_rows[i]['label'], 'b': passed_rows[j]['label'], 'baseline_cm': round(d, 3)})
    min_baseline = min((p['baseline_cm'] for p in pairwise), default=0.0)
    passed = len(passed_rows) >= required_views and min_baseline >= min_baseline_gate and (board_z_spread <= max_board_z_spread)
    report = {'method': 'NBA metric target opening + full rim conic + full backboard outline / court-plane camera solve', 'key_changes': ['uses the complete projected 18-inch rim circle rather than two rim points or a rim bounding box', 'uses the 20x14-inch clean opening inside the regulation 2-inch 24x18 target stripe', 'uses the full 6ft x 3.5ft backboard outline on near-basket views to break focal/distance ambiguity', 'treats backboard bottom elevation as an estimated nuisance value and checks cross-view agreement rather than hard-coding it', 'uses independent seeded +/-0.5 pixel perturbations to test camera-centre conditioning', 'Left Above Rim is constrained by target plus regulation baseline/free-throw lane floor anchors'], 'view_count': len(rows), 'passed_view_count': len(passed_rows), 'pairwise_camera_baselines': pairwise, 'minimum_pairwise_camera_baseline_cm': min_baseline, 'cross_view_backboard_bottom_spread_cm': round(float(board_z_spread), 4), 'gate': {'max_landmark_rmse_px': max_rmse, 'max_rim_center_error_px': max_rim_center, 'max_rim_major_axis_error_px': max_rim_major, 'max_rim_minor_axis_error_px': max_rim_minor, 'max_rim_angle_error_deg': max_rim_angle, 'max_backboard_corner_rmse_px': max_board_rmse, 'max_half_pixel_camera_center_shift_cm': max_sensitivity, 'max_cross_view_backboard_bottom_spread_cm': max_board_z_spread, 'minimum_required_passed_views': required_views, 'min_distinct_camera_baseline_cm': min_baseline_gate, 'pass': bool(passed)}, 'views': rows}
    (args.out / 'nba_geometry_proof_v3.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
