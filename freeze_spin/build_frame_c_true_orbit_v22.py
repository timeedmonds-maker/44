from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from build_portable_moge_pnp_freeview_v12 import (
    W, H, detect_dynamic_and_ball, moge_infer,
    scaled_world_cloud, reference_cloud, angle_between,
)
from build_portable_moge_pnp_freeview_v13 import solve_target_from_reference_reciprocal
from build_portable_moge_true_orbit_v16 import true_orbit_pose, project_point
from build_portable_moge_true_orbit_v18 import raster_cloud_bounded, safe_static_fill


LOCKED_RIGHT_SLASH_TIME = 8.275733
LOCKED_RIGHT_SLASH_FRAME = 248
LOCKED_CHOOSER_OPTION = "C"
LOCKED_EVENT = {"game_id": "0022500301", "event_id": 489, "date": "2025-11-30"}

# Manually verified spatial anchor in the synchronized Left Slash rendering of Frame C.
# This is not a timing selector. Timing is hard-locked above; this pixel only identifies
# the basketball/action pivot within that already-selected physical instant.
DEFAULT_LEFT_SLASH_BALL_PIXEL = (500.0, 104.0)

# Broadcast-family variants are useful source pixels but are not counted as independent
# physical baselines for selecting the orbit direction.
NON_INDEPENDENT_BASELINE_LABELS = {
    "Broadcast", "Other Broadcast", "Mobile Broadcast", "Play by Play"
}


def fov_degrees(K: np.ndarray) -> tuple[float, float]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    return (
        math.degrees(2.0 * math.atan(W / (2.0 * fx))),
        math.degrees(2.0 * math.atan(H / (2.0 * fy))),
    )


def serialise_solve(s: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}


def validate_locked_manifest(d: dict) -> None:
    event = d.get("event", {})
    chosen = d.get("chosen_timing", {})
    if str(event.get("game_id")) != LOCKED_EVENT["game_id"] or int(event.get("event_id", -1)) != 489:
        raise RuntimeError(f"Wrong event in synchronized Frame C manifest: {event}")
    if chosen.get("source_camera") != "Right Slash" or chosen.get("option") != LOCKED_CHOOSER_OPTION:
        raise RuntimeError(f"Frame C authority changed: {chosen}")
    if abs(float(chosen.get("right_slash_local_time", -1)) - LOCKED_RIGHT_SLASH_TIME) > 5e-7:
        raise RuntimeError(f"Right Slash Frame C time changed: {chosen}")
    if int(chosen.get("decoded_frame_index_from_manual_chooser", -1)) != LOCKED_RIGHT_SLASH_FRAME:
        raise RuntimeError(f"Right Slash Frame C index changed: {chosen}")


def choose_reference_ball(ref: dict, expected: tuple[float, float], max_distance: float = 48.0):
    ex, ey = expected
    candidates = []
    for b in ref.get("balls", []):
        cx, cy = float(b["cx"]), float(b["cy"])
        dist = math.hypot(cx - ex, cy - ey)
        candidates.append((dist, b))
    candidates.sort(key=lambda x: x[0])
    if candidates and candidates[0][0] <= max_distance:
        b = candidates[0][1]
        return float(b["cx"]), float(b["cy"]), {
            "method": "Mask R-CNN sports-ball detection nearest manual Frame-C spatial anchor",
            "expected_pixel": [ex, ey],
            "detected": b,
            "distance_px": float(candidates[0][0]),
        }
    return ex, ey, {
        "method": "manual Frame-C spatial anchor fallback; timing unchanged",
        "expected_pixel": [ex, ey],
        "detector_candidates": [b for _, b in candidates[:5]],
    }


def target_from_pixel(ref: dict, u: float, v: float) -> np.ndarray:
    x = int(np.clip(round(u), 0, W - 1))
    y = int(np.clip(round(v), 0, H - 1))
    target = ref["points"][y, x].astype(np.float64)
    if np.all(np.isfinite(target)) and target[2] > 0.25:
        return target
    patch = ref["points"][max(0, y - 5):min(H, y + 6), max(0, x - 5):min(W, x + 6)].reshape(-1, 3)
    good = patch[np.all(np.isfinite(patch), axis=1) & (patch[:, 2] > 0.25)]
    if not len(good):
        raise RuntimeError("No valid MoGe depth around locked Frame-C action pivot")
    return np.median(good, axis=0).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--options-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="Left Slash")
    ap.add_argument("--pivot-u", type=float, default=DEFAULT_LEFT_SLASH_BALL_PIXEL[0])
    ap.add_argument("--pivot-v", type=float, default=DEFAULT_LEFT_SLASH_BALL_PIXEL[1])
    ap.add_argument("--tokens", type=int, default=1600)
    ap.add_argument("--max-degree", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=31)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.options_json.read_text())
    validate_locked_manifest(manifest)

    images = {}
    source_rows = {}
    for row in manifest.get("options", []):
        label = row["camera"]
        p = args.frames_dir / row["file"]
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"Missing synchronized Frame-C image for {label}: {p}")
        if im.shape[:2] != (H, W):
            raise RuntimeError(f"{label} is not native 960x540: {im.shape}")
        images[label] = im
        source_rows[label] = row
    if len(images) != 12:
        raise RuntimeError(f"Expected 12 synchronized Frame-C cameras, found {len(images)}")
    if args.reference not in images:
        raise RuntimeError(f"Reference {args.reference} unavailable")
    rs = source_rows.get("Right Slash", {})
    if abs(float(rs.get("requested_local_time", -1)) - LOCKED_RIGHT_SLASH_TIME) > 5e-7 or int(rs.get("decoded_frame_index", -1)) != 248:
        raise RuntimeError(f"Runtime synchronized Right Slash no longer decodes locked C: {rs}")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    detector = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, progress=True
    ).eval()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()

    views = {}
    for label, image in images.items():
        dyn, balls = detect_dynamic_and_ball(detector, image)
        depth, points, valid, K, Kn = moge_infer(moge, image, args.tokens)
        views[label] = {
            "label": label, "image": image, "dynamic": dyn, "balls": balls,
            "depth": depth, "points": points, "valid": valid, "K": K, "Kn": Kn,
        }
        cv2.imwrite(str(args.out / f"{label.replace(' ', '_')}_dynamic_mask.png"), dyn.astype(np.uint8) * 255)

    ref = views[args.reference]
    bu, bv, pivot_obs = choose_reference_ball(ref, (args.pivot_u, args.pivot_v))
    target = target_from_pixel(ref, bu, bv)

    solves = {}
    for label, view in views.items():
        if label == args.reference:
            continue
        s = solve_target_from_reference_reciprocal(ref, view)
        solves[label] = s
        print("CAMERA_SOLVE", label, {k: v for k, v in s.items() if k not in ("R", "t", "C")}, flush=True)

    passed = [(lab, s) for lab, s in solves.items() if s.get("passed")]
    if not passed:
        raise RuntimeError("No secondary camera passed reciprocal static-PnP calibration")

    baselines = {}
    for lab, s in passed:
        C = s["C"].astype(np.float64)
        baselines[lab] = float(angle_between(-target, C - target))

    independent = [
        (ang, lab, solves[lab])
        for lab, ang in baselines.items()
        if lab not in NON_INDEPENDENT_BASELINE_LABELS and ang >= 3.0
    ]
    if not independent:
        raise RuntimeError("No independent reciprocal-PnP camera proves a >=3 degree physical baseline")

    # Prefer a real solved endpoint that covers the requested 5-degree proof. If none
    # reaches it, use the largest proven baseline and report the limitation rather than
    # extrapolating beyond solved physical support.
    requested_max = float(args.max_degree)
    covering = [x for x in independent if x[0] >= requested_max]
    if covering:
        covering.sort(key=lambda x: x[0])
        baseline_angle, target_label, target_solve = covering[0]
        render_max = requested_max
        coverage_status = "covers_requested_arc"
    else:
        independent.sort(key=lambda x: x[0], reverse=True)
        baseline_angle, target_label, target_solve = independent[0]
        render_max = min(requested_max, float(baseline_angle))
        coverage_status = "limited_to_largest_proven_physical_baseline"
    if render_max < 3.0:
        raise RuntimeError(f"Proven render arc only {render_max:.4f} degrees")

    C_target = target_solve["C"].astype(np.float64)
    clouds = {args.reference: reference_cloud(ref)}
    for lab, s in passed:
        clouds[lab] = scaled_world_cloud(views[lab], s)

    # Secondary cameras are static-only. This is deliberately stricter than v18:
    # they may reveal arena/court support but can never repaint player/ball pixels.
    fill_order = [target_label] + [lab for lab, _ in passed if lab != target_label]

    K0 = ref["K"].astype(np.float64)
    fx0, fy0 = float(K0[0, 0]), float(K0[1, 1])
    fovx0, fovy0 = fov_degrees(K0)
    C0 = np.zeros(3, np.float64)
    radius0 = float(np.linalg.norm(C0 - target))
    pivot0 = project_point(K0, np.eye(3), C0, target)
    pose_rows = []

    def pose(deg: float):
        R, C, _ = true_orbit_pose(target, C_target, deg)
        radius = float(np.linalg.norm(C - target))
        pivot = project_point(K0, R, C, target)
        actual = float(angle_between(C0 - target, C - target))
        row = {
            "requested_degree": float(deg),
            "actual_angular_displacement_deg": actual,
            "camera_center": C.tolist(),
            "radius": radius,
            "radius_drift": float(radius - radius0),
            "focal_px": [fx0, fy0],
            "fov_deg": [fovx0, fovy0],
            "projected_pivot": pivot.tolist(),
            "pivot_drift_px": float(np.linalg.norm(pivot - pivot0)),
        }
        pose_rows.append(row)
        return R, C, row

    def render(deg: float):
        if deg <= 1e-12:
            row = {
                "requested_degree": 0.0,
                "actual_angular_displacement_deg": 0.0,
                "camera_center": C0.tolist(),
                "radius": radius0,
                "radius_drift": 0.0,
                "focal_px": [fx0, fy0],
                "fov_deg": [fovx0, fovy0],
                "projected_pivot": pivot0.tolist(),
                "pivot_drift_px": 0.0,
            }
            pose_rows.append(row)
            return ref["image"].copy(), np.full((H, W), 255, np.uint8), [], row
        R, C, prow = pose(deg)
        base, mask, _ = raster_cloud_bounded(clouds[args.reference], K0, R, C, radius=1)
        reports = []
        for lab in fill_order:
            cand, cmask, cdyn = raster_cloud_bounded(clouds[lab], K0, R, C, radius=1)
            take, stats = safe_static_fill(base, mask, cand, cmask, cdyn)
            base[take] = cand[take]
            mask[take] = 255
            reports.append({
                "label": lab,
                "physical_baseline_deg": baselines[lab],
                "static_safe_fill": int(take.sum()),
                "static_rejected": int(stats["static_rejected"]),
                "dynamic_fill": 0,
                "policy": "secondary dynamic/player/ball pixels forbidden",
            })
        return base, mask, reports, prow

    still_rows = []
    for nominal in [0, 1, 2, 3, 5]:
        actual = min(float(nominal), render_max)
        image, mask, reports, prow = render(actual)
        cv2.imwrite(str(args.out / f"frame_c_true_orbit_{nominal:02d}deg_native.png"), image)
        cv2.imwrite(str(args.out / f"frame_c_unresolved_{nominal:02d}deg.png"), (mask == 0).astype(np.uint8) * 255)
        still_rows.append({
            "nominal_degree": nominal,
            "actual_degree": actual,
            "resolved_fraction": float((mask > 0).mean()),
            "unresolved_pixels": int((mask == 0).sum()),
            "fills": reports,
            "pose": prow,
        })

    # 31-frame proof deliberately goes 0 -> render_max -> 0 so the viewer can judge
    # lateral parallax in both directions without a camera cut or hidden transition.
    for i in range(args.frames):
        phase = i / max(args.frames - 1, 1)
        degree = render_max * math.sin(math.pi * phase)
        frame, _, _, _ = render(degree)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)

    max_radius_drift = max(abs(float(r["radius_drift"])) for r in pose_rows)
    max_pivot_drift = max(float(r["pivot_drift_px"]) for r in pose_rows)
    focal_set = {(round(float(r["focal_px"][0]), 9), round(float(r["focal_px"][1]), 9)) for r in pose_rows}
    fov_set = {(round(float(r["fov_deg"][0]), 9), round(float(r["fov_deg"][1]), 9)) for r in pose_rows}
    max_actual = max(float(r["actual_angular_displacement_deg"]) for r in pose_rows)

    if max_radius_drift > 1e-6:
        raise RuntimeError(f"HARD NO-ZOOM FAIL: camera/action radius drift {max_radius_drift}")
    if max_pivot_drift > 0.05:
        raise RuntimeError(f"HARD NO-ZOOM FAIL: action pivot drift {max_pivot_drift}px")
    if len(focal_set) != 1 or len(fov_set) != 1:
        raise RuntimeError("HARD NO-ZOOM FAIL: focal length or FOV changed")
    if max_actual + 1e-6 < 3.0:
        raise RuntimeError(f"HARD NO-ZOOM FAIL: actual spatial displacement only {max_actual} deg")

    qa = {
        "prototype": "frame_c_reciprocal_static_true_orbit_v22",
        "event": LOCKED_EVENT,
        "freeze_timing_lock": {
            "authority_camera": "Right Slash",
            "chooser_option": "C",
            "right_slash_local_time": LOCKED_RIGHT_SLASH_TIME,
            "decoded_right_slash_frame_index": LOCKED_RIGHT_SLASH_FRAME,
            "policy": "immutable user-selected physical instant; no apex/rim/audio detector may change it",
        },
        "synchronized_camera_set": manifest.get("options", []),
        "source_resolution": [W, H],
        "render_resolution": [W, H],
        "resolution_policy": "native 960x540 only",
        "reference_camera": args.reference,
        "reference_spatial_pivot": pivot_obs,
        "pivot_pixel_used": [bu, bv],
        "pivot_reference_world": target.tolist(),
        "camera_method": "MoGe-2 depth prior + dynamic-masked static SIFT + forward/reverse solvePnPRansac + SE(3) reciprocal closure",
        "camera_coordinate_scale": "MoGe-relative reference coordinate system; not claimed as court-feet metric calibration",
        "camera_solves": {lab: serialise_solve(s) for lab, s in solves.items()},
        "passed_camera_baselines_deg": baselines,
        "independent_baseline_policy": "Broadcast/Other Broadcast/Mobile Broadcast/Play by Play excluded from independent-baseline selection",
        "target_direction_camera": target_label,
        "target_real_baseline_angle_deg": float(baseline_angle),
        "coverage_status": coverage_status,
        "render_max_degree": float(render_max),
        "orbit_method": "rigid constant-radius camera-centre rotation around fixed Frame-C 3D action pivot; fixed reference K/FOV; no radial interpolation",
        "reference_camera_center": C0.tolist(),
        "reference_radius": radius0,
        "reference_focal_px": [fx0, fy0],
        "reference_fov_deg": [fovx0, fovy0],
        "reference_projected_pivot": pivot0.tolist(),
        "max_radius_drift": max_radius_drift,
        "max_pivot_drift_px": max_pivot_drift,
        "focal_constant": len(focal_set) == 1,
        "fov_constant": len(fov_set) == 1,
        "actual_max_angular_displacement_deg": max_actual,
        "edge_splat_policy": "explicit bounded shifts only; no np.roll/wraparound",
        "secondary_fill_policy": "STATIC ONLY + proximity/Lab consistency gate; all secondary dynamic pixels forbidden",
        "pose_qa": pose_rows,
        "stills": still_rows,
        "generation_policy": "no generated appearance, diffusion, generative fill, optical-flow morph, focal zoom, crop animation or radial dolly; output colours derive from official NBA source pixels",
    }
    (args.out / "frame_c_true_orbit_qa_v22.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({
        "reference": args.reference,
        "target_direction_camera": target_label,
        "target_real_baseline_deg": baseline_angle,
        "render_max_degree": render_max,
        "passed_cameras": list(baselines),
        "max_radius_drift": max_radius_drift,
        "max_pivot_drift_px": max_pivot_drift,
        "actual_max_angular_displacement_deg": max_actual,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
