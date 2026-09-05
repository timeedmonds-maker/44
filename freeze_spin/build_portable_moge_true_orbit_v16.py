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
    W, H, safe, label_from_frame, detect_dynamic_and_ball, moge_infer,
    solve_target_from_reference, scaled_world_cloud, reference_cloud,
    raster_cloud, angle_between, safe_fill_gate,
)


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-10:
        return np.eye(3, dtype=np.float64)
    rvec = axis / n * float(angle_rad)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R.astype(np.float64)


def true_orbit_pose(target: np.ndarray, target_camera_center: np.ndarray, degree: float):
    """Rigidly rotate the reference camera around target at constant radius.

    Reference world == reference camera coordinates, so C0=[0,0,0], R0=I.
    Q rotates the complete camera rig about the target.  The new world-to-camera
    rotation is Q.T.  This preserves target camera coordinates exactly while
    changing viewpoint, so there is no radial dolly/zoom component.
    """
    target = np.asarray(target, dtype=np.float64)
    C0 = np.zeros(3, dtype=np.float64)
    v0 = C0 - target
    v1 = np.asarray(target_camera_center, dtype=np.float64) - target
    axis = np.cross(v0, v1)
    baseline = angle_between(v0, v1)
    if baseline < 1e-6:
        raise RuntimeError("Target camera has negligible angular baseline")
    signed_deg = float(np.clip(degree, 0.0, baseline))
    Q = axis_angle_matrix(axis, math.radians(signed_deg))
    C = target + Q @ v0
    R = Q.T
    return R, C, baseline


def project_point(K: np.ndarray, R: np.ndarray, C: np.ndarray, X: np.ndarray) -> np.ndarray:
    q = K @ (R @ (np.asarray(X, dtype=np.float64) - np.asarray(C, dtype=np.float64)))
    if q[2] <= 1e-9:
        return np.array([np.nan, np.nan])
    return q[:2] / q[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="In Arena")
    ap.add_argument("--tokens", type=int, default=1400)
    ap.add_argument("--max-degree", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=31)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(list(args.locked_dir.glob("*_apex.png")) + list(args.locked_dir.glob("*_predunk.png")))
    images = {}
    for p in paths:
        label = label_from_frame(p).replace(" apex", "")
        label = p.stem.replace("_apex", "").replace("_predunk", "").replace("_", " ")
        im = cv2.imread(str(p))
        if im is not None:
            images[label] = im
    if args.reference not in images:
        raise RuntimeError(f"Reference {args.reference} unavailable: {list(images)}")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    detector = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, progress=True).eval()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()

    views = {}
    for label, image in images.items():
        if image.shape[1] != W or image.shape[0] != H:
            raise RuntimeError(f"{label} not native {W}x{H}: {image.shape}")
        dyn, balls = detect_dynamic_and_ball(detector, image)
        depth, points, valid, K, Kn = moge_infer(moge, image, args.tokens)
        views[label] = {"label": label, "image": image, "dynamic": dyn, "balls": balls,
                        "depth": depth, "points": points, "valid": valid, "K": K, "Kn": Kn}

    ref = views[args.reference]
    solves = {}
    for label, v in views.items():
        if label == args.reference:
            continue
        s = solve_target_from_reference(ref, v)
        solves[label] = s
        print(label, {k: v for k, v in s.items() if k not in ("R", "t", "C")}, flush=True)
    passed = [(lab, s) for lab, s in solves.items() if s.get("passed")]
    if not passed:
        raise RuntimeError("No secondary camera passed portable MoGe+SIFT+PnP calibration")

    if not ref["balls"]:
        raise RuntimeError("No basketball detected in reference freeze frame")
    b = ref["balls"][0]
    bx = int(np.clip(round(b["cx"]), 0, W - 1)); by = int(np.clip(round(b["cy"]), 0, H - 1))
    target = ref["points"][by, bx].astype(np.float64)
    if not np.all(np.isfinite(target)) or target[2] <= 0.25:
        ys = slice(max(0, by - 3), min(H, by + 4)); xs = slice(max(0, bx - 3), min(W, bx + 4))
        pts = ref["points"][ys, xs].reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0.25)]
        if not len(pts):
            raise RuntimeError("No valid reference MoGe depth at basketball")
        target = np.median(pts, axis=0)

    options = []
    for lab, s in passed:
        C = s["C"].astype(np.float64)
        ang = angle_between(-target, C - target)
        if ang >= 4.0:
            options.append((ang, lab, s))
    if not options:
        options = [(angle_between(-target, s["C"].astype(np.float64) - target), lab, s) for lab, s in passed]
    options.sort(key=lambda x: x[0])
    baseline_angle, target_label, target_solve = options[0]
    render_max = min(float(args.max_degree), float(baseline_angle))
    C_target = target_solve["C"].astype(np.float64)

    clouds = {args.reference: reference_cloud(ref)}
    for lab, s in passed:
        clouds[lab] = scaled_world_cloud(views[lab], s)
    fill_order = [target_label] + [lab for lab, _ in passed if lab != target_label]

    C0 = np.zeros(3, dtype=np.float64)
    radius0 = float(np.linalg.norm(C0 - target))
    pivot0 = project_point(ref["K"], np.eye(3), C0, target)
    focal0 = [float(ref["K"][0, 0]), float(ref["K"][1, 1])]

    pose_qa = []
    def pose_for_degree(deg: float):
        R, C, baseline = true_orbit_pose(target, C_target, deg)
        radius = float(np.linalg.norm(C - target))
        pivot = project_point(ref["K"], R, C, target)
        pose_qa.append({
            "degree": float(deg),
            "camera_center": C.tolist(),
            "radius": radius,
            "radius_drift": float(radius - radius0),
            "pivot_pixel": pivot.tolist(),
            "pivot_drift_px": float(np.linalg.norm(pivot - pivot0)),
            "fx": focal0[0], "fy": focal0[1],
        })
        return R, C

    def render(deg: float):
        if deg <= 1e-9:
            return ref["image"].copy(), np.full((H, W), 255, np.uint8), {"secondary": []}
        R, C = pose_for_degree(deg)
        base, mask, dyn = raster_cloud(clouds[args.reference], ref["K"], R, C, radius=1)
        report = []
        for lab in fill_order:
            im, cm, cd = raster_cloud(clouds[lab], ref["K"], R, C, radius=1)
            take, stats = safe_fill_gate(base, mask, im, cm, cd)
            base[take] = im[take]; mask[take] = 255
            stats["label"] = lab; stats["accepted_total"] = int(take.sum()); report.append(stats)
        return base, mask, {"secondary": report}

    still = []
    for deg in [0, 1, 2, 3, 5]:
        d = min(float(deg), render_max)
        frame, mask, rr = render(d)
        cv2.imwrite(str(args.out / f"true_orbit_{deg:02d}deg_native.png"), frame)
        cv2.imwrite(str(args.out / f"true_orbit_unresolved_{deg:02d}deg.png"), (mask == 0).astype(np.uint8) * 255)
        still.append({"degree": deg, "actual_degree": d,
                      "resolved_fraction": float((mask > 0).mean()),
                      "unresolved_pixels": int((mask == 0).sum()), "fills": rr["secondary"]})

    for i in range(args.frames):
        phase = i / max(args.frames - 1, 1)
        deg = render_max * math.sin(math.pi * phase)
        frame, _, _ = render(deg)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)

    max_radius_drift = max(abs(r["radius_drift"]) for r in pose_qa) if pose_qa else 0.0
    max_pivot_drift = max(r["pivot_drift_px"] for r in pose_qa) if pose_qa else 0.0
    if max_radius_drift > 1e-6:
        raise RuntimeError(f"Orbit radius drifted: {max_radius_drift}")
    if max_pivot_drift > 0.05:
        raise RuntimeError(f"Action pivot drifted {max_pivot_drift:.4f}px; not a rigid orbit")

    serial = {}
    for lab, s in solves.items():
        serial[lab] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    qa = {
        "prototype": "portable_moge_true_orbit_v16",
        "event": {"game_id": "0022500301", "event_id": 489,
                  "description": "Steven Adams dunk vs Utah immediately after Adams block"},
        "source_resolution": [W, H], "render_resolution": [W, H], "resolution_policy": "native only",
        "reference": args.reference, "detected_ball_ref": b, "target_world_m": target.tolist(),
        "camera_method": "MoGe-2 reference 3D + person-masked static SIFT + PnP",
        "orbit_method": "constant-radius rigid rotation of reference camera about 3D action pivot; R=Q^T; reference K/FOV held fixed",
        "forbidden_motion_removed": "no radius interpolation, no camera-distance blend, no focal interpolation, no Ken-Burns/push-in",
        "camera_solves": serial, "passed_secondary_count": len(passed),
        "target_direction_camera": target_label, "real_baseline_angle_deg": baseline_angle,
        "render_max_degree": render_max, "reference_radius": radius0,
        "reference_focal_px": focal0, "max_radius_drift": max_radius_drift,
        "max_pivot_drift_px": max_pivot_drift, "pose_qa": pose_qa,
        "fill_policy": "reference real pixels first; secondary dynamic pixels may fill disocclusions; static fill requires narrow-hole proximity plus local Lab agreement",
        "generation_policy": "no generated appearance, no diffusion, no optical-flow morph; every output colour is an NBA source pixel",
        "stills": still,
        "success_gate": "constant radius to 1e-6; fixed intrinsics; action pivot projection drift <=0.05px; visually obvious lateral parallax at 3-5 degrees; no zoom-only interpretation",
    }
    (args.out / "portable_moge_true_orbit_qa_v16.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"target_direction_camera": target_label, "baseline_angle_deg": baseline_angle,
                      "render_max_degree": render_max, "max_radius_drift": max_radius_drift,
                      "max_pivot_drift_px": max_pivot_drift, "stills": still}, indent=2), flush=True)


if __name__ == "__main__":
    main()
