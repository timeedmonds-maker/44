from __future__ import annotations

"""v67 diagnostic: metric-anchored monocular-depth static orbit.

This intentionally does *not* estimate camera pose or world scale from MoGe.
The accepted Left Above Rim v41/v42 camera and v35 NBA floor homography are the
metric truth. MoGe-2 supplies only a per-pixel depth-shape field. The learned
depth is aligned to the accepted metric camera from held-out regulation-floor
samples, then original source pixels are reprojected through a rigid constant-
radius orbit around the regulation rim centre.

No secondary-camera dynamic pixels, generative fill, temporal synthesis, camera
promotion, or replay permission is introduced here. The proof is diagnostic
and exposes disocclusion holes explicitly.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from moge.model.v2 import MoGeModel
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from build_portable_moge_pnp_freeview_v12 import detect_dynamic_and_ball, moge_infer

W, H = 960, 540
FT = 30.48
IN = 2.54
RIM = np.asarray([15.0 * IN, 0.0, 10.0 * FT], dtype=np.float64)
ANGLES = (0.0, 3.0, 6.0, 9.0, 12.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def K_matrix(f: float, pp) -> np.ndarray:
    return np.asarray([[f, 0.0, pp[0]], [0.0, f, pp[1]], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_h(Hm: np.ndarray, xy: np.ndarray) -> np.ndarray:
    q = (Hm @ np.column_stack([xy, np.ones(len(xy))]).T).T
    return q[:, :2] / q[:, 2:3]


def floor_grid(Hm: np.ndarray):
    rows = []
    for ix, x in enumerate(np.linspace(-4.0 * FT, 30.0 * FT, 24)):
        for iy, y in enumerate(np.linspace(-25.0 * FT, 25.0 * FT, 31)):
            rows.append((ix, iy, x, y))
    xy = np.asarray([[r[2], r[3]] for r in rows], dtype=np.float64)
    uv = project_h(Hm, xy)
    keep = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 12.0)
        & (uv[:, 0] <= W - 13.0)
        & (uv[:, 1] >= 12.0)
        & (uv[:, 1] <= H - 13.0)
    )
    P = np.column_stack([xy, np.zeros(len(xy))])[keep]
    U = uv[keep]
    meta = [rows[i] for i in np.where(keep)[0]]
    return P, U, meta


def decompose_rotation_seed(Hm: np.ndarray, K: np.ndarray) -> np.ndarray:
    A = np.linalg.inv(K) @ Hm
    a1, a2 = A[:, 0], A[:, 1]
    s = 2.0 / max(np.linalg.norm(a1) + np.linalg.norm(a2), 1e-12)
    r1 = s * a1
    r2 = s * a2
    r3 = np.cross(r1, r2)
    R0 = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(R0)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    rv, _ = cv2.Rodrigues(R)
    return rv.reshape(3)


def project_camera(C: np.ndarray, K: np.ndarray, rv: np.ndarray, P: np.ndarray):
    R, _ = cv2.Rodrigues(rv.reshape(3, 1))
    Xc = (R @ (P - C).T).T
    uvh = (K @ Xc.T).T
    uv = uvh[:, :2] / uvh[:, 2:3]
    return uv, Xc, R


def recover_accepted_rotation(C: np.ndarray, K: np.ndarray, Hm: np.ndarray):
    P, U, _ = floor_grid(Hm)
    seed = decompose_rotation_seed(Hm, K)
    candidates = [seed]
    candidates.extend([
        seed + np.asarray([math.pi, 0.0, 0.0]),
        seed + np.asarray([0.0, math.pi, 0.0]),
        seed + np.asarray([0.0, 0.0, math.pi]),
    ])
    roots = []
    for start in candidates:
        def fun(rv):
            try:
                pred, Xc, _ = project_camera(C, K, rv, P)
                depth_pen = np.minimum(Xc[:, 2] - 20.0, 0.0) * 5.0
                return np.r_[(pred - U).ravel(), depth_pen]
            except Exception:
                return np.full(len(P) * 3, 1e6, dtype=np.float64)
        fit = least_squares(fun, start, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=12000)
        pred, Xc, R = project_camera(C, K, fit.x, P)
        e = np.linalg.norm(pred - U, axis=1)
        roots.append({
            "rv": fit.x,
            "R": R,
            "rms_px": float(np.sqrt(np.mean(e ** 2))),
            "p95_px": float(np.percentile(e, 95)),
            "max_px": float(np.max(e)),
            "min_depth_cm": float(np.min(Xc[:, 2])),
            "cost": float(fit.cost),
        })
    roots.sort(key=lambda x: (x["p95_px"], x["cost"]))
    return roots[0], roots


def floor_depth_samples(Hm, C, K, rv, moge_depth, moge_valid, dynamic):
    P, U, meta = floor_grid(Hm)
    pred, Xc, _ = project_camera(C, K, rv, P)
    reproj = np.linalg.norm(pred - U, axis=1)
    x = np.rint(U[:, 0]).astype(int)
    y = np.rint(U[:, 1]).astype(int)
    good = (
        (x >= 0) & (x < W) & (y >= 0) & (y < H)
        & moge_valid[y, x]
        & np.isfinite(moge_depth[y, x])
        & (moge_depth[y, x] > 0.05)
        & (~dynamic[y, x])
        & (Xc[:, 2] > 20.0)
        & (reproj <= 1.0)
    )
    P, U, Xc, x, y, reproj = P[good], U[good], Xc[good], x[good], y[good], reproj[good]
    meta = [meta[i] for i in np.where(good)[0]]
    d = moge_depth[y, x].astype(np.float64)
    z = Xc[:, 2].astype(np.float64)
    hold = np.asarray([((int(m[0]) + 2 * int(m[1])) % 5) == 0 for m in meta], dtype=bool)
    if int((~hold).sum()) < 40 or int(hold.sum()) < 10:
        raise RuntimeError(f"insufficient clean floor depth anchors train={int((~hold).sum())} held={int(hold.sum())}")
    return {
        "P": P, "U": U, "moge": d, "metric_z": z, "hold": hold,
        "reproj": reproj, "meta": meta,
    }


def fit_depth_models(d: np.ndarray, z: np.ndarray, train: np.ndarray):
    rows = []
    dt, zt = d[train], z[train]

    scale = float(np.median(zt / np.maximum(dt, 1e-8)))
    rows.append({"name": "scale", "params": [scale]})

    def affine_res(p): return p[0] * dt + p[1] - zt
    af = least_squares(affine_res, [scale, 0.0], loss="soft_l1", f_scale=20.0, max_nfev=5000)
    rows.append({"name": "affine", "params": af.x.tolist()})

    def apply(row, values):
        values = np.asarray(values, dtype=np.float64)
        if row["name"] == "scale":
            return row["params"][0] * values
        return row["params"][0] * values + row["params"][1]

    train_ids = np.where(train)[0]
    for row in rows:
        pred = apply(row, d)
        trerr = np.abs(pred[train] - z[train])
        row["train_median_cm"] = float(np.median(trerr))
        row["train_p95_cm"] = float(np.percentile(trerr, 95))
        folds = []
        for fold in range(4):
            val_ids = train_ids[np.arange(len(train_ids)) % 4 == fold]
            fit_ids = np.setdiff1d(train_ids, val_ids)
            mask = np.zeros(len(d), bool); mask[fit_ids] = True
            if row["name"] == "scale":
                p = [float(np.median(z[mask] / np.maximum(d[mask], 1e-8)))]
            else:
                rr = least_squares(lambda q: q[0]*d[mask]+q[1]-z[mask], row["params"], loss="soft_l1", f_scale=20.0, max_nfev=3000)
                p = rr.x.tolist()
            temp = {"name": row["name"], "params": p}
            e = np.abs(apply(temp, d[val_ids]) - z[val_ids])
            folds.extend(e.tolist())
        row["cv_median_cm"] = float(np.median(folds))
        row["cv_p95_cm"] = float(np.percentile(folds, 95))
        row["positive_fraction_all"] = float(np.mean(np.isfinite(pred) & (pred > 20.0)))
    rows.sort(key=lambda r: (r["cv_p95_cm"], r["cv_median_cm"], len(r["params"])))
    return rows[0], rows, apply


def metric_cloud(image, depth_metric, valid, K, R, C):
    yy, xx = np.indices((H, W))
    ok = valid & np.isfinite(depth_metric) & (depth_metric > 20.0) & (depth_metric < 12000.0)
    ys, xs = np.where(ok)
    z = depth_metric[ys, xs].astype(np.float64)
    xn = (xs.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (ys.astype(np.float64) - K[1, 2]) / K[1, 1]
    Xc = np.column_stack([xn * z, yn * z, z])
    Xw = (R.T @ Xc.T).T + C
    return Xw.astype(np.float32), image[ys, xs].copy(), np.column_stack([xs, ys]).astype(np.int32)


def orbit_pose(C0: np.ndarray, R0: np.ndarray, pivot: np.ndarray, azimuth_deg: float):
    t = math.radians(float(azimuth_deg))
    Q = np.asarray([
        [math.cos(t), -math.sin(t), 0.0],
        [math.sin(t),  math.cos(t), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    C = pivot + Q @ (C0 - pivot)
    R = R0 @ Q.T
    return R, C


def project_points(K, R, C, X):
    Xc = (R @ (X.astype(np.float64) - C).T).T
    q = (K @ Xc.T).T
    uv = q[:, :2] / q[:, 2:3]
    return uv, Xc[:, 2]


def raster_source_cloud(cloud, K, R, C, radius=1):
    X, colours, src_uv = cloud
    uv, z = project_points(K, R, C, X)
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    ok = np.isfinite(uv).all(axis=1) & (z > 20.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ids = np.where(ok)[0]
    image = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    provenance = np.full((H, W, 2), -1, np.int32)
    zbuf = np.full(H * W, np.inf, np.float32)
    if len(ids):
        pix = v[ids] * W + u[ids]
        np.minimum.at(zbuf, pix, z[ids].astype(np.float32))
        winners = ids[z[ids] <= zbuf[pix] + 1e-4]
        image[v[winners], u[winners]] = colours[winners]
        mask[v[winners], u[winners]] = 255
        provenance[v[winners], u[winners]] = src_uv[winners]
    for _ in range(int(radius)):
        base_img, base_mask, base_prov = image.copy(), mask.copy(), provenance.copy()
        holes = mask == 0
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            simg = np.roll(np.roll(base_img, dy, 0), dx, 1)
            sm = np.roll(np.roll(base_mask, dy, 0), dx, 1)
            sp = np.roll(np.roll(base_prov, dy, 0), dx, 1)
            take = holes & (mask == 0) & (sm > 0)
            image[take] = simg[take]; mask[take] = 255; provenance[take] = sp[take]
    return image, mask, provenance


def image_metrics(rendered, mask, pivot_px):
    covered = mask > 0
    roi = np.zeros((H, W), bool)
    cx, cy = [int(round(v)) for v in pivot_px]
    x0, x1 = max(0, cx - 220), min(W, cx + 221)
    y0, y1 = max(0, cy - 190), min(H, cy + 191)
    roi[y0:y1, x0:x1] = True
    return {
        "resolved_fraction_full": float(np.mean(covered)),
        "resolved_fraction_action_roi": float(np.mean(covered[roi])) if np.any(roi) else 0.0,
        "unresolved_pixels_full": int((~covered).sum()),
        "unresolved_pixels_action_roi": int((roi & ~covered).sum()),
        "action_roi_xyxy": [x0, y0, x1, y1],
    }


def json_safe(x):
    if isinstance(x, dict): return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--floor-proof", type=Path, required=True)
    ap.add_argument("--camera-registry", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=1600)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("v67 requires immutable native 960x540 Left Above Rim Frame C")
    floor = json.loads(args.floor_proof.read_text())
    reg = json.loads(args.camera_registry.read_text())
    if floor.get("status") != "PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35":
        raise RuntimeError("v35 floor proof is not accepted")
    cam = reg.get("accepted_cameras", {}).get("Left Above Rim", {})
    if not cam.get("permissions", {}).get("physical_camera_center_allowed") or not cam.get("permissions", {}).get("metric_event_camera_489_allowed"):
        raise RuntimeError("Left Above Rim v41/v42 metric camera is not accepted in registry")
    event = cam["event_489"]
    if event.get("freeze_frame") != args.frame.name:
        raise RuntimeError(f"wrong accepted Frame C: {args.frame.name} != {event.get('freeze_frame')}")
    C = np.asarray(cam["physical_camera_center_prior_cm"], dtype=np.float64)
    K = K_matrix(float(event["focal_px"]), event["principal_point_px"])
    Hm = np.asarray(floor["floor_homography_world_to_image"], dtype=np.float64)

    rot, roots = recover_accepted_rotation(C, K, Hm)
    if rot["p95_px"] > 0.55 or rot["min_depth_cm"] <= 20.0:
        raise RuntimeError(f"could not reproduce accepted v42 orientation from v35+registry: {rot}")
    rv = rot["rv"]; R = rot["R"]

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    detector = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, progress=True).eval()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    dynamic, balls = detect_dynamic_and_ball(detector, image)
    depth, _, valid, K_moge, _ = moge_infer(moge, image, args.tokens)
    cv2.imwrite(str(args.out / "left_above_rim_dynamic_mask_v67.png"), dynamic.astype(np.uint8) * 255)

    samples = floor_depth_samples(Hm, C, K, rv, depth, valid, dynamic)
    train = ~samples["hold"]
    best_model, model_rows, apply_model = fit_depth_models(samples["moge"], samples["metric_z"], train)
    pred = apply_model(best_model, samples["moge"])
    held_err = np.abs(pred[samples["hold"]] - samples["metric_z"][samples["hold"]])
    train_err = np.abs(pred[train] - samples["metric_z"][train])

    metric_depth = apply_model(best_model, depth.astype(np.float64))
    metric_valid = valid & np.isfinite(metric_depth) & (metric_depth > 20.0) & (metric_depth < 12000.0)
    cloud = metric_cloud(image, metric_depth, metric_valid, K, R, C)

    pivot_uv, pivot_depth = project_points(K, R, C, RIM[None, :])
    if pivot_depth[0] <= 20.0:
        raise RuntimeError("regulation rim pivot is behind accepted camera")
    pivot_px0 = pivot_uv[0]

    stills = []
    zero_roundtrip = None
    for deg in ANGLES:
        Rn, Cn = orbit_pose(C, R, RIM, deg)
        pivot_new, _ = project_points(K, Rn, Cn, RIM[None, :])
        rendered, mask, provenance = raster_source_cloud(cloud, K, Rn, Cn, radius=1)
        cv2.imwrite(str(args.out / f"metric_anchor_depth_orbit_{int(deg):02d}deg_native.png"), rendered)
        cv2.imwrite(str(args.out / f"metric_anchor_depth_orbit_{int(deg):02d}deg_unresolved.png"), (mask == 0).astype(np.uint8) * 255)
        m = image_metrics(rendered, mask, pivot_new[0])
        radius0 = float(np.linalg.norm(C - RIM)); radiusn = float(np.linalg.norm(Cn - RIM))
        ray0 = C - RIM; rayn = Cn - RIM
        actual_angle = math.degrees(math.acos(float(np.clip(np.dot(ray0, rayn)/(np.linalg.norm(ray0)*np.linalg.norm(rayn)), -1.0, 1.0))))
        if deg == 0.0:
            same = np.all(rendered == image, axis=2) & (mask > 0)
            zero_roundtrip = {
                "covered_fraction": float(np.mean(mask > 0)),
                "exact_source_colour_fraction_of_covered": float(np.mean(same[mask > 0])) if np.any(mask > 0) else 0.0,
            }
        stills.append({
            "requested_azimuth_deg": deg,
            "actual_3d_angular_displacement_deg": actual_angle,
            "camera_center_cm": Cn.tolist(),
            "radius_cm": radiusn,
            "radius_drift_cm": radiusn - radius0,
            "projected_pivot_px": pivot_new[0].tolist(),
            "pivot_drift_px": float(np.linalg.norm(pivot_new[0] - pivot_px0)),
            **m,
        })

    held_med = float(np.median(held_err)); held_p95 = float(np.percentile(held_err, 95))
    gates = {
        "accepted_metric_camera_reproduced_floor_p95_at_most_0_55px": rot["p95_px"] <= 0.55,
        "clean_floor_depth_anchor_count_at_least_50": len(samples["moge"]) >= 50,
        "heldout_floor_depth_median_at_most_35cm": held_med <= 35.0,
        "heldout_floor_depth_p95_at_most_90cm": held_p95 <= 90.0,
        "zero_degree_covered_fraction_at_least_0_97": zero_roundtrip is not None and zero_roundtrip["covered_fraction"] >= 0.97,
        "zero_degree_source_colour_identity_at_least_0_995": zero_roundtrip is not None and zero_roundtrip["exact_source_colour_fraction_of_covered"] >= 0.995,
        "orbit_radius_constant": max(abs(s["radius_drift_cm"]) for s in stills) <= 1e-6,
        "rim_pivot_constant": max(s["pivot_drift_px"] for s in stills) <= 0.05,
    }
    passed = bool(all(gates.values()))
    safe_angles = [s["requested_azimuth_deg"] for s in stills if s["resolved_fraction_action_roi"] >= 0.92]
    safe_max = max(safe_angles) if safe_angles else 0.0
    report = {
        "schema_version": 1,
        "status": "PASS_DIAGNOSTIC_METRIC_ANCHORED_DEPTH_ORBIT_V67" if passed else "FAIL_DIAGNOSTIC_METRIC_ANCHORED_DEPTH_ORBIT_V67",
        "game_id": "0022500301", "event_id": 489, "camera": "Left Above Rim",
        "frame": args.frame.name, "frame_sha256": sha256(args.frame),
        "method": "accepted v41/v42 NBA metric camera + accepted v35 floor + MoGe-2 depth shape aligned to metric floor; original source-pixel z-buffer orbit only",
        "strategic_change": "learned model may estimate depth shape but may not establish camera pose, world scale, or camera promotion",
        "accepted_metric_camera": {
            "center_cm": C.tolist(), "focal_px": float(K[0,0]), "principal_point_px": [float(K[0,2]), float(K[1,2])],
            "recovered_rvec": rv.tolist(), "floor_reproduction": {k:v for k,v in rot.items() if k not in ("rv","R")},
            "root_diagnostics": [{k:v for k,v in r.items() if k not in ("rv","R")} for r in roots],
        },
        "moge": {"model": "Ruicheng/moge-2-vits-normal", "tokens": args.tokens, "reported_intrinsics_px": K_moge.tolist(), "valid_fraction": float(np.mean(valid))},
        "depth_metric_alignment": {
            "clean_floor_anchor_count": int(len(samples["moge"])), "train_count": int(train.sum()), "heldout_count": int(samples["hold"].sum()),
            "selected_model": best_model, "candidate_models": model_rows,
            "train_median_abs_cm": float(np.median(train_err)), "train_p95_abs_cm": float(np.percentile(train_err,95)),
            "heldout_median_abs_cm": held_med, "heldout_p95_abs_cm": held_p95, "heldout_max_abs_cm": float(np.max(held_err)),
        },
        "pivot": {"world_cm": RIM.tolist(), "source_px": pivot_px0.tolist(), "meaning": "regulation rim centre; fixed static pivot, not player/ball fit"},
        "zero_roundtrip": zero_roundtrip,
        "stills": stills,
        "coverage_assessment": {"action_roi_min_resolved_fraction_for_safe_angle": 0.92, "safe_max_azimuth_deg": safe_max, "note": "coverage is reported honestly and does not modify geometry gates"},
        "gates": gates,
        "permissions": {
            "physical_camera_promotion_allowed": False,
            "metric_camera_promotion_allowed": False,
            "product_static_novel_view_allowed": False,
            "replay_render_allowed": False,
        },
        "appearance_policy": "original Left Above Rim source pixels only; one-pixel nearest-source splat; no generated fill, no secondary dynamic pixels",
    }
    report = json_safe(report)
    (args.out / "metric_anchor_depth_orbit_v67.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "depth_model": best_model["name"], "heldout_depth_median_cm": held_med,
        "heldout_depth_p95_cm": held_p95, "safe_max_azimuth_deg": safe_max,
        "coverage": {str(s["requested_azimuth_deg"]): s["resolved_fraction_action_roi"] for s in stills}, "gates": gates,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
