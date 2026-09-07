from __future__ import annotations

"""Diagnostic three-camera free-view preview for Adams-Jazz Frame C.

This is an explicit diagnostic exception requested by the user while Right Slash
(camera #4) remains unsolved. It consumes only the three cameras already accepted
in adams_jazz_game_camera_registry_v5.json:
  * Left Above Rim
  * Right Above Rim
  * Broadcast

No camera is promoted here. No registry permissions are changed. Appearance is
source-grounded: original NBA pixels are reprojected from metric-aligned per-view
MoGe depth. Secondary views fill only holes left by the Left Above Rim anchor;
there is no crossfade, optical-flow morph, or generative fill.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from moge.model.v2 import MoGeModel

from freeze_spin.build_metric_anchor_depth_orbit_v67 import (
    H,
    W,
    RIM,
    K_matrix,
    orbit_pose,
    project_points,
    raster_source_cloud,
    recover_accepted_rotation,
)
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer
from freeze_spin import prove_broadcast_shared_center_v90 as v90
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44
from freeze_spin import solve_broadcast_direct_target_lines_v87 as v87
from freeze_spin import diagnose_broadcast_homography_conditioning_v85 as v85

ACCEPTED = ("Left Above Rim", "Right Above Rim", "Broadcast")
TARGET_KEYS = ("target_top", "target_left", "target_right")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_native(path: Path) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None or im.shape[:2] != (H, W):
        raise RuntimeError(f"missing/non-native source frame: {path}")
    return im


def camera_row(label: str, C, K, R, source_frame: str, evidence: str):
    return {
        "label": label,
        "center_cm": np.asarray(C, float).tolist(),
        "K": np.asarray(K, float).tolist(),
        "R_world_to_camera": np.asarray(R, float).tolist(),
        "source_frame": source_frame,
        "evidence": evidence,
    }


def recover_broadcast_rotation(C, f, pp, floor_spec, target_spec):
    # Reproduce v90 geometry conventions, but keep the accepted v90 optical
    # centre, focal and principal point fixed. Only Frame-C orientation is fit.
    v85.patch_line_aware_geometry()
    train, _ = v44.split_groups(floor_spec["observations_px"], floor_spec["held_out_indices"])
    target_obs = {
        k: np.asarray(target_spec["observed_line_samples_px"][k], dtype=np.float64)
        for k in TARGET_KEYS
    }
    seed_state = v90.find_frame_c_seed(train, target_obs, target_spec)
    rv0 = np.asarray(seed_state[:3], dtype=np.float64)
    logf = math.log(float(f))
    cx, cy = [float(x) for x in pp]

    def residual(rv):
        p = v90.state_from_center(np.asarray(C, float), rv, logf, cx, cy)
        return v90.state_residual(p, train, target_obs)

    fit = least_squares(
        residual, rv0, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=12000
    )
    rv = np.asarray(fit.x, dtype=np.float64)
    R = cv2.Rodrigues(rv.reshape(3, 1))[0]
    abs_res = np.abs(residual(rv))
    qa = {
        "orientation_fit_cost": float(fit.cost),
        "residual_median_abs_px": float(np.median(abs_res)),
        "residual_p95_abs_px": float(np.percentile(abs_res, 95)),
        "rvec": rv.tolist(),
    }
    return R, qa


def build_cameras(registry, lar_floor, rar_report, broadcast_report, broadcast_floor, broadcast_target, frames_dir):
    names = tuple(registry.get("accepted_camera_names", []))
    if registry.get("accepted_camera_count") != 3 or names != ACCEPTED:
        raise RuntimeError(f"authoritative accepted-camera frontier changed: {names}")

    rows = {}

    lar_reg = registry["accepted_cameras"]["Left Above Rim"]
    lar_ev = lar_reg["event_489"]
    lar_path = frames_dir / lar_ev["freeze_frame"]
    C = np.asarray(lar_reg["physical_camera_center_prior_cm"], dtype=np.float64)
    K = K_matrix(float(lar_ev["focal_px"]), lar_ev["principal_point_px"])
    Hm = np.asarray(lar_floor["floor_homography_world_to_image"], dtype=np.float64)
    rot, roots = recover_accepted_rotation(C, K, Hm)
    if rot["p95_px"] > 0.55 or rot["min_depth_cm"] <= 20.0:
        raise RuntimeError(f"Left Above Rim accepted orientation reproduction failed: {rot}")
    rows["Left Above Rim"] = camera_row(
        "Left Above Rim", C, K, rot["R"], lar_path.name, "v41/v42 centre+intrinsics; v35 floor orientation"
    )
    rows["Left Above Rim"]["orientation_qa"] = {
        "floor_p95_px": float(rot["p95_px"]),
        "floor_rms_px": float(rot["rms_px"]),
        "root_count": len(roots),
    }

    if rar_report.get("status") != "PASS_RIGHT_ABOVE_RIM_METRIC_CAMERA_V73":
        raise RuntimeError("Right Above Rim v73 artifact is not accepted")
    rar_reg = registry["accepted_cameras"]["Right Above Rim"]
    C = np.asarray(rar_report["physical_center_cm"], dtype=np.float64)
    target = rar_report["target_frame_c"]
    K = K_matrix(float(target["focal_px"]), target["principal_point_px"])
    rv = np.asarray(target["rvec"], dtype=np.float64)
    R = cv2.Rodrigues(rv.reshape(3, 1))[0]
    rar_name = "L_Right_Above_Rim_8.562013s_frame0257.png"
    if np.linalg.norm(C - np.asarray(rar_reg["physical_camera_center_cm"], float)) > 1e-6:
        raise RuntimeError("Right Above Rim v73 centre disagrees with authoritative v5 registry")
    rows["Right Above Rim"] = camera_row(
        "Right Above Rim", C, K, R, rar_name, "v73 accepted Frame-C state; v74 fixed-mount centre"
    )
    rows["Right Above Rim"]["orientation_qa"] = {
        "heldout_dense_rim_p95_px": float(target["heldout_dense_11520"]["rim"]["p95_px"]),
        "heldout_dense_ft_p95_px": float(target["heldout_dense_11520"]["ft"]["p95_px"]),
    }

    if broadcast_report.get("status") != "PASS_BROADCAST_SHARED_OPTICAL_CENTER_V90":
        raise RuntimeError("Broadcast v90 artifact is not accepted")
    b_reg = registry["accepted_cameras"]["Broadcast"]
    C = np.asarray(broadcast_report["shared_camera_center_cm"], dtype=np.float64)
    f = float(broadcast_report["frame_c_focal_px"])
    pp = np.asarray(broadcast_report["shared_principal_point_px"], dtype=np.float64)
    K = K_matrix(f, pp)
    R, qa = recover_broadcast_rotation(C, f, pp, broadcast_floor, broadcast_target)
    reg_cm = np.asarray(b_reg["shared_camera_center_ft"], float) * 30.48
    if np.linalg.norm(C - reg_cm) > 1e-5:
        raise RuntimeError("Broadcast v90 centre disagrees with authoritative v5 registry")
    b_name = "A_Broadcast_9.194613s_frame0276.png"
    rows["Broadcast"] = camera_row(
        "Broadcast", C, K, R, b_name, "v90 accepted shared centre/f/pp; Frame-C orientation reproduced from v86/v87 evidence"
    )
    rows["Broadcast"]["orientation_qa"] = qa

    for row in rows.values():
        load_native(frames_dir / row["source_frame"])
    return rows


def regulation_floor_points():
    xs = np.linspace(-4.0 * 30.48, 34.0 * 30.48, 61)
    ys = np.linspace(-25.0 * 30.48, 25.0 * 30.48, 71)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])


def robust_depth_mapping(depth, valid, cam):
    K = np.asarray(cam["K"], dtype=np.float64)
    R = np.asarray(cam["R_world_to_camera"], dtype=np.float64)
    C = np.asarray(cam["center_cm"], dtype=np.float64)
    P = regulation_floor_points()
    uv, z = project_points(K, R, C, P)
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    good = (
        np.isfinite(uv).all(axis=1)
        & (z > 20.0)
        & (x >= 2) & (x < W - 2) & (y >= 2) & (y < H - 2)
    )
    ids = np.where(good)[0]
    x, y, z = x[good], y[good], z[good]
    d = depth[y, x].astype(np.float64)
    ok = valid[y, x] & np.isfinite(d) & (d > 0.02)
    ids, d, z = ids[ok], d[ok], z[ok]
    if len(d) < 35:
        raise RuntimeError(f"{cam['label']} insufficient metric floor/depth anchors: {len(d)}")

    hold = (ids % 7) == 0
    if int((~hold).sum()) < 25 or int(hold.sum()) < 5:
        hold = (np.arange(len(d)) % 7) == 0
    train = ~hold
    scale0 = float(np.median(z[train] / np.maximum(d[train], 1e-6)))
    fit = least_squares(
        lambda p: p[0] * d[train] + p[1] - z[train],
        [scale0, 0.0], loss="soft_l1", f_scale=45.0, max_nfev=6000
    )
    p = np.asarray(fit.x, dtype=np.float64)
    residual = np.abs(p[0] * d + p[1] - z)
    med = float(np.median(residual[train]))
    mad = float(np.median(np.abs(residual[train] - med)))
    thresh = max(75.0, med + 4.0 * 1.4826 * max(mad, 1.0))
    support = train & (residual <= thresh)
    if int(support.sum()) >= 25:
        fit2 = least_squares(
            lambda q: q[0] * d[support] + q[1] - z[support],
            p, loss="soft_l1", f_scale=35.0, max_nfev=6000
        )
        p = np.asarray(fit2.x, dtype=np.float64)
    pred = p[0] * d + p[1]
    err = np.abs(pred - z)
    held = err[hold]
    return p, {
        "candidate_anchor_count": int(len(d)),
        "training_support_count": int(support.sum()),
        "heldout_count": int(hold.sum()),
        "scale": float(p[0]),
        "offset_cm": float(p[1]),
        "heldout_median_abs_cm": float(np.median(held)) if len(held) else None,
        "heldout_p95_abs_cm": float(np.percentile(held, 95)) if len(held) else None,
        "all_median_abs_cm": float(np.median(err)),
        "all_p95_abs_cm": float(np.percentile(err, 95)),
    }


def build_cloud(image, depth, valid, mapping, cam, stride=2):
    scale, offset = [float(x) for x in mapping]
    z = scale * depth.astype(np.float64) + offset
    K = np.asarray(cam["K"], dtype=np.float64)
    R = np.asarray(cam["R_world_to_camera"], dtype=np.float64)
    C = np.asarray(cam["center_cm"], dtype=np.float64)
    yy, xx = np.indices((H, W))
    sample = ((xx % stride) == 0) & ((yy % stride) == 0)
    ok = sample & valid & np.isfinite(z) & (z > 20.0) & (z < 12000.0)
    ys, xs = np.where(ok)
    zz = z[ys, xs]
    xn = (xs.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (ys.astype(np.float64) - K[1, 2]) / K[1, 1]
    Xc = np.column_stack([xn * zz, yn * zz, zz])
    Xw = (R.T @ Xc.T).T + C
    colours = image[ys, xs].copy()
    src_uv = np.column_stack([xs, ys]).astype(np.int32)
    return Xw.astype(np.float32), colours, src_uv


def composite_hole_fill(renders, order):
    first = order[0]
    out = renders[first][0].copy()
    mask = renders[first][1] > 0
    provenance = np.zeros((H, W), dtype=np.uint8)
    provenance[mask] = ACCEPTED.index(first) + 1
    for label in order[1:]:
        image, m = renders[label][0], renders[label][1] > 0
        take = (~mask) & m
        out[take] = image[take]
        provenance[take] = ACCEPTED.index(label) + 1
        mask |= take
    return out, mask, provenance


def action_roi_fraction(mask, pivot_px):
    cx, cy = [int(round(x)) for x in pivot_px]
    x0, x1 = max(0, cx - 230), min(W, cx + 231)
    y0, y1 = max(0, cy - 200), min(H, cy + 201)
    roi = np.zeros((H, W), bool)
    roi[y0:y1, x0:x1] = True
    return float(mask[roi].mean()) if np.any(roi) else 0.0, [x0, y0, x1, y1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--lar-floor", type=Path, required=True)
    ap.add_argument("--rar-report", type=Path, required=True)
    ap.add_argument("--broadcast-report", type=Path, required=True)
    ap.add_argument("--broadcast-floor", type=Path, required=True)
    ap.add_argument("--broadcast-target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--max-degree", type=float, default=25.0)
    ap.add_argument("--tokens", type=int, default=1200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    registry = read_json(args.registry)
    cameras = build_cameras(
        registry,
        read_json(args.lar_floor),
        read_json(args.rar_report),
        read_json(args.broadcast_report),
        read_json(args.broadcast_floor),
        read_json(args.broadcast_target),
        args.frames_dir,
    )
    images = {label: load_native(args.frames_dir / cameras[label]["source_frame"]) for label in ACCEPTED}

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()

    clouds = {}
    depth_qa = {}
    for label in ACCEPTED:
        depth, _, valid, kmoge, _ = moge_infer(model, images[label], args.tokens)
        mapping, qa = robust_depth_mapping(depth, valid, cameras[label])
        clouds[label] = build_cloud(images[label], depth, valid, mapping, cameras[label], stride=2)
        qa["moge_valid_fraction"] = float(np.mean(valid))
        qa["moge_reported_intrinsics"] = np.asarray(kmoge).tolist()
        qa["cloud_points"] = int(len(clouds[label][0]))
        depth_qa[label] = qa

    anchor = cameras["Left Above Rim"]
    K0 = np.asarray(anchor["K"], dtype=np.float64)
    R0 = np.asarray(anchor["R_world_to_camera"], dtype=np.float64)
    C0 = np.asarray(anchor["center_cm"], dtype=np.float64)

    def render_degree(degree: float):
        if abs(degree) < 1e-9:
            full = np.ones((H, W), dtype=bool)
            provenance = np.ones((H, W), dtype=np.uint8)
            return images["Left Above Rim"].copy(), full, provenance, {
                "Left Above Rim": 1.0, "Broadcast": 0.0, "Right Above Rim": 0.0
            }
        Rt, Ct = orbit_pose(C0, R0, RIM, degree)
        renders = {}
        source_coverage = {}
        for label in ACCEPTED:
            im, mask, _ = raster_source_cloud(clouds[label], K0, Rt, Ct, radius=1)
            renders[label] = (im, mask)
            source_coverage[label] = float(np.mean(mask > 0))
        out, mask, provenance = composite_hole_fill(
            renders, ("Left Above Rim", "Broadcast", "Right Above Rim")
        )
        return out, mask, provenance, source_coverage

    key_rows = []
    for degree in (0, 5, 10, 15, 20, 25):
        frame, mask, provenance, sc = render_degree(float(degree))
        Rt, Ct = orbit_pose(C0, R0, RIM, float(degree))
        pivot, _ = project_points(K0, Rt, Ct, RIM[None, :])
        af, roi = action_roi_fraction(mask, pivot[0])
        cv2.imwrite(str(args.out / f"three_camera_{degree:02d}deg_native.png"), frame)
        cv2.imwrite(str(args.out / f"three_camera_{degree:02d}deg_unresolved.png"), (~mask).astype(np.uint8) * 255)
        cv2.imwrite(str(args.out / f"three_camera_{degree:02d}deg_provenance.png"), provenance * 80)
        key_rows.append({
            "degree": degree,
            "resolved_fraction_full": float(mask.mean()),
            "resolved_fraction_action_roi": af,
            "action_roi_xyxy": roi,
            "source_raw_coverage": sc,
            "provenance_pixels": {
                "Left Above Rim": int(np.sum(provenance == 1)),
                "Right Above Rim": int(np.sum(provenance == 2)),
                "Broadcast": int(np.sum(provenance == 3)),
                "unresolved": int(np.sum(provenance == 0)),
            },
        })

    degrees = []
    for i in range(args.frames):
        phase = i / max(args.frames - 1, 1)
        degree = float(args.max_degree * math.sin(math.pi * phase))
        degrees.append(degree)
        frame, _, _, _ = render_degree(degree)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)

    report = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_THREE_ACCEPTED_CAMERA_PREVIEW_RENDERED",
        "game_id": "0022500301",
        "event_id": 489,
        "basketball_moment": "Steven Adams dunk vs Utah immediately after his block; Frame-C pre-completion freeze",
        "explicit_user_exception": "Render the three solved cameras now without changing or stopping the Right Slash fourth-camera solve.",
        "accepted_cameras_used": list(ACCEPTED),
        "accepted_camera_count": 3,
        "fourth_camera": "Right Slash remains outside this render and remains the unsolved active frontier.",
        "certification_scope": "diagnostic visual sufficiency test only; does not grant replay_render_allowed and does not alter registry v5",
        "source_resolution": [960, 540],
        "render_resolution_native": [960, 540],
        "appearance_policy": "official NBA Frame-C source pixels only; no generative fill, crossfade, optical-flow morph, or appearance averaging",
        "geometry_policy": "accepted metric cameras establish source rays; MoGe-2 supplies per-view depth shape aligned to regulation NBA floor depth; Left Above Rim is the appearance anchor; Broadcast then Right Above Rim fill only unresolved pixels",
        "camera_states": cameras,
        "depth_alignment_qa": depth_qa,
        "key_stills": key_rows,
        "motion": {
            "frames": args.frames,
            "fps": 30,
            "max_degree": args.max_degree,
            "path": "0 -> +max-degree -> 0 around fixed regulation rim pivot",
            "degree_min": float(min(degrees)),
            "degree_max": float(max(degrees)),
        },
        "diagnostic_gate": {
            "camera_count_exactly_three_accepted": True,
            "zero_degree_exact_source_frame": True,
            "secondary_views_only_fill_holes": True,
            "unresolved_regions_remain_unfilled": True,
            "no_fourth_camera_claim": True,
        },
    }
    (args.out / "three_camera_freeview_preview_v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "camera_states_three_accepted.json").write_text(
        json.dumps(cameras, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "accepted_cameras_used": report["accepted_cameras_used"],
        "depth_alignment_qa": depth_qa,
        "key_stills": key_rows,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
