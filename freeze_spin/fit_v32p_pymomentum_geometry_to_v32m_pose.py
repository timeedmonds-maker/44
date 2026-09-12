from __future__ import annotations

"""v32p: stable low-level PyMomentum fit of the MHR body to accepted v32m Adams geometry.

The high-level MHR.from_files path segfaulted on the hosted CPU runner before
optimization.  v32n proved that direct pymomentum.geometry Character.load_fbx,
forward kinematics and skinning are stable on the same runner.  v32p therefore
uses only that proven low-level API plus scipy least_squares.

Inputs:
- accepted v32m metric Adams joints (source-grounded; no generated geometry)
- exact B32 three-camera calibration and native 960x540 source frames
- public MHR LOD1 rig/mesh

Outputs:
- fitted articulated MHR triangle mesh in NBA world centimetres
- three native source-view silhouette overlays for visual QA
- fail-closed fit metrics

No novel view, generated RGB, inpainting, capsule body, Gaussian body, or fourth
camera is used here.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

W, H = 960, 540
CAMS = ("Left Above Rim", "Broadcast", "Right Above Rim")
FRAME_NAMES = {
    "Left Above Rim": "v32_chosen_Left_Above_Rim_frame0260.png",
    "Broadcast": "v32_chosen_Broadcast_frame0276.png",
    "Right Above Rim": "v32_chosen_Right_Above_Rim_frame0256.png",
}
MHR_MAP = {
    "left_shoulder": "l_uparm",
    "right_shoulder": "r_uparm",
    "left_elbow": "l_lowarm",
    "right_elbow": "r_lowarm",
    "left_wrist": "l_wrist",
    "right_wrist": "r_wrist",
    "left_hip": "l_upleg",
    "right_hip": "r_upleg",
    "left_knee": "l_lowleg",
    "right_knee": "r_lowleg",
    "left_ankle": "l_foot",
    "right_ankle": "r_foot",
}
PREFERRED_FIT = (
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
SEGMENTS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


def make_frame(left: np.ndarray, right: np.ndarray, up_point: np.ndarray) -> np.ndarray:
    x = unit(np.asarray(right, float) - np.asarray(left, float))
    mid = 0.5 * (np.asarray(left, float) + np.asarray(right, float))
    yr = np.asarray(up_point, float) - mid
    y = unit(yr - x * float(np.dot(x, yr)))
    z = unit(np.cross(x, y))
    y = unit(np.cross(z, x))
    return np.column_stack([x, y, z])


def camera(scene: dict, label: str):
    d = scene["cameras"][label]
    return (
        np.asarray(d["K_px"], float),
        np.asarray(d["R_world_to_camera"], float),
        np.asarray(d["C_world_cm"], float),
    )


def project(K: np.ndarray, R: np.ndarray, C: np.ndarray, X: np.ndarray):
    X = np.asarray(X, float)
    xc = (R @ (X - C).T).T
    z = xc[:, 2]
    q = (K @ xc.T).T
    uv = np.full((len(X), 2), np.nan, float)
    ok = np.abs(z) > 1e-8
    uv[ok] = q[ok, :2] / q[ok, 2:3]
    return uv, z


def mesh_mask(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    tri = uv[faces]
    valid = np.all(np.isfinite(tri), axis=(1, 2)) & np.all(np.abs(tri) < 6000, axis=(1, 2))
    for t in tri[valid]:
        p = np.rint(t).astype(np.int32)
        cv2.fillConvexPoly(mask, p, 255, lineType=cv2.LINE_8)
    return mask


def draw_overlay(img: np.ndarray, mask: np.ndarray, joint_uv: dict, label: str) -> np.ndarray:
    out = img.copy()
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (255, 255, 0), 2, cv2.LINE_AA)
    for p in joint_uv.values():
        if p is None or not np.all(np.isfinite(p)):
            continue
        x, y = np.rint(p).astype(int)
        if -20 <= x < W + 20 and -20 <= y < H + 20:
            cv2.circle(out, (x, y), 3, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.putText(
        out,
        f"v32p direct PyMomentum MHR | {label} | cyan mesh silhouette, magenta fitted joints",
        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def ensure_assets(work: Path):
    import pymomentum.geometry as geo
    from mhr.io import get_mhr_fbx_path, get_mhr_model_path

    assets = work / "assets"
    fbx = Path(get_mhr_fbx_path(assets, 1))
    model_path = Path(get_mhr_model_path(assets))
    if not fbx.exists() or not model_path.exists():
        subprocess.run(["mhr-download-assets", "--dest", str(work)], check=True)
    assert fbx.exists(), fbx
    assert model_path.exists(), model_path
    character = geo.Character.load_fbx(str(fbx), str(model_path), load_blendshapes=True)
    return geo, character, fbx, model_path


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{float(v[0])} {float(v[1])} {float(v[2])}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v32m-root", type=Path, required=True)
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nfev", type=int, default=180)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)

    q = json.loads((a.v32m_root / "v32m_single_anchor_qa.json").read_text())
    assert q["status"] == "PASS_V32M_SINGLE_ANCHOR_ARTICULATED_POSITION"
    assert q["surface_stage_unlocked"] is True
    target_all = {k: np.asarray(v["world_cm"], np.float64) for k, v in q["joints"].items()}
    fit_names = [n for n in PREFERRED_FIT if n in target_all]
    if len(fit_names) < 8:
        raise RuntimeError(f"insufficient trusted v32m body landmarks: {fit_names}")

    stage = a.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    assert scene["resolution"] == [W, H]
    assert set(scene["cameras"]) == set(CAMS)
    images = {c: cv2.imread(str(stage / FRAME_NAMES[c])) for c in CAMS}
    assert all(im is not None for im in images.values())

    geo, character, fbx, model_path = ensure_assets(a.work)
    names = list(character.skeleton.joint_names)
    ji = {n: i for i, n in enumerate(names)}
    pnames = list(character.parameter_transform.names)
    pi = {n: i for i, n in enumerate(pnames)}
    missing = [MHR_MAP[n] for n in fit_names if MHR_MAP[n] not in ji]
    if missing:
        raise RuntimeError(f"MHR joint mapping missing: {missing}")

    nparam = len(pnames)
    zero = np.zeros(nparam, np.float64)
    fit_joint_idx = np.asarray([ji[MHR_MAP[n]] for n in fit_names], np.int32)
    offsets = np.zeros((len(fit_joint_idx), 3), np.float64)
    neutral_pts = np.asarray(
        geo.model_parameters_to_positions(character, zero, fit_joint_idx, offsets), float
    )
    neutral_state = np.asarray(geo.model_parameters_to_skeleton_state(character, zero), float)
    neutral_all = neutral_state[:, :3]

    # Build a physically sensible global Sim(3) initialization. MHR is Y-up;
    # the accepted NBA world is Z-up. Hip axis + head/nose determines the frame.
    lh = neutral_all[ji["l_upleg"]]
    rh = neutral_all[ji["r_upleg"]]
    head = neutral_all[ji["c_head"]]
    tl = target_all["left_hip"]
    tr = target_all["right_hip"]
    tup = target_all.get("nose", target_all["left_shoulder"])
    Fs = make_frame(lh, rh, head)
    Ft = make_frame(tl, tr, tup)
    R0 = Ft @ Fs.T

    ratios = []
    neutral_by_name = {n: neutral_pts[i] for i, n in enumerate(fit_names)}
    for aa, bb in SEGMENTS:
        if aa in target_all and bb in target_all and aa in neutral_by_name and bb in neutral_by_name:
            td = float(np.linalg.norm(target_all[aa] - target_all[bb]))
            nd = float(np.linalg.norm(neutral_by_name[aa] - neutral_by_name[bb]))
            if nd > 1e-5:
                ratios.append(td / nd)
    s0 = float(np.median(ratios)) if ratios else 1.10
    s0 = float(np.clip(s0, .80, 1.55))
    t0 = 0.5 * (tl + tr) - s0 * (R0 @ (0.5 * (lh + rh)))
    r0 = Rotation.from_matrix(R0).as_rotvec()

    # Optimize only body parameters that can affect the measured joints, plus
    # MHR's six interpretable body dimension flex DOFs. Never touch fingers or
    # generated appearance/identity channels.
    driven = np.asarray(character.parameters_for_joints(fit_joint_idx.tolist()), bool)
    pose = np.asarray(character.parameter_transform.pose_parameters, bool)
    active_mask = driven & pose
    for n in (
        "spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible",
        "arm_length_flexible", "hip_width_flexible", "leg_length_flexible",
    ):
        if n in pi:
            active_mask[pi[n]] = True
    for i, n in enumerate(pnames):
        if i < 6 or "thumb" in n or "index" in n or "middle" in n or "ring" in n or "pinky" in n:
            active_mask[i] = False
    active_idx = np.flatnonzero(active_mask)
    active_names = [pnames[i] for i in active_idx]

    # Model-parameter bounds from MHR, with conservative finite caps for any
    # unbounded angle/flexible parameter. External Sim(3) is bounded separately.
    lo_full, hi_full = character.model_parameter_limits
    lo_full = np.asarray(lo_full, float)
    hi_full = np.asarray(hi_full, float)
    ilo, ihi = [], []
    for i in active_idx:
        lo, hi = float(lo_full[i]), float(hi_full[i])
        if not np.isfinite(lo) or abs(lo) > 1e10:
            lo = -3.2
        if not np.isfinite(hi) or abs(hi) > 1e10:
            hi = 3.2
        if "_flexible" in pnames[i]:
            lo, hi = max(lo, -2.5), min(hi, 2.5)
        ilo.append(lo); ihi.append(hi)

    # x = [rotvec(3), log_scale(1), translation(3), active MHR params]
    x0 = np.concatenate([r0, [math.log(s0)], t0, np.zeros(len(active_idx))])
    lower = np.concatenate([
        r0 - 1.8,
        [math.log(.72)],
        t0 - np.array([220., 220., 220.]),
        np.asarray(ilo),
    ])
    upper = np.concatenate([
        r0 + 1.8,
        [math.log(1.75)],
        t0 + np.array([220., 220., 220.]),
        np.asarray(ihi),
    ])
    target = np.stack([target_all[n] for n in fit_names])
    weights = np.asarray([
        1.20 if "hip" in n else 1.15 if "knee" in n or "ankle" in n else 1.0
        for n in fit_names
    ], float)

    def decode(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        scale = math.exp(float(x[3]))
        trans = x[4:7]
        mp = zero.copy()
        mp[active_idx] = x[7:]
        mp = np.asarray(character.apply_model_param_limits(mp), float)
        return R, scale, trans, mp

    eval_counter = {"n": 0}
    def residual(x):
        eval_counter["n"] += 1
        R, scale, trans, mp = decode(x)
        local = np.asarray(
            geo.model_parameters_to_positions(character, mp, fit_joint_idx, offsets), float
        )
        pred = scale * (local @ R.T) + trans
        data = ((pred - target) * weights[:, None]).reshape(-1)
        # Light regularization keeps unconstrained limbs near the anatomical prior.
        # Units are cm-equivalent residuals, deliberately weaker than landmark data.
        reg = []
        for value, idx in zip(x[7:], active_idx):
            name = pnames[idx]
            lam = .55 if "_flexible" in name else .16
            reg.append(lam * float(value))
        return np.concatenate([data, np.asarray(reg, float)])

    initial_res = residual(x0)
    result = least_squares(
        residual, x0, bounds=(lower, upper), method="trf",
        loss="soft_l1", f_scale=4.0, x_scale="jac",
        max_nfev=a.max_nfev, verbose=2,
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    R, scale, trans, mp = decode(result.x)
    local = np.asarray(geo.model_parameters_to_positions(character, mp, fit_joint_idx, offsets), float)
    pred = scale * (local @ R.T) + trans
    err = np.linalg.norm(pred - target, axis=1)

    # Skin the real MHR triangle mesh using the same low-level path proven in v32n.
    state = np.asarray(geo.model_parameters_to_skeleton_state(character, mp), float)
    verts_local = np.asarray(character.skin_points(state), float)
    verts_world = scale * (verts_local @ R.T) + trans
    faces = np.asarray(character.mesh.faces, np.int32)
    if not np.all(np.isfinite(verts_world)):
        raise RuntimeError("non-finite MHR vertices")

    write_ply(a.out / "v32p_adams_mhr_world_cm.ply", verts_world, faces)
    np.savez_compressed(
        a.out / "v32p_fit.npz",
        model_parameters=mp.astype(np.float32), rotation=R.astype(np.float32),
        scale=np.array([scale], np.float32), translation=trans.astype(np.float32),
        vertices_world_cm=verts_world.astype(np.float32), faces=faces,
    )

    joint_world = {n: pred[i] for i, n in enumerate(fit_names)}
    overlays, camera_qa = [], {}
    for c in CAMS:
        K, Rc, Cc = camera(scene, c)
        uv, z = project(K, Rc, Cc, verts_world)
        mask = mesh_mask(uv, faces)
        juv = {
            n: project(K, Rc, Cc, p.reshape(1, 3))[0][0]
            for n, p in joint_world.items()
        }
        ov = draw_overlay(images[c], mask, juv, c)
        cv2.imwrite(str(a.out / f"v32p_{c.replace(' ', '_')}_mesh_overlay.png"), ov)
        overlays.append(ov)
        ys, xs = np.where(mask > 0)
        camera_qa[c] = {
            "mesh_projected_pixels": int(len(xs)),
            "mesh_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "finite_vertex_fraction": float(np.mean(np.all(np.isfinite(uv), axis=1))),
            "signed_depth_median_cm": float(np.nanmedian(z)),
        }
    cv2.imwrite(str(a.out / "v32p_three_camera_mesh_overlay.png"), np.hstack(overlays))

    per = {n: float(e) for n, e in zip(fit_names, err)}
    med = float(np.median(err)); p90 = float(np.percentile(err, 90)); mx = float(np.max(err))
    fit_pass = bool(
        result.success and len(fit_names) >= 8 and med <= 7.0 and p90 <= 12.0 and mx <= 18.0
        and np.all(np.isfinite(verts_world))
    )
    qa = {
        "version": "v32p_direct_pymomentum_geometry_fit",
        "status": "PASS_V32P_MHR_ANATOMICAL_SEED" if fit_pass else "FAIL_CLOSED_V32P_MHR_ANATOMICAL_SEED",
        "source_pose": "v32m accepted source-grounded geometry only",
        "native_resolution": [W, H],
        "cameras": list(CAMS),
        "generated_rgb": False,
        "novel_view_rendered": False,
        "surface_type": "MHR LOD1 skinned triangle mesh via pymomentum.geometry direct path",
        "loader": "pymomentum.geometry.Character.load_fbx",
        "optimizer": "scipy.optimize.least_squares soft_l1",
        "mhr_fbx": str(fbx), "mhr_model": str(model_path),
        "vertex_count": int(len(verts_world)), "face_count": int(len(faces)),
        "fit_joint_names": fit_names,
        "mhr_joint_mapping": {n: MHR_MAP[n] for n in fit_names},
        "active_model_parameter_count": int(len(active_idx)),
        "active_model_parameters": active_names,
        "optimized_model_parameters": {pnames[i]: float(mp[i]) for i in active_idx if abs(float(mp[i])) > 1e-7},
        "external_rotation_matrix": R.tolist(),
        "external_scale": float(scale), "external_translation_cm": trans.tolist(),
        "per_joint_error_cm": per,
        "median_joint_error_cm": med, "p90_joint_error_cm": p90, "max_joint_error_cm": mx,
        "optimizer_result": {
            "success": bool(result.success), "status": int(result.status), "message": str(result.message),
            "cost": float(result.cost), "optimality": float(result.optimality),
            "nfev": int(result.nfev), "njev": None if result.njev is None else int(result.njev),
            "function_calls_observed": int(eval_counter["n"]),
            "initial_residual_rms": float(np.sqrt(np.mean(initial_res ** 2))),
            "final_data_rms_cm": float(np.sqrt(np.mean((pred - target) ** 2))),
        },
        "camera_projection_qa": camera_qa,
        "gate": {
            "optimizer_success": bool(result.success),
            "trusted_body_joints_ge_8": len(fit_names) >= 8,
            "median_joint_error_le_7cm": med <= 7.0,
            "p90_joint_error_le_12cm": p90 <= 12.0,
            "max_joint_error_le_18cm": mx <= 18.0,
            "mesh_finite": bool(np.all(np.isfinite(verts_world))),
            "silhouette_stage_unlocked": fit_pass,
        },
        "warning": "Passing v32p only establishes an anatomical surface seed. Source-silhouette validation in all three cameras remains mandatory before novel-view rendering.",
    }
    (a.out / "v32p_fit_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps({
        "status": qa["status"], "fit_joint_names": fit_names,
        "median_joint_error_cm": med, "p90_joint_error_cm": p90,
        "max_joint_error_cm": mx, "external_scale": scale,
        "optimizer": qa["optimizer_result"], "gate": qa["gate"],
    }, indent=2), flush=True)
    if not fit_pass:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
