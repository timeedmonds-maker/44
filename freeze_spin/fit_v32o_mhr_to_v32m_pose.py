from __future__ import annotations

"""v32o2: fit the real MHR anatomical rig to the accepted v32m Adams pose.

The first v32o attempt used MHR.from_files(), whose PyTorch/PyMomentum wrapper
segfaulted on the hosted CPU runner before optimization.  This implementation
uses the lower-level PyMomentum Character + IK path already proven stable by
v32n.  It fits only source-grounded v32m metric joints, then optionally passes
the solved 204 MHR pose parameters through Meta's public TorchScript MHR LOD1
model for pose-corrected surface vertices.

No image pixels are synthesized.  This stage creates no novel view.  It only
asks whether a genuine articulated human triangle mesh can occupy the measured
Adams geometry and project coherently into the three calibrated NBA cameras.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot

import pymomentum.geometry as geo
import pymomentum.solver as solver

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

# v32m directly measures these.  Right wrist/ankle are retained but downweighted
# because their corresponding proximal right-side joints are occluded at t+00.
FIT_NAMES = (
    "left_shoulder", "left_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "left_ankle", "right_ankle",
)
FIT_WEIGHTS = {
    "left_shoulder": 1.10,
    "left_elbow": 1.20,
    "left_wrist": 1.10,
    "right_wrist": 0.45,
    "left_hip": 1.35,
    "right_hip": 1.35,
    "left_knee": 1.25,
    "left_ankle": 1.10,
    "right_ankle": 0.50,
}


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


def make_frame(left, right, up_point) -> np.ndarray:
    x = unit(np.asarray(right, float) - np.asarray(left, float))
    mid = 0.5 * (np.asarray(left, float) + np.asarray(right, float))
    yr = np.asarray(up_point, float) - mid
    y = unit(yr - x * float(np.dot(x, yr)))
    z = unit(np.cross(x, y))
    y = unit(np.cross(z, x))
    return np.column_stack([x, y, z])


def camera(scene, label):
    d = scene["cameras"][label]
    return (
        np.asarray(d["K_px"], float),
        np.asarray(d["R_world_to_camera"], float),
        np.asarray(d["C_world_cm"], float),
    )


def project(K, R, C, X):
    X = np.asarray(X, float)
    xc = (R @ (X - C).T).T
    z = xc[:, 2]
    q = (K @ xc.T).T
    uv = np.full((len(X), 2), np.nan, float)
    ok = np.abs(z) > 1e-7
    uv[ok] = q[ok, :2] / q[ok, 2:3]
    return uv, z


def mesh_mask(uv, faces):
    mask = np.zeros((H, W), np.uint8)
    tri = uv[faces]
    valid = np.all(np.isfinite(tri), axis=(1, 2)) & np.all(np.abs(tri) < 6000, axis=(1, 2))
    for t in tri[valid]:
        p = np.rint(t).astype(np.int32)
        cv2.fillConvexPoly(mask, p, 255, lineType=cv2.LINE_8)
    return mask


def draw_overlay(img, mask, joint_uv, label):
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
        f"v32o2 MHR mesh projection | {label} | cyan mesh silhouette",
        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def find_assets(work: Path) -> Path:
    assets = work / "assets"
    if not (assets / "lod1.fbx").exists():
        subprocess.run(["mhr-download-assets", "--dest", str(work)], check=True)
    assert (assets / "lod1.fbx").exists(), assets
    assert (assets / "compact_v6_1.model").exists(), assets
    return assets


def load_character(assets: Path):
    return geo.Character.load_fbx(
        str(assets / "lod1.fbx"),
        str(assets / "compact_v6_1.model"),
        load_blendshapes=True,
    )


def neutral_positions(character, joint_indices):
    z = np.zeros((character.parameter_transform.size,), np.float32)
    parents = np.asarray(joint_indices, np.int64)
    offsets = np.zeros((len(joint_indices), 3), np.float32)
    return geo.model_parameters_to_positions(character, z, parents, offsets)


def build_initial_pose(character, ji, target_all):
    n = character.parameter_transform.size
    init = np.zeros((n,), np.float64)
    ids = [ji["l_upleg"], ji["r_upleg"], ji["c_head"]]
    sl, sr, sh = neutral_positions(character, ids)
    tl = target_all["left_hip"]
    tr = target_all["right_hip"]
    tup = target_all.get("nose", target_all["left_shoulder"])
    Fs = make_frame(sl, sr, sh)
    Ft = make_frame(tl, tr, tup)
    R0 = Ft @ Fs.T
    # Momentum root parameters are XYZ Euler rotations in radians.
    eul = SciRot.from_matrix(R0).as_euler("xyz", degrees=False)
    init[3:6] = eul
    neutral_mid = 0.5 * (sl + sr)
    target_mid = 0.5 * (tl + tr)
    init[0:3] = target_mid - (R0 @ neutral_mid)
    return init, R0


def run_ik(character, ji, target_all, fit_names, init):
    n = character.parameter_transform.size
    parents = torch.tensor([ji[MHR_MAP[x]] for x in fit_names], dtype=torch.int64)
    offsets = torch.zeros((len(fit_names), 3), dtype=torch.float64)
    targets = torch.tensor(
        np.stack([target_all[x] for x in fit_names])[None, ...], dtype=torch.float64
    )
    pos_w = torch.tensor(
        [[FIT_WEIGHTS.get(x, 1.0) for x in fit_names]], dtype=torch.float64
    )

    active = torch.zeros((n,), dtype=torch.bool)
    active[:68] = True  # rigid + torso/arms/legs/neck, no fingers
    if n >= 136:
        active[130:136] = True  # interpretable flexible body dimensions only

    mp0 = torch.tensor(init[None, :], dtype=torch.float64)
    motion_targets = mp0.clone()
    motion_w = torch.zeros((1, n), dtype=torch.float64)
    motion_w[:, 6:68] = 0.10
    if n >= 136:
        motion_w[:, 130:136] = 0.20
    # Keep unobserved DOFs close to initialization while allowing root to move freely.

    active_err = [
        solver.ErrorFunctionType.Limit,
        solver.ErrorFunctionType.Position,
        solver.ErrorFunctionType.Motion,
    ]
    err_w = torch.tensor([[1.0, 1.0, 0.025]], dtype=torch.float64)
    opts = solver.SolverOptions()
    # Defaults are robust here; explicitly increase iterations when exposed.
    for attr, val in (("max_iterations", 120), ("min_iterations", 5)):
        if hasattr(opts, attr):
            setattr(opts, attr, val)

    out = solver.solve_ik(
        character=character,
        active_parameters=active,
        model_parameters_init=mp0,
        active_error_functions=active_err,
        error_function_weights=err_w,
        options=opts,
        position_cons_parents=parents,
        position_cons_offsets=offsets,
        position_cons_weights=pos_w,
        position_cons_targets=targets,
        motion_targets=motion_targets,
        motion_weights=motion_w,
    )
    return out.detach().cpu().numpy()[0].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v32m-root", type=Path, required=True)
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("v32o_work"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    q = json.loads((args.v32m_root / "v32m_single_anchor_qa.json").read_text())
    assert q["status"] == "PASS_V32M_SINGLE_ANCHOR_ARTICULATED_POSITION"
    assert q["surface_stage_unlocked"] is True
    target_all = {k: np.asarray(v["world_cm"], np.float32) for k, v in q["joints"].items()}
    fit_names = [n for n in FIT_NAMES if n in target_all]
    if len(fit_names) < 8:
        raise RuntimeError(f"insufficient trusted body joints: {fit_names}")

    stage = args.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    assert scene["resolution"] == [W, H] and set(scene["cameras"]) == set(CAMS)
    images = {c: cv2.imread(str(stage / FRAME_NAMES[c])) for c in CAMS}
    assert all(v is not None for v in images.values())

    assets = find_assets(args.work)
    character = load_character(assets)
    names = list(character.skeleton.joint_names)
    ji = {n: i for i, n in enumerate(names)}
    missing = [MHR_MAP[n] for n in fit_names if MHR_MAP[n] not in ji]
    if missing:
        raise RuntimeError(f"MHR joint mapping missing: {missing}")

    init, R0 = build_initial_pose(character, ji, target_all)
    solved = run_ik(character, ji, target_all, fit_names, init)

    pidx = np.asarray([ji[MHR_MAP[n]] for n in fit_names], np.int64)
    zeros = np.zeros((len(fit_names), 3), np.float32)
    solved_pts = geo.model_parameters_to_positions(character, solved, pidx, zeros)
    target = np.stack([target_all[n] for n in fit_names])
    err = np.linalg.norm(solved_pts - target, axis=1)

    # Stable geometry path: direct PyMomentum skinned vertices.
    skel = geo.model_parameters_to_skeleton_state(character, solved)
    raw_verts = np.asarray(character.skin_points(skel), np.float32)
    faces = np.asarray(character.mesh.faces, np.int32)
    vworld = raw_verts
    surface_source = "PyMomentum MHR skinned mesh"

    # Prefer public TorchScript LOD1 pose-corrected surface when it is usable.
    ts_path = assets / "mhr_model.pt"
    ts_error = None
    if ts_path.exists():
        try:
            ts = torch.jit.load(str(ts_path), map_location="cpu")
            ts.eval()
            with torch.no_grad():
                vv, _ = ts(
                    torch.zeros((1, 45), dtype=torch.float32),
                    torch.tensor(solved[None, :], dtype=torch.float32),
                    torch.zeros((1, 72), dtype=torch.float32),
                )
            vv = vv[0].cpu().numpy().astype(np.float32)
            if len(vv) == len(raw_verts) and np.all(np.isfinite(vv)):
                vworld = vv
                surface_source = "MHR public TorchScript LOD1 with pose correctives"
        except Exception as e:
            ts_error = repr(e)

    ply = args.out / "v32o_adams_mhr_world_cm.ply"
    with ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vworld)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in vworld:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    joint_world = {n: solved_pts[i] for i, n in enumerate(fit_names)}
    overlays, camera_qa = [], {}
    for c in CAMS:
        K, Rc, Cc = camera(scene, c)
        uv, z = project(K, Rc, Cc, vworld)
        mask = mesh_mask(uv, faces)
        juv = {n: project(K, Rc, Cc, p.reshape(1, 3))[0][0] for n, p in joint_world.items()}
        ov = draw_overlay(images[c], mask, juv, c)
        cv2.imwrite(str(args.out / f"v32o_{c.replace(' ', '_')}_mesh_overlay.png"), ov)
        cv2.imwrite(str(args.out / f"v32o_{c.replace(' ', '_')}_mesh_mask.png"), mask)
        overlays.append(ov)
        camera_qa[c] = {
            "mesh_projected_pixels": int(np.sum(mask > 0)),
            "mesh_vertex_finite_fraction": float(np.mean(np.all(np.isfinite(uv), axis=1))),
            "signed_depth_median_cm": float(np.nanmedian(z)),
        }
    cv2.imwrite(str(args.out / "v32o_three_camera_mesh_overlay.png"), np.hstack(overlays))

    per = {n: float(e) for n, e in zip(fit_names, err)}
    med = float(np.median(err))
    p90 = float(np.percentile(err, 90))
    mx = float(np.max(err))
    fit_pass = (
        len(fit_names) >= 8
        and med <= 6.0
        and p90 <= 10.0
        and mx <= 15.0
        and np.all(np.isfinite(vworld))
        and len(vworld) > 10000
    )
    qa = {
        "version": "v32o2_direct_pymomentum_mhr_fit",
        "status": "PASS_V32O_MHR_ANATOMICAL_SEED" if fit_pass else "FAIL_CLOSED_V32O_MHR_ANATOMICAL_SEED",
        "source_pose": "v32m accepted source-grounded metric pose",
        "native_resolution": [W, H],
        "generated_rgb": False,
        "novel_view_rendered": False,
        "surface_type": surface_source,
        "torchscript_fallback_error": ts_error,
        "vertex_count": int(len(vworld)),
        "face_count": int(len(faces)),
        "fit_joint_names": fit_names,
        "mhr_joint_mapping": {n: MHR_MAP[n] for n in fit_names},
        "per_joint_error_cm": per,
        "median_joint_error_cm": med,
        "p90_joint_error_cm": p90,
        "max_joint_error_cm": mx,
        "initial_root_rotation_matrix": R0.tolist(),
        "solved_model_parameters": {
            character.parameter_transform.names[i]: float(x)
            for i, x in enumerate(solved)
            if abs(float(x)) > 1e-6
        },
        "camera_projection_qa": camera_qa,
        "gate": {
            "trusted_body_joints_ge_8": len(fit_names) >= 8,
            "median_joint_error_le_6cm": med <= 6.0,
            "p90_joint_error_le_10cm": p90 <= 10.0,
            "max_joint_error_le_15cm": mx <= 15.0,
            "mesh_finite": bool(np.all(np.isfinite(vworld))),
            "mesh_is_real_surface": len(vworld) > 10000,
            "silhouette_stage_unlocked": bool(fit_pass),
        },
        "warning": "Passing this gate only proves an anatomical surface seed. Source-silhouette agreement in all three cameras remains mandatory before any novel-view render.",
    }
    (args.out / "v32o_mhr_fit_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out / "v32o_mhr_fit.npz",
        model_parameters=solved,
        vertices_world_cm=vworld,
        faces=faces,
        joint_names=np.asarray(fit_names),
        joints_world_cm=solved_pts,
    )
    print(json.dumps({
        "status": qa["status"],
        "surface_type": surface_source,
        "median_joint_error_cm": med,
        "p90_joint_error_cm": p90,
        "max_joint_error_cm": mx,
        "gate": qa["gate"],
    }, indent=2), flush=True)
    if not fit_pass:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
