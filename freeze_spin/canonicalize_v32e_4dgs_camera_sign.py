from __future__ import annotations

"""V32e: canonicalize projective camera signs for 4DGaussians positive depth.

A calibrated pinhole camera P and -P have identical image projections.  The
upstream NBA camera solves can therefore be geometrically correct while using
opposite camera-depth signs.  The 4DGaussians CUDA rasterizer, however, rejects
points with non-positive view-space z.  This utility chooses the equivalent
3x4 sign independently for each solved camera so the validated focal action is
in positive depth, while proving that:

- native pixels / intrinsics are untouched;
- every projected seed point is unchanged to numerical precision;
- each physical camera centre is unchanged;
- the initialized action cloud is in positive rasterizer depth.

No image synthesis, resize, warp, interpolation or learned operation occurs.
"""

import argparse
import json
from pathlib import Path

import numpy as np


META_NAMES = (
    "train_meta.json",
    "test_meta.json",
    "train_meta_holdout_left_above_rim.json",
    "test_meta_holdout_left_above_rim.json",
    "train_meta_holdout_broadcast.json",
    "test_meta_holdout_broadcast.json",
    "train_meta_holdout_right_above_rim.json",
    "test_meta_holdout_right_above_rim.json",
)


def project(K: np.ndarray, M: np.ndarray, xyz: np.ndarray):
    X = np.concatenate([xyz, np.ones((len(xyz), 1), np.float64)], axis=1)
    Xc = (M @ X.T).T[:, :3]
    q = (K @ Xc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    return uv, Xc[:, 2]


def camera_center(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)[:3, 3]


def canonicalize(M: np.ndarray, reference_world_m: np.ndarray):
    ref_h = np.append(reference_world_m, 1.0)
    z = float((M @ ref_h)[2])
    out = M.copy()
    flipped = z < 0.0
    if flipped:
        # Negating the first three homogeneous camera rows changes P to -P.
        # Pixel projection is exactly invariant, while view-space depth changes
        # sign.  Keep the homogeneous [0,0,0,1] row unchanged so inv(M) returns
        # the same Euclidean camera centre.
        out[:3, :] *= -1.0
    return out, flipped, z, float((out @ ref_h)[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--min-positive-fraction", type=float, default=0.995)
    args = ap.parse_args()
    root = args.dataset

    backend_path = root / "v32d_backend.json"
    backend = json.loads(backend_path.read_text())
    ref_cm = np.asarray(backend["focal_player_seed"]["center_world_cm"], np.float64)
    reference = ref_cm / 100.0
    cloud = np.load(root / "init_pt_cld.npz")["data"][:, :3].astype(np.float64)

    main_path = root / "train_meta.json"
    original_main = json.loads(main_path.read_text())
    camera_rows = {}
    for K, M, cid in zip(original_main["k"][0], original_main["w2c"][0], original_main["cam_id"][0]):
        camera_rows[int(cid)] = (np.asarray(K, np.float64), np.asarray(M, np.float64))
    if sorted(camera_rows) != [0, 1, 2]:
        raise RuntimeError(f"expected camera ids 0,1,2; got {sorted(camera_rows)}")

    fixed_by_id = {}
    qa_cameras = {}
    for cid, (K, M) in camera_rows.items():
        fixed, flipped, ref_before, ref_after = canonicalize(M, reference)
        uv0, z0 = project(K, M, cloud)
        uv1, z1 = project(K, fixed, cloud)
        finite = np.isfinite(uv0).all(axis=1) & np.isfinite(uv1).all(axis=1)
        if not finite.any():
            raise RuntimeError(f"camera {cid}: no finite seed projections")
        max_uv_delta = float(np.max(np.linalg.norm(uv1[finite] - uv0[finite], axis=1)))
        c0 = camera_center(M); c1 = camera_center(fixed)
        center_shift = float(np.linalg.norm(c1 - c0))
        positive_fraction = float(np.mean(z1 > 0.2))
        if max_uv_delta > 1e-8:
            raise RuntimeError(f"camera {cid}: projective invariance failed {max_uv_delta}px")
        if center_shift > 1e-9:
            raise RuntimeError(f"camera {cid}: camera centre moved {center_shift}m")
        if ref_after <= 0.2:
            raise RuntimeError(f"camera {cid}: focal reference depth still invalid {ref_after}")
        if positive_fraction < args.min_positive_fraction:
            raise RuntimeError(f"camera {cid}: only {positive_fraction:.6f} of seed cloud is positive depth")
        fixed_by_id[cid] = fixed
        qa_cameras[str(cid)] = {
            "projective_sign_flipped": bool(flipped),
            "reference_depth_before_m": ref_before,
            "reference_depth_after_m": ref_after,
            "seed_positive_depth_fraction_after": positive_fraction,
            "seed_min_depth_after_m": float(np.min(z1)),
            "seed_median_depth_after_m": float(np.median(z1)),
            "max_seed_projection_change_px": max_uv_delta,
            "camera_center_shift_m": center_shift,
            "camera_center_m": c1.tolist(),
            "linear_determinant_after": float(np.linalg.det(fixed[:3, :3])),
        }

    rewritten = []
    for name in META_NAMES:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"missing expected metadata {path}")
        meta = json.loads(path.read_text())
        for ti, ids in enumerate(meta["cam_id"]):
            for j, cid in enumerate(ids):
                cid = int(cid)
                if cid not in fixed_by_id:
                    raise RuntimeError(f"{name}: unknown camera id {cid}")
                meta["w2c"][ti][j] = fixed_by_id[cid].tolist()
        path.write_text(json.dumps(meta, indent=2))
        rewritten.append(name)

    # Re-open every metadata file and enforce the positive-depth contract at all
    # repeated timestamps.  Camera poses are static for this event, but the
    # exhaustive check prevents a malformed holdout manifest from slipping in.
    for name in META_NAMES:
        meta = json.loads((root / name).read_text())
        for ti, (Ks, Ms, ids) in enumerate(zip(meta["k"], meta["w2c"], meta["cam_id"])):
            for K, M, cid in zip(Ks, Ms, ids):
                _, z = project(np.asarray(K, np.float64), np.asarray(M, np.float64), cloud)
                frac = float(np.mean(z > 0.2))
                if frac < args.min_positive_fraction:
                    raise RuntimeError(f"{name} t={ti} cam={cid}: positive-depth fraction {frac}")

    qa = {
        "version": "v32e_4dgs_camera_depth_canonical",
        "source_dataset_version": backend.get("version"),
        "reference": "validated v31 focal-player skeleton median",
        "reference_world_m": reference.tolist(),
        "camera_id_labels": {"0": "Left Above Rim", "1": "Broadcast", "2": "Right Above Rim"},
        "cameras": qa_cameras,
        "metadata_files_rewritten": rewritten,
        "pixel_data_modified": False,
        "intrinsics_modified": False,
        "point_cloud_modified": False,
        "projective_geometry_modified": False,
        "purpose": "choose the P versus -P representation accepted by the positive-z Gaussian rasterizer",
        "status": "PASS_V32E_4DGS_CAMERA_DEPTH_CANONICAL",
    }
    (root / "v32e_camera_depth_canonical_qa.json").write_text(json.dumps(qa, indent=2))
    backend["v32e_camera_depth_canonicalization"] = qa
    backend["version"] = "v32e_native_action_crops_focal_seeded_depth_canonical"
    backend_path.write_text(json.dumps(backend, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
