from __future__ import annotations

"""Build a freeze-only Gaussian dataset from the validated v32e package.

The product target here is one static free-view moment, so dynamic time modelling
is unnecessary for the first decisive swivel proof.  This exporter keeps only
the synchronized t+00 images from the three solved cameras, retains the v32e
camera-depth canonicalization and seeded point cloud, and adds six calibrated
virtual cameras at 0,5,10,15,20,25 degrees around the focal action.

No source RGB is generated, resized, warped, interpolated or upscaled.  The six
orbit image files are explicit loader placeholders copied byte-for-byte from the
real Left-Above-Rim freeze frame; they are never ground truth and must not be
shown as rendered output.  Novel RGB is produced only later by the trained
Gaussian renderer.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ANGLES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
CAMERA_LABELS = {0: "Left Above Rim", 1: "Broadcast", 2: "Right Above Rim"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def zrot(deg: float) -> np.ndarray:
    a = np.deg2rad(float(deg))
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float64)


def rigid_orbit_w2c(M0: np.ndarray, pivot: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rigidly orbit a calibrated physical camera about world +Z.

    If U0 is camera-to-world orientation, U(theta)=Q U0 and
    C(theta)=pivot+Q(C0-pivot).  Therefore R(theta)=R0 Q^T.
    This exactly reproduces M0 at 0 degrees and keeps the camera's relation to
    the focal action rigid while moving through a true 3D arc.
    """
    M0 = np.asarray(M0, np.float64)
    R0 = M0[:3, :3]
    C0 = np.linalg.inv(M0)[:3, 3]
    Q = zrot(angle_deg)
    C = pivot + Q @ (C0 - pivot)
    R = R0 @ Q.T
    t = -R @ C
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def subset_row(meta: dict, time_index: int, camera_ids: tuple[int, ...]) -> dict:
    ids = [int(x) for x in meta["cam_id"][time_index]]
    pos = {cid: i for i, cid in enumerate(ids)}
    keep = [pos[cid] for cid in camera_ids]
    return {
        "w": int(meta["w"]),
        "h": int(meta["h"]),
        "fn": [[meta["fn"][time_index][i] for i in keep]],
        "k": [[meta["k"][time_index][i] for i in keep]],
        "w2c": [[meta["w2c"][time_index][i] for i in keep]],
        "cam_id": [[ids[i] for i in keep]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="extracted v32e dataset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--freeze-index", type=int, default=6)
    args = ap.parse_args()

    src, out = args.input, args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ims").mkdir(exist_ok=True)

    backend = json.loads((src / "v32d_backend.json").read_text())
    canon = backend.get("v32e_camera_depth_canonicalization", {})
    if canon.get("status") != "PASS_V32E_4DGS_CAMERA_DEPTH_CANONICAL":
        raise RuntimeError("input is not a passed v32e depth-canonical dataset")
    meta = json.loads((src / "train_meta.json").read_text())
    fi = int(args.freeze_index)
    if fi < 0 or fi >= len(meta["fn"]):
        raise RuntimeError(f"bad freeze index {fi}")
    ids = [int(x) for x in meta["cam_id"][fi]]
    if ids != [0, 1, 2]:
        raise RuntimeError(f"expected solved camera order [0,1,2], got {ids}")
    if not all("t+00_" in fn for fn in meta["fn"][fi]):
        raise RuntimeError(f"freeze index does not resolve to t+00: {meta['fn'][fi]}")

    # Copy only the three real synchronized source images needed for the static model.
    source_hashes = {}
    for fn in meta["fn"][fi]:
        p = src / "ims" / fn
        q = out / "ims" / fn
        shutil.copy2(p, q)
        source_hashes[fn] = sha256(q)

    shutil.copy2(src / "init_pt_cld.npz", out / "init_pt_cld.npz")

    train_all = subset_row(meta, fi, (0, 1, 2))
    train_rar = subset_row(meta, fi, (0, 1))
    test_rar = subset_row(meta, fi, (2,))
    (out / "train_meta.json").write_text(json.dumps(train_all, indent=2))
    (out / "test_meta.json").write_text(json.dumps(train_all, indent=2))
    (out / "train_meta_holdout_right_above_rim.json").write_text(json.dumps(train_rar, indent=2))
    (out / "test_meta_holdout_right_above_rim.json").write_text(json.dumps(test_rar, indent=2))

    # Calibrated 0..25 degree camera arc, anchored exactly at Left Above Rim.
    K0 = np.asarray(train_all["k"][0][0], np.float64)
    M0 = np.asarray(train_all["w2c"][0][0], np.float64)
    pivot = np.asarray(backend["focal_player_seed"]["center_world_cm"], np.float64) / 100.0
    orbit_ms = [rigid_orbit_w2c(M0, pivot, a) for a in ANGLES]

    # Exact 0-degree invariants.
    if not np.allclose(orbit_ms[0], M0, atol=1e-10, rtol=0.0):
        raise RuntimeError("0-degree orbit pose does not exactly reproduce Left Above Rim")
    c0 = np.linalg.inv(M0)[:3, 3]
    radii = [float(np.linalg.norm(np.linalg.inv(M)[:3, 3] - pivot)) for M in orbit_ms]
    if max(abs(r - radii[0]) for r in radii) > 1e-9:
        raise RuntimeError("orbit radius is not rigid")

    # The PanopticSports loader requires a file name for each camera even when
    # rendering.  Copy the real freeze image byte-for-byte as a non-GT loader
    # placeholder.  Keeping all six virtual views in ONE metadata row forces
    # time=0 for every camera.
    anchor_fn = train_all["fn"][0][0]
    anchor = out / "ims" / anchor_fn
    orbit_fns, placeholder_hashes = [], {}
    for i, a in enumerate(ANGLES):
        fn = f"orbit_placeholder_{i:02d}_{int(a):02d}deg.png"
        dst = out / "ims" / fn
        shutil.copy2(anchor, dst)
        if sha256(dst) != source_hashes[anchor_fn]:
            raise RuntimeError("orbit placeholder is not a byte-identical source copy")
        orbit_fns.append(fn)
        placeholder_hashes[fn] = sha256(dst)

    orbit_meta = {
        "w": int(train_all["w"]),
        "h": int(train_all["h"]),
        "fn": [orbit_fns],
        "k": [[K0.tolist() for _ in ANGLES]],
        "w2c": [[M.tolist() for M in orbit_ms]],
        "cam_id": [[100 + i for i in range(len(ANGLES))]],
    }
    (out / "test_meta_orbit_0_25.json").write_text(json.dumps(orbit_meta, indent=2))

    qa = {
        "version": "v32f_freeze_only_static_gaussian",
        "source_backend_version": backend.get("version"),
        "freeze_index": fi,
        "freeze_files": train_all["fn"][0],
        "training_camera_ids": train_all["cam_id"][0],
        "heldout_test_camera": "Right Above Rim",
        "heldout_training_camera_ids": train_rar["cam_id"][0],
        "heldout_test_camera_ids": test_rar["cam_id"][0],
        "time_steps": 1,
        "real_training_images": 3,
        "source_resolution": [int(train_all["w"]), int(train_all["h"])],
        "source_hashes_sha256": source_hashes,
        "orbit_angles_deg": list(ANGLES),
        "orbit_pivot_world_m": pivot.tolist(),
        "orbit_radius_m": radii[0],
        "zero_degree_w2c_max_abs_error": float(np.max(np.abs(orbit_ms[0] - M0))),
        "max_orbit_radius_error_m": float(max(abs(r - radii[0]) for r in radii)),
        "orbit_loader_placeholders_are_ground_truth": False,
        "orbit_placeholder_hashes_sha256": placeholder_hashes,
        "native_pixels_preserved": True,
        "generated_rgb": False,
        "resized": False,
        "warped": False,
        "interpolated": False,
        "upscaled": False,
        "point_cloud_modified": False,
        "camera_solution_modified": False,
        "status": "PASS_V32F_FREEZE_ONLY_DATASET_AND_ORBIT",
    }
    (out / "v32f_freeze_only_qa.json").write_text(json.dumps(qa, indent=2))
    (out / "v32f_backend.json").write_text(json.dumps({
        **backend,
        "version": "v32f_freeze_only_static_gaussian",
        "v32f": qa,
    }, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
