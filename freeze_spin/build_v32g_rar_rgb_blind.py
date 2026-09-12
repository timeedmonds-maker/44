from __future__ import annotations

"""Build a clean known-pose held-out-RGB dataset for the v32 freeze frame.

V32f was useful as a GPU package, but its initial point cloud and orbit pivot
inherit target-camera subject geometry from earlier three-view work.  This
builder deliberately does not copy that initializer.  Camera 2 (Right Above
Rim) is retained only as a known evaluation pose and RGB ground truth.

The training initializer uses only:
  * authoritative NBA metric court/basket geometry,
  * calibrated camera 0/1 poses/intrinsics,
  * camera 0/1 RGB.

All 384x448 crops are chosen from the projected regulation rim.  No player,
ball, detector box, target-camera pixel, or target-derived subject geometry is
used to decide a crop or seed a Gaussian.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from nba_geometry import (
    BACKBOARD_FACE_X_NEAR_CM,
    COURT_CENTER_Y_CM,
    RIM_INSIDE_RADIUS_CM,
    basket,
)

CROP_W, CROP_H = 384, 448
CAM_LABELS = {0: "Left Above Rim", 1: "Broadcast", 2: "Right Above Rim"}

_b = basket("near")
# V32 camera calibration uses the same baseline-X metric frame as nba_geometry,
# with Y recentered so the court centreline is zero.
RIM = np.array(
    [_b.rim_center_x_cm, _b.y_cm - COURT_CENTER_Y_CM, _b.z_cm],
    dtype=np.float64,
) / 100.0
RIM_R = RIM_INSIDE_RADIUS_CM / 100.0
BOARD_X = BACKBOARD_FACE_X_NEAR_CM / 100.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def project(K: np.ndarray, M: np.ndarray, p: np.ndarray):
    q = M @ np.r_[p, 1.0]
    uv = K @ q[:3]
    return uv[:2] / uv[2], float(q[2])


def camera_center(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)[:3, 3]


def metric_crop(K: np.ndarray, M: np.ndarray, w: int, h: int):
    uv, depth = project(K, M, RIM)
    # Put the rim at 28% of crop height, leaving most pixels for the court/action
    # below it.  The constants are protocol constants, not image-content choices.
    x0 = int(round(float(uv[0]) - CROP_W / 2.0))
    y0 = int(round(float(uv[1]) - CROP_H * 0.28))
    x0 = max(0, min(w - CROP_W, x0))
    y0 = max(0, min(h - CROP_H, y0))
    return [x0, y0, x0 + CROP_W, y0 + CROP_H], uv.tolist(), depth


def crop_K(K: np.ndarray, box: list[int]) -> np.ndarray:
    out = K.copy()
    out[0, 2] -= box[0]
    out[1, 2] -= box[1]
    return out


def ray_world(K: np.ndarray, M: np.ndarray, u: float, v: float):
    dcam = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
    R = M[:3, :3]
    C = camera_center(M)
    d = R.T @ dcam
    d /= np.linalg.norm(d)
    return C, d


def ray_box(C: np.ndarray, d: np.ndarray, bmin: np.ndarray, bmax: np.ndarray):
    inv = np.where(np.abs(d) > 1e-12, 1.0 / d, 1e30)
    t0 = (bmin - C) * inv
    t1 = (bmax - C) * inv
    lo = max(float(np.max(np.minimum(t0, t1))), 0.0)
    hi = float(np.min(np.maximum(t0, t1)))
    return (lo, hi) if hi > lo else None


def sample_rgb(im: np.ndarray, u: float, v: float) -> np.ndarray:
    x = int(np.clip(round(float(u)), 0, im.shape[1] - 1))
    y = int(np.clip(round(float(v)), 0, im.shape[0] - 1))
    return im[y, x, :3].astype(np.float32) / 255.0


def source_colour(p: np.ndarray, cams, ims) -> np.ndarray:
    colours = []
    for cid, (K, M) in cams.items():
        uv, z = project(K, M, p)
        if z > 0 and 0 <= uv[0] < CROP_W and 0 <= uv[1] < CROP_H:
            colours.append(sample_rgb(ims[cid], *uv))
    if colours:
        return np.mean(colours, axis=0)
    return np.array([0.45, 0.32, 0.55], dtype=np.float32)


def find_full_frame(root: Path, label: str) -> Path:
    token = label.replace(" ", "_")
    matches = sorted(root.glob(f"v32_chosen_{token}_frame*.png"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one full frame for {label}, got {matches}")
    return matches[0]


def zrot(deg: float) -> np.ndarray:
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def orbit(M0: np.ndarray, pivot: np.ndarray, deg: float) -> np.ndarray:
    R0 = M0[:3, :3]
    C0 = camera_center(M0)
    Q = zrot(deg)
    C = pivot + Q @ (C0 - pivot)
    R = R0 @ Q.T
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = -R @ C
    return M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v32f", type=Path, required=True)
    ap.add_argument("--full-frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = args.v32f
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "ims").mkdir(parents=True)

    all_meta = json.load(open(src / "train_meta.json"))
    backend = json.load(open(src / "v32f_backend.json"))
    if all_meta.get("cam_id") != [[0, 1, 2]]:
        raise RuntimeError("expected v32f camera ids [[0,1,2]]")

    w2cs = {
        int(cid): np.asarray(all_meta["w2c"][0][i], dtype=np.float64)
        for i, cid in enumerate(all_meta["cam_id"][0])
    }
    # Read only the calibrated full-frame intrinsics from v32f_backend.  The
    # leaky focal-player seed, ball solution, old point cloud and old orbit pivot
    # are never consumed by this builder.
    full_K = {
        cid: np.asarray(backend["full_frame_K_px"][label], dtype=np.float64)
        for cid, label in CAM_LABELS.items()
    }

    source_images: dict[int, np.ndarray] = {}
    crop_Ks: dict[int, np.ndarray] = {}
    crop_info = {}
    crop_hashes = {}

    for cid in (0, 1, 2):
        label = CAM_LABELS[cid]
        full_path = find_full_frame(args.full_frames, label)
        im = Image.open(full_path).convert("RGB")
        w, h = im.size
        if (w, h) != (960, 540):
            raise RuntimeError(f"unexpected full-frame size for {label}: {(w,h)}")
        box, rim_uv, rim_depth = metric_crop(full_K[cid], w2cs[cid], w, h)
        x0, y0, x1, y1 = box
        cropped = im.crop((x0, y0, x1, y1))
        fn = f"t+00_{label.replace(' ', '_')}.png"
        dst = out / "ims" / fn
        cropped.save(dst, optimize=False)
        crop_Ks[cid] = crop_K(full_K[cid], box)
        crop_hashes[label] = sha256(dst)
        crop_info[label] = {
            "xyxy": box,
            "rim_full_uv": rim_uv,
            "rim_depth_m": rim_depth,
            "derived_from": "NBA metric rim + calibrated camera only",
        }
        # Camera 2 is intentionally NOT converted to an array.  Its RGB bytes
        # are materialized only as ground truth and are unavailable to seeding.
        if cid in (0, 1):
            source_images[cid] = np.asarray(cropped)

    source_cams = {cid: (crop_Ks[cid], w2cs[cid]) for cid in (0, 1)}
    points = []
    categories = []

    # Regulation floor around the near basket. Colours are sampled only from the
    # two training views; geometry is target-independent.
    for x in np.arange(-0.8, 6.01, 0.16):
        for y in np.arange(-4.0, 4.01, 0.16):
            p = np.array([x, y, 0.0], dtype=np.float64)
            points.append(np.r_[p, source_colour(p, source_cams, source_images)])
            categories.append("floor")

    # Regulation backboard plane.  NBA board is 3.5 ft tall; near-basket board
    # face X is authoritative in nba_geometry.  The 9.5..13 ft vertical span is
    # the regulation board placement around a 10 ft rim.
    for y in np.arange(-0.9144, 0.9145, 0.10):
        for z in np.arange(2.8956, 3.9625, 0.10):
            p = np.array([BOARD_X, y, z], dtype=np.float64)
            points.append(np.r_[p, source_colour(p, source_cams, source_images)])
            categories.append("backboard")

    # Regulation rim tube.  Even its initial colour is taken from source views.
    for zoff in (-0.018, 0.0, 0.018):
        for th in np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False):
            p = RIM + np.array([RIM_R * np.cos(th), RIM_R * np.sin(th), zoff])
            points.append(np.r_[p, source_colour(p, source_cams, source_images)])
            categories.append("rim")

    # Target-blind action volume.  Every sample is back-projected from a source
    # pixel and clipped to a fixed metric box around the basket.  This is a broad
    # initializer, not a recovered target-view skeleton or target-view visual hull.
    bmin = np.array([0.20, -1.80, 0.15], dtype=np.float64)
    bmax = np.array([3.20, 1.80, 3.90], dtype=np.float64)
    ray_counts = {}
    corners = np.asarray(
        [[x, y, z] for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])]
    )
    for cid in (0, 1):
        K, M = source_cams[cid]
        im = source_images[cid]
        projected = [project(K, M, p)[0] for p in corners if project(K, M, p)[1] > 0]
        uvs = np.asarray(projected)
        xlo = max(0, int(np.floor(uvs[:, 0].min())))
        xhi = min(CROP_W - 1, int(np.ceil(uvs[:, 0].max())))
        ylo = max(0, int(np.floor(uvs[:, 1].min())))
        yhi = min(CROP_H - 1, int(np.ceil(uvs[:, 1].max())))
        count = 0
        for v in range(ylo, yhi + 1, 7):
            for u in range(xlo, xhi + 1, 7):
                C, d = ray_world(K, M, u, v)
                hit = ray_box(C, d, bmin, bmax)
                if not hit:
                    continue
                lo, hi = hit
                colour = sample_rgb(im, u, v)
                for fraction in (0.20, 0.40, 0.60, 0.80):
                    p = C + (lo + fraction * (hi - lo)) * d
                    points.append(np.r_[p, colour])
                    categories.append(f"source_ray_cam{cid}")
                    count += 1
        ray_counts[str(cid)] = count

    data = np.asarray(points, dtype=np.float32)
    if not np.isfinite(data).all() or data.shape[1] != 6:
        raise RuntimeError("invalid point cloud")
    # Deterministic 2 cm voxel de-duplication.
    key = np.round(data[:, :3] / 0.02).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    data = data[np.sort(idx)]
    np.savez_compressed(out / "init_pt_cld.npz", data=data)

    def make_meta(ids):
        fns, Ks, Ms = [], [], []
        for cid in ids:
            fns.append(f"t+00_{CAM_LABELS[cid].replace(' ', '_')}.png")
            Ks.append(crop_Ks[cid].tolist())
            Ms.append(w2cs[cid].tolist())
        return {"w": CROP_W, "h": CROP_H, "fn": [fns], "k": [Ks], "w2c": [Ms], "cam_id": [ids]}

    # Make the safe two-view setup the canonical metadata.  The all-three file is
    # explicitly labelled post-gate so an accidental default launch cannot train
    # on the held-out target.
    train = make_meta([0, 1])
    test = make_meta([2])
    all_three = make_meta([0, 1, 2])
    (out / "train_meta.json").write_text(json.dumps(train, indent=2))
    (out / "test_meta.json").write_text(json.dumps(test, indent=2))
    (out / "train_meta_all_three_POST_GATE_ONLY.json").write_text(json.dumps(all_three, indent=2))

    # Rebuild the prospective 61-view camera path about the regulation rim, not
    # the contaminated v32f subject-centre pivot.  Rendering remains post-gate.
    angles = np.linspace(0.0, 25.0, 61)
    K0, M0 = crop_Ks[0], w2cs[0]
    Ms = [orbit(M0, RIM, float(deg)) for deg in angles]
    anchor = out / "ims" / train["fn"][0][0]
    orbit_fns = []
    for i, deg in enumerate(angles):
        fn = f"dense_orbit_placeholder_{i:03d}_{deg:06.3f}deg.png"
        shutil.copy2(anchor, out / "ims" / fn)
        orbit_fns.append(fn)
    orbit_meta = {
        "w": CROP_W,
        "h": CROP_H,
        "fn": [orbit_fns],
        "k": [[K0.tolist() for _ in angles]],
        "w2c": [[M.tolist() for M in Ms]],
        "cam_id": [[1000 + i for i in range(len(angles))]],
    }
    (out / "test_meta_orbit_dense_0_25_POST_GATE_ONLY.json").write_text(json.dumps(orbit_meta, indent=2))

    for fn in ("v32f_freeze_static_4dgs_config.py", "evaluate_v32d_holdout.py"):
        shutil.copy2(src / fn, out / fn)

    counts = {}
    for category in categories:
        counts[category] = counts.get(category, 0) + 1
    qa = {
        "version": "v32g_known_pose_rar_rgb_blind",
        "status": "PASS_V32G_RAR_RGB_BLIND_PACKAGE",
        "test_definition": "known-pose held-out-RGB novel-view synthesis",
        "train_camera_ids": [0, 1],
        "heldout_camera_id": 2,
        "heldout_rgb_materialized_as_evaluation_gt": True,
        "heldout_rgb_used_for_crop_selection": False,
        "heldout_rgb_used_for_initializer": False,
        "heldout_rgb_used_for_training": False,
        "heldout_subject_geometry_used_for_initializer": False,
        "heldout_ball_geometry_used_for_initializer": False,
        "heldout_subject_colour_used_for_initializer": False,
        "heldout_camera_pose_used_for_evaluation_only": True,
        "initializer_sources": [
            "NBA regulation court/basket metric geometry",
            "camera 0/1 calibrated poses/intrinsics",
            "camera 0/1 RGB only",
        ],
        "forbidden_upstream_sources": [
            "v31 Broadcast+RAR focal skeleton",
            "v31 three-view ball solution",
            "v32b all-camera world-union tracks",
            "v32f init_pt_cld.npz",
            "v32f focal_player_seed orbit pivot",
        ],
        "metric_rim_world_m": RIM.tolist(),
        "metric_action_box_m": {"min": bmin.tolist(), "max": bmax.tolist()},
        "metric_crop_rule": "384x448 from projected NBA metric rim; rim at 28% crop height; clamp only to 960x540 bounds",
        "crops": crop_info,
        "crop_sha256": crop_hashes,
        "init_points": int(len(data)),
        "pre_dedupe_category_counts": counts,
        "ray_sample_points_before_dedupe": ray_counts,
        "init_bounds_m": {"min": data[:, :3].min(0).tolist(), "max": data[:, :3].max(0).tolist()},
        "orbit": {
            "frames": 61,
            "degrees": [0.0, 25.0],
            "pivot_world_m": RIM.tolist(),
            "anchor_camera_id": 0,
            "rendering_allowed_only_after_visual_holdout_pass": True,
        },
        "note": "Target RGB is evaluation ground truth only. Its calibrated pose/intrinsics are intentionally known at evaluation time.",
    }
    (out / "v32g_provenance_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
