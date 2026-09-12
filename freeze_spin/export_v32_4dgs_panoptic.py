from __future__ import annotations

"""Export the v32 Stage-A package to a 4DGaussians PanopticSports-style dataset.

Why this backend:
- unlike the v12-v31 focal anatomy mesh, it is a dynamic *whole-scene* Gaussian
  representation, so every on-court player can be learned jointly;
- 4DGaussians already contains a PanopticSports reader that accepts calibrated
  multi-camera video through train_meta.json/test_meta.json plus init_pt_cld.npz;
- our three accepted NBA cameras are already calibrated, so we write exact K/w2c
  directly rather than asking COLMAP to re-solve them.

The exporter does not train or synthesize anything. It only reformats native
960x540 official source frames and deterministic camera/scene priors for a later
CUDA training job.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ("Left Above Rim", "Broadcast", "Right Above Rim")
CAM_IDS = {"Left Above Rim": 0, "Broadcast": 1, "Right Above Rim": 2}
W, H = 960, 540


def safe(label: str) -> str:
    return label.replace(" ", "_")


def w2c_from_manifest(cam: dict) -> np.ndarray:
    R = np.asarray(cam["R_world_to_camera"], np.float64)
    C_cm = np.asarray(cam["C_world_cm"], np.float64)
    C_m = C_cm / 100.0
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = -(R @ C_m)
    return M


def project(P_m: np.ndarray, cam: dict):
    K = np.asarray(cam["K_px"], np.float64)
    M = w2c_from_manifest(cam)
    X = np.concatenate([P_m, np.ones((len(P_m), 1), np.float64)], axis=1)
    Xc = (M @ X.T).T[:, :3]
    q = (K @ Xc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    ok = (
        np.isfinite(uv).all(axis=1)
        & (Xc[:, 2] > 0.05)
        & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1)
        & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    )
    return uv, ok


def source_image_for(stage_a: Path, label: str):
    rows = sorted(stage_a.glob(f"v32_chosen_{safe(label)}_frame*.png"))
    if len(rows) != 1:
        raise RuntimeError(f"chosen source ambiguity for {label}: {rows}")
    im = cv2.imread(str(rows[0]), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (H, W):
        raise RuntimeError(f"bad chosen image {rows[0]}")
    return im


def court_initial_points(manifest: dict, stage_a: Path):
    # Regulation floor region around the active basket, in the accepted metric
    # world frame.  Keep the initialization sparse; the Gaussian model densifies.
    xs = np.arange(-3.0, 16.01, 0.30, dtype=np.float64)
    ys = np.arange(-8.0, 8.01, 0.30, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    P = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    images = {lab: source_image_for(stage_a, lab) for lab in CAMERAS}
    samples = []
    valids = []
    for lab in CAMERAS:
        uv, ok = project(P, manifest["cameras"][lab])
        c = np.zeros((len(P), 3), np.float32)
        ii = np.where(ok)[0]
        if len(ii):
            u = np.clip(np.rint(uv[ii, 0]).astype(int), 0, W - 1)
            v = np.clip(np.rint(uv[ii, 1]).astype(int), 0, H - 1)
            # OpenCV BGR -> RGB, normalized.
            c[ii] = images[lab][v, u][:, ::-1].astype(np.float32) / 255.0
        samples.append(c)
        valids.append(ok)
    S = np.stack(samples, axis=0)
    V = np.stack(valids, axis=0)
    rgb = np.zeros((len(P), 3), np.float32)
    keep = np.any(V, axis=0)
    for i in np.where(keep)[0]:
        rgb[i] = np.median(S[V[:, i], i], axis=0)
    return P[keep].astype(np.float32), rgb[keep]


def crop_color(stage_a: Path, obs: dict):
    p = stage_a / obs["rgba_crop"]
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return np.asarray([0.45, 0.45, 0.45], np.float32)
    alpha = im[:, :, 3] > 0
    if int(alpha.sum()) < 10:
        return np.asarray([0.45, 0.45, 0.45], np.float32)
    pix = im[:, :, :3][alpha][:, ::-1].astype(np.float32) / 255.0
    return np.median(pix, axis=0).astype(np.float32)


def player_initial_points(manifest: dict, stage_a: Path):
    pts = []
    cols = []
    tracks = manifest["all_on_court_people"]["loose_world_union_tracks"]
    for tr in tracks:
        xy = np.asarray(tr["centroid_xy_cm"], np.float64) / 100.0
        colors = [crop_color(stage_a, o) for o in tr.get("observations", [])]
        col = np.median(np.stack(colors), axis=0) if colors else np.asarray([.45, .45, .45], np.float32)
        # A very sparse person-shaped initialization, not final body geometry.
        # 4DGS learns/densifies the actual dynamic appearance from images.
        for z in np.arange(0.10, 2.21, 0.18):
            # Taper the cloud near head/feet.
            r = 0.10 if z < 0.25 else (0.12 if z > 1.85 else 0.22)
            n = 8 if r >= 0.2 else 5
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
                pts.append([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), z])
                cols.append(col)
    if not pts:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    return np.asarray(pts, np.float32), np.asarray(cols, np.float32)


def find_ball_center_cm(ball: dict | None):
    if not isinstance(ball, dict):
        return None
    candidates = ["center_world_cm", "world_cm", "center_cm", "xyz_cm", "triangulated_center_cm"]
    for k in candidates:
        v = ball.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            a = np.asarray(v[:3], np.float64)
            if np.isfinite(a).all():
                return a
    for v in ball.values():
        if isinstance(v, dict):
            x = find_ball_center_cm(v)
            if x is not None:
                return x
    return None


def ball_initial_points(manifest: dict):
    b = find_ball_center_cm(manifest.get("ball", {}).get("carried_three_view_solution_from_v31"))
    if b is None:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    c = b / 100.0
    # NBA ball radius ~0.12 m; initialize a tiny orange sphere.
    pts = []
    for phi in np.linspace(0.25, np.pi - 0.25, 7):
        for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            pts.append(c + 0.12 * np.asarray([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)]))
    rgb = np.tile(np.asarray([[0.78, 0.34, 0.08]], np.float32), (len(pts), 1))
    return np.asarray(pts, np.float32), rgb


def make_init_cloud(manifest: dict, stage_a: Path, out: Path):
    pc, cc = court_initial_points(manifest, stage_a)
    pp, cp = player_initial_points(manifest, stage_a)
    pb, cb = ball_initial_points(manifest)
    xyz = np.concatenate([pc, pp, pb], axis=0)
    rgb = np.concatenate([cc, cp, cb], axis=0)
    data = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    np.savez_compressed(out / "init_pt_cld.npz", data=data)
    return {"total_points": int(len(data)), "court_points": int(len(pc)), "player_seed_points": int(len(pp)), "ball_seed_points": int(len(pb))}


def copy_images_and_build_times(manifest: dict, stage_a: Path, out: Path):
    ims = out / "ims"
    ims.mkdir(parents=True, exist_ok=True)
    # map relative frame -> camera row; all bursts are required to be synchronized
    per_cam = {lab: {int(r["relative_frame"]): r for r in manifest["burst"][lab]} for lab in CAMERAS}
    rels = sorted(set.intersection(*[set(x.keys()) for x in per_cam.values()]))
    times = []
    for rel in rels:
        fns, ks, w2cs, ids = [], [], [], []
        for lab in CAMERAS:
            src = stage_a / per_cam[lab][rel]["file"]
            fn = f"t{rel:+03d}_{safe(lab)}.png"
            shutil.copy2(src, ims / fn)
            fns.append(fn)
            ks.append(manifest["cameras"][lab]["K_px"])
            w2cs.append(w2c_from_manifest(manifest["cameras"][lab]).tolist())
            ids.append(CAM_IDS[lab])
        times.append({"relative_frame": int(rel), "fn": fns, "k": ks, "w2c": w2cs, "cam_id": ids})
    return times


def meta_payload(times, keep_cam_ids=None):
    fn, k, w2c, cam_id = [], [], [], []
    for t in times:
        ids = list(range(len(t["cam_id"]))) if keep_cam_ids is None else [i for i, cid in enumerate(t["cam_id"]) if cid in keep_cam_ids]
        fn.append([t["fn"][i] for i in ids])
        k.append([t["k"][i] for i in ids])
        w2c.append([t["w2c"][i] for i in ids])
        cam_id.append([t["cam_id"][i] for i in ids])
    return {"w": W, "h": H, "fn": fn, "k": k, "w2c": w2c, "cam_id": cam_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = json.loads((args.stage_a / "v32_scene_manifest.json").read_text())
    if m["resolution"] != [W, H]:
        raise RuntimeError(f"v32 native resolution violation: {m['resolution']}")

    times = copy_images_and_build_times(m, args.stage_a, args.out)
    all3 = meta_payload(times)
    (args.out / "train_meta.json").write_text(json.dumps(all3, indent=2))
    # For final-model smoke rendering we keep the same three cameras in test_meta.
    # Held-out-camera manifests below are the actual generalization gates.
    (args.out / "test_meta.json").write_text(json.dumps(all3, indent=2))

    holdouts = {}
    for lab in CAMERAS:
        hid = CAM_IDS[lab]
        train = meta_payload(times, keep_cam_ids={x for x in CAM_IDS.values() if x != hid})
        test = meta_payload(times, keep_cam_ids={hid})
        stem = safe(lab).lower()
        tr = args.out / f"train_meta_holdout_{stem}.json"
        te = args.out / f"test_meta_holdout_{stem}.json"
        tr.write_text(json.dumps(train, indent=2)); te.write_text(json.dumps(test, indent=2))
        holdouts[lab] = {"train_meta": tr.name, "test_meta": te.name, "train_camera_ids": sorted({x for x in CAM_IDS.values() if x != hid}), "test_camera_id": hid}

    cloud = make_init_cloud(m, args.stage_a, args.out)
    backend = {
        "backend": "hustvl/4DGaussians PanopticSports input contract",
        "reason": "dynamic calibrated multi-camera whole-scene Gaussian representation; preserves all court players rather than focal-only human mesh",
        "native_resolution": [W, H],
        "time_steps": int(len(times)),
        "source_cameras": list(CAMERAS),
        "training_images": int(len(times) * len(CAMERAS)),
        "freeze_relative_frame": 0,
        "initial_point_cloud": cloud,
        "held_out_camera_manifests": holdouts,
        "gpu_requirement": "CUDA GPU; actual 4D Gaussian optimization is intentionally not attempted on CPU GitHub-hosted runner",
        "evaluation_order": [
            "train three held-out-camera models (2 source cameras -> 1 real camera ground truth)",
            "reject backend if people disappear, limbs fail, or sharp freeze identity is not retained",
            "only after held-out pass, train all-three-camera model and render 0/5/10/15/20/25 degree arc",
        ],
        "upscale": False,
        "uhd": False,
    }
    (args.out / "v32_4dgs_backend.json").write_text(json.dumps(backend, indent=2))
    print(json.dumps(backend, indent=2), flush=True)


if __name__ == "__main__":
    main()
