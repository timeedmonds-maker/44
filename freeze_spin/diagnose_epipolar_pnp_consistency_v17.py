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

import build_portable_moge_pnp_freeview_v12 as base


def rot_angle_deg(R: np.ndarray) -> float:
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(c))


def vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def norm_points(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(uv, np.float64).reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)


def sampson_px(E: np.ndarray, n1: np.ndarray, n2: np.ndarray, fpx: float) -> np.ndarray:
    x1 = np.column_stack([n1, np.ones(len(n1))])
    x2 = np.column_stack([n2, np.ones(len(n2))])
    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T
    num = np.sum(x2 * Ex1, axis=1) ** 2
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12)) * fpx


def diagnose(ref: dict, tgt: dict) -> dict:
    static_ref = (~ref["dynamic"]) & ref["valid"]
    static_tgt = (~tgt["dynamic"]) & tgt["valid"]
    _, uv1, uv2 = base.sift_matches(
        ref["image"], tgt["image"], static_ref, static_tgt
    )
    out = {"static_sift_matches": int(len(uv1))}
    if len(uv1) < 35:
        out.update({"passed": False, "reason": "insufficient static SIFT matches"})
        return out

    # Existing forward PnP uses only reference-view MoGe depth, not target depth.
    pnp = base.solve_target_from_reference(ref, tgt)
    out["forward_pnp"] = {
        k: v
        for k, v in pnp.items()
        if k not in ("R", "t", "C")
    }
    if "R" not in pnp or "t" not in pnp:
        out.update({"passed": False, "reason": "forward PnP did not produce a pose"})
        return out

    n1 = norm_points(uv1, ref["K"])
    n2 = norm_points(uv2, tgt["K"])
    avg_f = float(np.mean([ref["K"][0, 0], ref["K"][1, 1], tgt["K"][0, 0], tgt["K"][1, 1]]))
    threshold_norm = 2.0 / max(avg_f, 1.0)
    E, mask_e = cv2.findEssentialMat(
        n1,
        n2,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.9999,
        threshold=threshold_norm,
        maxIters=10000,
    )
    if E is None or mask_e is None:
        out.update({"passed": False, "reason": "essential matrix failed"})
        return out
    if E.shape[0] > 3:
        E = E[:3, :3]
    mask_e = mask_e.reshape(-1).astype(bool)
    if int(mask_e.sum()) < 25:
        out.update({
            "passed": False,
            "reason": "essential matrix insufficient inliers",
            "essential_inliers": int(mask_e.sum()),
        })
        return out

    _, Re, te, mask_pose = cv2.recoverPose(
        E,
        n1,
        n2,
        np.eye(3),
        mask=mask_e.astype(np.uint8).reshape(-1, 1),
    )
    pose_inliers = int((mask_pose.reshape(-1) > 0).sum())
    Rf = np.asarray(pnp["R"], np.float64)
    tf = np.asarray(pnp["t"], np.float64).reshape(3)
    rot_delta = rot_angle_deg(Re @ Rf.T)
    trans_delta = min(vec_angle_deg(te.reshape(3), tf), vec_angle_deg(-te.reshape(3), tf))
    serr = sampson_px(E, n1[mask_e], n2[mask_e], avg_f)

    # A low epipolar residual plus agreement with reference-depth PnP means
    # the camera pose itself is supported even if the target monocular depth
    # cloud is not mutually rigid with the reference cloud.
    passed = bool(
        int(mask_e.sum()) >= 25
        and pose_inliers >= 25
        and float(np.median(serr)) <= 1.5
        and float(np.percentile(serr, 95)) <= 3.5
        and rot_delta <= 5.0
        and trans_delta <= 25.0
    )
    out.update({
        "passed": passed,
        "reason": "epipolar and reference-depth PnP poses agree" if passed else "epipolar/PnP pose disagreement",
        "essential_inliers": int(mask_e.sum()),
        "recover_pose_inliers": pose_inliers,
        "essential_inlier_fraction": float(mask_e.mean()),
        "median_sampson_px": float(np.median(serr)),
        "p95_sampson_px": float(np.percentile(serr, 95)),
        "pnp_vs_essential_rotation_deg": float(rot_delta),
        "pnp_vs_essential_translation_direction_deg": float(trans_delta),
        "essential_R_ref_to_target": Re.tolist(),
        "essential_t_direction_ref_to_target": te.reshape(3).tolist(),
        "pnp_R_ref_to_target": Rf.tolist(),
        "pnp_t_ref_to_target": tf.tolist(),
        "policy": "Target MoGe depth is not used by this pose-consistency gate; it tests whether wide-baseline camera orientation/translation direction are independently supported by static epipolar geometry and reference-depth PnP.",
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="In Arena")
    ap.add_argument("--targets", nargs="+", default=["Broadcast", "Other Broadcast", "Right Above Rim"])
    ap.add_argument("--tokens", type=int, default=1600)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = [args.reference] + args.targets
    images = {}
    for label in labels:
        p = args.locked_dir / f"{label.replace(' ', '_')}_apex.png"
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"Missing locked apex frame: {p}")
        if im.shape[:2] != (540, 960):
            raise RuntimeError(f"Expected native 960x540 frame for {label}: {im.shape}")
        images[label] = im

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    detector = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, progress=True
    ).eval()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()

    views = {}
    for label, image in images.items():
        dynamic, balls = base.detect_dynamic_and_ball(detector, image)
        depth, points, valid, K, Kn = base.moge_infer(moge, image, args.tokens)
        views[label] = {
            "label": label,
            "image": image,
            "dynamic": dynamic,
            "balls": balls,
            "depth": depth,
            "points": points,
            "valid": valid,
            "K": K,
            "Kn": Kn,
        }

    ref = views[args.reference]
    results = {}
    for label in args.targets:
        results[label] = diagnose(ref, views[label])
        print(label, json.dumps(results[label], indent=2), flush=True)

    payload = {
        "prototype": "epipolar_vs_reference_depth_pnp_v17",
        "event": {"game_id": "0022500301", "event_id": 489},
        "state": "accepted_ball_apex",
        "reference": args.reference,
        "targets": args.targets,
        "native_resolution": [960, 540],
        "results": results,
        "decision_rule": "Do not promote a wide-baseline camera cloud from this test. A pass only promotes its pose as a candidate for a new depth/geometry representation.",
    }
    (args.out / "epipolar_pnp_consistency_v17.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
