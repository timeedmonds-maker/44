from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import build_portable_moge_pnp_freeview_v12 as base


def rotation_angle_deg(R: np.ndarray) -> float:
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(c))


def solve_target_from_reference_reciprocal(ref, tgt):
    """v13 camera gate.

    Keep v12's forward static-SIFT + reference-MoGe PnP proof, but do not
    accept/reject a distinct camera merely because its independent MoGe scale
    happens to sit on one side of the old 2.8 absolute ceiling.

    Instead, a camera with a non-unit MoGe scale must prove that the target
    MoGe cloud, after applying the robust forward scale, solves back to the
    reference and closes the SE(3) transform. This is a geometry check, not a
    relaxed scale gate.
    """
    static_ref = (~ref["dynamic"]) & ref["valid"]
    static_tgt = (~tgt["dynamic"]) & tgt["valid"]
    _, uv_ref, uv_tgt = base.sift_matches(
        ref["image"], tgt["image"], static_ref, static_tgt
    )
    if len(uv_ref) < 35:
        return {
            "passed": False,
            "reason": f"only {len(uv_ref)} static SIFT matches",
            "reciprocal_gate": "not reached",
        }

    keep_ref, X_ref = base.sample_point_map(ref["points"], ref["valid"], uv_ref)
    uv_tgt_fwd = uv_tgt[keep_ref]
    if len(X_ref) < 30:
        return {
            "passed": False,
            "reason": f"only {len(X_ref)} valid reference-depth matches",
            "matches": int(len(X_ref)),
            "reciprocal_gate": "not reached",
        }

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X_ref,
        uv_tgt_fwd,
        tgt["K"],
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=4.0,
        confidence=0.999,
        iterationsCount=2500,
    )
    if not ok or inliers is None or len(inliers) < 25:
        return {
            "passed": False,
            "reason": "forward PnP RANSAC insufficient inliers",
            "matches": int(len(X_ref)),
            "inliers": 0 if inliers is None else int(len(inliers)),
            "reciprocal_gate": "not reached",
        }

    fids = inliers.reshape(-1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            X_ref[fids], uv_tgt_fwd[fids], tgt["K"], None, rvec, tvec
        )
    except Exception:
        pass
    Rf, _ = cv2.Rodrigues(rvec)
    tf = tvec.reshape(3)
    fproj, _ = cv2.projectPoints(
        X_ref[fids], rvec, tvec, tgt["K"], None
    )
    fproj = fproj.reshape(-1, 2)
    ferr = np.linalg.norm(fproj - uv_tgt_fwd[fids], axis=1)

    # Robust target MoGe scale from the same forward PnP inliers.
    x2 = np.rint(uv_tgt_fwd[fids, 0]).astype(int)
    y2 = np.rint(uv_tgt_fwd[fids, 1]).astype(int)
    pred_z = (Rf @ X_ref[fids].T).T[:, 2] + tf[2]
    d2 = tgt["depth"][y2, x2]
    good_scale = (
        np.isfinite(d2)
        & (d2 > 0.25)
        & np.isfinite(pred_z)
        & (pred_z > 0.25)
    )
    ratios = pred_z[good_scale] / d2[good_scale]
    scale = float(np.median(ratios)) if len(ratios) >= 10 else float("nan")
    mad = (
        float(np.median(np.abs(ratios - scale)))
        if len(ratios) >= 10
        else float("inf")
    )
    scale_mad_fraction = mad / max(abs(scale), 1e-6)

    C = -Rf.T @ tf
    forward_z = (Rf @ X_ref.T).T[:, 2] + tf[2]
    forward_cheirality = float(np.mean(forward_z > 0.25))

    # Reciprocal solve: target MoGe 3D -> reference 2D, using the forward
    # scale only as a unit conversion. A wrong scale/pose should not compose
    # back to identity.
    keep_tgt, X_tgt_raw = base.sample_point_map(
        tgt["points"], tgt["valid"], uv_tgt
    )
    if not np.isfinite(scale) or not (0.15 < scale < 8.0):
        return {
            "passed": False,
            "reason": "non-finite or grossly implausible target MoGe scale",
            "matches": int(len(X_ref)),
            "inliers": int(len(fids)),
            "inlier_fraction": float(len(fids) / len(X_ref)),
            "median_reprojection_px": float(np.median(ferr)),
            "p95_reprojection_px": float(np.percentile(ferr, 95)),
            "depth_scale": scale,
            "depth_scale_mad": mad,
            "depth_scale_mad_fraction": scale_mad_fraction,
            "forward_cheirality": forward_cheirality,
            "R": Rf,
            "t": tf,
            "C": C,
            "reciprocal_gate": "scale sanity fail",
        }
    if len(X_tgt_raw) < 30:
        return {
            "passed": False,
            "reason": f"only {len(X_tgt_raw)} valid target-depth reciprocal matches",
            "matches": int(len(X_ref)),
            "inliers": int(len(fids)),
            "inlier_fraction": float(len(fids) / len(X_ref)),
            "median_reprojection_px": float(np.median(ferr)),
            "p95_reprojection_px": float(np.percentile(ferr, 95)),
            "depth_scale": scale,
            "depth_scale_mad": mad,
            "depth_scale_mad_fraction": scale_mad_fraction,
            "forward_cheirality": forward_cheirality,
            "R": Rf,
            "t": tf,
            "C": C,
            "reciprocal_gate": "insufficient target depth",
        }

    X_tgt = X_tgt_raw * scale
    uv_ref_rev = uv_ref[keep_tgt]
    rok, rrvec, rtvec, rinliers = cv2.solvePnPRansac(
        X_tgt,
        uv_ref_rev,
        ref["K"],
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=4.0,
        confidence=0.999,
        iterationsCount=2500,
    )
    if not rok or rinliers is None or len(rinliers) < 25:
        return {
            "passed": False,
            "reason": "reciprocal PnP RANSAC insufficient inliers",
            "matches": int(len(X_ref)),
            "inliers": int(len(fids)),
            "inlier_fraction": float(len(fids) / len(X_ref)),
            "median_reprojection_px": float(np.median(ferr)),
            "p95_reprojection_px": float(np.percentile(ferr, 95)),
            "depth_scale": scale,
            "depth_scale_mad": mad,
            "depth_scale_mad_fraction": scale_mad_fraction,
            "forward_cheirality": forward_cheirality,
            "reciprocal_matches": int(len(X_tgt)),
            "reciprocal_inliers": 0 if rinliers is None else int(len(rinliers)),
            "R": Rf,
            "t": tf,
            "C": C,
            "reciprocal_gate": "PnP fail",
        }

    rids = rinliers.reshape(-1)
    try:
        rrvec, rtvec = cv2.solvePnPRefineLM(
            X_tgt[rids], uv_ref_rev[rids], ref["K"], None, rrvec, rtvec
        )
    except Exception:
        pass
    Rr, _ = cv2.Rodrigues(rrvec)
    tr = rtvec.reshape(3)
    rproj, _ = cv2.projectPoints(
        X_tgt[rids], rrvec, rtvec, ref["K"], None
    )
    rproj = rproj.reshape(-1, 2)
    rerr = np.linalg.norm(rproj - uv_ref_rev[rids], axis=1)

    reverse_z = (Rr @ X_tgt.T).T[:, 2] + tr[2]
    reverse_cheirality = float(np.mean(reverse_z > 0.25))
    Rclose = Rr @ Rf
    tclose = Rr @ tf + tr
    rotation_closure_deg = rotation_angle_deg(Rclose)
    scene_scale = max(float(np.median(np.linalg.norm(X_ref, axis=1))), 1e-6)
    translation_closure_fraction = float(np.linalg.norm(tclose) / scene_scale)

    forward_pass = (
        len(fids) >= 25
        and float(len(fids) / len(X_ref)) >= 0.35
        and float(np.median(ferr)) <= 2.5
        and float(np.percentile(ferr, 95)) <= 6.0
        and scale_mad_fraction <= 0.08
        and forward_cheirality >= 0.95
    )
    reverse_pass = (
        len(rids) >= 25
        and float(len(rids) / len(X_tgt)) >= 0.35
        and float(np.median(rerr)) <= 2.5
        and float(np.percentile(rerr, 95)) <= 6.0
        and reverse_cheirality >= 0.95
    )
    closure_pass = (
        rotation_closure_deg <= 2.0
        and translation_closure_fraction <= 0.08
    )
    passed = bool(forward_pass and reverse_pass and closure_pass)

    return {
        "passed": passed,
        "reason": "passed reciprocal closure" if passed else "reciprocal geometry closure failed",
        "matches": int(len(X_ref)),
        "inliers": int(len(fids)),
        "inlier_fraction": float(len(fids) / len(X_ref)),
        "median_reprojection_px": float(np.median(ferr)),
        "p95_reprojection_px": float(np.percentile(ferr, 95)),
        "depth_scale": scale,
        "depth_scale_mad": mad,
        "depth_scale_mad_fraction": scale_mad_fraction,
        "forward_cheirality": forward_cheirality,
        "reciprocal_matches": int(len(X_tgt)),
        "reciprocal_inliers": int(len(rids)),
        "reciprocal_inlier_fraction": float(len(rids) / len(X_tgt)),
        "reciprocal_median_reprojection_px": float(np.median(rerr)),
        "reciprocal_p95_reprojection_px": float(np.percentile(rerr, 95)),
        "reverse_cheirality": reverse_cheirality,
        "rotation_closure_deg": rotation_closure_deg,
        "translation_closure_fraction": translation_closure_fraction,
        "forward_gate_passed": bool(forward_pass),
        "reverse_gate_passed": bool(reverse_pass),
        "closure_gate_passed": bool(closure_pass),
        "reciprocal_gate": "passed" if passed else "failed",
        "R": Rf,
        "t": tf,
        "C": C,
    }


def output_dir_from_argv() -> Path | None:
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--out":
            return Path(sys.argv[i + 1])
    return None


if __name__ == "__main__":
    out = output_dir_from_argv()
    base.solve_target_from_reference = solve_target_from_reference_reciprocal
    base.main()
    if out is not None:
        p12 = out / "portable_moge_pnp_qa_v12.json"
        if p12.exists():
            q = json.loads(p12.read_text())
            q["prototype"] = "portable_moge_sift_pnp_reciprocal_freeview_v13"
            q["camera_method"] = (
                "MoGe-2 reference 3D + person-masked static SIFT + forward PnP; "
                "target MoGe scale is accepted only after reverse PnP and SE(3) closure. "
                "No Jazz camera pose is accepted by widening the old absolute scale ceiling."
            )
            q["success_gate"] = (
                "forward and reciprocal PnP plus transform closure; then native 3-5 degree "
                "coverage must improve without relaxing the conservative static fill gate; "
                "visual QA remains authoritative."
            )
            (out / "portable_moge_pnp_qa_v13.json").write_text(
                json.dumps(q, indent=2)
            )
