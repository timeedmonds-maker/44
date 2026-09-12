from __future__ import annotations

"""v33a: transfer the accepted v73 RAR camera from certified frame0257 to
B32's exact freeze frame0256 using STATIC image structure only.

The v32 package uses the v73 frame-C metric camera, but its selected RAR RGB is
one decoded frame earlier.  For a fixed physical optical centre, two image
states are related by a 3x3 image homography even when pan/tilt/zoom changes.
This diagnostic estimates that homography from static court/backboard/support
features, composes it with the accepted v73 projection matrix, decomposes the
result back to K/R/C, and updates only the RAR camera state in a writable copy
of the B32 scene manifest.

Moving-player pixels are explicitly masked from the registration.  No player
measurement, body fit, Broadcast evidence, or generated RGB can influence the
camera correction.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RAR = "Right Above Rim"


def camera_projection(cam: dict) -> np.ndarray:
    K = np.asarray(cam["K_px"], float)
    R = np.asarray(cam["R_world_to_camera"], float)
    C = np.asarray(cam["C_world_cm"], float)
    t = -R @ C
    return K @ np.c_[R, t]


def decompose_projection(P: np.ndarray):
    K, R, Ch, *_ = cv2.decomposeProjectionMatrix(np.asarray(P, float))
    K = K / K[2, 2]
    C = (Ch[:3] / Ch[3]).reshape(3)
    # cv2 returns a proper rotation for this well-conditioned camera.  Fail
    # rather than repair silently if that invariant is broken.
    if np.linalg.det(R) < 0.999 or np.linalg.det(R) > 1.001:
        raise RuntimeError(f"v33a improper decomposed rotation det={np.linalg.det(R)}")
    return K, R, C


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xh = np.c_[np.asarray(X, float), np.ones(len(X))]
    q = (P @ Xh.T).T
    return q[:, :2] / q[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--v73-frame0257", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    stage = a.b32_root / "stage_a"
    manifest_path = stage / "v32_scene_manifest.json"
    scene = json.loads(manifest_path.read_text())
    if scene.get("resolution") != [960, 540]:
        raise RuntimeError(f"v33a unexpected B32 resolution {scene.get('resolution')}")
    f256 = stage / "v32_chosen_Right_Above_Rim_frame0256.png"
    if not f256.is_file():
        raise RuntimeError(f"v33a exact B32 RAR frame missing: {f256}")
    if "frame0257" not in a.v73_frame0257.name:
        raise RuntimeError(f"v33a expected certified v73 frame0257, got {a.v73_frame0257.name}")

    im256 = cv2.imread(str(f256), cv2.IMREAD_GRAYSCALE)
    im257 = cv2.imread(str(a.v73_frame0257), cv2.IMREAD_GRAYSCALE)
    if im256 is None or im257 is None or im256.shape != (540, 960) or im257.shape != (540, 960):
        raise RuntimeError("v33a could not load exact native RAR frames")

    # Static registration mask.  The excluded rectangles conservatively remove
    # the live players/ball while retaining the court, rim/backboard support,
    # baseline structure, floor texture and painted geometry.
    mask = np.full(im256.shape, 255, np.uint8)
    mask[120:350, 300:680] = 0
    mask[0:180, 0:350] = 0
    mask[150:330, 0:180] = 0
    mask[360:540, 500:720] = 0

    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=10)
    k1, d1 = orb.detectAndCompute(im256, mask)
    k2, d2 = orb.detectAndCompute(im257, mask)
    if d1 is None or d2 is None:
        raise RuntimeError("v33a static ORB descriptors unavailable")
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < 0.70 * n.distance]
    if len(good) < 500:
        raise RuntimeError(f"v33a insufficient static matches {len(good)}")
    p256 = np.float32([k1[m.queryIdx].pt for m in good])
    p257 = np.float32([k2[m.trainIdx].pt for m in good])
    H_256_to_257, inl = cv2.findHomography(
        p256, p257, cv2.RANSAC, 1.0, maxIters=10000, confidence=0.999
    )
    if H_256_to_257 is None or inl is None:
        raise RuntimeError("v33a static homography failed")
    keep = inl.ravel().astype(bool)
    pred257 = cv2.perspectiveTransform(p256.reshape(-1, 1, 2), H_256_to_257).reshape(-1, 2)
    err = np.linalg.norm(pred257 - p257, axis=1)[keep]
    static = {
        "ratio_test_matches": int(len(good)),
        "ransac_inliers": int(np.sum(keep)),
        "inlier_fraction": float(np.mean(keep)),
        "median_px": float(np.median(err)),
        "p90_px": float(np.percentile(err, 90)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
    }
    static_gate = bool(
        static["ransac_inliers"] >= 800
        and static["inlier_fraction"] >= 0.50
        and static["median_px"] <= 0.60
        and static["p95_px"] <= 1.00
        and static["max_px"] <= 1.20
    )
    if not static_gate:
        raise RuntimeError(f"v33a fail-closed static registration {static}")

    # v73/B32 camera is the certified 0257 state.  Convert its image mapping to
    # the exact 0256 state: P256 = H(257->256) * P257.
    H_257_to_256 = np.linalg.inv(H_256_to_257)
    H_257_to_256 /= H_257_to_256[2, 2]
    old = scene["cameras"][RAR]
    P257 = camera_projection(old)
    P256 = H_257_to_256 @ P257
    K256, R256, C256 = decompose_projection(P256)
    Cold = np.asarray(old["C_world_cm"], float)
    center_delta = float(np.linalg.norm(C256 - Cold))

    # Direct projection-equivalence check on non-coplanar regulation-world
    # samples.  This ensures decomposition did not change the composed camera.
    Xcheck = np.array([
        [38.1, 0.0, 304.8], [60.96, 0.0, 304.8], [15.24, 0.0, 304.8],
        [38.1, 100.0, 304.8], [38.1, -100.0, 304.8], [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0], [38.1, 0.0, 200.0],
    ], float)
    old_uv = project(P257, Xcheck)
    expected_uv = cv2.perspectiveTransform(old_uv.reshape(-1, 1, 2), H_257_to_256).reshape(-1, 2)
    rebuilt_P = K256 @ np.c_[R256, -R256 @ C256]
    rebuilt_uv = project(rebuilt_P, Xcheck)
    decomposition_max = float(np.max(np.linalg.norm(expected_uv - rebuilt_uv, axis=1)))

    # Quantify how much the stale one-frame camera moves the exact B32 image.
    probes257 = np.array([[[480.0, 259.0]], [[480.0, 289.0]], [[500.0, 350.0]]], np.float64)
    probes256 = cv2.perspectiveTransform(probes257, H_257_to_256).reshape(-1, 2)
    probe_shift = probes256 - probes257.reshape(-1, 2)

    camera_gate = bool(
        center_delta <= 1e-6
        and decomposition_max <= 1e-6
        and 450.0 <= K256[0, 0] <= 750.0
        and 450.0 <= K256[1, 1] <= 750.0
        and 400.0 <= K256[0, 2] <= 560.0
        and 200.0 <= K256[1, 2] <= 330.0
    )
    if not camera_gate:
        raise RuntimeError(
            f"v33a fail-closed camera transfer center={center_delta} decomp={decomposition_max} K={K256.tolist()}"
        )

    corrected = dict(old)
    corrected["C_world_cm"] = C256.tolist()
    corrected["R_world_to_camera"] = R256.tolist()
    corrected["K_px"] = K256.tolist()
    corrected["extrinsic_world_to_camera_3x4"] = np.c_[R256, -R256 @ C256].tolist()
    corrected["v33a_exact_frame_transfer"] = {
        "from_certified_frame": a.v73_frame0257.name,
        "to_b32_exact_frame": f256.name,
        "method": "static-only ORB/RANSAC shared-center image homography; moving-player regions masked",
        "H_256_to_257": H_256_to_257.tolist(),
        "H_257_to_256": H_257_to_256.tolist(),
        "static_registration": static,
        "physical_center_delta_cm": center_delta,
        "projection_decomposition_max_px": decomposition_max,
    }

    audit = {
        "version": "v33a_exact_rar_frame_camera",
        "status": "PASS_V33A_EXACT_RAR_FRAME_STATIC_TRANSFER",
        "policy": "static geometry only; preserve accepted v73 physical optical centre; no player/Broadcast evidence",
        "source_camera_state": old,
        "corrected_camera_state": corrected,
        "static_registration": static,
        "H_256_to_257": H_256_to_257.tolist(),
        "H_257_to_256": H_257_to_256.tolist(),
        "probe_257_to_256_shift_px": probe_shift.tolist(),
        "physical_center_delta_cm": center_delta,
        "projection_decomposition_max_px": decomposition_max,
        "gates": {"static_registration": static_gate, "camera_transfer": camera_gate},
    }
    (a.out / "v33a_exact_rar_camera_audit.json").write_text(json.dumps(audit, indent=2))

    # Visual static QA: warp certified 0257 into exact 0256 and write a 50/50
    # overlay.  Dynamic regions are not used to assess the registration.
    warped257 = cv2.warpPerspective(im257, H_257_to_256, (960, 540))
    overlay = cv2.addWeighted(im256, 0.5, warped257, 0.5, 0.0)
    cv2.imwrite(str(a.out / "v33a_rar_static_alignment_overlay.png"), overlay)
    cv2.imwrite(str(a.out / "v33a_rar_static_registration_mask.png"), mask)

    if a.apply:
        backup = stage / "v32_scene_manifest_pre_v33a.json"
        if not backup.exists():
            backup.write_text(json.dumps(scene, indent=2))
        scene["cameras"][RAR] = corrected
        scene.setdefault("v33a", {})["exact_rar_frame_camera"] = audit
        manifest_path.write_text(json.dumps(scene, indent=2))

    print(json.dumps({
        "status": audit["status"], "static_registration": static,
        "probe_257_to_256_shift_px": probe_shift.tolist(),
        "old_K": old["K_px"], "new_K": K256.tolist(),
        "center_delta_cm": center_delta, "applied": bool(a.apply),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
