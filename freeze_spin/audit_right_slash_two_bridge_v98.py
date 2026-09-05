from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

cv2.setRNGSeed(20260905)


def stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "median_px": None, "p90_px": None, "p95_px": None, "fraction_under_3px": 0.0}
    return {
        "n": int(v.size),
        "median_px": float(np.median(v)),
        "p90_px": float(np.percentile(v, 90)),
        "p95_px": float(np.percentile(v, 95)),
        "fraction_under_3px": float(np.mean(v <= 3.0)),
    }


def load(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (540, 960):
        raise RuntimeError(f"invalid native frame {p}: {None if im is None else im.shape}")
    return im


def sharpness(im: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.70 * n.distance]
    return (
        np.float32([ka[m.queryIdx].pt for m in good]),
        np.float32([kb[m.trainIdx].pt for m in good]),
    )


def masks(p: np.ndarray) -> dict[str, np.ndarray]:
    x, y = p[:, 0], p[:, 1]
    # Historical v96 rectangle retained only as an independent image-space
    # upper-static holdout. It is NOT semantic basket validation.
    hold = (x > 330) & (x < 800) & (y > 20) & (y < 220)
    action = (x > 280) & (x < 760) & (y > 180) & (y < 520)
    return {"train": ~hold & ~action, "hold": hold, "action": action}


def fit(a: np.ndarray, b: np.ndarray) -> dict:
    p, q = match(a, b)
    m = masks(p)
    if int(m["train"].sum()) < 8:
        raise RuntimeError("insufficient training support")
    H, inl = cv2.findHomography(
        p[m["train"]], q[m["train"]], cv2.RANSAC, 1.5,
        maxIters=20000, confidence=0.999,
    )
    if H is None or inl is None:
        raise RuntimeError("homography fit failed")
    keep = inl.ravel().astype(bool)
    tp = p[m["train"]][keep]
    tq = q[m["train"]][keep]
    te = np.linalg.norm(cv2.perspectiveTransform(tp[:, None, :], H)[:, 0, :] - tq, axis=1)
    hp = p[m["hold"]]
    hq = q[m["hold"]]
    he = np.linalg.norm(cv2.perspectiveTransform(hp[:, None, :], H)[:, 0, :] - hq, axis=1)
    H = H / H[2, 2]
    return {
        "ratio_test_matches": int(len(p)),
        "training_matches": int(m["train"].sum()),
        "training_inliers": int(keep.sum()),
        "training_residual": stats(te),
        "heldout_upper_static": stats(he),
        "H": H,
    }


def hop_gate(r: dict, min_inliers: int, min_holdout: int) -> dict:
    t = r["training_residual"]
    h = r["heldout_upper_static"]
    checks = {
        "training_inliers": r["training_inliers"] >= min_inliers,
        "training_p95_le_1_5": t["p95_px"] is not None and t["p95_px"] <= 1.5,
        "heldout_matches": h["n"] >= min_holdout,
        "heldout_median_le_2_0": h["median_px"] is not None and h["median_px"] <= 2.0,
        "heldout_p90_le_3_0": h["p90_px"] is not None and h["p90_px"] <= 3.0,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def direct_validation(source: np.ndarray, target: np.ndarray, H: np.ndarray) -> dict:
    p, q = match(source, target)
    m = masks(p)
    hp, hq = p[m["hold"]], q[m["hold"]]
    he = np.linalg.norm(cv2.perspectiveTransform(hp[:, None, :], H)[:, 0, :] - hq, axis=1)
    s = stats(he)
    checks = {
        "direct_heldout_matches_ge_40": s["n"] >= 40,
        "direct_heldout_median_le_2_0": s["median_px"] is not None and s["median_px"] <= 2.0,
        "direct_heldout_p90_le_3_0": s["p90_px"] is not None and s["p90_px"] <= 3.0,
    }
    return {"ratio_test_matches": int(len(p)), "heldout_upper_static": s, "gate": {"passed": bool(all(checks.values())), "checks": checks}}


def clean(r: dict) -> dict:
    return {**r, "H": r["H"].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("game_id") != "0022500301" or int(manifest.get("event_id")) != 489 or manifest.get("camera_label") != "Right Slash":
        raise SystemExit("wrong immutable v93 source artifact")

    files = {
        "freeze": "b40_8.276s.png",
        "bridge_050": "b41_8.326s.png",
        "bridge_100": "b42_8.376s.png",
    }
    ims = {k: load(args.frames / v) for k, v in files.items()}
    candidates = ["b43_8.426s.png", "b44_8.476s.png", "b45_8.526s.png"]

    # Do not relax v97's 60-inlier bridge gate. Instead split the difficult
    # +0.100 -> freeze hop into two independently held-out +0.050 s hops.
    r_100_050 = fit(ims["bridge_100"], ims["bridge_050"])
    r_050_freeze = fit(ims["bridge_050"], ims["freeze"])
    g_100_050 = hop_gate(r_100_050, min_inliers=60, min_holdout=40)
    g_050_freeze = hop_gate(r_050_freeze, min_inliers=60, min_holdout=40)

    rows = []
    freeze_sharp = sharpness(ims["freeze"])
    for name in candidates:
        src = load(args.frames / name)
        r_src_100 = fit(src, ims["bridge_100"])
        g_src_100 = hop_gate(r_src_100, min_inliers=100, min_holdout=100)
        H = r_050_freeze["H"] @ r_100_050["H"] @ r_src_100["H"]
        H = H / H[2, 2]
        direct = direct_validation(src, ims["freeze"], H)
        gain = sharpness(src) / freeze_sharp
        checks = {
            "source_to_100ms_hop": g_src_100["passed"],
            "100ms_to_050ms_hop": g_100_050["passed"],
            "050ms_to_freeze_hop": g_050_freeze["passed"],
            "direct_composed_holdout": direct["gate"]["passed"],
            "sharpness_gain_ge_2_5x": gain >= 2.5,
        }
        rows.append({
            "file": name,
            "sharpness": sharpness(src),
            "sharpness_gain_vs_freeze": gain,
            "source_to_bridge_100": clean(r_src_100),
            "source_to_bridge_100_gate": g_src_100,
            "composed_source_to_freeze_H": H.tolist(),
            "direct_composed_validation": direct,
            "candidate_gate": {"passed": bool(all(checks.values())), "checks": checks},
        })

    passed = [r for r in rows if r["candidate_gate"]["passed"]]
    passed.sort(key=lambda r: (-r["sharpness"], r["direct_composed_validation"]["heldout_upper_static"]["median_px"]))
    selected = passed[0] if passed else None
    status = "PASS_RIGHT_SLASH_TWO_BRIDGE_V98" if selected else "FAIL_RIGHT_SLASH_TWO_BRIDGE_V98"

    report = {
        "status": status,
        "game_id": "0022500301",
        "event_id": 489,
        "camera_label": "Right Slash",
        "source_artifact": "adams-jazz-right-slash-event489-burst-v93-33955558281",
        "freeze": {"file": files["freeze"], "sharpness": freeze_sharp},
        "bridge_100_to_050": {"from": files["bridge_100"], "to": files["bridge_050"], "fit": clean(r_100_050), "gate": g_100_050},
        "bridge_050_to_freeze": {"from": files["bridge_050"], "to": files["freeze"], "fit": clean(r_050_freeze), "gate": g_050_freeze},
        "candidates": rows,
        "selected_sharp_geometry_frame": None if selected is None else {
            "file": selected["file"],
            "sharpness": selected["sharpness"],
            "sharpness_gain_vs_freeze": selected["sharpness_gain_vs_freeze"],
            "direct_composed_holdout": selected["direct_composed_validation"]["heldout_upper_static"],
        },
        "qa": {
            "v97_failure_preserved": "v97 failed under OpenCV 4.10 because b42->b40 produced 54 training inliers against the predeclared minimum of 60, despite passing its independent held-out error gates.",
            "gate_policy": "v98 does not lower that threshold. It introduces b41 (+0.050 s) so both bridge hops must independently satisfy the original >=60-inlier / <=1.5 px training-p95 / held-out transfer gates, and the complete chain must still predict direct sharp-to-freeze held-out correspondences within <=2 px median and <=3 px p90.",
            "semantic_correction": "heldout_upper_static is not called a basket region and is not basket-geometry QA.",
        },
        "permissions": {
            "right_slash_sharp_geometry_frame_attempt_allowed": bool(selected),
            "right_slash_shared_center_metric_attempt_allowed": bool(selected),
            "right_slash_metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "next_action": "If PASS, use the selected sharp event-489 frame for regulation NBA basket/court metric calibration and transfer the resulting event intrinsics/orientation back to the synchronized freeze with the validated temporal chain. Metric promotion still requires full-rim/board/court visual QA and camera-centre consistency.",
    }
    (args.out / "right_slash_two_bridge_v98.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
