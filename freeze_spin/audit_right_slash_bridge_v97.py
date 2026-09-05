from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def robust(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "median_px": None, "p90_px": None, "p95_px": None, "fraction_under_3px": 0.0}
    return {
        "n": int(v.size),
        "median_px": float(np.median(v)),
        "p90_px": float(np.percentile(v, 90)),
        "p95_px": float(np.percentile(v, 95)),
        "fraction_under_3px": float(np.mean(v <= 3.0)),
    }


def load(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"missing image: {path}")
    if im.shape[:2] != (540, 960):
        raise RuntimeError(f"expected native 960x540 frame: {path} got {im.shape[:2][::-1]}")
    return im


def sharpness(im: np.ndarray) -> float:
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def sift_matches(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.70 * n.distance]
    p = np.float32([ka[m.queryIdx].pt for m in good])
    q = np.float32([kb[m.trainIdx].pt for m in good])
    return p, q


def partitions(p: np.ndarray) -> dict[str, np.ndarray]:
    x, y = p[:, 0], p[:, 1]
    # Preserve the v96 independent holdout exactly for comparability. It is an
    # image-space static-support band, NOT a semantic basket validation region.
    hold = (x > 330) & (x < 800) & (y > 20) & (y < 220)
    action = (x > 280) & (x < 760) & (y > 180) & (y < 520)
    train = ~hold & ~action
    return {
        "train": train,
        "heldout_upper_static": hold,
        "dynamic_action_exclusion": action,
        "upper_left": (x <= 330) & (y < 220),
        "upper_mid": (x > 330) & (x < 800) & (y < 180),
        "lower_left_static": (x < 280) & (y >= 180) & (y < 500),
    }


def transfer(ref: np.ndarray, target: np.ndarray) -> dict:
    p, q = sift_matches(ref, target)
    if len(p) < 20:
        raise RuntimeError(f"insufficient matches: {len(p)}")
    parts = partitions(p)
    train = parts["train"]
    if int(train.sum()) < 8:
        raise RuntimeError(f"insufficient training matches: {int(train.sum())}")
    H, mask = cv2.findHomography(p[train], q[train], cv2.RANSAC, 1.5, maxIters=20000, confidence=0.999)
    if H is None or mask is None:
        raise RuntimeError("homography fit failed")
    inl = mask.ravel().astype(bool)
    train_p = p[train][inl]
    train_q = q[train][inl]
    pred = cv2.perspectiveTransform(train_p[:, None, :], H)[:, 0, :]
    train_err = np.linalg.norm(pred - train_q, axis=1)
    regions = {}
    for name, sel in parts.items():
        if name in ("train", "dynamic_action_exclusion"):
            continue
        if not np.any(sel):
            regions[name] = robust(np.array([], dtype=float))
            continue
        pred = cv2.perspectiveTransform(p[sel, None, :], H)[:, 0, :]
        regions[name] = robust(np.linalg.norm(pred - q[sel], axis=1))
    return {
        "ratio_test_matches": int(len(p)),
        "training_matches": int(train.sum()),
        "training_inliers": int(inl.sum()),
        "training_residual": robust(train_err),
        "regions": regions,
        "H": H,
    }


def composed_validation(source: np.ndarray, target: np.ndarray, H_source_to_target: np.ndarray) -> dict:
    p, q = sift_matches(source, target)
    if len(p) == 0:
        return {"ratio_test_matches": 0, "regions": {}}
    parts = partitions(p)
    regions = {}
    for name, sel in parts.items():
        if name in ("train", "dynamic_action_exclusion"):
            continue
        if not np.any(sel):
            regions[name] = robust(np.array([], dtype=float))
            continue
        pred = cv2.perspectiveTransform(p[sel, None, :], H_source_to_target)[:, 0, :]
        regions[name] = robust(np.linalg.norm(pred - q[sel], axis=1))
    static = ~parts["dynamic_action_exclusion"]
    pred = cv2.perspectiveTransform(p[static, None, :], H_source_to_target)[:, 0, :]
    regions["all_static_outside_action"] = robust(np.linalg.norm(pred - q[static], axis=1))
    return {"ratio_test_matches": int(len(p)), "regions": regions}


def transfer_gate(row: dict, *, min_train_inliers: int, min_holdout_matches: int) -> dict:
    t = row["training_residual"]
    h = row["regions"]["heldout_upper_static"]
    checks = {
        "training_inliers": row["training_inliers"] >= min_train_inliers,
        "training_p95_le_1_5": t["p95_px"] is not None and t["p95_px"] <= 1.5,
        "heldout_matches": h["n"] >= min_holdout_matches,
        "heldout_median_le_2_0": h["median_px"] is not None and h["median_px"] <= 2.0,
        "heldout_p90_le_3_0": h["p90_px"] is not None and h["p90_px"] <= 3.0,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def compose_gate(row: dict) -> dict:
    h = row["regions"].get("heldout_upper_static", {})
    checks = {
        "direct_heldout_matches_ge_40": h.get("n", 0) >= 40,
        "direct_heldout_median_le_2_0": h.get("median_px") is not None and h["median_px"] <= 2.0,
        "direct_heldout_p90_le_3_0": h.get("p90_px") is not None and h["p90_px"] <= 3.0,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def jsonable_transfer(row: dict) -> dict:
    return {k: (v.tolist() if k == "H" else v) for k, v in row.items()}


def draw_composite(source: np.ndarray, bridge: np.ndarray, target: np.ndarray, H: np.ndarray, label: str, out: Path) -> None:
    warped = cv2.warpPerspective(source, H, (960, 540))
    blend = cv2.addWeighted(target, 0.5, warped, 0.5, 0.0)
    panels = [source.copy(), bridge.copy(), target.copy(), warped, blend]
    names = ["sharp source", "b42 bridge +0.100s", "freeze b40", "sharp warped to freeze", "50/50 static alignment"]
    for im, name in zip(panels, names):
        cv2.rectangle(im, (0, 0), (959, 36), (0, 0, 0), -1)
        cv2.putText(im, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([np.hstack(panels[:3]), np.hstack([panels[3], panels[4], np.zeros_like(target)])])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 36), (0, 0, 0), -1)
    cv2.putText(canvas, label, (470, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("game_id") != "0022500301" or int(manifest.get("event_id")) != 489 or manifest.get("camera_label") != "Right Slash":
        raise SystemExit("wrong immutable v93 source artifact")

    target_path = args.frames / args.target
    bridge_path = args.frames / args.bridge
    target = load(target_path)
    bridge = load(bridge_path)
    target_sharpness = sharpness(target)
    bridge_sharpness = sharpness(bridge)

    bridge_to_target = transfer(bridge, target)
    bridge_gate = transfer_gate(bridge_to_target, min_train_inliers=60, min_holdout_matches=40)

    rows = []
    for name in args.candidates:
        source = load(args.frames / name)
        s = sharpness(source)
        source_to_bridge = transfer(source, bridge)
        hop_gate = transfer_gate(source_to_bridge, min_train_inliers=100, min_holdout_matches=100)
        H = bridge_to_target["H"] @ source_to_bridge["H"]
        H = H / H[2, 2]
        comp = composed_validation(source, target, H)
        comp_gate = compose_gate(comp)
        sharp_gain = s / target_sharpness if target_sharpness > 0 else 0.0
        checks = {
            "bridge_to_target_passed": bridge_gate["passed"],
            "source_to_bridge_passed": hop_gate["passed"],
            "composed_direct_heldout_passed": comp_gate["passed"],
            "sharpness_gain_ge_2_5x": sharp_gain >= 2.5,
        }
        rows.append({
            "file": name,
            "sharpness": s,
            "sharpness_gain_vs_freeze": sharp_gain,
            "source_to_bridge": jsonable_transfer(source_to_bridge),
            "source_to_bridge_gate": hop_gate,
            "composed_source_to_freeze": {"H": H.tolist(), **comp},
            "composed_gate": comp_gate,
            "candidate_gate": {"passed": bool(all(checks.values())), "checks": checks},
        })

    passed = [r for r in rows if r["candidate_gate"]["passed"]]
    passed.sort(key=lambda r: (-r["sharpness"], r["composed_source_to_freeze"]["regions"]["heldout_upper_static"]["median_px"]))
    selected = passed[0] if passed else None
    status = "PASS_RIGHT_SLASH_FREEZE_TO_SHARP_BRIDGE_V97" if selected else "FAIL_RIGHT_SLASH_FREEZE_TO_SHARP_BRIDGE_V97"

    report = {
        "status": status,
        "game_id": "0022500301",
        "event_id": 489,
        "camera_label": "Right Slash",
        "source_artifact": "adams-jazz-right-slash-event489-burst-v93-33955558281",
        "target_freeze": {"file": args.target, "sharpness": target_sharpness},
        "temporal_bridge": {
            "file": args.bridge,
            "delta_from_freeze_s": 0.100,
            "sharpness": bridge_sharpness,
            "transfer_to_freeze": jsonable_transfer(bridge_to_target),
            "gate": bridge_gate,
        },
        "candidate_sharp_frames": rows,
        "selected_sharp_geometry_frame": None if selected is None else {
            "file": selected["file"],
            "sharpness": selected["sharpness"],
            "sharpness_gain_vs_freeze": selected["sharpness_gain_vs_freeze"],
            "composed_heldout": selected["composed_source_to_freeze"]["regions"]["heldout_upper_static"],
        },
        "interpretation": "A passing result establishes a reproducible projective bridge from the synchronized blurred freeze frame through b42 (+0.100 s) into at least one materially sharper same-camera frame, with the composed transform tested on direct source-to-freeze correspondences that were not used to fit either hop. This authorizes a sharp-frame metric geometry attempt constrained back to the freeze; it does not itself prove a metric camera centre or validate basket geometry.",
        "important_qa_correction": "The legacy v95/v96 rectangular 'basket' band is treated here only as an image-space held-out upper-static region because it contains spectators. No result from that region is described as semantic basket validation.",
        "permissions": {
            "right_slash_sharp_geometry_frame_attempt_allowed": bool(selected),
            "right_slash_shared_center_metric_attempt_allowed": bool(selected),
            "right_slash_metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "next_action": "Use the selected sharp frame for regulation basket/court metric calibration, then constrain the event-489 freeze camera through the verified b42 bridge. Require independent full-rim/board/court visual QA and camera-centre stability before promotion.",
    }
    (args.out / "right_slash_temporal_bridge_v97.json").write_text(json.dumps(report, indent=2) + "\n")

    if selected:
        H = np.asarray(selected["composed_source_to_freeze"]["H"], dtype=np.float64)
        source = load(args.frames / selected["file"])
        draw_composite(source, bridge, target, H, f"Right Slash v97 | {selected['file']} -> b42 -> b40", args.out / "right_slash_temporal_bridge_v97.png")

    print(json.dumps(report, indent=2), flush=True)
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
