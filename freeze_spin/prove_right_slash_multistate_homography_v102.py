from __future__ import annotations

"""v102: distributed fixed-optical-centre projective proof for Right Slash.

For a pinhole camera whose optical centre is fixed, any change in pan/tilt/zoom
between two images of a static 3-D scene is related by one image homography,
independent of scene depth.  v102 tests that prediction on immutable native
Right Slash states recovered by v101.  Dynamic players are not used as anchors:
RANSAC must find one transform with broad spatial support simultaneously in the
stands/basket zone and on the court floor.

Passing this test only authorizes the subsequent metric shared-centre solve.  It
cannot promote a metric camera or permit a replay render.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
ANCHOR = (416, "f06.png", "325a02876fb09c89de6657a711e3241ef5382fbf39fcc1696c95686a642d2668")
STATES = [
    (169, "f02.png", "74b02d5706a89ab3eeb7153db78030295fbbda87bbc2ce13275af5c3844c1c66"),
    (375, "f01.png", "20b6cc30e1fa49299566d53c591e404bd8b8ef5d19a7019b7c004e5c51a370cc"),
    (457, "f03.png", "e84b789e012d2a1bab6b0f3be8d13858aeec1cb682ca0c81956ddc53c57b8013"),
    (540, "f00.png", "2c10a5be6096181fd423b7d7a8b6136c4b90c97ca8cf9d3442f04ae434cd2bcd"),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_native(p: Path, expected_sha: str) -> np.ndarray:
    if sha256(p) != expected_sha:
        raise RuntimeError(f"immutable v101 frame SHA mismatch: {p}")
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (H, W):
        raise RuntimeError(f"missing/non-native v101 frame: {p}")
    return im


def features(im: np.ndarray):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.008, edgeThreshold=12)
    return sift.detectAndCompute(g, None)


def fit_homography(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    ka, da = features(a)
    kb, db = features(b)
    if da is None or db is None:
        raise RuntimeError("SIFT descriptors unavailable")
    good = []
    for pair in cv2.BFMatcher().knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        raise RuntimeError("insufficient ratio-test matches")
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(seed)
    M, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 1.5, maxIters=30000, confidence=0.9995)
    if M is None or mask is None:
        raise RuntimeError("RANSAC homography failed")
    M = M / M[2, 2]
    inl = mask.ravel().astype(bool)
    pred = cv2.perspectiveTransform(pa[:, None, :], M)[:, 0, :]
    err = np.linalg.norm(pred - pb, axis=1)
    x, y = pa[:, 0], pa[:, 1]
    regions = {
        "stands": y < 260,
        "basket_zone": (x > 280) & (x < 580) & (y < 250),
        "floor": y > 360,
        "left_third": x < 320,
        "middle_third": (x >= 320) & (x < 640),
        "right_third": x >= 640,
    }
    rr = {}
    for name, region in regions.items():
        z = region & inl
        rr[name] = {
            "inliers": int(z.sum()),
            "median_px": float(np.median(err[z])) if z.any() else None,
            "p95_px": float(np.percentile(err[z], 95)) if z.any() else None,
        }
    cells = sorted({(min(int(px // 120), 7), min(int(py // 90), 5)) for px, py in pa[inl]})
    return {
        "H": M,
        "ratio_test_matches": len(good),
        "ransac_inliers": int(inl.sum()),
        "inlier_fraction": float(inl.mean()),
        "median_px": float(np.median(err[inl])),
        "p95_px": float(np.percentile(err[inl], 95)),
        "max_px": float(np.max(err[inl])),
        "spatial_cell_count_8x6": len(cells),
        "spatial_cells_8x6": [list(q) for q in cells],
        "regions": rr,
    }


def roundtrip_metrics(Hab: np.ndarray, Hba: np.ndarray) -> dict:
    gx, gy = np.meshgrid(np.linspace(40, 920, 12), np.linspace(30, 510, 8))
    p = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    q = cv2.perspectiveTransform(p[:, None, :], Hab)[:, 0, :]
    r = cv2.perspectiveTransform(q[:, None, :], Hba)[:, 0, :]
    d = np.linalg.norm(r - p, axis=1)
    return {
        "count": int(len(d)),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(np.max(d)),
    }


def draw_inlier_support(anchor: np.ndarray, rec: dict, out: Path) -> None:
    ov = anchor.copy()
    for cx, cy in rec["spatial_cells_8x6"]:
        x0, y0 = int(cx * 120), int(cy * 90)
        cv2.rectangle(ov, (x0, y0), (min(x0 + 119, 959), min(y0 + 89, 539)), (0, 255, 0), 2)
    cv2.putText(ov, f"inliers={rec['ransac_inliers']} p95={rec['p95_px']:.3f}px cells={rec['spatial_cell_count_8x6']}",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(ov, f"inliers={rec['ransac_inliers']} p95={rec['p95_px']:.3f}px cells={rec['spatial_cell_count_8x6']}",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), ov)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v101-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ae, af, ash = ANCHOR
    apath = args.v101_root / f"event_{ae}_selected" / af
    anchor = load_native(apath, ash)
    rows = []
    for event_id, filename, expected_sha in STATES:
        p = args.v101_root / f"event_{event_id}_selected" / filename
        target = load_native(p, expected_sha)
        forward = fit_homography(anchor, target, 102000 + event_id)
        reverse = fit_homography(target, anchor, 102500 + event_id)
        rt = roundtrip_metrics(forward["H"], reverse["H"])
        gates = {
            "ratio_matches_at_least_100": forward["ratio_test_matches"] >= 100,
            "ransac_inliers_at_least_80": forward["ransac_inliers"] >= 80,
            "global_p95_at_most_1_6px": forward["p95_px"] <= 1.6,
            "spatial_cells_at_least_15": forward["spatial_cell_count_8x6"] >= 15,
            "stands_inliers_at_least_50": forward["regions"]["stands"]["inliers"] >= 50,
            "stands_p95_at_most_1_6px": (forward["regions"]["stands"]["p95_px"] or 1e9) <= 1.6,
            "basket_zone_inliers_at_least_30": forward["regions"]["basket_zone"]["inliers"] >= 30,
            "basket_zone_p95_at_most_1_6px": (forward["regions"]["basket_zone"]["p95_px"] or 1e9) <= 1.6,
            "floor_inliers_at_least_8": forward["regions"]["floor"]["inliers"] >= 8,
            "floor_p95_at_most_1_6px": (forward["regions"]["floor"]["p95_px"] or 1e9) <= 1.6,
            "reverse_ransac_inliers_at_least_80": reverse["ransac_inliers"] >= 80,
            "reverse_global_p95_at_most_1_6px": reverse["p95_px"] <= 1.6,
            "forward_reverse_roundtrip_p95_at_most_1px": rt["p95_px"] <= 1.0,
        }
        passed = all(gates.values())
        rec = {
            "event_id": event_id,
            "file": filename,
            "sha256": expected_sha,
            "forward_anchor416_to_state": {**{k:v for k,v in forward.items() if k != "H"}, "H": forward["H"].tolist()},
            "reverse_state_to_anchor416": {**{k:v for k,v in reverse.items() if k != "H"}, "H": reverse["H"].tolist()},
            "forward_reverse_roundtrip": rt,
            "gates": gates,
            "pass": passed,
        }
        rows.append(rec)
        draw_inlier_support(anchor, forward, args.out / f"anchor416_support_to_event{event_id}.png")
        print("V102", event_id, "PASS" if passed else "FAIL", "inliers", forward["ransac_inliers"],
              "p95", round(forward["p95_px"], 4), "floor", forward["regions"]["floor"]["inliers"],
              "roundtrip", round(rt["p95_px"], 4), flush=True)

    passing = [r for r in rows if r["pass"]]
    gates = {
        "all_four_predeclared_distributed_states_pass": len(passing) == len(STATES),
        "at_least_four_independent_states_pass": len(passing) >= 4,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "status": "PASS_RIGHT_SLASH_DISTRIBUTED_FIXED_CENTER_V102" if passed else "FAIL_RIGHT_SLASH_DISTRIBUTED_FIXED_CENTER_V102",
        "game_id": "0022500301",
        "camera_label": "Right Slash",
        "anchor": {"event_id": ae, "file": af, "sha256": ash},
        "method": "single bidirectionally verified image homography across broad static 3-D scene support; same-centre pan/tilt/zoom projective test",
        "interpretation": "A pass is strong evidence consistent with one fixed optical centre across the predeclared distributed Right Slash states. It authorizes the metric shared-centre solve only; it is not a metric calibration.",
        "guardrails": [
            "native 960x540 source pixels only",
            "no player or ball landmarks",
            "camera label is not used as physical-camera proof",
            "one homography must span stands/basket zone and court floor",
            "no metric camera or replay promotion in v102",
        ],
        "states": rows,
        "passing_state_count": len(passing),
        "gates": gates,
        "permissions": {
            "shared_center_metric_attempt_allowed": passed,
            "right_slash_metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
    }
    (args.out / "right_slash_distributed_fixed_center_v102.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL", report["status"], "passing", len(passing), "of", len(STATES), flush=True)
    if not passed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
