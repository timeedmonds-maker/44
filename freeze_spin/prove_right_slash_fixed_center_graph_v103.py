from __future__ import annotations

"""v103: corrected distributed fixed-centre graph proof for Adams/Jazz Right Slash.

v102 deliberately failed closed, but its reverse test fitted two independent
homographies from different SIFT correspondence sets and compared their
composition over a full-frame grid, including unsupported extrapolation.  That
is not the invariant implied by a fixed optical centre.

v103 keeps the original per-edge image-quality gates unchanged.  It tests one
homography symmetrically on the SAME immutable correspondences, normalizing the
inverse-transfer error into destination-pixel units by the local homography
Jacobian.  It then requires a connected graph spanning four predeclared states
and an independent cycle-consistency check on actually supported inlier points.

A pass authorizes only the metric shared-centre solve.  It does not promote a
metric camera or replay render.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FRAMES = {
    375: ("f01.png", "20b6cc30e1fa49299566d53c591e404bd8b8ef5d19a7019b7c004e5c51a370cc"),
    416: ("f06.png", "325a02876fb09c89de6657a711e3241ef5382fbf39fcc1696c95686a642d2668"),
    457: ("f03.png", "e84b789e012d2a1bab6b0f3be8d13858aeec1cb682ca0c81956ddc53c57b8013"),
    540: ("f00.png", "2c10a5be6096181fd423b7d7a8b6136c4b90c97ca8cf9d3442f04ae434cd2bcd"),
}
# These three strong edges form a connected tree across all four states.
REQUIRED_EDGES = [(375, 540), (416, 540), (416, 457)]
# The fourth edge is used only to close an independent consistency cycle.
CYCLE_EDGE = (457, 540)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_native(root: Path, event_id: int) -> np.ndarray:
    fn, expected = FRAMES[event_id]
    p = root / f"event_{event_id}_selected" / fn
    if sha256(p) != expected:
        raise RuntimeError(f"immutable v101 frame SHA mismatch: {p}")
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (H, W):
        raise RuntimeError(f"missing/non-native v101 frame: {p}")
    return im


def sift_features(im: np.ndarray):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.008, edgeThreshold=12)
    return sift.detectAndCompute(g, None)


def local_area_scale(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = []
    for x, y in np.asarray(pts, dtype=np.float64):
        den = M[2, 0] * x + M[2, 1] * y + M[2, 2]
        u = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / den
        v = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / den
        J = np.array([
            [(M[0, 0] - u * M[2, 0]) / den, (M[0, 1] - u * M[2, 1]) / den],
            [(M[1, 0] - v * M[2, 0]) / den, (M[1, 1] - v * M[2, 1]) / den],
        ])
        out.append(math.sqrt(abs(float(np.linalg.det(J)))))
    return np.asarray(out, dtype=np.float64)


def fit_edge(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    ka, da = sift_features(a)
    kb, db = sift_features(b)
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
    ferr = np.linalg.norm(pred - pb, axis=1)

    invM = np.linalg.inv(M)
    invM = invM / invM[2, 2]
    back = cv2.perspectiveTransform(pb[inl, None, :], invM)[:, 0, :]
    berr = np.linalg.norm(back - pa[inl], axis=1)
    scale = local_area_scale(M, pa[inl])
    inverse_destination_equivalent = berr * scale

    x, y = pa[:, 0], pa[:, 1]
    regions = {
        "stands": y < 260,
        "basket_zone": (x > 280) & (x < 580) & (y < 250),
        "floor": y > 360,
    }
    rr = {}
    for name, region in regions.items():
        z = region & inl
        rr[name] = {
            "inliers": int(z.sum()),
            "median_px": float(np.median(ferr[z])) if z.any() else None,
            "p95_px": float(np.percentile(ferr[z], 95)) if z.any() else None,
        }
    cells = sorted({(min(int(px // 120), 7), min(int(py // 90), 5)) for px, py in pa[inl]})
    return {
        "H": M,
        "source_inliers_px": pa[inl],
        "destination_inliers_px": pb[inl],
        "ratio_test_matches": len(good),
        "ransac_inliers": int(inl.sum()),
        "forward_median_px": float(np.median(ferr[inl])),
        "forward_p95_px": float(np.percentile(ferr[inl], 95)),
        "inverse_destination_equivalent_p95_px": float(np.percentile(inverse_destination_equivalent, 95)),
        "spatial_cell_count_8x6": len(cells),
        "spatial_cells_8x6": [list(q) for q in cells],
        "regions": rr,
    }


def edge_gates(r: dict) -> dict:
    return {
        "ratio_matches_at_least_100": r["ratio_test_matches"] >= 100,
        "ransac_inliers_at_least_80": r["ransac_inliers"] >= 80,
        "forward_p95_at_most_1_6px": r["forward_p95_px"] <= 1.6,
        "inverse_destination_equivalent_p95_at_most_1_6px": r["inverse_destination_equivalent_p95_px"] <= 1.6,
        "spatial_cells_at_least_15": r["spatial_cell_count_8x6"] >= 15,
        "stands_inliers_at_least_50": r["regions"]["stands"]["inliers"] >= 50,
        "stands_p95_at_most_1_6px": (r["regions"]["stands"]["p95_px"] or 1e9) <= 1.6,
        "basket_zone_inliers_at_least_30": r["regions"]["basket_zone"]["inliers"] >= 30,
        "basket_zone_p95_at_most_1_6px": (r["regions"]["basket_zone"]["p95_px"] or 1e9) <= 1.6,
        "floor_inliers_at_least_8": r["regions"]["floor"]["inliers"] >= 8,
        "floor_p95_at_most_1_6px": (r["regions"]["floor"]["p95_px"] or 1e9) <= 1.6,
    }


def cycle_metrics(direct: dict, first: dict, second: dict) -> dict:
    comp = second["H"] @ first["H"]
    comp = comp / comp[2, 2]
    pts = direct["source_inliers_px"].astype(np.float32)
    a = cv2.perspectiveTransform(pts[:, None, :], direct["H"])[:, 0, :]
    b = cv2.perspectiveTransform(pts[:, None, :], comp)[:, 0, :]
    d = np.linalg.norm(a - b, axis=1)
    return {
        "count": int(len(d)),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(np.max(d)),
    }


def serializable_edge(r: dict) -> dict:
    return {
        "H": r["H"].tolist(),
        "ratio_test_matches": r["ratio_test_matches"],
        "ransac_inliers": r["ransac_inliers"],
        "forward_median_px": r["forward_median_px"],
        "forward_p95_px": r["forward_p95_px"],
        "inverse_destination_equivalent_p95_px": r["inverse_destination_equivalent_p95_px"],
        "spatial_cell_count_8x6": r["spatial_cell_count_8x6"],
        "spatial_cells_8x6": r["spatial_cells_8x6"],
        "regions": r["regions"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v101-root", type=Path, required=True)
    ap.add_argument("--v102-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    q102 = json.loads(args.v102_report.read_text())
    if q102.get("status") != "FAIL_RIGHT_SLASH_DISTRIBUTED_FIXED_CENTER_V102":
        raise RuntimeError("v103 requires the exact fail-closed v102 prerequisite")
    if q102.get("permissions", {}).get("shared_center_metric_attempt_allowed") is not False:
        raise RuntimeError("v102 prerequisite permissions changed")

    images = {eid: load_native(args.v101_root, eid) for eid in FRAMES}
    edges = {}
    for a, b in REQUIRED_EDGES + [CYCLE_EDGE]:
        r = fit_edge(images[a], images[b], 103000 + 31 * a + b)
        edges[(a, b)] = r
        g = edge_gates(r)
        print("V103 EDGE", a, b, "PASS" if all(g.values()) else "FAIL",
              "inliers", r["ransac_inliers"], "p95", round(r["forward_p95_px"], 4),
              "inverse_eq", round(r["inverse_destination_equivalent_p95_px"], 4),
              "floor", r["regions"]["floor"]["inliers"], flush=True)

    required = []
    for e in REQUIRED_EDGES:
        r = edges[e]
        g = edge_gates(r)
        required.append({"edge": list(e), "metrics": serializable_edge(r), "gates": g, "pass": bool(all(g.values()))})

    # 416 -> 457 -> 540 must agree with the direct 416 -> 540 transform on
    # direct-edge inliers.  This is an independent composition check and is
    # evaluated only where the direct transform has real correspondence support.
    cyc = cycle_metrics(edges[(416, 540)], edges[(416, 457)], edges[(457, 540)])
    cycle_gate = cyc["count"] >= 80 and cyc["p95_px"] <= 2.5

    all_required = all(r["pass"] for r in required)
    nodes = sorted({x for e in REQUIRED_EDGES for x in e})
    graph_gate = all_required and nodes == [375, 416, 457, 540]
    passed = bool(graph_gate and cycle_gate)

    report = {
        "schema_version": 1,
        "status": "PASS_RIGHT_SLASH_FIXED_CENTER_GRAPH_V103" if passed else "FAIL_RIGHT_SLASH_FIXED_CENTER_GRAPH_V103",
        "game_id": "0022500301",
        "camera_label": "Right Slash",
        "prerequisite": {
            "v102_status": q102["status"],
            "v102_preserved_as_failed_diagnostic": True,
            "correction": "Replace independently-fitted forward/reverse full-grid roundtrip with one-H symmetric transfer on identical correspondences plus supported cycle consistency.",
        },
        "frames": {str(k): {"file": FRAMES[k][0], "sha256": FRAMES[k][1]} for k in sorted(FRAMES)},
        "required_edges": required,
        "cycle_diagnostic_edge_457_to_540": serializable_edge(edges[(457, 540)]),
        "cycle_416_to_457_to_540_vs_direct_416_to_540": cyc,
        "gates": {
            "all_three_required_edges_pass_original_per_edge_thresholds": all_required,
            "required_edges_form_connected_four_state_graph": graph_gate,
            "cycle_supported_inliers_at_least_80": cyc["count"] >= 80,
            "cycle_supported_p95_at_most_2_5px": cyc["p95_px"] <= 2.5,
        },
        "interpretation": "Passing is strong projective evidence consistent with one fixed Right Slash optical centre across four distributed same-game states. It authorizes a metric shared-centre attempt only.",
        "guardrails": [
            "native 960x540 immutable v101 pixels only",
            "no player or ball landmarks",
            "per-edge 1.6 px and 80-inlier thresholds are not relaxed from v102",
            "cycle is evaluated on supported direct-edge inliers, not unsupported full-frame extrapolation",
            "no metric-camera or replay promotion in v103",
        ],
        "permissions": {
            "shared_center_metric_attempt_allowed": passed,
            "right_slash_metric_camera_allowed": False,
            "replay_render_allowed": False,
        },
    }
    (args.out / "right_slash_fixed_center_graph_v103.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL", report["status"], "cycle_p95", round(cyc["p95_px"], 4), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
