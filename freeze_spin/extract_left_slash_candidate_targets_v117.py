from __future__ import annotations

"""v117a: direct target-plane observations for v116 Left Slash PTZ states.

The v116 full-scene homography is used only to predict a local search window.
Each candidate's four backboard target inner edges/corners are then re-observed
from that candidate's own native pixels via the deterministic line fitter used
by v113. No rim observation is consumed as metric evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import extract_left_slash_event75_geometry_v112 as base

FRAME_C_SHA256 = "2ced6fbf7108459e4c8acda1d62ab8a4a77a455971287c1fcfcdc6ea7b6a272f"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def perspective(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def draw(im: np.ndarray, pred: np.ndarray, obs: np.ndarray, label: str, out: Path) -> None:
    q = im.copy()
    cv2.polylines(q, [np.round(pred).astype(np.int32)], True, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.polylines(q, [np.round(obs).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for p in np.round(obs).astype(np.int32):
        cv2.circle(q, tuple(p), 3, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(q, label[:150], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(q, label[:150], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v116-json", type=Path, required=True)
    ap.add_argument("--v116-root", type=Path, required=True)
    ap.add_argument("--frame-c-geometry", type=Path, required=True)
    ap.add_argument("--min-states", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    v116 = json.loads(args.v116_json.read_text())
    fc = json.loads(args.frame_c_geometry.read_text())
    if v116.get("game_id") != "0022500527" or v116.get("camera_label") != "Left Slash":
        raise SystemExit("v117 requires HOU-POR 0022500527 Left Slash v116 evidence")
    if v116.get("status") != "DISCOVERY_ONLY_NO_PROMOTION":
        raise SystemExit("unexpected v116 status")
    if fc.get("status") != "PASS_LEFT_SLASH_FRAME_C_SOURCE_GEOMETRY_V113":
        raise SystemExit("v117 requires passing v113 Frame-C source geometry")
    if fc.get("frame_c_sha256") != FRAME_C_SHA256:
        raise SystemExit("v113 immutable Frame-C provenance mismatch")

    target_fc = np.asarray(fc["target"]["source_observed_inner_corners_px"], np.float64)
    selected = []
    seen_events = set()
    for r in v116.get("top_candidates", []):
        a = r.get("analysis", {})
        if not a.get("credible_fixed_center_state_candidate") or not a.get("materially_different_optical_state"):
            continue
        eid = int(r["event_probe"])
        if eid in seen_events:
            continue
        seen_events.add(eid)
        selected.append(r)
        if len(selected) >= args.max_states:
            break
    if len(selected) < args.min_states:
        raise SystemExit(f"only {len(selected)} independent credible v116 events; need {args.min_states}")

    states = []
    overlays = []
    for rank, r in enumerate(selected, 1):
        p = args.v116_root / r["selected_frame"]
        if not p.exists():
            raise SystemExit(f"missing selected candidate {p}")
        expected = r.get("selected_frame_sha256")
        if expected and sha256(p) != expected:
            raise SystemExit(f"candidate SHA mismatch: {p.name}")
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (540, 960):
            raise SystemExit(f"bad native candidate {p}")
        H = np.asarray(r["analysis"]["homography"], np.float64)
        pred = perspective(H, target_fc)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fit = base.choose_target_lines(gray, pred)
        if fit is None:
            states.append({"event_probe": int(r["event_probe"]), "sample_name": r["sample_name"], "status": "DIRECT_TARGET_FIT_FAILED"})
            continue
        _lines, diag, obs = fit
        shift = np.linalg.norm(obs - pred, axis=1)
        angle_max = max(float(d["angle_error_deg"]) for d in diag)
        mid_max = max(float(d["midline_distance_px"]) for d in diag)
        gates = {
            "corner_shift_from_homography_prior_max_at_most_8px": float(shift.max()) <= 8.0,
            "line_angle_error_max_at_most_12deg": angle_max <= 12.0,
            "line_mid_distance_max_at_most_12px": mid_max <= 12.0,
        }
        passed = all(gates.values())
        ov = args.out / f"state_{rank:02d}_event_{int(r['event_probe']):04d}_target.png"
        draw(im, pred, obs, f"event {r['event_probe']} {r['sample_name']} pass={passed} shift={shift.max():.2f}px", ov)
        overlays.append(ov.name)
        states.append({
            "event_probe": int(r["event_probe"]),
            "title": r.get("title"),
            "sample_name": r["sample_name"],
            "selected_frame": r["selected_frame"],
            "selected_frame_sha256": sha256(p),
            "status": "PASS_DIRECT_TARGET_V117" if passed else "FAIL_DIRECT_TARGET_V117",
            "homography_prior": H.tolist(),
            "homography_transform": r["analysis"].get("transform", {}),
            "homography_quality": {
                "ransac_inliers": r["analysis"].get("ransac_inliers"),
                "inlier_ratio": r["analysis"].get("inlier_ratio"),
                "grid_cells_6x4": r["analysis"].get("grid_cells_6x4"),
                "inlier_residual": r["analysis"].get("inlier_residual"),
            },
            "transfer_prior_inner_corners_px": pred.tolist(),
            "source_observed_inner_corners_px": obs.tolist(),
            "corner_shift_from_prior_px": shift.tolist(),
            "corner_shift_max_px": float(shift.max()),
            "line_diagnostics": diag,
            "gates": gates,
            "overlay": ov.name,
        })

    passed_states = [s for s in states if s.get("status") == "PASS_DIRECT_TARGET_V117"]
    payload = {
        "status": "PASS_LEFT_SLASH_CANDIDATE_TARGETS_V117" if len(passed_states) >= args.min_states else "FAIL_LEFT_SLASH_CANDIDATE_TARGETS_V117",
        "purpose": "Native-pixel target-plane observations for independent same-game Left Slash PTZ states",
        "game_id": "0022500527",
        "camera_label": "Left Slash",
        "immutable_frame_c_sha256": FRAME_C_SHA256,
        "frame_c_source_observed_inner_corners_px": target_fc.tolist(),
        "candidate_states": states,
        "passing_state_count": len(passed_states),
        "permissions": {"metric_camera_promotion_allowed": False, "replay_render_allowed": False},
        "guardrail": "Homographies are local search priors only; metric solve consumes direct target lines. Rim centreline is excluded from metric evidence.",
        "overlays": overlays,
    }
    (args.out / "left_slash_candidate_targets_v117.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"status": payload["status"], "passing_state_count": len(passed_states), "states": [(s.get("event_probe"), s.get("status"), s.get("corner_shift_max_px")) for s in states]}, indent=2))
    raise SystemExit(0 if len(passed_states) >= args.min_states else 2)


if __name__ == "__main__":
    main()
