from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.triangulate_locked_ball_v1 import (
    CAMERA_INDEX,
    STATIC_ROI,
    estimate_selected_to_anchor_homography,
    project,
)


def camera_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["label"]: row for row in payload["cameras"]}


def solve_point(cameras: dict[str, dict], observations: dict[str, np.ndarray], sigmas: dict[str, float]) -> tuple[np.ndarray, float]:
    labels = list(observations)
    A = []
    for label in labels:
        P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
        u, v = observations[label]
        A.extend([u * P[2] - P[0], v * P[2] - P[1]])
    _, _, vt = np.linalg.svd(np.asarray(A))
    h = vt[-1]
    x0 = h[:3] / h[3]

    def residual(X: np.ndarray) -> np.ndarray:
        rows = []
        for label in labels:
            P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
            rows.extend((project(P, X) - observations[label]) / sigmas[label])
        return np.asarray(rows)

    fit = least_squares(residual, x0, loss="huber", f_scale=1.0)
    return fit.x, float(fit.cost)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--ball", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--anchor-images", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cameras = camera_map(args.cameras)
    spec = json.loads(args.landmarks.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    ball = np.asarray(json.loads(args.ball.read_text(encoding="utf-8"))["ball_center_world_cm"], dtype=np.float64)

    H = {}
    selected_images = {}
    motion_qa = {}
    for label in CAMERA_INDEX:
        frame = int(state["selected_frames"][label])
        anchor = cv2.imread(str(args.anchor_images / f"{CAMERA_INDEX[label]}.png"))
        selected = cv2.imread(str(args.locked_images / f"{label.replace(' ', '_')}_F{frame}.png"))
        if anchor is None or selected is None:
            raise FileNotFoundError(label)
        H[label], motion_qa[label] = estimate_selected_to_anchor_homography(anchor, selected, STATIC_ROI[label])
        selected_images[label] = selected

    points = {}
    for name, row in spec["landmarks"].items():
        obs = {}
        sigma = {}
        for label, item in row["views"].items():
            p = np.asarray(item["pixel_xy_selected_frame"], dtype=np.float64)
            obs[label] = cv2.perspectiveTransform(p.reshape(1, 1, 2), H[label])[0, 0]
            sigma[label] = float(item["uncertainty_px"])
        X, cost = solve_point(cameras, obs, sigma)
        views = {}
        max_norm = 0.0
        for label in obs:
            P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
            pred_anchor = project(P, X)
            err = float(np.linalg.norm(pred_anchor - obs[label]))
            norm = err / sigma[label]
            max_norm = max(max_norm, norm)
            pred_selected = cv2.perspectiveTransform(pred_anchor.reshape(1, 1, 2), np.linalg.inv(H[label]))[0, 0]
            views[label] = {
                "reprojection_error_px": round(err, 4),
                "normalized_error_sigma": round(norm, 4),
                "predicted_selected_px": [round(float(v), 3) for v in pred_selected],
            }
        points[name] = {
            "player": row["player"],
            "joint": row["joint"],
            "world_cm": [round(float(v), 6) for v in X],
            "optimizer_cost": round(cost, 8),
            "max_normalized_error_sigma": round(max_norm, 4),
            "views": views,
        }

    chains = {}
    for chain, names in spec["chains"].items():
        rows = []
        for a, b in zip(names, names[1:]):
            A = np.asarray(points[a]["world_cm"])
            B = np.asarray(points[b]["world_cm"])
            rows.append({"a": a, "b": b, "length_cm": round(float(np.linalg.norm(A - B)), 6)})
        chains[chain] = rows

    ah = np.asarray(points["adams_block_hand"]["world_cm"])
    ch = np.asarray(points["cissoko_ball_hand"]["world_cm"])
    interaction = {
        "adams_hand_to_ball_center_cm": round(float(np.linalg.norm(ah - ball)), 6),
        "cissoko_hand_to_ball_center_cm": round(float(np.linalg.norm(ch - ball)), 6),
        "hand_to_hand_cm": round(float(np.linalg.norm(ah - ch)), 6),
    }

    gate = {
        "all_points_use_at_least_three_views": all(len(row["views"]) >= 3 for row in spec["landmarks"].values()),
        "all_points_within_1_5_sigma": all(row["max_normalized_error_sigma"] <= 1.5 for row in points.values()),
        "both_hands_near_ball": 5.0 <= interaction["adams_hand_to_ball_center_cm"] <= 22.0 and 5.0 <= interaction["cissoko_hand_to_ball_center_cm"] <= 22.0,
        "hands_close_at_interaction": interaction["hand_to_hand_cm"] <= 15.0,
    }
    for chain, rows in chains.items():
        gate[f"{chain}_broad_segment_lengths"] = all(7.0 <= row["length_cm"] <= 50.0 for row in rows)
    gate["pass"] = bool(all(gate.values()))

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "provisional_forearm_metric_proof_only",
        "identity_guardrail": "Shoulder/torso anchors or another separated view are required before full player identity is accepted.",
        "points": points,
        "chains": chains,
        "interaction": interaction,
        "motion_qa": motion_qa,
        "metric_gate": gate,
        "full_player_identity_gate": {"pass": False},
    }
    (args.out / "semantic_points_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
