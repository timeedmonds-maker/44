from __future__ import annotations

"""Mine recurring physical optical states inside an NBA camera-labelled feed.

Discovery only. NBA feed labels can contain cuts between multiple physical cameras.
This tool ranks individual native frames against an immutable target frame using
static-scene feature geometry. It deliberately does not promote a camera centre,
metric camera, novel view, or replay.

The cross-depth diagnostic fits a homography from upper-court correspondences and
measures how well it predicts elevated basket/support correspondences. In raw,
potentially distorted pixels this is a diagnostic rather than a proof; large
cross-depth residual is a fail-closed warning against treating a visually similar
state as a fixed optical centre.
"""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.scan_same_game_camera_priors import discover_events, extract_frames, safe_label, w

IMAGE_W = 960
IMAGE_H = 540


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grid_spread(points: np.ndarray, cols: int = 6, rows: int = 4) -> int:
    if points.size == 0:
        return 0
    cells = set()
    for x, y in points:
        gx = min(cols - 1, max(0, int(float(x) / IMAGE_W * cols)))
        gy = min(rows - 1, max(0, int(float(y) / IMAGE_H * rows)))
        cells.add((gx, gy))
    return len(cells)


def robust_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "median_px": None, "p75_px": None, "p90_px": None, "p95_px": None}
    return {
        "n": int(values.size),
        "median_px": float(np.median(values)),
        "p75_px": float(np.percentile(values, 75)),
        "p90_px": float(np.percentile(values, 90)),
        "p95_px": float(np.percentile(values, 95)),
    }


def analyze_frame(target_kp, target_desc, source: Path, ratio: float = 0.72, ransac_px: float = 2.5) -> dict:
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"status": "bad_image"}
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=10)
    source_kp, source_desc = sift.detectAndCompute(gray, None)
    if source_desc is None or len(source_kp) < 20:
        return {"status": "insufficient_features"}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(target_desc, source_desc, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 24:
        return {"status": "insufficient_matches", "good_matches": len(good)}

    target_xy = np.float32([target_kp[m.queryIdx].pt for m in good])
    source_xy = np.float32([source_kp[m.trainIdx].pt for m in good])

    H_all, mask_all = cv2.findHomography(
        target_xy, source_xy, cv2.RANSAC, ransac_px, maxIters=10000, confidence=0.999
    )
    if H_all is None or mask_all is None:
        return {"status": "full_homography_failed", "good_matches": len(good)}
    in_all = mask_all.ravel().astype(bool)
    predicted_all = cv2.perspectiveTransform(target_xy.reshape(-1, 1, 2), H_all).reshape(-1, 2)
    err_all = np.linalg.norm(predicted_all - source_xy, axis=1)
    full_err = err_all[in_all]

    # Upper court is visually separated from the elevated rim/backboard/support in
    # the immutable Frame C target. Fit only this floor-dominated region, then
    # evaluate the elevated region without using it to estimate the homography.
    court_fit = target_xy[:, 1] < 190.0
    support_eval = (
        (target_xy[:, 1] >= 350.0)
        & (target_xy[:, 0] >= 120.0)
        & (target_xy[:, 0] <= 840.0)
    )

    cross_depth = {
        "status": "insufficient_court_or_support_matches",
        "court_candidate_matches": int(np.sum(court_fit)),
        "support_candidate_matches": int(np.sum(support_eval)),
    }
    if int(np.sum(court_fit)) >= 12 and int(np.sum(support_eval)) >= 20:
        H_court, mask_court = cv2.findHomography(
            target_xy[court_fit], source_xy[court_fit], cv2.RANSAC, ransac_px,
            maxIters=10000, confidence=0.999
        )
        if H_court is not None and mask_court is not None:
            pred = cv2.perspectiveTransform(target_xy.reshape(-1, 1, 2), H_court).reshape(-1, 2)
            err = np.linalg.norm(pred - source_xy, axis=1)
            support_err = err[support_eval]
            cross_depth = {
                "status": "diagnostic_only_raw_pixel_homography",
                "court_candidate_matches": int(np.sum(court_fit)),
                "court_ransac_inliers": int(np.sum(mask_court)),
                "support_candidate_matches": int(np.sum(support_eval)),
                "support_residual": robust_stats(support_err),
                "support_fraction_under_3px": float(np.mean(support_err <= 3.0)),
                "support_fraction_under_5px": float(np.mean(support_err <= 5.0)),
                "interpretation": "Raw-pixel cross-depth consistency diagnostic only; lens distortion can contribute residual. Not a fixed-centre proof.",
            }

    full = {
        "good_matches": len(good),
        "ransac_inliers": int(np.sum(in_all)),
        "inlier_ratio": float(np.mean(in_all)),
        "inlier_spatial_grid_cells_6x4": grid_spread(target_xy[in_all]),
        "inlier_residual": robust_stats(full_err),
    }

    support_med = cross_depth.get("support_residual", {}).get("median_px")
    # Discovery score only. Prefer broad, numerous full-scene agreement and low
    # court->support residual. Missing cross-depth evidence is strongly penalized.
    if support_med is None:
        score = 1e6
    else:
        score = (
            float(support_med)
            + 0.35 * float(full["inlier_residual"]["p95_px"] or 50.0)
            + 18.0 / math.sqrt(max(1, full["ransac_inliers"]))
            + max(0, 10 - full["inlier_spatial_grid_cells_6x4"]) * 2.0
        )

    credible_state_candidate = bool(
        full["ransac_inliers"] >= 60
        and full["inlier_spatial_grid_cells_6x4"] >= 8
        and cross_depth.get("court_ransac_inliers", 0) >= 15
        and cross_depth.get("support_candidate_matches", 0) >= 30
    )
    return {
        "status": "ok",
        "discovery_score_lower_is_better": float(score),
        "credible_state_candidate": credible_state_candidate,
        "full_scene": full,
        "cross_depth": cross_depth,
    }


def make_montage(target: Path, candidates: list[dict], out: Path) -> None:
    ims = []
    target_im = cv2.imread(str(target))
    if target_im is None:
        return
    cv2.putText(target_im, "IMMUTABLE FRAME C TARGET", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(target_im, "IMMUTABLE FRAME C TARGET", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    ims.append(target_im)
    for c in candidates:
        im = cv2.imread(c["selected_frame"])
        if im is None:
            continue
        m = c["analysis"]
        support = m.get("cross_depth", {}).get("support_residual", {}).get("median_px")
        text = f"event {c['event_probe']} {c['sample_name']} score={m['discovery_score_lower_is_better']:.2f} supportMed={support if support is not None else -1:.2f}px"
        cv2.putText(im, text[:120], (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(im, text[:120], (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        ims.append(im)
    if not ims:
        return
    cell_w, cell_h = IMAGE_W, IMAGE_H
    cols = 3
    rows = int(math.ceil(len(ims) / cols))
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
    for i, im in enumerate(ims):
        y, x = divmod(i, cols)
        canvas[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w] = im
    cv2.imwrite(str(out), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--camera-label", required=True)
    ap.add_argument("--target-frame", type=Path, required=True)
    ap.add_argument("--target-sha256")
    ap.add_argument("--count", type=int, default=36)
    ap.add_argument("--event-start", type=int, default=5)
    ap.add_argument("--event-stop", type=int, default=1205)
    ap.add_argument("--event-step", type=int, default=20)
    ap.add_argument("--samples-per-clip", type=int, default=7)
    ap.add_argument("--keep-top", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    selected_dir = args.out / "selected_frames"
    selected_dir.mkdir(exist_ok=True)

    actual_target_sha = sha256(args.target_frame)
    if args.target_sha256 and actual_target_sha != args.target_sha256:
        raise SystemExit(f"Immutable target SHA mismatch: {actual_target_sha}")

    target_gray = cv2.imread(str(args.target_frame), cv2.IMREAD_GRAYSCALE)
    if target_gray is None or target_gray.shape != (IMAGE_H, IMAGE_W):
        raise SystemExit(f"Bad immutable target geometry: {None if target_gray is None else target_gray.shape}")
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=10)
    target_kp, target_desc = sift.detectAndCompute(target_gray, None)
    if target_desc is None:
        raise SystemExit("No target SIFT descriptors")

    discovered = discover_events(
        args.game_id, args.camera_label, args.count,
        args.event_start, args.event_stop, args.event_step,
    )

    event_results = []
    slug = safe_label(args.camera_label)
    work = args.out / "work"
    work.mkdir(exist_ok=True)

    for rank, d in enumerate(discovered, 1):
        eid = int(d["event_id"])
        rec = {"rank": rank, "event_probe": eid, "title": d["title"]}
        clip = work / f"event_{eid}_{slug}_SOURCE.mp4"
        frame_dir = work / f"event_{eid}_frames"
        try:
            w.download_hls_source(d["url"], clip)
            q = w.probe_video(clip)
            if not q.get("ok"):
                raise RuntimeError(q.get("reason"))
            frames = extract_frames(clip, frame_dir, n=args.samples_per_clip)
            frame_analyses = []
            for p in frames:
                a = analyze_frame(target_kp, target_desc, p)
                frame_analyses.append({"path": p, "analysis": a})
            valid = [x for x in frame_analyses if x["analysis"].get("status") == "ok"]
            if not valid:
                raise RuntimeError("No target-matchable samples")
            valid.sort(key=lambda x: x["analysis"].get("discovery_score_lower_is_better", 1e9))
            best = valid[0]
            selected_name = f"event_{eid:04d}_{slug}_{best['path'].name}"
            selected_path = selected_dir / selected_name
            shutil.copy2(best["path"], selected_path)
            rec.update({
                "status": "ok",
                "probe": q,
                "sample_name": best["path"].name,
                "selected_frame": str(selected_path),
                "selected_frame_sha256": sha256(selected_path),
                "analysis": best["analysis"],
                "all_sample_summaries": [
                    {
                        "sample_name": x["path"].name,
                        "status": x["analysis"].get("status"),
                        "score": x["analysis"].get("discovery_score_lower_is_better"),
                        "credible_state_candidate": x["analysis"].get("credible_state_candidate", False),
                        "full_inliers": x["analysis"].get("full_scene", {}).get("ransac_inliers"),
                        "support_median_px": x["analysis"].get("cross_depth", {}).get("support_residual", {}).get("median_px"),
                    }
                    for x in frame_analyses
                ],
            })
        except Exception as e:
            rec.update({"status": "failed", "error": repr(e)})
        finally:
            clip.unlink(missing_ok=True)
            shutil.rmtree(frame_dir, ignore_errors=True)
        event_results.append(rec)
        print(f"[{rank}/{len(discovered)}] event {eid}: {rec['status']}", flush=True)

    ok = [r for r in event_results if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["analysis"].get("discovery_score_lower_is_better", 1e9))
    top = ok[:args.keep_top]
    keep_names = {Path(r["selected_frame"]).name for r in top}
    for p in selected_dir.glob("*.png"):
        if p.name not in keep_names:
            p.unlink()

    # Convert stored paths to artifact-relative paths after pruning.
    for r in event_results:
        if r.get("status") == "ok":
            p = Path(r["selected_frame"])
            r["selected_frame"] = str(Path("selected_frames") / p.name)
    for r in top:
        p = Path(r["selected_frame"])
        r["selected_frame"] = str(args.out / p)

    make_montage(args.target_frame, top, args.out / "top_state_candidates_montage.png")
    for r in top:
        r["selected_frame"] = str(Path(r["selected_frame"]).relative_to(args.out))

    payload = {
        "game_id": args.game_id,
        "camera_label": args.camera_label,
        "immutable_target": {
            "file": args.target_frame.name,
            "sha256": actual_target_sha,
            "geometry": f"{IMAGE_W}x{IMAGE_H}",
        },
        "purpose": "Mine recurring physical optical states hidden inside a camera-labelled NBA feed before any metric-camera proof.",
        "method": {
            "event_count_requested": args.count,
            "event_probe_range": [args.event_start, args.event_stop, args.event_step],
            "samples_per_clip": args.samples_per_clip,
            "sift_lowe_ratio": 0.72,
            "ransac_threshold_px": 2.5,
            "cross_depth_diagnostic": "fit raw-pixel homography on target y<190 court matches; evaluate target y>=350 support matches without fitting them",
            "warning": "Cross-depth raw-pixel residual is discovery evidence only. Lens distortion can contribute. It cannot authorize a fixed camera centre.",
        },
        "events": event_results,
        "top_candidates": [
            {
                "event_probe": r["event_probe"],
                "title": r["title"],
                "sample_name": r["sample_name"],
                "selected_frame": r["selected_frame"],
                "selected_frame_sha256": r["selected_frame_sha256"],
                "analysis": r["analysis"],
            }
            for r in top
        ],
        "credible_candidate_event_count": int(sum(r["analysis"].get("credible_state_candidate", False) for r in ok)),
        "permissions": {
            "fixed_camera_center_prior_allowed": False,
            "metric_camera_promotion_allowed": False,
            "static_novel_view_allowed": False,
            "replay_render_allowed": False,
        },
        "status": "DISCOVERY_ONLY_NO_PROMOTION",
    }
    (args.out / "same_game_physical_state_mining_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "successful_events": len(ok),
        "credible_candidate_event_count": payload["credible_candidate_event_count"],
        "top": [(r["event_probe"], r["sample_name"], round(r["analysis"]["discovery_score_lower_is_better"], 3)) for r in top],
    }, indent=2), flush=True)
    if len(ok) < max(5, args.count // 3):
        raise SystemExit(f"Only {len(ok)}/{args.count} events yielded target-matchable samples")


if __name__ == "__main__":
    main()
