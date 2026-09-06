from __future__ import annotations

"""Preflight a per-game physical-camera registry from real NBA source pixels.

For each official camera label, compare the immutable target freeze against several
same-game events. If a camera has a fixed optical centre, pan/tilt/zoom changes are
related by one global image homography for all static scene depths. We therefore fit
H only on conservative background features and validate it on spatially separated
held-out static features. Moving players are never used as metric anchors and a pass
here grants only fixed-centre *candidate* status, never metric camera promotion.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
CLIP_RE = re.compile(r"^\d+_R(\d+)_(\d+)_(\d+)_(.+)_SOURCE\.mp4$")


def label_from_token(token: str) -> str:
    return token.replace("_", " ")


def safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def read_frame(cap: cv2.VideoCapture, frac: float) -> np.ndarray | None:
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        return None
    idx = int(np.clip(round(frac * (count - 1)), 0, count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[1] != W or frame.shape[0] != H:
        return None
    return frame


def extract_event_samples(clips: Path, out: Path, target_event: int) -> dict[str, dict[int, list[Path]]]:
    out.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for clip in sorted(clips.glob("*_SOURCE.mp4")):
        m = CLIP_RE.match(clip.name)
        if not m:
            continue
        _, _, event_s, token = m.groups()
        event_id = int(event_s)
        label = label_from_token(token)
        if event_id == target_event:
            # The target is represented by the separately synchronized immutable Frame C.
            continue
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            continue
        for j, frac in enumerate((0.25, 0.50, 0.75)):
            frame = read_frame(cap, frac)
            if frame is None:
                continue
            dst = out / f"{safe_name(label)}__event{event_id:04d}__s{j}.png"
            cv2.imwrite(str(dst), frame)
            grouped[label][event_id].append(dst)
        cap.release()
    return grouped


def target_frames_by_label(target_dir: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(target_dir.glob("*.png")):
        stem = p.stem
        # e.g. I_Left_HandHeld_8.875603s_frame0266
        m = re.match(r"^[A-L]_(.+?)_\d+\.\d+s_frame\d+$", stem)
        if not m:
            continue
        out[label_from_token(m.group(1))] = p
    return out


def sift_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.015)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    raw = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < 0.72 * n.distance]
    if not good:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    # Enforce one-to-one target assignment.
    best = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return pa, pb


def err_stats(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {"n": 0, "median_px": None, "p90_px": None, "p95_px": None}
    return {
        "n": int(len(err)),
        "median_px": float(np.median(err)),
        "p90_px": float(np.percentile(err, 90)),
        "p95_px": float(np.percentile(err, 95)),
    }


def transfer_err(Hm: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if len(p) == 0:
        return np.empty(0, np.float64)
    pred = cv2.perspectiveTransform(p[:, None, :], Hm)[:, 0]
    return np.linalg.norm(pred - q, axis=1)


def audit_pair(source_path: Path, target_path: Path) -> dict:
    a = cv2.imread(str(source_path))
    b = cv2.imread(str(target_path))
    rec = {"source": str(source_path), "target": str(target_path), "pass": False}
    if a is None or b is None or a.shape[:2] != (H, W) or b.shape[:2] != (H, W):
        rec["status"] = "unreadable_or_non_native"
        return rec
    p, q = sift_points(a, b)
    rec["match_count"] = int(len(p))
    if len(p) < 30:
        rec["status"] = "insufficient_matches"
        return rec

    # Exclude the lower central action region from model fitting in BOTH images.
    def action_core(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        return (x > 0.20 * W) & (x < 0.80 * W) & (y > 0.48 * H) & (y < 0.98 * H)

    # Training comes from arena/crowd and image edges. The held-out band sits at a
    # different collection of scene depths, including basket/court features when visible.
    xa, ya = p[:, 0], p[:, 1]
    xb, yb = q[:, 0], q[:, 1]
    train_geom = (ya < 0.46 * H) | (xa < 0.14 * W) | (xa > 0.86 * W)
    train_geom &= (yb < 0.46 * H) | (xb < 0.14 * W) | (xb > 0.86 * W)
    training = train_geom & ~action_core(p) & ~action_core(q)
    withheld = ~training & ~action_core(p) & ~action_core(q)
    rec["training_count"] = int(training.sum())
    rec["withheld_count"] = int(withheld.sum())
    if int(training.sum()) < 12:
        rec["status"] = "insufficient_background_training"
        return rec

    Hm, mask = cv2.findHomography(p[training], q[training], cv2.RANSAC, 1.5, maxIters=30000, confidence=0.999)
    if Hm is None or mask is None:
        rec["status"] = "homography_failed"
        return rec
    inlier = mask.ravel().astype(bool)
    train_inliers = int(inlier.sum())
    tr_err = transfer_err(Hm, p[training][inlier], q[training][inlier])
    wh_err = transfer_err(Hm, p[withheld], q[withheld])
    tr = err_stats(tr_err)
    wh = err_stats(wh_err)
    rec.update({"training_inliers": train_inliers, "training_error": tr, "withheld_error": wh})

    gates = {
        "training_inliers_at_least_24": train_inliers >= 24,
        "training_p95_at_most_1_5px": tr["p95_px"] is not None and tr["p95_px"] <= 1.5,
        "withheld_matches_at_least_10": wh["n"] >= 10,
        "withheld_median_at_most_2_5px": wh["median_px"] is not None and wh["median_px"] <= 2.5,
        "withheld_p90_at_most_4px": wh["p90_px"] is not None and wh["p90_px"] <= 4.0,
    }
    rec["gates"] = gates
    rec["pass"] = bool(all(gates.values()))
    rec["status"] = "fixed_center_transfer_candidate" if rec["pass"] else "transfer_rejected"
    rec["H_source_to_target"] = Hm.tolist()
    return rec


def best_event_pair(paths: list[Path], target: Path) -> dict:
    rows = [audit_pair(p, target) for p in paths]
    def score(r: dict) -> tuple:
        passed = 1 if r.get("pass") else 0
        wh = r.get("withheld_error") or {}
        med = wh.get("median_px")
        p90 = wh.get("p90_px")
        return (passed, int(r.get("training_inliers", 0)), -(med if med is not None else 1e9), -(p90 if p90 is not None else 1e9))
    rows.sort(key=score, reverse=True)
    return rows[0] if rows else {"pass": False, "status": "no_samples"}


def make_sheet(label: str, target: Path, event_paths: dict[int, list[Path]], out: Path) -> None:
    tiles = []
    target_im = cv2.imread(str(target))
    if target_im is not None:
        im = target_im.copy(); cv2.putText(im, "FRAME C TARGET", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA); tiles.append(im)
    for event_id in sorted(event_paths):
        if not event_paths[event_id]:
            continue
        im = cv2.imread(str(event_paths[event_id][len(event_paths[event_id]) // 2]))
        if im is None:
            continue
        im = im.copy(); cv2.putText(im, f"EVENT {event_id}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA); tiles.append(im)
    if not tiles:
        return
    tw, th = 480, 270
    tiles = [cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA) for x in tiles]
    cols = 3; rows = math.ceil(len(tiles) / cols)
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, cols); canvas[y*th:(y+1)*th, x*tw:(x+1)*tw] = tile
    cv2.putText(canvas, label, (12, canvas.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--target-frames", type=Path, required=True)
    ap.add_argument("--target-event", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sample_root = args.out / "samples"
    grouped = extract_event_samples(args.clips, sample_root, args.target_event)
    targets = target_frames_by_label(args.target_frames)

    labels = sorted(set(grouped) & set(targets))
    report = {
        "method": "same-game global-homography fixed-optical-centre preflight",
        "semantics": "positive evidence can nominate a fixed-centre candidate; failure does not prove a camera is mobile; no metric camera is promoted here",
        "target_event": args.target_event,
        "source_resolution": [W, H],
        "labels": {},
    }

    fixed_candidates = []
    for label in labels:
        event_results = []
        for event_id, paths in sorted(grouped[label].items()):
            best = best_event_pair(paths, targets[label])
            best["event_id"] = event_id
            event_results.append(best)
        passing = [r for r in event_results if r.get("pass")]
        total = len(event_results)
        coverage = len(passing) / total if total else 0.0
        if len(passing) >= 3 and coverage >= 0.50:
            status = "FIXED_CENTER_CANDIDATE"
            fixed_candidates.append(label)
        elif len(passing) >= 1:
            status = "MIXED_OR_INSUFFICIENT"
        else:
            status = "NO_FIXED_CENTER_EVIDENCE"
        rec = {
            "status": status,
            "same_game_event_count": total,
            "target_transfer_pass_count": len(passing),
            "target_transfer_coverage": coverage,
            "best_passing_event": None,
            "events": event_results,
        }
        if passing:
            passing.sort(key=lambda r: (int(r.get("training_inliers", 0)), -float((r.get("withheld_error") or {}).get("median_px") or 1e9)), reverse=True)
            rec["best_passing_event"] = passing[0].get("event_id")
        report["labels"][label] = rec
        make_sheet(label, targets[label], grouped[label], args.out / f"contact_{safe_name(label)}.jpg")
        print(label, status, len(passing), "/", total, "best", rec["best_passing_event"], flush=True)

    report["fixed_center_candidates"] = fixed_candidates
    report["fixed_center_candidate_count"] = len(fixed_candidates)
    (args.out / "game_camera_registry_preflight_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixed_center_candidates": fixed_candidates, "count": len(fixed_candidates)}, indent=2))


if __name__ == "__main__":
    main()
