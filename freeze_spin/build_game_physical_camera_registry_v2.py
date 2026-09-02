from __future__ import annotations

"""Build a physical-camera registry without trusting NBA angle labels.

The same optical centre may appear under several HLS labels, while one HLS label may
contain different physical views across events.  For a pinhole camera with a fixed
optical centre, arbitrary pan/tilt/zoom views of a static 3D scene are related by a
single image homography, independent of scene depth.  This script therefore:

1. clusters the exact synchronized Frame-C target images by strong, broad, multi-depth
   homography consistency;
2. samples every label from multiple other events in the same game;
3. searches those samples against each target physical-centre anchor regardless of
   label name; and
4. promotes only recurring *physical-centre candidates*.  It does not produce metric
   XYZ camera calibration and does not render novel views.

Moving-player regions are excluded conservatively.  A failure means only that this
particular sampled evidence did not establish a reusable centre.
"""

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
CLIP_RE = re.compile(r"^\d+_R(\d+)_(\d+)_(\d+)_(.+)_SOURCE\.mp4$")
TARGET_RE = re.compile(r"^([A-L])_(.+?)_(\d+\.\d+)s_frame(\d+)\.png$")
SAMPLE_FRACTIONS = (0.28, 0.50, 0.72)


def label_from_token(s: str) -> str:
    return s.replace("_", " ")


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


@dataclass
class FeatureFrame:
    path: Path
    label: str
    event_id: int | None
    sample_fraction: float | None
    image: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None


def action_core_mask_xy(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros(0, bool)
    x, y = xy[:, 0], xy[:, 1]
    # The exact event's moving bodies occupy the central/lower court.  This mask is
    # intentionally generous; static arena evidence is plentiful elsewhere.
    return (x > 0.18 * W) & (x < 0.84 * W) & (y > 0.43 * H) & (y < 0.99 * H)


def make_sift():
    return cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.018, edgeThreshold=10)


def featurize(path: Path, label: str, event_id: int | None, frac: float | None, sift) -> FeatureFrame | None:
    im = cv2.imread(str(path))
    if im is None or im.shape[:2] != (H, W):
        return None
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)
    return FeatureFrame(path, label, event_id, frac, im, kp or [], desc)


def points_from_matches(a: FeatureFrame, b: FeatureFrame, good: list) -> tuple[np.ndarray, np.ndarray]:
    pa = np.float32([a.keypoints[m.queryIdx].pt for m in good])
    pb = np.float32([b.keypoints[m.trainIdx].pt for m in good])
    return pa, pb


def match_features(a: FeatureFrame, b: FeatureFrame) -> tuple[np.ndarray, np.ndarray]:
    if a.descriptors is None or b.descriptors is None or len(a.descriptors) < 2 or len(b.descriptors) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        raw = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    except cv2.error:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.72 * n.distance:
            good.append(m)
    # One-to-one target descriptor assignment.
    best = {}
    for m in good:
        old = best.get(m.trainIdx)
        if old is None or m.distance < old.distance:
            best[m.trainIdx] = m
    good = list(best.values())
    if not good:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return points_from_matches(a, b, good)


def bbox_fraction(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    xmin, ymin = np.min(p, axis=0)
    xmax, ymax = np.max(p, axis=0)
    return float(max(0.0, xmax - xmin) * max(0.0, ymax - ymin) / (W * H))


def band_counts(p: np.ndarray) -> list[int]:
    if len(p) == 0:
        return [0, 0, 0]
    y = p[:, 1]
    return [int((y < H / 3).sum()), int(((y >= H / 3) & (y < 2 * H / 3)).sum()), int((y >= 2 * H / 3).sum())]


def homography_evidence(a: FeatureFrame, b: FeatureFrame, *, strict: bool) -> dict:
    pa, pb = match_features(a, b)
    rec = {
        "source": str(a.path), "target": str(b.path),
        "source_label": a.label, "target_label": b.label,
        "source_event_id": a.event_id, "source_fraction": a.sample_fraction,
        "raw_good_matches": int(len(pa)), "pass": False,
    }
    if len(pa) < 30:
        rec["status"] = "insufficient_matches"
        return rec
    keep = ~action_core_mask_xy(pa) & ~action_core_mask_xy(pb)
    pa, pb = pa[keep], pb[keep]
    rec["static_candidate_matches"] = int(len(pa))
    if len(pa) < 24:
        rec["status"] = "insufficient_static_matches"
        return rec
    Hm, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 1.5, maxIters=30000, confidence=0.999)
    if Hm is None or mask is None:
        rec["status"] = "homography_failed"
        return rec
    inl = mask.ravel().astype(bool)
    ia, ib = pa[inl], pb[inl]
    n = int(inl.sum())
    if n:
        pred = cv2.perspectiveTransform(ia[:, None, :], Hm)[:, 0]
        err = np.linalg.norm(pred - ib, axis=1)
        med = float(np.median(err)); p95 = float(np.percentile(err, 95))
    else:
        med = p95 = float("inf")
    ratio = float(n / max(len(pa), 1))
    src_cov = bbox_fraction(ia); tgt_cov = bbox_fraction(ib)
    src_bands = band_counts(ia); tgt_bands = band_counts(ib)
    src_active = int(sum(c >= 6 for c in src_bands)); tgt_active = int(sum(c >= 6 for c in tgt_bands))
    src_span = float((np.max(ia[:, 1]) - np.min(ia[:, 1])) / H) if n else 0.0
    tgt_span = float((np.max(ib[:, 1]) - np.min(ib[:, 1])) / H) if n else 0.0
    rec.update({
        "inliers": n, "inlier_ratio": ratio,
        "median_error_px": med, "p95_error_px": p95,
        "source_bbox_area_fraction": src_cov, "target_bbox_area_fraction": tgt_cov,
        "source_y_band_inliers": src_bands, "target_y_band_inliers": tgt_bands,
        "source_active_y_bands": src_active, "target_active_y_bands": tgt_active,
        "source_vertical_span_fraction": src_span, "target_vertical_span_fraction": tgt_span,
        "H_source_to_target": Hm.tolist(),
    })
    if strict:
        gates = {
            "inliers_at_least_40": n >= 40,
            "inlier_ratio_at_least_0_55": ratio >= 0.55,
            "p95_at_most_1_5px": p95 <= 1.5,
            "source_bbox_at_least_0_12": src_cov >= 0.12,
            "target_bbox_at_least_0_12": tgt_cov >= 0.12,
            "source_vertical_span_at_least_0_35": src_span >= 0.35,
            "target_vertical_span_at_least_0_35": tgt_span >= 0.35,
            "source_two_depth_bands": src_active >= 2,
            "target_two_depth_bands": tgt_active >= 2,
        }
    else:
        gates = {
            "inliers_at_least_32": n >= 32,
            "inlier_ratio_at_least_0_45": ratio >= 0.45,
            "p95_at_most_1_6px": p95 <= 1.6,
            "source_bbox_at_least_0_08": src_cov >= 0.08,
            "target_bbox_at_least_0_08": tgt_cov >= 0.08,
            "source_vertical_span_at_least_0_28": src_span >= 0.28,
            "target_vertical_span_at_least_0_28": tgt_span >= 0.28,
            "source_two_depth_bands": src_active >= 2,
            "target_two_depth_bands": tgt_active >= 2,
        }
    rec["gates"] = gates
    rec["pass"] = bool(all(gates.values()))
    rec["status"] = "same_optical_centre_evidence" if rec["pass"] else "rejected"
    return rec


class UnionFind:
    def __init__(self, keys):
        self.p = {k: k for k in keys}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def target_frames(root: Path, sift) -> dict[str, FeatureFrame]:
    out = {}
    for p in sorted(root.glob("*.png")):
        m = TARGET_RE.match(p.name)
        if not m:
            continue
        _, token, _, _ = m.groups()
        label = label_from_token(token)
        f = featurize(p, label, None, None, sift)
        if f is not None:
            out[label] = f
    return out


def extract_samples(clips: Path, root: Path, target_event: int, sift) -> list[FeatureFrame]:
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for clip in sorted(clips.glob("*_SOURCE.mp4")):
        m = CLIP_RE.match(clip.name)
        if not m:
            continue
        _, _, event_s, token = m.groups()
        event_id = int(event_s)
        if event_id == target_event:
            continue
        label = label_from_token(token)
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            continue
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            cap.release(); continue
        for frac in SAMPLE_FRACTIONS:
            idx = int(np.clip(round(frac * (count - 1)), 0, count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, im = cap.read()
            if not ok or im is None or im.shape[:2] != (H, W):
                continue
            p = root / f"{safe(label)}__event{event_id:04d}__f{frac:.2f}.jpg"
            cv2.imwrite(str(p), im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)
            frames.append(FeatureFrame(p, label, event_id, frac, im, kp or [], desc))
        cap.release()
    return frames


def evidence_score(r: dict) -> tuple:
    return (
        1 if r.get("pass") else 0,
        int(r.get("inliers", 0)),
        float(r.get("inlier_ratio", 0.0)),
        -float(r.get("p95_error_px", 1e9)),
        float(r.get("target_bbox_area_fraction", 0.0)),
    )


def contact_sheet(title: str, target: FeatureFrame, rows: list[dict], sample_lookup: dict[str, FeatureFrame], out: Path) -> None:
    items = [(target.image.copy(), f"TARGET {target.label}")]
    for r in rows[:11]:
        f = sample_lookup.get(r["source"])
        if f is None:
            continue
        txt = f"E{f.event_id} {f.label} @{f.sample_fraction:.2f} inl={r.get('inliers',0)}"
        items.append((f.image.copy(), txt))
    tw, th = 480, 270
    tiles = []
    for im, txt in items:
        x = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(x, (0, 0), (tw, 34), (0, 0, 0), -1)
        cv2.putText(x, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        tiles.append(x)
    cols = 3; rows_n = math.ceil(len(tiles)/cols)
    canvas = np.zeros((rows_n*th + 40, cols*tw, 3), np.uint8)
    for i,t in enumerate(tiles):
        y,x = divmod(i, cols); canvas[y*th:(y+1)*th, x*tw:(x+1)*tw] = t
    cv2.putText(canvas, title, (10, rows_n*th+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--target-frames", type=Path, required=True)
    ap.add_argument("--target-event", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sift = make_sift()

    targets = target_frames(args.target_frames, sift)
    if len(targets) < 8:
        raise RuntimeError(f"Expected >=8 exact target views, got {len(targets)}")

    labels = sorted(targets)
    uf = UnionFind(labels)
    target_pairs = []
    for i,a_label in enumerate(labels):
        for b_label in labels[i+1:]:
            r = homography_evidence(targets[a_label], targets[b_label], strict=True)
            r["a"] = a_label; r["b"] = b_label
            target_pairs.append(r)
            if r["pass"]:
                uf.union(a_label, b_label)

    raw_clusters = {}
    for lab in labels:
        raw_clusters.setdefault(uf.find(lab), []).append(lab)
    cluster_members = sorted([sorted(v) for v in raw_clusters.values()], key=lambda v: (v[0], len(v)))

    samples = extract_samples(args.clips, args.out / "samples", args.target_event, sift)
    sample_lookup = {str(s.path): s for s in samples}

    # Representative preference: exact Broadcast if present in a duplicate cluster,
    # otherwise the member with the most SIFT features in the target frame.
    clusters = []
    evidence_dir = args.out / "evidence"; evidence_dir.mkdir(exist_ok=True)
    for ci, members in enumerate(cluster_members, 1):
        if "Broadcast" in members:
            rep_label = "Broadcast"
        else:
            rep_label = max(members, key=lambda l: len(targets[l].keypoints))
        target = targets[rep_label]
        event_best = {}
        for s in samples:
            r = homography_evidence(s, target, strict=False)
            event = int(s.event_id)
            if event not in event_best or evidence_score(r) > evidence_score(event_best[event]):
                event_best[event] = r
        ordered = sorted(event_best.values(), key=evidence_score, reverse=True)
        passing = [r for r in ordered if r.get("pass")]
        passing_events = sorted({int(r["source_event_id"]) for r in passing})
        source_labels = sorted({r["source_label"] for r in passing})
        status = "RECURRING_FIXED_CENTRE_CANDIDATE" if len(passing_events) >= 3 else ("LIMITED_FIXED_CENTRE_EVIDENCE" if passing_events else "NO_RECURRING_EVIDENCE")

        copied = []
        for rank,r in enumerate(passing[:12],1):
            src = Path(r["source"])
            dst = evidence_dir / f"cluster{ci:02d}_{rank:02d}_event{int(r['source_event_id']):04d}_{safe(r['source_label'])}.jpg"
            if src.exists():
                shutil.copy2(src, dst); copied.append(str(dst))
        contact_sheet(f"PHYSICAL CLUSTER {ci}: {', '.join(members)}", target, passing, sample_lookup, args.out / f"cluster_{ci:02d}_contact.jpg")
        clusters.append({
            "cluster_id": ci,
            "exact_target_members": members,
            "representative_target_label": rep_label,
            "representative_target_file": str(target.path),
            "target_feature_count": len(target.keypoints),
            "status": status,
            "passing_distinct_event_count": len(passing_events),
            "passing_event_ids": passing_events,
            "contributing_source_labels": source_labels,
            "best_event_evidence": ordered[:16],
            "copied_passing_evidence": copied,
        })

    recurring = [c for c in clusters if c["status"] == "RECURRING_FIXED_CENTRE_CANDIDATE"]
    report = {
        "method": "label-independent same-optical-centre clustering via broad multi-depth static-scene homography",
        "semantics": "HLS angle labels are treated as non-authoritative metadata; cluster IDs represent pixel-supported recurring optical-centre candidates, not yet metric XYZ cameras",
        "target_event": args.target_event,
        "source_resolution": [W,H],
        "sample_fractions": list(SAMPLE_FRACTIONS),
        "exact_target_view_count": len(targets),
        "target_pair_count": len(target_pairs),
        "target_equivalent_pairs": [{k:v for k,v in r.items() if k not in ("H_source_to_target",)} for r in target_pairs if r.get("pass")],
        "target_pairs": target_pairs,
        "physical_target_cluster_count": len(clusters),
        "recurring_fixed_centre_candidate_count": len(recurring),
        "clusters": clusters,
        "critical_policy": {
            "nba_angle_label_is_physical_camera_id": False,
            "same_label_required_for_recurrence": False,
            "metric_xyz_promoted_by_this_stage": False,
            "render_allowed_by_this_stage": False,
        },
    }
    (args.out / "game_physical_camera_registry_v2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("TARGET_EQUIVALENT_PAIRS")
    for r in report["target_equivalent_pairs"]:
        print(r["a"], "<=>", r["b"], "inliers",r.get("inliers"),"p95",round(float(r.get("p95_error_px",999)),3),"coverage",round(float(r.get("target_bbox_area_fraction",0)),3))
    for c in clusters:
        print("PHYSICAL_CLUSTER",c["cluster_id"],c["exact_target_members"],c["status"],"events",c["passing_event_ids"],"labels",c["contributing_source_labels"])
    print("RECURRING_CLUSTER_COUNT", len(recurring))


if __name__ == "__main__":
    main()
