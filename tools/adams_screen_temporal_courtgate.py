#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kd_double_team_ballhandler_onnx import YoloOnnx, extract_frames
from adams_screen_temporal_onnx import (
    associate, cluster_tracks, detect_sequences, by_tid
)


def tracks_to_list(tracked):
    acc = defaultdict(lambda: {"frames": [], "boxes": []})
    for fi, dets in enumerate(tracked):
        for d in dets:
            acc[int(d["tid"])] ["frames"].append(fi)
            # nbacv court gate only uses first 4 values; keep confidence if present
            acc[int(d["tid"])] ["boxes"].append(list(map(float, d["box"])))
    return [{"track_id": tid, "frames": rec["frames"], "boxes": rec["boxes"]}
            for tid, rec in acc.items()]


def court_calibrate(frame_paths, court_model, device="cpu"):
    from nbacv.court import _court_infer, calibrate_video
    kps = {}
    good_size = 640
    sizes = [640, 960]
    hw = None
    for i, p in enumerate(frame_paths):
        fr = cv2.imread(str(p))
        if fr is None:
            kps[i] = None
            continue
        hw = fr.shape[:2]
        kp = None
        for sz in [good_size] + [x for x in sizes if x != good_size]:
            cand = _court_infer(court_model, fr, sz, device)
            if cand is not None and (cand[1] >= 0.5).sum() >= 6:
                kp = cand
                good_size = sz
                break
            if kp is None:
                kp = cand
        kps[i] = kp
    if hw is None:
        return {}, 0.0
    calib = calibrate_video(kps, frame_hw=hw)
    cov = sum(1 for v in calib.values() if v.get("H")) / max(len(calib), 1)
    return calib, cov


def court_filter(frame_paths, tracked, calib, fps):
    from nbacv.court_gate import court_roi_roles
    tracks = tracks_to_list(tracked)
    if not tracks:
        return tracked, {}, {}, {}
    keep, near_frac, inside_frac = court_roi_roles(tracks, calib, fps=fps)

    # Screen participants should be players on the floor, not coaches/crowd.
    # Keep the upstream robust near-court gate, plus a light strict-inside cue.
    # A track with no calibrated samples defaults upstream to 1.0 and remains.
    playable = {}
    for t in tracks:
        tid = t["track_id"]
        playable[tid] = bool(keep.get(tid, True) and inside_frac.get(tid, 1.0) >= 0.20)

    filtered = []
    for dets in tracked:
        filtered.append([d for d in dets if playable.get(int(d["tid"]), True)])
    return filtered, playable, near_frac, inside_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--onnx-model", required=True)
    ap.add_argument("--court-model", required=True)
    ap.add_argument("--nbacv-src", required=True)
    ap.add_argument("--sample-fps", type=float, default=6.0)
    ap.add_argument("--max-seconds", type=float, default=16.0)
    ap.add_argument("--target-hls-width", type=int, default=960)
    ap.add_argument("--person-conf", type=float, default=0.14)
    ap.add_argument("--ball-conf", type=float, default=0.03)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(a.nbacv_src)))

    from ultralytics import YOLO
    court_model = YOLO(a.court_model)
    model = YoloOnnx(a.onnx_model)

    df = pd.read_csv(a.input, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    rows, seq_rows = [], []

    with tempfile.TemporaryDirectory(prefix="adams_courtgate_") as td:
        root = Path(td)
        for _, r in df.iterrows():
            t0 = time.perf_counter()
            events = []
            raw = str(r.get("screen_event_nums", ""))
            for x in raw.split("|"):
                if x.strip().isdigit() and int(x) not in events:
                    events.append(int(x))
            if not events:
                # opener universe stores an exact anchor column in some builds
                for col in ("video_event_num", "anchor_event_num", "event_num"):
                    v = r.get(col)
                    if pd.notna(v) and str(v).replace(".0", "").isdigit():
                        events = [int(float(v))]
                        break

            total_sequences = 0
            best = None
            sampled = bh_obs = 0
            raw_tracks = kept_tracks = 0
            mean_calib = []
            errors = []

            for ev in events:
                evdir = root / f"{r.game_id}_{ev}"
                try:
                    meta = extract_frames(str(r.game_id), ev, evdir, a.sample_fps,
                                          a.max_seconds, a.target_hls_width)
                    paths = meta["frames"]
                    people_pf, balls_pf = [], []
                    from kd_double_team_ballhandler_onnx import players_on_courtish
                    for p in paths:
                        fr = cv2.imread(str(p))
                        if fr is None:
                            people_pf.append([]); balls_pf.append([]); continue
                        people_pf.append(players_on_courtish(
                            model.detect_class(fr, 0, a.person_conf), fr.shape[0], max_players=18))
                        balls_pf.append(model.detect_class(fr, 32, a.ball_conf, 0.35))

                    tracked = associate(people_pf)
                    raw_ids = {d["tid"] for ds in tracked for d in ds}
                    raw_tracks += len(raw_ids)
                    calib, cov = court_calibrate(paths, court_model)
                    mean_calib.append(cov)
                    gated, playable, near_frac, inside_frac = court_filter(
                        paths, tracked, calib, a.sample_fps)
                    kept_ids = {d["tid"] for ds in gated for d in ds}
                    kept_tracks += len(kept_ids)

                    labels, centers = cluster_tracks(paths, gated)
                    seqs, bhs = detect_sequences(paths, gated, balls_pf, labels, centers,
                                                  a.sample_fps)
                    # Require persistence across distinct samples; old first pass
                    # could accidentally merge several defender hypotheses in one frame.
                    seqs = [s for s in seqs if s["end_frame"] > s["start_frame"]]
                    sampled += len(paths)
                    bh_obs += sum(x is not None for x in bhs)
                    total_sequences += len(seqs)

                    for si, s in enumerate(seqs):
                        b = s["best"]
                        rec = {
                            "game_id": str(r.game_id), "possession_uid": r.possession_uid,
                            "event_num": ev, "sequence_index": si,
                            "start_sample": s["start_frame"], "end_sample": s["end_frame"],
                            "start_s": round(s["start_frame"] / a.sample_fps, 3),
                            "end_s": round(s["end_frame"] / a.sample_fps, 3),
                            "ballhandler_tid": s["ballhandler_tid"],
                            "screener_tid": s["screener_tid"],
                            "defender_tid": b.get("defender_tid"),
                            "n_hits": len({h["frame"] for h in s["hits"]}),
                            "best_score": b["score"],
                            "calibration_coverage": round(cov, 3),
                            "screener_near_frac": round(float(near_frac.get(s["screener_tid"], 0)), 3),
                            "screener_inside_frac": round(float(inside_frac.get(s["screener_tid"], 0)), 3),
                            "defender_near_frac": round(float(near_frac.get(b.get("defender_tid"), 0)), 3),
                            "defender_inside_frac": round(float(inside_frac.get(b.get("defender_tid"), 0)), 3),
                        }
                        seq_rows.append(rec)
                        if best is None or rec["best_score"] > best["best_score"]:
                            best = rec
                except Exception as e:
                    errors.append(f"event {ev}: {type(e).__name__}: {e}")
                finally:
                    shutil.rmtree(evdir, ignore_errors=True)

            rows.append({
                "game_id": str(r.game_id), "period": r.get("period"),
                "possession_uid": r.possession_uid, "start_time": r.get("start_time"),
                "end_time": r.get("end_time"), "pts_poss": r.get("pts_poss"),
                "type_end": r.get("type_end"), "screen_event_nums": "|".join(map(str, events)),
                "court_screen_sequences": total_sequences,
                "court_candidate": total_sequences > 0,
                "best_event_num": None if best is None else best["event_num"],
                "best_start_s": None if best is None else best["start_s"],
                "best_end_s": None if best is None else best["end_s"],
                "best_screen_score": None if best is None else best["best_score"],
                "sampled_frames": sampled, "ballhandler_observed_frames": bh_obs,
                "raw_track_count": raw_tracks, "kept_track_count": kept_tracks,
                "mean_calibration_coverage": round(float(np.mean(mean_calib)), 3) if mean_calib else None,
                "errors": " || ".join(errors),
                "runtime_seconds": round(time.perf_counter() - t0, 2),
            })
            print(json.dumps(rows[-1], default=str), flush=True)

    out = pd.DataFrame(rows)
    seq = pd.DataFrame(seq_rows)
    out.to_csv(a.out / "court_screen_candidates.csv", index=False)
    seq.to_csv(a.out / "court_screen_sequences.csv", index=False)
    qa = {
        "rows": len(out),
        "candidate_possessions": int(out.court_candidate.sum()) if len(out) else 0,
        "candidate_rate": float(out.court_candidate.mean()) if len(out) else None,
        "total_sequences": int(out.court_screen_sequences.sum()) if len(out) else 0,
        "mean_runtime_seconds": float(out.runtime_seconds.mean()) if len(out) else None,
        "mean_calibration_coverage": float(out.mean_calibration_coverage.dropna().mean()) if len(out) and out.mean_calibration_coverage.notna().any() else None,
        "error_rows": int(out.errors.fillna("").astype(str).ne("").sum()) if len(out) else 0,
        "note": "Screen geometry is evaluated only after robust court-ROI participation gating; candidate is not yet Adams-confirmed."
    }
    (a.out / "qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
