#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kd_double_team_ballhandler_onnx import YoloOnnx, extract_frames, players_on_courtish
from adams_screen_temporal_onnx import associate, ballhandler_tid

GAME_ID = "0022500001"
ADAMS_ID = 203500

# Prefer made FTs / made FGs so the final stable ballhandler before the ball
# leaves a player is the named PBP actor. Two independent seeds are used where
# possible so the gallery can be checked by leave-one-seed-out similarity.
SEEDS = [
    {"player_id": 203500, "name": "Steven Adams", "jersey": "12", "event": 253, "kind": "ft"},
    {"player_id": 203500, "name": "Steven Adams", "jersey": "12", "event": 460, "kind": "made_fg"},
    {"player_id": 201142, "name": "Kevin Durant", "jersey": "7", "event": 682, "kind": "ft"},
    {"player_id": 201142, "name": "Kevin Durant", "jersey": "7", "event": 308, "kind": "made_fg"},
    {"player_id": 1630578, "name": "Alperen Sengun", "jersey": "28", "event": 342, "kind": "ft"},
    {"player_id": 1630578, "name": "Alperen Sengun", "jersey": "28", "event": 417, "kind": "made_fg"},
    {"player_id": 1641708, "name": "Amen Thompson", "jersey": "1", "event": 45, "kind": "ft"},
    {"player_id": 1641708, "name": "Amen Thompson", "jersey": "1", "event": 646, "kind": "made_fg"},
    {"player_id": 1629006, "name": "Josh Okogie", "jersey": "20", "event": 757, "kind": "ft"},
    {"player_id": 1631095, "name": "Jabari Smith Jr.", "jersey": "10", "event": 554, "kind": "made_fg"},
    {"player_id": 1631095, "name": "Jabari Smith Jr.", "jersey": "10", "event": 817, "kind": "made_fg"},
    {"player_id": 1642263, "name": "Reed Sheppard", "jersey": "15", "event": 247, "kind": "made_fg"},
    # Tari had no made FG/FT in this game. Keep two shot attempts as lower-
    # confidence negative-gallery seeds; they are never used to confirm Adams.
    {"player_id": 1631106, "name": "Tari Eason", "jersey": "17", "event": 445, "kind": "miss_fg"},
    {"player_id": 1631106, "name": "Tari Eason", "jersey": "17", "event": 453, "kind": "miss_fg"},
]


def l2(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-9)


def torso_crop(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    # Slightly wider / taller than jersey OCR crop: appearance needs shoulders,
    # jersey, arms and some head/hair context.
    cx1 = x1 + int(0.08 * w)
    cx2 = x2 - int(0.08 * w)
    cy1 = y1 + int(0.02 * h)
    cy2 = y1 + int(0.72 * h)
    c = frame[max(0, cy1):max(0, cy2), max(0, cx1):max(0, cx2)]
    return c if c.size else None


def choose_seed_tid(tracked, balls_pf):
    """Choose the PBP actor as the final meaningful ballhandler run.

    For made FGs/FTs the event clip normally ends after the attempt, so the
    last stable player-ball association before a sustained no-handler gap is
    the shooter. We fall back to the most frequently observed ballhandler.
    """
    bhs = [ballhandler_tid(tracked[i], balls_pf[i]) for i in range(len(tracked))]
    runs = []
    start = 0
    while start < len(bhs):
        tid = bhs[start]
        end = start
        while end + 1 < len(bhs) and bhs[end + 1] == tid:
            end += 1
        runs.append((start, end, tid))
        start = end + 1

    candidates = []
    for ri, (s, e, tid) in enumerate(runs):
        if tid is None:
            continue
        run_len = e - s + 1
        if run_len < 1:
            continue
        none_after = 0
        if ri + 1 < len(runs) and runs[ri + 1][2] is None:
            none_after = runs[ri + 1][1] - runs[ri + 1][0] + 1
        # Later release and a trailing no-handler gap are strong cues.
        score = e + 2.5 * min(none_after, 6) + 0.5 * run_len
        candidates.append((score, e, run_len, tid))

    if candidates:
        candidates.sort(reverse=True)
        chosen = candidates[0][3]
    else:
        cnt = Counter(t for t in bhs if t is not None)
        chosen = cnt.most_common(1)[0][0] if cnt else None
    return chosen, bhs


def collect_tid_crops(frame_paths, tracked, tid, max_crops=20):
    items = []
    for fi, (pth, dets) in enumerate(zip(frame_paths, tracked)):
        rec = next((d for d in dets if d["tid"] == tid), None)
        if rec is None:
            continue
        fr = cv2.imread(str(pth))
        if fr is None:
            continue
        c = torso_crop(fr, rec["box"])
        if c is None or c.shape[0] < 28 or c.shape[1] < 14:
            continue
        items.append((fi, c))
    if not items:
        return []
    idx = np.linspace(0, len(items) - 1, min(max_crops, len(items))).astype(int)
    return [items[i] for i in idx]


def load_siglip(device="cpu"):
    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel
    model_name = "google/siglip-base-patch16-224"
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name).to(device).eval()
    return torch, proc, model


def embed_images(images, torch, proc, model, device="cpu", batch_size=32):
    if not images:
        return np.zeros((0, 768), np.float32)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images[i:i + batch_size]]
            inp = proc(images=batch, return_tensors="pt").to(device)
            out = model(**inp).pooler_output.float().cpu().numpy()
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
            vecs.append(out)
    return np.concatenate(vecs, axis=0)


def parse_lineup_ids(text):
    out = []
    for part in str(text).split(","):
        p = part.strip().split(" ", 1)[0]
        if p.isdigit():
            out.append(int(p))
    return out


def build_seed_vectors(model, outdir: Path, sample_fps=6.0, max_seconds=16.0):
    seed_records = []
    seed_crops = {}
    with tempfile.TemporaryDirectory(prefix="adams_gallery_seed_") as td:
        root = Path(td)
        for s in SEEDS:
            evdir = root / str(s["event"])
            try:
                meta = extract_frames(GAME_ID, int(s["event"]), evdir, sample_fps, max_seconds, 960)
                frame_paths = meta["frames"]
                people_pf, balls_pf = [], []
                for pth in frame_paths:
                    fr = cv2.imread(str(pth))
                    if fr is None:
                        people_pf.append([]); balls_pf.append([]); continue
                    people_pf.append(players_on_courtish(model.detect_class(fr, 0, 0.14), fr.shape[0]))
                    balls_pf.append(model.detect_class(fr, 32, 0.025, 0.35))
                tracked = associate(people_pf)
                tid, bhs = choose_seed_tid(tracked, balls_pf)
                crops = collect_tid_crops(frame_paths, tracked, tid) if tid is not None else []
                key = f"{s['player_id']}_{s['event']}"
                seed_crops[key] = crops
                # Save a small visual audit strip for each seed.
                sdir = outdir / "seed_crops" / key
                sdir.mkdir(parents=True, exist_ok=True)
                for j, (fi, im) in enumerate(crops):
                    cv2.imwrite(str(sdir / f"{j:02d}_f{fi:03d}.jpg"), im)
                seed_records.append({**s, "track_id": tid, "frames": len(frame_paths),
                                     "ballhandler_frames": sum(x is not None for x in bhs),
                                     "chosen_tid_ballhandler_frames": sum(x == tid for x in bhs) if tid is not None else 0,
                                     "crop_count": len(crops), "angle": meta.get("angle"), "error": ""})
            except Exception as e:
                seed_records.append({**s, "track_id": None, "frames": 0, "ballhandler_frames": 0,
                                     "chosen_tid_ballhandler_frames": 0, "crop_count": 0, "angle": None,
                                     "error": f"{type(e).__name__}: {e}"})
            finally:
                shutil.rmtree(evdir, ignore_errors=True)
    return seed_records, seed_crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-crop-root", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--ocr-evidence", type=Path, required=True)
    ap.add_argument("--onnx-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    yolo = YoloOnnx(a.onnx_model)
    seed_records, seed_crops = build_seed_vectors(yolo, a.out)
    pd.DataFrame(seed_records).to_csv(a.out / "seed_qa.csv", index=False)

    torch, proc, siglip = load_siglip("cpu")

    # Embed seed crops and form player prototypes.
    seed_vec_records = []
    player_vecs = defaultdict(list)
    for rec in seed_records:
        key = f"{rec['player_id']}_{rec['event']}"
        crops = [im for _, im in seed_crops.get(key, [])]
        V = embed_images(crops, torch, proc, siglip)
        if len(V):
            v = l2(V.mean(axis=0))
            player_vecs[int(rec["player_id"])].append((int(rec["event"]), v))
            seed_vec_records.append({"player_id": int(rec["player_id"]), "name": rec["name"],
                                     "event": int(rec["event"]), "crop_count": len(V), "vec": v})

    prototypes = {}
    for pid, evs in player_vecs.items():
        prototypes[pid] = l2(np.mean(np.stack([v for _, v in evs]), axis=0))

    # Leave-one-seed-out diagnostics: this tells us whether the gallery can
    # distinguish same-player from same-uniform teammates in THIS game/feed.
    loo = []
    all_pids = sorted(prototypes)
    for rec in seed_vec_records:
        pid = rec["player_id"]
        others_same = [v for ev, v in player_vecs[pid] if ev != rec["event"]]
        if not others_same:
            continue
        same_proto = l2(np.mean(np.stack(others_same), axis=0))
        same = float(rec["vec"] @ same_proto)
        impostors = [(p, float(rec["vec"] @ prototypes[p])) for p in all_pids if p != pid]
        impostors.sort(key=lambda x: x[1], reverse=True)
        loo.append({"player_id": pid, "name": rec["name"], "event": rec["event"],
                    "same_cos": same, "best_other_pid": impostors[0][0] if impostors else None,
                    "best_other_cos": impostors[0][1] if impostors else None,
                    "margin": same - impostors[0][1] if impostors else None})
    pd.DataFrame(loo).to_csv(a.out / "gallery_leave_one_out.csv", index=False)

    # Adams-specific conservative thresholds learned only from independent
    # Adams seeds. If the two Adams seeds do not recognize each other, we do
    # not auto-confirm any candidate from appearance alone.
    adams_loo = [x for x in loo if x["player_id"] == ADAMS_ID]
    gallery_gate_ok = len(adams_loo) >= 2 and min(x["margin"] for x in adams_loo) > 0
    adams_min_same = min((x["same_cos"] for x in adams_loo), default=1.0)
    adams_min_margin = min((x["margin"] for x in adams_loo), default=1.0)

    cand = pd.read_csv(a.candidates, dtype={"game_id": str})
    ocr = json.loads(a.ocr_evidence.read_text())
    ocr_by = {(x["candidate_id"], x["role"]): x for x in ocr}

    rows = []
    for _, r in cand.iterrows():
        cid = str(r.candidate_id)
        files = sorted((a.screen_crop_root / cid / "screener").glob("*.jpg"))
        ims = [cv2.imread(str(p)) for p in files]
        ims = [im for im in ims if im is not None]
        V = embed_images(ims, torch, proc, siglip)
        if len(V) == 0:
            rows.append({"candidate_id": cid, "appearance_status": "unobservable", "crop_count": 0})
            continue
        q = l2(V.mean(axis=0))
        lineup = parse_lineup_ids(r.lineup_team)
        sims = [(pid, float(q @ prototypes[pid])) for pid in lineup if pid in prototypes]
        sims.sort(key=lambda x: x[1], reverse=True)
        top_pid, top_cos = sims[0] if sims else (None, math.nan)
        second_cos = sims[1][1] if len(sims) > 1 else math.nan
        margin = top_cos - second_cos if len(sims) > 1 else math.nan
        adams_cos = next((s for pid, s in sims if pid == ADAMS_ID), math.nan)
        adams_rank = next((i + 1 for i, (pid, _) in enumerate(sims) if pid == ADAMS_ID), None)

        # OCR remains secondary evidence. Only exact roster-valid #12 can
        # independently support Adams; invalid reads are never fuzzy-forced.
        oe = ocr_by.get((cid, "screener"), {})
        exact12_weight = float((oe.get("votes") or {}).get("12", 0.0))

        status = "appearance_ambiguous"
        if gallery_gate_ok and top_pid == ADAMS_ID:
            # Require candidate to look at least almost as similar as the worst
            # independent Adams seed and preserve a positive teammate margin.
            if top_cos >= adams_min_same - 0.04 and margin >= max(0.01, 0.35 * adams_min_margin):
                status = "appearance_confirmed_adams"
            elif exact12_weight >= 1.0 and margin > 0:
                status = "appearance_probable_adams"
        elif top_pid is not None and top_pid != ADAMS_ID and margin >= 0.015:
            status = "appearance_rejected_other_screener"

        row = {"candidate_id": cid, "period": r.period, "start_time": r.start_time,
               "end_time": r.end_time, "event_num": int(r.event_num), "pts_poss": r.pts_poss,
               "type_end": r.type_end, "crop_count": len(V), "top_player_id": top_pid,
               "top_cos": round(top_cos, 5) if not math.isnan(top_cos) else None,
               "second_cos": round(second_cos, 5) if not math.isnan(second_cos) else None,
               "margin": round(margin, 5) if not math.isnan(margin) else None,
               "adams_cos": round(adams_cos, 5) if not math.isnan(adams_cos) else None,
               "adams_rank": adams_rank, "ocr_exact12_weight": round(exact12_weight, 4),
               "appearance_status": status, "lineup_team": r.lineup_team}
        for pid, s in sims:
            row[f"cos_{pid}"] = round(s, 5)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(a.out / "screen_gallery_identity.csv", index=False)
    qa = {
        "seed_events_requested": len(SEEDS),
        "seed_vectors_built": len(seed_vec_records),
        "player_prototypes": len(prototypes),
        "adams_loo": adams_loo,
        "gallery_gate_ok": gallery_gate_ok,
        "adams_min_same": adams_min_same,
        "adams_min_margin": adams_min_margin,
        "candidates": len(out),
        "confirmed_adams": int((out.appearance_status == "appearance_confirmed_adams").sum()) if len(out) else 0,
        "probable_adams": int((out.appearance_status == "appearance_probable_adams").sum()) if len(out) else 0,
        "rejected_other": int((out.appearance_status == "appearance_rejected_other_screener").sum()) if len(out) else 0,
        "ambiguous_or_unobservable": int(out.appearance_status.isin(["appearance_ambiguous", "unobservable"]).sum()) if len(out) else 0,
        "note": "Appearance confirmation is allowed only if independent Adams seeds recognize each other with positive teammate margin. OCR #12 is supporting evidence only; invalid reads are never forced to a roster number."
    }
    (a.out / "qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
