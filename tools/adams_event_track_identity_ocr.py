#!/usr/bin/env python3
"""Resolve player names for screen-overlay tracklets from jersey evidence.

This is deliberately lineup-constrained and abstaining. It does not infer a
name from appearance. It samples native-video torso crops for substantial
tracklets, runs the existing PARSeq/legibility jersey reader, and maps only a
high-confidence jersey number that is unique in the exact ten-player lineup.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_lineup_ids(text):
    return [int(x) for x in re.findall(r'(?<!\d)(\d{6,7})(?!\d)', str(text))]


def load_roster(path: Path):
    raw = json.loads(path.read_text())
    out = {}
    for team, rec in raw['teams'].items():
        for p in rec['players']:
            q = dict(p)
            q['id'] = int(q['id'])
            q['jersey'] = str(q['jersey'])
            q['team'] = team
            out[q['id']] = q
    return out


def crop_torso(frame, box, variant=0):
    x1, y1, x2, y2 = [float(x) for x in box]
    h = max(2.0, y2 - y1)
    w = max(2.0, x2 - x1)
    if variant == 0:
        ya, yb, xa, xb = .08, .62, .10, .90
    else:
        ya, yb, xa, xb = .16, .58, .02, .98
    a = max(0, int(round(y1 + ya * h)))
    b = min(frame.shape[0], int(round(y1 + yb * h)))
    c = max(0, int(round(x1 + xa * w)))
    d = min(frame.shape[1], int(round(x1 + xb * w)))
    if b - a < 12 or d - c < 8:
        return None
    crop = frame[a:b, c:d]
    if crop.size == 0:
        return None
    scale = max(1.0, 180.0 / max(1, crop.shape[0]))
    if scale > 1.05:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return crop


def make_contact(crops, labels, out: Path):
    if not crops:
        return
    thumbs = []
    for im, lab in zip(crops, labels):
        h, w = im.shape[:2]
        canvas = np.zeros((150, 140, 3), np.uint8)
        s = min(132 / max(w, 1), 116 / max(h, 1))
        rs = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        yy = 4 + (116 - rs.shape[0]) // 2
        xx = 4 + (132 - rs.shape[1]) // 2
        canvas[yy:yy + rs.shape[0], xx:xx + rs.shape[1]] = rs
        cv2.putText(canvas, lab[:21], (4, 140), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    cols = 6
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 150, cols * 140, 3), np.uint8)
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r*150:(r+1)*150, c*140:(c+1)*140] = im
    cv2.imwrite(str(out), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--context', type=Path, required=True)
    ap.add_argument('--roster', type=Path, required=True)
    ap.add_argument('--nbacv-src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-frames-per-track', type=int, default=22)
    ap.add_argument('--min-track-rows', type=int, default=8)
    ap.add_argument('--min-median-height', type=float, default=52.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    crop_root = a.out / 'crops'
    crop_root.mkdir(exist_ok=True)

    sys.path.insert(0, str(a.nbacv_src))
    from nbacv.jersey import load_models, _legibility_scores, _parseq_read, canon_number, LEGIBLE_TH

    tracks = pd.read_csv(a.tracks)
    ctx = json.loads(a.context.read_text())
    pbp = ctx.get('pbp') or {}
    roster = load_roster(a.roster)
    lineup_ids = set(parse_lineup_ids(pbp.get('lineup_home'))) | set(parse_lineup_ids(pbp.get('lineup_away')))
    exact = {pid: roster[pid] for pid in lineup_ids if pid in roster}
    by_jersey = defaultdict(list)
    for pid, p in exact.items():
        by_jersey[str(p['jersey'])].append(p)
    allowed = {j: ps[0] for j, ps in by_jersey.items() if len(ps) == 1}

    stats = []
    for tid, g in tracks.groupby('track_id'):
        heights = g['y2'] - g['y1']
        stats.append({
            'track_id': int(tid), 'rows': int(len(g)),
            'median_height': float(heights.median()), 'median_conf': float(g['conf'].median()),
            'first_frame': int(g['frame'].min()), 'last_frame': int(g['frame'].max())
        })
    stat_df = pd.DataFrame(stats).sort_values(['rows','median_height'], ascending=[False,False])
    candidates = stat_df[(stat_df['rows'] >= a.min_track_rows) & (stat_df['median_height'] >= a.min_median_height)].copy()
    stat_df.to_csv(a.out/'track_stats.csv', index=False)

    cap = cv2.VideoCapture(str(a.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    track_crops = {}
    for _, s in candidates.iterrows():
        tid = int(s.track_id)
        g = tracks[tracks.track_id == tid].copy()
        g['height'] = g['y2'] - g['y1']
        g['quality'] = g['height'] * g['conf']
        # Prefer strong samples while forcing temporal spread.
        g = g.sort_values('quality', ascending=False)
        chosen = []
        min_gap = max(2, int(round(fps * .18)))
        for _, r in g.iterrows():
            fr = int(r.frame)
            if all(abs(fr - x) >= min_gap for x in chosen):
                chosen.append(fr)
            if len(chosen) >= a.max_frames_per_track:
                break
        crops, labels = [], []
        tdir = crop_root / f'track_{tid:02d}'
        tdir.mkdir(exist_ok=True)
        for fr in sorted(chosen):
            r = g[g.frame == fr].iloc[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, image = cap.read()
            if not ok:
                continue
            box = (r.x1, r.y1, r.x2, r.y2)
            for variant in (0, 1):
                c = crop_torso(image, box, variant)
                if c is None:
                    continue
                p = tdir / f'f{fr:04d}_v{variant}.jpg'
                cv2.imwrite(str(p), c)
                crops.append(c)
                labels.append(f'T{tid} f{fr} v{variant}')
        track_crops[tid] = (crops, labels)
        make_contact(crops, labels, a.out/f'track_{tid:02d}_contact.jpg')
    cap.release()

    parseq, leg, device = load_models(device='cpu')
    rows, evidence_all = [], []
    for _, s in candidates.iterrows():
        tid = int(s.track_id)
        crops, labels = track_crops.get(tid, ([], []))
        votes = defaultdict(float)
        reads_detail = []
        legible_count = 0
        if crops:
            scores = _legibility_scores(leg, crops, device)
            legible = [(c, float(sc), i) for i, (c, sc) in enumerate(zip(crops, scores)) if sc >= LEGIBLE_TH]
            legible.sort(key=lambda x: -x[1])
            legible_count = len(legible)
            if legible:
                reads = _parseq_read(parseq, [x[0] for x in legible], device)
                for (text, conf), (_, ls, idx) in zip(reads, legible):
                    num = canon_number(text)
                    lab = labels[idx] if idx < len(labels) else str(idx)
                    reads_detail.append({'crop': lab, 'legibility': round(ls,3), 'raw': text, 'canon': num, 'parseq_conf': round(float(conf),3)})
                    if num is not None and float(conf) >= .50 and str(num) in allowed:
                        votes[str(num)] += float(conf) * max(.25, ls)

        jersey = None
        player = None
        vote_share = 0.0
        best_weight = 0.0
        if votes:
            total = sum(votes.values())
            jersey, best_weight = max(votes.items(), key=lambda kv: kv[1])
            vote_share = best_weight / max(total, 1e-9)
            # Conservative gate: at least two useful reads and dominant vote.
            support = sum(1 for d in reads_detail if str(d.get('canon')) == str(jersey) and d.get('parseq_conf',0) >= .50)
            if support >= 2 and vote_share >= .67 and best_weight >= .90:
                player = allowed.get(str(jersey))
            else:
                jersey = None

        rec = dict(s)
        rec.update({
            'jersey': jersey,
            'player_id': None if player is None else int(player['id']),
            'name': None if player is None else player['name'],
            'team': None if player is None else player['team'],
            'vote_share': round(vote_share,3),
            'vote_weight': round(best_weight,3),
            'crop_count': len(crops),
            'legible_crops': legible_count,
        })
        rows.append(rec)
        evidence_all.append({'track_id': tid, 'votes': dict(sorted(votes.items(), key=lambda kv:-kv[1])), 'reads': reads_detail})

    out = pd.DataFrame(rows)
    out.to_csv(a.out/'track_identity.csv', index=False)
    (a.out/'ocr_evidence.json').write_text(json.dumps(evidence_all, indent=2))
    named = out[out['name'].notna()] if len(out) else out
    dup_names = int(named['name'].duplicated().sum()) if len(named) else 0
    qa = {
        'exact_lineup_players': len(exact),
        'candidate_tracklets': int(len(candidates)),
        'named_tracklets': int(len(named)),
        'unique_named_players': int(named['name'].nunique()) if len(named) else 0,
        'duplicate_named_tracklets': dup_names,
        'lineup': sorted([{'id':int(pid),'name':p['name'],'jersey':p['jersey'],'team':p['team']} for pid,p in exact.items()], key=lambda x:(x['team'],x['jersey'])),
        'rule': 'Names require multi-frame PARSeq jersey evidence and a unique exact-lineup jersey match; otherwise abstain.'
    }
    (a.out/'qa.json').write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
