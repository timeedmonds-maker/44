#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kd_double_team_ballhandler_onnx import YoloOnnx, extract_frames, players_on_courtish
from adams_screen_candidate_onnx import screen_events
from adams_screen_temporal_onnx import associate, cluster_tracks, detect_sequences


def raw_court_polygons(frame_paths, court_model, device='cpu', conf=0.32, min_kp=5):
    from nbacv.court import _court_infer
    hulls = [None] * len(frame_paths)
    good_size = 640
    sizes = [640, 960]
    detected = 0
    for i, p in enumerate(frame_paths):
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        best = None
        for sz in [good_size] + [x for x in sizes if x != good_size]:
            cand = _court_infer(court_model, fr, sz, device)
            if cand is None:
                continue
            xy, cf = cand
            sel = (cf >= conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
            if int(sel.sum()) >= min_kp:
                best = xy[sel].astype(np.float32)
                good_size = sz
                break
        if best is not None:
            hulls[i] = cv2.convexHull(best)
            detected += 1
    good = [i for i, h in enumerate(hulls) if h is not None]
    if good:
        for i, h in enumerate(hulls):
            if h is not None:
                continue
            j = min(good, key=lambda g: abs(g - i))
            if abs(j - i) <= 3:
                hulls[i] = hulls[j]
    covered = sum(h is not None for h in hulls)
    return hulls, detected / max(len(hulls), 1), covered / max(len(hulls), 1)


def polygon_gate(people_pf, hulls, frame_paths, margin_frac=0.085):
    out = []
    tested = kept = removed = 0
    for pth, people, hull in zip(frame_paths, people_pf, hulls):
        fr = cv2.imread(str(pth))
        if fr is None or hull is None:
            out.append(people)
            continue
        margin_px = max(24.0, margin_frac * fr.shape[0])
        row = []
        for b in people:
            x = float((b[0] + b[2]) * 0.5)
            y = float(b[3])
            signed = cv2.pointPolygonTest(hull, (x, y), True)
            tested += 1
            if signed >= -margin_px:
                row.append(b); kept += 1
            else:
                removed += 1
        out.append(row)
    return out, {'tested': tested, 'kept': kept, 'removed': removed}


def keep_two_player_clusters(tracked, labels):
    mass = Counter()
    for dets in tracked:
        for d in dets:
            lab = labels.get(d['tid'])
            if lab is not None:
                mass[int(lab)] += 1
    keep = {lab for lab, _ in mass.most_common(2)}
    if len(keep) < 2:
        return tracked, keep, dict(mass)
    filtered = [[d for d in dets if labels.get(d['tid']) in keep] for dets in tracked]
    return filtered, keep, dict(mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--onnx-model', required=True)
    ap.add_argument('--court-model', required=True)
    ap.add_argument('--nbacv-src', required=True)
    ap.add_argument('--sample-fps', type=float, default=6.0)
    ap.add_argument('--max-seconds', type=float, default=16.0)
    ap.add_argument('--target-hls-width', type=int, default=960)
    ap.add_argument('--person-conf', type=float, default=0.14)
    ap.add_argument('--ball-conf', type=float, default=0.03)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.nbacv_src)))

    from ultralytics import YOLO
    court_model = YOLO(args.court_model)
    model = YoloOnnx(args.onnx_model)

    df = pd.read_csv(args.input, dtype={'game_id': str})
    df['game_id'] = df.game_id.str.zfill(10)
    rows, seq_rows = [], []

    with tempfile.TemporaryDirectory(prefix='adams_polycourt_') as td:
        root = Path(td)
        for _, r in df.iterrows():
            t0 = time.perf_counter()
            events = screen_events(r)
            total_sequences = 0
            best = None
            sampled = bh_obs = 0
            raw_tracks = player_tracks = 0
            raw_poly_cov = []
            bridged_poly_cov = []
            poly_removed = 0
            errors = []

            for ev in events:
                evdir = root / f'{r.game_id}_{ev}'
                try:
                    meta = extract_frames(str(r.game_id), int(ev), evdir,
                                          args.sample_fps, args.max_seconds,
                                          args.target_hls_width)
                    paths = meta['frames']
                    people_pf, balls_pf = [], []
                    for p in paths:
                        fr = cv2.imread(str(p))
                        if fr is None:
                            people_pf.append([]); balls_pf.append([]); continue
                        people_pf.append(players_on_courtish(
                            model.detect_class(fr, 0, args.person_conf),
                            fr.shape[0], max_players=18))
                        balls_pf.append(model.detect_class(fr, 32, args.ball_conf, 0.35))

                    hulls, raw_cov, bridge_cov = raw_court_polygons(paths, court_model)
                    raw_poly_cov.append(raw_cov)
                    bridged_poly_cov.append(bridge_cov)
                    court_people, gate_stats = polygon_gate(people_pf, hulls, paths)
                    poly_removed += gate_stats['removed']

                    tracked = associate(court_people)
                    raw_ids = {d['tid'] for ds in tracked for d in ds}
                    raw_tracks += len(raw_ids)

                    labels, centers = cluster_tracks(paths, tracked)
                    tracked2, keep_labs, cluster_mass = keep_two_player_clusters(tracked, labels)
                    kept_ids = {d['tid'] for ds in tracked2 for d in ds}
                    player_tracks += len(kept_ids)

                    seqs, bhs = detect_sequences(paths, tracked2, balls_pf,
                                                  labels, centers,
                                                  args.sample_fps)
                    seqs = [s for s in seqs
                            if s['end_frame'] > s['start_frame'] and
                            len({h['frame'] for h in s['hits']}) >= 2]
                    sampled += len(paths)
                    bh_obs += sum(x is not None for x in bhs)
                    total_sequences += len(seqs)

                    for si, s in enumerate(seqs):
                        b = s['best']
                        rec = {
                            'game_id': str(r.game_id),
                            'possession_uid': r.possession_uid,
                            'event_num': int(ev),
                            'sequence_index': si,
                            'start_sample': s['start_frame'],
                            'end_sample': s['end_frame'],
                            'start_s': round(s['start_frame'] / args.sample_fps, 3),
                            'end_s': round(s['end_frame'] / args.sample_fps, 3),
                            'ballhandler_tid': s['ballhandler_tid'],
                            'screener_tid': s['screener_tid'],
                            'defender_tid': b.get('defender_tid'),
                            'n_hits': len({h['frame'] for h in s['hits']}),
                            'best_score': b['score'],
                            'raw_polygon_coverage': round(raw_cov, 3),
                            'bridged_polygon_coverage': round(bridge_cov, 3),
                            'player_cluster_labels': '|'.join(map(str, sorted(keep_labs))),
                            'cluster_mass': json.dumps(cluster_mass, sort_keys=True),
                        }
                        seq_rows.append(rec)
                        if best is None or rec['best_score'] > best['best_score']:
                            best = rec
                except Exception as e:
                    errors.append(f'event {ev}: {type(e).__name__}: {e}')
                finally:
                    shutil.rmtree(evdir, ignore_errors=True)

            rows.append({
                'game_id': str(r.game_id), 'period': r.get('period'),
                'possession_uid': r.possession_uid,
                'start_time': r.get('start_time'), 'end_time': r.get('end_time'),
                'duration_s': r.get('duration_s'), 'pts_poss': r.get('pts_poss'),
                'type_end': r.get('type_end'), 'lineup_team': r.get('lineup_team'),
                'lineup_opp': r.get('lineup_opp'),
                'screen_event_nums': '|'.join(map(str, events)),
                'court_screen_sequences': total_sequences,
                'court_candidate': total_sequences > 0,
                'best_event_num': None if best is None else best['event_num'],
                'best_start_s': None if best is None else best['start_s'],
                'best_end_s': None if best is None else best['end_s'],
                'best_screen_score': None if best is None else best['best_score'],
                'sampled_frames': sampled,
                'ballhandler_observed_frames': bh_obs,
                'raw_track_count': raw_tracks,
                'player_track_count': player_tracks,
                'mean_raw_polygon_coverage': round(float(np.mean(raw_poly_cov)), 3) if raw_poly_cov else None,
                'mean_bridged_polygon_coverage': round(float(np.mean(bridged_poly_cov)), 3) if bridged_poly_cov else None,
                'polygon_removed_detections': poly_removed,
                'errors': ' || '.join(errors),
                'runtime_seconds': round(time.perf_counter() - t0, 2),
            })
            print(json.dumps(rows[-1], default=str), flush=True)

    out = pd.DataFrame(rows)
    seq = pd.DataFrame(seq_rows)
    out.to_csv(args.out / 'court_screen_candidates.csv', index=False)
    seq.to_csv(args.out / 'court_screen_sequences.csv', index=False)
    qa = {
        'rows': len(out),
        'candidate_possessions': int(out.court_candidate.sum()) if len(out) else 0,
        'candidate_rate': float(out.court_candidate.mean()) if len(out) else None,
        'total_sequences': int(out.court_screen_sequences.sum()) if len(out) else 0,
        'mean_runtime_seconds': float(out.runtime_seconds.mean()) if len(out) else None,
        'mean_raw_polygon_coverage': float(out.mean_raw_polygon_coverage.dropna().mean()) if len(out) and out.mean_raw_polygon_coverage.notna().any() else None,
        'mean_bridged_polygon_coverage': float(out.mean_bridged_polygon_coverage.dropna().mean()) if len(out) and out.mean_bridged_polygon_coverage.notna().any() else None,
        'polygon_removed_detections': int(out.polygon_removed_detections.sum()) if len(out) else 0,
        'error_rows': int(out.errors.fillna('').astype(str).ne('').sum()) if len(out) else 0,
        'note': 'Raw court-landmark polygon gate plus two dominant uniform clusters; candidate is not yet Adams-confirmed.'
    }
    (args.out / 'qa.json').write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
