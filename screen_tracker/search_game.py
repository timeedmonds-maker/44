#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

PBP_URL_DEFAULT = 'https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'


def norm_game(v):
    s = str(v or '').strip()
    if s.endswith('.0'):
        s = s[:-2]
    digits = ''.join(ch for ch in s if ch.isdigit())
    return digits.zfill(10) if digits else ''


def pid_in_lineup(pid: str, lineup: str) -> bool:
    pid = str(pid)
    for part in str(lineup or '').split(','):
        head = part.strip().split(' ', 1)[0]
        if head == pid:
            return True
    return False


def event_pid_present(pid: str, row: dict) -> bool:
    pid = str(pid)
    return any(str(row.get(k) or '').strip().startswith(pid + ' ') for k in ('player1_name','player2_name','player3_name'))


def as_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def load_game_rows(game_id: str, pbp_url: str):
    req = urllib.request.Request(pbp_url, headers={'User-Agent':'Mozilla/5.0'})
    out = []
    with urllib.request.urlopen(req, timeout=240) as resp:
        reader = csv.DictReader(io.TextIOWrapper(resp, encoding='utf-8-sig', newline=''))
        for r in reader:
            if norm_game(r.get('game_id')) == game_id:
                out.append(r)
    if not out:
        raise RuntimeError(f'No PBP rows found for game {game_id}')
    return out


def build_manifest(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    game = norm_game(args.game_id)
    screener = str(args.screener_id)
    target = str(args.target_id)
    rows = load_game_rows(game, args.pbp_url)

    side_votes = defaultdict(int)
    team_votes = defaultdict(int)
    for r in rows:
        lh, la = str(r.get('lineup_home') or ''), str(r.get('lineup_away') or '')
        if pid_in_lineup(screener, lh) and pid_in_lineup(target, lh):
            side_votes['home'] += 1
            if r.get('team_home'): team_votes[str(r.get('team_home'))] += 1
        if pid_in_lineup(screener, la) and pid_in_lineup(target, la):
            side_votes['away'] += 1
            if r.get('team_away'): team_votes[str(r.get('team_away'))] += 1
    if not side_votes:
        raise RuntimeError(f'Players {screener} and {target} were never found together in an exact lineup for game {game}')
    pair_side = max(side_votes, key=side_votes.get)
    pair_team = max(team_votes, key=team_votes.get) if team_votes else None

    groups = defaultdict(list)
    for r in rows:
        poss = r.get('possession')
        period = as_int(r.get('period'))
        if poss in (None, '', 'nan') or period is None:
            continue
        pnum = as_int(poss)
        pkey = str(pnum if pnum is not None else poss)
        groups[(period, pkey)].append(r)

    possessions = []
    scan_rows = []
    for (period, pkey), grows in groups.items():
        pair_rows = []
        for r in grows:
            lineup = str(r.get('lineup_home') or '') if pair_side == 'home' else str(r.get('lineup_away') or '')
            if not (pid_in_lineup(screener, lineup) and pid_in_lineup(target, lineup)):
                continue
            off = str(r.get('off_team_abb') or '')
            team = str(r.get('team_home') or '') if pair_side == 'home' else str(r.get('team_away') or '')
            if off and team and off != team:
                continue
            pair_rows.append(r)
        if not pair_rows:
            continue

        evmap = {}
        for r in grows:
            ev = as_int(r.get('event_num'))
            if ev is None:
                continue
            prev = evmap.get(ev)
            if prev is None or len(str(r.get('description') or '')) > len(str(prev.get('description') or '')):
                evmap[ev] = r
        events = sorted(evmap)
        if not events:
            continue

        secs = [as_float(r.get('secs_game')) for r in grows]
        secs = [x for x in secs if x is not None]
        duration = max(secs) - min(secs) if len(secs) >= 2 else max([as_float(r.get('secs_played')) or 0 for r in grows] or [0])
        uid = f'{game}_P{period}_POS{pkey}'
        anchor = events[-1]
        last = evmap[anchor]
        start_clock = next((str(r.get('start_poss')) for r in grows if str(r.get('start_poss') or '').strip()), str(grows[0].get('clock') or ''))
        end_clock = str(last.get('clock') or '')
        target_events = [ev for ev,r in evmap.items() if event_pid_present(target, r)]
        screener_events = [ev for ev,r in evmap.items() if event_pid_present(screener, r)]
        target_scoring_events = []
        for ev,r in evmap.items():
            if str(r.get('player1_name') or '').startswith(target+' ') and (as_float(r.get('shot_pts')) or 0) > 0:
                target_scoring_events.append(ev)

        poss_rec = {
            'season':'2025-26','game_id':game,'period':period,'possession_uid':uid,
            'possession_number':pkey,'pair_side':pair_side,'pair_team':pair_team or '',
            'screener_id':screener,'screener_name':args.screener_name,
            'target_id':target,'target_name':args.target_name,
            'start_time':start_clock,'end_time':end_clock,'duration_s':round(float(duration),3),
            'anchor_event_num':anchor,'window_event_nums':'|'.join(map(str,events)),
            'event_count':len(events),'target_event_nums':'|'.join(map(str,target_events)),
            'target_scoring_event_nums':'|'.join(map(str,target_scoring_events)),
            'screener_event_nums':'|'.join(map(str,screener_events)),
            'terminal_description':str(last.get('description') or ''),
            'pts_poss':max([as_float(r.get('pts_poss')) or 0 for r in grows] or [0]),
            'type_end':str(last.get('type_end') or last.get('description') or '')
        }
        possessions.append(poss_rec)

        for ev in events:
            r = evmap[ev]
            target_p1 = str(r.get('player1_name') or '').startswith(target+' ')
            target_any = event_pid_present(target, r)
            screener_any = event_pid_present(screener, r)
            shot_pts = as_float(r.get('shot_pts')) or 0
            scan_rows.append({
                **poss_rec,
                'event_num':ev,
                'window_event_nums':str(ev),
                'anchor_event_num':ev,
                'duration_s':0,
                'event_description':str(r.get('description') or ''),
                'event_clock':str(r.get('clock') or ''),
                'player1_name':str(r.get('player1_name') or ''),
                'player2_name':str(r.get('player2_name') or ''),
                'player3_name':str(r.get('player3_name') or ''),
                'shot_pts':shot_pts,
                'target_primary_actor':target_p1,
                'target_any_actor':target_any,
                'target_scored':bool(target_p1 and shot_pts > 0),
                'screener_any_actor':screener_any,
                'screener_assist_signal':bool(target_p1 and shot_pts > 0 and (str(r.get('player2_name') or '').startswith(screener+' ') or str(r.get('player3_name') or '').startswith(screener+' '))),
            })

    possessions.sort(key=lambda x:(x['period'], str(x['possession_number'])))
    scan_rows.sort(key=lambda x:(x['period'], str(x['possession_number']), x['event_num']))
    if not possessions:
        raise RuntimeError(f'No offensive possessions found with both players on court in game {game}')

    pd.DataFrame(possessions).to_csv(outdir/'joint_possessions.csv', index=False)
    pd.DataFrame(scan_rows).to_csv(outdir/'event_scan_manifest.csv', index=False)
    manifest = {
        'tool_id':'SCREEN_TRACKER_SEARCH',
        'search_contract_version':'1.0.0',
        'game_id':game,
        'pair_side':pair_side,
        'pair_team':pair_team,
        'screener':{'player_id':screener,'name':args.screener_name},
        'target':{'player_id':target,'name':args.target_name},
        'method':[
            'exact PBP join filters all offensive possessions with both players in the exact lineup',
            'every event number in every retained possession is submitted to clips.nba.com and its fresh signed lrmedia HLS',
            'deterministic frame sampling and temporal screen geometry scan runs on each HLS event clip',
            'candidate ranking combines visual screen evidence with PBP actor relevance; final identity is validated by the full Screen Tracker application before render'
        ],
        'possession_count':len(possessions),
        'event_scan_count':len(scan_rows),
        'pbp_url':args.pbp_url,
    }
    (outdir/'search_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _truth(v):
    if isinstance(v, bool): return v
    return str(v).strip().lower() == 'true'


def rank_results(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.event_manifest, dtype={'game_id':str,'screener_id':str,'target_id':str})
    scan_root = Path(args.scan_root)
    cand_files = list(scan_root.rglob('temporal_candidates.csv'))
    seq_files = list(scan_root.rglob('screen_sequences.csv'))
    if not cand_files:
        raise RuntimeError(f'No temporal_candidates.csv found under {scan_root}')
    c = pd.concat([pd.read_csv(p, dtype={'game_id':str}) for p in cand_files], ignore_index=True)
    s_frames = [pd.read_csv(p, dtype={'game_id':str}) for p in seq_files if p.stat().st_size > 1]
    seq = pd.concat(s_frames, ignore_index=True) if s_frames else pd.DataFrame()

    c['submitted_event_num'] = pd.to_numeric(c['screen_event_nums'].astype(str).str.split('|').str[0], errors='coerce')
    c['event_num_merge'] = c['submitted_event_num']
    m = manifest.copy()
    m['event_num_merge'] = pd.to_numeric(m['event_num'], errors='coerce')
    merged = c.merge(m, on=['possession_uid','event_num_merge'], how='left', suffixes=('_scan',''))

    if not seq.empty:
        seq['event_num_merge'] = pd.to_numeric(seq['event_num'], errors='coerce')
        seqagg = seq.groupby(['possession_uid','event_num_merge']).agg(
            sequence_count=('sequence_index','count'),
            max_sequence_hits=('n_hits','max'),
            max_visual_score=('best_score','max'),
            best_sequence_start_s=('start_s','min'),
            best_sequence_end_s=('end_s','max'),
        ).reset_index()
        merged = merged.merge(seqagg, on=['possession_uid','event_num_merge'], how='left')
    else:
        merged['sequence_count'] = 0
        merged['max_sequence_hits'] = 0
        merged['max_visual_score'] = pd.to_numeric(merged.get('best_screen_score'), errors='coerce')

    for col in ('sequence_count','max_sequence_hits','max_visual_score'):
        if col not in merged: merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)
    for col in ('target_primary_actor','target_any_actor','target_scored','screener_assist_signal'):
        merged[col] = merged[col].map(_truth)
    merged['visual_screen'] = merged['sequence_count'] > 0
    merged['pair_search_score'] = (
        merged['max_visual_score']
        + 1.25 * merged['max_sequence_hits'].clip(upper=6)
        + 4.0 * merged['target_primary_actor'].astype(int)
        + 1.5 * merged['target_any_actor'].astype(int)
        + 2.0 * merged['target_scored'].astype(int)
        + 4.0 * merged['screener_assist_signal'].astype(int)
    )
    def tier(r):
        if not r['visual_screen']: return 'NO_VISUAL_SCREEN'
        if r['target_primary_actor'] and r['screener_assist_signal']: return 'A'
        if r['target_primary_actor']: return 'B'
        if r['target_any_actor']: return 'C'
        return 'D'
    merged['candidate_tier'] = merged.apply(tier, axis=1)
    merged = merged.sort_values(['visual_screen','pair_search_score','max_visual_score'], ascending=[False,False,False])
    merged.to_csv(outdir/'ranked_events.csv', index=False)

    visual = merged[merged['visual_screen']].copy()
    if len(visual):
        idx = visual.groupby('possession_uid')['pair_search_score'].idxmax()
        poss = visual.loc[idx].sort_values(['pair_search_score','max_visual_score'], ascending=False).reset_index(drop=True)
    else:
        poss = merged.head(0).copy()
    poss.insert(0, 'rank', range(1, len(poss)+1))
    poss.to_csv(outdir/'ranked_possessions.csv', index=False)

    top = poss.head(int(args.top_n))
    top.to_csv(outdir/'top_candidates.csv', index=False)
    summary = {
        'tool_id':'SCREEN_TRACKER_SEARCH',
        'search_contract_version':'1.0.0',
        'event_rows_scanned':int(len(merged)),
        'events_with_visual_screen':int(merged['visual_screen'].sum()),
        'candidate_possessions':int(len(poss)),
        'top_n':int(min(len(top), int(args.top_n))),
        'ranking_note':'visual screen geometry first; exact-PBP target/screener actor signals rank candidates; final pair identity must pass full Screen Tracker role/identity QA before render',
        'top_candidates':[
            {
                'rank':int(r['rank']),
                'game_id':str(r.get('game_id','')).zfill(10),
                'period':int(float(r.get('period',0))) if pd.notna(r.get('period')) else None,
                'possession_uid':str(r.get('possession_uid')),
                'event_num':int(float(r.get('event_num_merge'))) if pd.notna(r.get('event_num_merge')) else None,
                'clock':str(r.get('event_clock','')),
                'description':str(r.get('event_description','')),
                'tier':str(r.get('candidate_tier')),
                'score':round(float(r.get('pair_search_score',0)),3),
                'visual_score':round(float(r.get('max_visual_score',0)),3),
                'sequence_hits':int(float(r.get('max_sequence_hits',0))),
            } for _,r in top.iterrows()
        ]
    }
    (outdir/'search_qa.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description='Screen Tracker game search manifest/ranking tool')
    sub = ap.add_subparsers(dest='cmd', required=True)

    m = sub.add_parser('manifest')
    m.add_argument('--game-id', required=True)
    m.add_argument('--screener-id', required=True)
    m.add_argument('--target-id', required=True)
    m.add_argument('--screener-name', default='Screener')
    m.add_argument('--target-name', default='Target')
    m.add_argument('--pbp-url', default=PBP_URL_DEFAULT)
    m.add_argument('--out', required=True)
    m.set_defaults(func=build_manifest)

    r = sub.add_parser('rank')
    r.add_argument('--event-manifest', required=True)
    r.add_argument('--scan-root', required=True)
    r.add_argument('--top-n', type=int, default=12)
    r.add_argument('--out', required=True)
    r.set_defaults(func=rank_results)

    args = ap.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
