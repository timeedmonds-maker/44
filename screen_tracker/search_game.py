#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
import pyreadr
import requests

POSS_URL_DEFAULT = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/possessions2026/data.rds'
PBP_URL_DEFAULT = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds'


def norm_game(v):
    s = str(v or '').strip()
    if s.endswith('.0'):
        s = s[:-2]
    digits = ''.join(ch for ch in s if ch.isdigit())
    return digits.zfill(10) if digits else ''


def lineup_has(pid: str, lineup) -> bool:
    if lineup is None or (isinstance(lineup, float) and pd.isna(lineup)):
        return False
    return bool(re.search(rf'(?<!\d){re.escape(str(pid))}(?!\d)', str(lineup)))


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
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def clock_sec(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return math.nan
    s = str(v).strip()
    m = re.match(r'^(\d+):(\d+(?:\.\d+)?)$', s)
    if m:
        return int(m.group(1))*60 + float(m.group(2))
    m = re.match(r'^PT(?:(\d+)M)?([0-9.]+)S$', s)
    if m:
        return int(m.group(1) or 0)*60 + float(m.group(2))
    return math.nan


def download(url: str, path: Path):
    with requests.get(url, stream=True, timeout=240) as r:
        r.raise_for_status()
        with path.open('wb') as f:
            for chunk in r.iter_content(8*1024*1024):
                if chunk:
                    f.write(chunk)


def load_rds(url: str, path: Path) -> pd.DataFrame:
    download(url, path)
    obj = pyreadr.read_r(str(path))
    path.unlink(missing_ok=True)
    if not obj:
        raise RuntimeError(f'No dataframe in {url}')
    return next(iter(obj.values()))


def build_manifest(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir/'tmp'
    tmp.mkdir(exist_ok=True)

    game = norm_game(args.game_id)
    screener = str(args.screener_id)
    target = str(args.target_id)

    poss = load_rds(args.possessions_url, tmp/'possessions.rds')
    pbp = load_rds(args.pbp_url, tmp/'pbp.rds')
    tmp.rmdir()

    poss['game_id_norm'] = poss['game_id'].map(norm_game)
    pbp['game_id_norm'] = pbp['game_id'].map(norm_game)

    # The possession table is the authoritative possession universe. lineup_team
    # is the offensive lineup, so this is exactly the set requested: every
    # offensive possession with both specified players on court together.
    u = poss[
        poss['game_id_norm'].eq(game)
        & poss['lineup_team'].map(lambda x: lineup_has(screener, x))
        & poss['lineup_team'].map(lambda x: lineup_has(target, x))
    ].copy().reset_index(drop=True)
    if u.empty:
        raise RuntimeError(f'No offensive possessions found with players {screener} and {target} together in game {game}')

    u['period_i'] = pd.to_numeric(u['period'], errors='coerce').astype('Int64')
    u['start_clock_sec'] = u['start_time'].map(clock_sec)
    u['end_clock_sec'] = u['end_time'].map(clock_sec)
    pair_team = str(u['team_poss'].dropna().astype(str).str.strip().str.upper().mode().iloc[0]) if u['team_poss'].notna().any() else ''

    p = pbp[pbp['game_id_norm'].eq(game)].copy()
    p['period_i'] = pd.to_numeric(p['period'], errors='coerce')
    p['event_num_i'] = pd.to_numeric(p['event_num'], errors='coerce')
    p['clock_sec'] = p['clock'].map(clock_sec)
    for c in ('description','player1_name','player2_name','player3_name','action_type','sub_type','off_team_abb','team_abb'):
        if c not in p.columns:
            p[c] = ''
    by_period = {int(per):df for per,df in p[p['period_i'].notna()].groupby('period_i')}

    possessions = []
    scan_rows = []
    window_audit = []

    for row_i, r in u.iterrows():
        if pd.isna(r['period_i']):
            continue
        period = int(r['period_i'])
        team_poss = str(r.get('team_poss') or pair_team).strip().upper()
        poss_num = r.get('poss_num_team')
        if pd.notna(poss_num):
            try:
                poss_label = f'{int(float(poss_num)):04d}'
            except Exception:
                poss_label = str(poss_num)
        else:
            poss_label = f'row{row_i:04d}'
        uid = f'{game}-P{period}-{team_poss}-{poss_label}'

        hi = float(r['start_clock_sec']) if pd.notna(r['start_clock_sec']) else math.nan
        lo = float(r['end_clock_sec']) if pd.notna(r['end_clock_sec']) else math.nan
        df = by_period.get(period, p.iloc[0:0])
        if math.isfinite(hi) and math.isfinite(lo):
            q = df[df['clock_sec'].between(min(hi,lo)-0.15, max(hi,lo)+0.15, inclusive='both')].copy()
        else:
            q = df.iloc[0:0].copy()

        evmap = {}
        for _, pr in q.iterrows():
            ev = as_int(pr.get('event_num_i'))
            if ev is None:
                continue
            rec = pr.to_dict()
            prev = evmap.get(ev)
            if prev is None or len(str(rec.get('description') or '')) > len(str(prev.get('description') or '')):
                evmap[ev] = rec
        events = sorted(evmap)

        anchor = None
        if events:
            qteam = [ev for ev in events if str(evmap[ev].get('off_team_abb') or '').strip().upper() == team_poss]
            anchor = qteam[-1] if qteam else events[-1]
        terminal = evmap.get(anchor, {}) if anchor is not None else {}

        target_events = [ev for ev in events if event_pid_present(target, evmap[ev])]
        screener_events = [ev for ev in events if event_pid_present(screener, evmap[ev])]
        target_scoring_events = []
        for ev in events:
            er = evmap[ev]
            if str(er.get('player1_name') or '').startswith(target+' ') and (as_float(er.get('shot_pts')) or 0) > 0:
                target_scoring_events.append(ev)

        duration = abs(hi-lo) if math.isfinite(hi) and math.isfinite(lo) else None
        poss_rec = {
            'season':'2025-26','game_id':game,'period':period,'possession_uid':uid,
            'poss_num_team':poss_num,'pair_team':team_poss,
            'screener_id':screener,'screener_name':args.screener_name,
            'target_id':target,'target_name':args.target_name,
            'start_time':str(r.get('start_time') or ''),'end_time':str(r.get('end_time') or ''),
            'duration_s':None if duration is None else round(float(duration),3),
            'anchor_event_num':anchor,
            'window_event_nums':'|'.join(map(str,events)),
            'event_count':len(events),
            'target_event_nums':'|'.join(map(str,target_events)),
            'target_scoring_event_nums':'|'.join(map(str,target_scoring_events)),
            'screener_event_nums':'|'.join(map(str,screener_events)),
            'terminal_description':str(terminal.get('description') or ''),
            'pts_poss':as_float(r.get('pts_poss')),
            'type_end':str(r.get('type_end') or ''),
            'lineup_team':str(r.get('lineup_team') or ''),
            'lineup_opp':str(r.get('lineup_opp') or ''),
            'opp':str(r.get('opp') or ''),
        }
        possessions.append(poss_rec)
        window_audit.append({'possession_uid':uid,'pbp_events':[evmap[ev] for ev in events]})

        for ev in events:
            er = evmap[ev]
            target_p1 = str(er.get('player1_name') or '').startswith(target+' ')
            target_any = event_pid_present(target, er)
            screener_any = event_pid_present(screener, er)
            shot_pts = as_float(er.get('shot_pts')) or 0
            scan_rows.append({
                **poss_rec,
                'event_num':ev,
                'window_event_nums':str(ev),
                'anchor_event_num':ev,
                'duration_s':0,
                'event_description':str(er.get('description') or ''),
                'event_clock':str(er.get('clock') or ''),
                'player1_name':str(er.get('player1_name') or ''),
                'player2_name':str(er.get('player2_name') or ''),
                'player3_name':str(er.get('player3_name') or ''),
                'shot_pts':shot_pts,
                'target_primary_actor':target_p1,
                'target_any_actor':target_any,
                'target_scored':bool(target_p1 and shot_pts > 0),
                'screener_any_actor':screener_any,
                'screener_assist_signal':bool(
                    target_p1 and shot_pts > 0 and (
                        str(er.get('player2_name') or '').startswith(screener+' ')
                        or str(er.get('player3_name') or '').startswith(screener+' ')
                    )
                ),
            })

    possessions.sort(key=lambda x:(x['period'], str(x['poss_num_team'])))
    scan_rows.sort(key=lambda x:(x['period'], str(x['poss_num_team']), x['event_num']))
    if not possessions:
        raise RuntimeError(f'No joined possessions produced for {game}')
    if not scan_rows:
        raise RuntimeError(f'Possessions found but no PBP events joined for {game}')

    pd.DataFrame(possessions).to_csv(outdir/'joint_possessions.csv', index=False)
    pd.DataFrame(scan_rows).to_csv(outdir/'event_scan_manifest.csv', index=False)
    (outdir/'pbp_windows.json').write_text(json.dumps(window_audit, indent=2, default=str))
    manifest = {
        'tool_id':'SCREEN_TRACKER_SEARCH',
        'search_contract_version':'1.0.1',
        'game_id':game,
        'pair_team':pair_team,
        'screener':{'player_id':screener,'name':args.screener_name},
        'target':{'player_id':target,'name':args.target_name},
        'method':[
            'authoritative possession table filters every offensive possession whose lineup_team contains both specified players',
            'exact PBP is joined by game, period and possession start/end clock window',
            'every exact PBP event number inside every retained possession is submitted independently to clips.nba.com and fresh signed lrmedia HLS',
            'deterministic temporal screen geometry scans every resolvable HLS event clip',
            'visual evidence is aggregated to possessions; PBP actor/outcome signals rank but never define the universe',
            'final selected candidate must pass full Screen Tracker role and identity QA before render'
        ],
        'possession_count':len(possessions),
        'possessions_with_pbp_events':sum(1 for x in possessions if x['event_count'] > 0),
        'event_scan_count':len(scan_rows),
        'possessions_url':args.possessions_url,
        'pbp_url':args.pbp_url,
    }
    (outdir/'search_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _truth(v):
    if isinstance(v, bool):
        return v
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
        if col not in merged:
            merged[col] = 0
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
        if not r['visual_screen']:
            return 'NO_VISUAL_SCREEN'
        if r['target_primary_actor'] and r['screener_assist_signal']:
            return 'A'
        if r['target_primary_actor']:
            return 'B'
        if r['target_any_actor']:
            return 'C'
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
        'search_contract_version':'1.0.1',
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
    m.add_argument('--possessions-url', default=POSS_URL_DEFAULT)
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
