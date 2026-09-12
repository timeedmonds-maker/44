from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyreadr
import requests

PBP_URL = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds'
XFG_URL = 'https://stats.gleague.nba.com/stats/shotqualityvideologs'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    s = getattr(THREAD_LOCAL, 'session', None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        THREAD_LOCAL.session = s
    return s


def norm_game_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)


def parse_player(s: pd.Series) -> pd.DataFrame:
    x = s.astype(str).str.extract(r'^\s*(\d+)\s+(.*)$')
    x.columns = ['player_id', 'player_name']
    x['player_id'] = pd.to_numeric(x['player_id'], errors='coerce').astype('Int64')
    return x


def fetch_pair(game_id: str, player_id: int, max_attempts: int = 5) -> tuple[dict | None, dict]:
    params = {'GameID': game_id, 'PlayerID': int(player_id)}
    last = None
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            r = session().get(XFG_URL, params=params, timeout=(6, 30))
            elapsed = time.time() - started
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and str(j.get('gameId', '')).zfill(10) == game_id and int(j.get('playerId') or 0) == int(player_id):
                    return j, {'status': 200, 'attempts': attempt, 'elapsed': elapsed}
                last = {'status': 200, 'attempts': attempt, 'elapsed': elapsed, 'error': 'unexpected_payload'}
            else:
                last = {'status': r.status_code, 'attempts': attempt, 'elapsed': elapsed}
        except Exception as e:
            last = {'status': None, 'attempts': attempt, 'elapsed': time.time() - started, 'error': repr(e)}
        if attempt < max_attempts:
            time.sleep(min(8.0, 0.65 * (2 ** (attempt - 1))))
    return None, last or {'error': 'unknown'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, required=True)
    ap.add_argument('--num-shards', type=int, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--min-fga', type=int, default=300)
    ap.add_argument('--out-dir', default='out')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rds = out / f'pbp_{args.shard_index}.rds'
    r = requests.get(PBP_URL, timeout=180)
    r.raise_for_status()
    rds.write_bytes(r.content)
    df = next(iter(pyreadr.read_r(str(rds)).values()))
    rds.unlink(missing_ok=True)

    needed = {'game_id','event_num','player1_name','is_field_goal','action_type','area_detail','shot_distance'}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f'Missing PBP columns: {missing}; available={list(df.columns)}')

    df['game_id'] = norm_game_id(df['game_id'])
    is_fg = pd.to_numeric(df['is_field_goal'], errors='coerce').fillna(0).eq(1)
    reg = df['game_id'].str.startswith('002')
    fga = df.loc[is_fg & reg, ['game_id','event_num','player1_name','action_type','area_detail','shot_distance']].copy()
    p = parse_player(fga['player1_name'])
    fga['player_id'] = p['player_id']
    fga['player_name'] = p['player_name']
    fga = fga.dropna(subset=['player_id','event_num']).copy()
    fga['player_id'] = fga['player_id'].astype(int)
    fga['event_num'] = pd.to_numeric(fga['event_num'], errors='coerce').astype('Int64')
    fga = fga.dropna(subset=['event_num']).copy()
    fga['event_num'] = fga['event_num'].astype(int)

    player_counts = (
        fga.groupby(['player_id','player_name'], as_index=False)
        .agg(pbp_fga=('event_num','size'))
    )
    eligible = player_counts.loc[player_counts['pbp_fga'] >= args.min_fga].copy()
    eligible_ids = set(eligible['player_id'].astype(int).tolist())
    eligible_lookup = eligible.set_index('player_id').to_dict('index')

    action = fga['action_type'].fillna('').astype(str).str.lower().str.strip()
    detail = fga['area_detail'].fillna('').astype(str).str.strip()
    is2 = action.eq('2pt')
    short = detail.str.match(r'^8-16(?:\s|$)', na=False)
    long = detail.str.match(r'^16-24(?:\s|$)', na=False)
    mid = fga.loc[is2 & (short | long) & fga['player_id'].isin(eligible_ids)].copy()
    mid['zone'] = ''
    mid.loc[short.loc[mid.index], 'zone'] = 'short'
    mid.loc[long.loc[mid.index], 'zone'] = 'long'

    games = sorted(fga['game_id'].unique().tolist())
    shard_games = set(games[args.shard_index::args.num_shards])
    mid = mid[mid['game_id'].isin(shard_games)].copy()

    zone_lookup = {
        (str(r.game_id), int(r.player_id), int(r.event_num)): str(r.zone)
        for r in mid.itertuples(index=False)
    }
    manifest = (
        mid[['game_id','player_id','player_name']]
        .drop_duplicates(['game_id','player_id'])
        .sort_values(['game_id','player_id'])
        .reset_index(drop=True)
    )

    acc = defaultdict(lambda: {
        'short_attempts': 0, 'short_makes': 0, 'short_xfg_sum': 0.0,
        'long_attempts': 0, 'long_makes': 0, 'long_xfg_sum': 0.0,
        'matched_midrange_events': 0,
        'successful_games': 0, 'failed_games': 0,
    })
    errors = []
    timings = []
    matched_keys = set()
    started = time.time()

    def apply_payload(game_id: str, pid: int, payload: dict) -> None:
        a = acc[pid]
        a['successful_games'] += 1
        for s in (payload.get('shotList') or []):
            try:
                ev = int(float(s.get('eventNum')))
                xfg = float(s.get('shotQuality'))
            except Exception:
                continue
            if math.isnan(xfg):
                continue
            key = (game_id, pid, ev)
            zone = zone_lookup.get(key)
            if not zone:
                continue
            made = int(s.get('success') or 0)
            matched_keys.add(key)
            a['matched_midrange_events'] += 1
            a[f'{zone}_attempts'] += 1
            a[f'{zone}_makes'] += made
            a[f'{zone}_xfg_sum'] += xfg

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_pair, str(row.game_id), int(row.player_id)): (str(row.game_id), int(row.player_id), str(row.player_name))
            for row in manifest.itertuples(index=False)
        }
        total = len(futs)
        for done, fut in enumerate(as_completed(futs), start=1):
            game_id, pid, pname = futs[fut]
            try:
                payload, meta = fut.result()
            except Exception as e:
                payload, meta = None, {'error': repr(e)}
            if meta.get('elapsed') is not None:
                timings.append(float(meta['elapsed']))
            if payload is None:
                acc[pid]['failed_games'] += 1
                errors.append({'game_id': game_id, 'player_id': pid, 'player_name': pname, **meta})
            else:
                apply_payload(game_id, pid, payload)
            if done % 200 == 0 or done == total:
                print(f'shard={args.shard_index} done={done}/{total} errors={len(errors)} matched={len(matched_keys)}/{len(zone_lookup)} elapsed={time.time()-started:.1f}s', flush=True)

    rows = []
    player_ids = set(mid['player_id'].astype(int).tolist()) | set(acc.keys())
    for pid in sorted(player_ids):
        a = acc[pid]
        meta = eligible_lookup.get(pid, {})
        rows.append({
            'player_id': pid,
            'player_name': meta.get('player_name') or '',
            'pbp_fga_global': int(meta.get('pbp_fga') or 0),
            'short_attempts': int(a['short_attempts']),
            'short_makes': int(a['short_makes']),
            'short_xfg_sum': float(a['short_xfg_sum']),
            'long_attempts': int(a['long_attempts']),
            'long_makes': int(a['long_makes']),
            'long_xfg_sum': float(a['long_xfg_sum']),
            'matched_midrange_events': int(a['matched_midrange_events']),
            'successful_games': int(a['successful_games']),
            'failed_games': int(a['failed_games']),
        })
    pd.DataFrame(rows).to_csv(out / f'partial_{args.shard_index}.csv', index=False)
    pd.DataFrame(errors).to_csv(out / f'errors_{args.shard_index}.csv', index=False)

    unmatched = len(zone_lookup) - len(matched_keys)
    summary = {
        'shard_index': args.shard_index,
        'num_shards': args.num_shards,
        'games': len(shard_games),
        'eligible_players_global': len(eligible_ids),
        'midrange_pbp_events': len(zone_lookup),
        'request_pairs': len(manifest),
        'matched_midrange_events': len(matched_keys),
        'unmatched_midrange_events': unmatched,
        'errors': len(errors),
        'elapsed_seconds': time.time() - started,
        'mean_request_seconds': (sum(timings) / len(timings)) if timings else None,
        'max_request_seconds': max(timings) if timings else None,
    }
    (out / f'summary_{args.shard_index}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
