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
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--min-fga', type=int, default=300)
    ap.add_argument('--out-dir', default='xfg_efg_shards')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rds = out / f'pbp_{args.shard_index}.rds'
    r = requests.get(PBP_URL, timeout=180)
    r.raise_for_status()
    rds.write_bytes(r.content)
    df = next(iter(pyreadr.read_r(str(rds)).values()))
    rds.unlink(missing_ok=True)

    df['game_id'] = norm_game_id(df['game_id'])
    is_fg = pd.to_numeric(df['is_field_goal'], errors='coerce').fillna(0).eq(1)
    reg = df['game_id'].str.startswith('002')
    fga = df.loc[is_fg & reg, ['game_id', 'event_num', 'player1_name']].copy()
    p = parse_player(fga['player1_name'])
    fga['player_id'] = p['player_id']
    fga['player_name'] = p['player_name']
    fga = fga.dropna(subset=['player_id']).copy()
    fga['player_id'] = fga['player_id'].astype(int)

    player_counts = (
        fga.groupby(['player_id', 'player_name'], as_index=False)
        .agg(pbp_fga=('event_num', 'size'))
    )
    eligible = player_counts.loc[player_counts['pbp_fga'] >= args.min_fga].copy()
    eligible_ids = set(eligible['player_id'].astype(int).tolist())
    eligible_lookup = eligible.set_index('player_id').to_dict('index')

    games = sorted(fga['game_id'].unique().tolist())
    shard_games = set(games[args.shard_index::args.num_shards])
    manifest = (
        fga.loc[fga['game_id'].isin(shard_games) & fga['player_id'].isin(eligible_ids), ['game_id', 'player_id', 'player_name']]
        .drop_duplicates(['game_id', 'player_id'])
        .sort_values(['game_id', 'player_id'])
        .reset_index(drop=True)
    )

    acc = defaultdict(lambda: {
        'tracked_fga': 0,
        'tracked_fgm': 0,
        'tracked_3pa': 0,
        'tracked_3pm': 0,
        'actual_efg_num': 0.0,
        'expected_efg_num': 0.0,
        'api_official_fga': 0,
        'unlisted_fga': 0,
        'successful_games': 0,
        'failed_games': 0,
        'player_name_api': None,
    })
    errors = []
    timings = []
    started = time.time()

    def apply_payload(pid: int, payload: dict) -> None:
        a = acc[pid]
        shot_list = payload.get('shotList') or []
        official = payload.get('shots')
        if isinstance(official, (int, float)) and not isinstance(official, bool):
            official_i = int(official)
            a['api_official_fga'] += official_i
            a['unlisted_fga'] += max(0, official_i - len(shot_list))
        a['successful_games'] += 1
        a['player_name_api'] = payload.get('playerName') or a['player_name_api']
        for s in shot_list:
            try:
                xfg = float(s.get('shotQuality'))
            except Exception:
                continue
            if math.isnan(xfg):
                continue
            made = int(s.get('success') or 0)
            is3 = str(s.get('shotType') or '').upper().startswith('3PT')
            weight = 1.5 if is3 else 1.0
            a['tracked_fga'] += 1
            a['tracked_fgm'] += made
            a['tracked_3pa'] += int(is3)
            a['tracked_3pm'] += int(is3 and made)
            a['actual_efg_num'] += made * weight
            a['expected_efg_num'] += xfg * weight

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
                apply_payload(pid, payload)
            if done % 250 == 0 or done == total:
                print(f'shard={args.shard_index} done={done}/{total} errors={len(errors)} elapsed={time.time()-started:.1f}s', flush=True)

    rows = []
    for pid, a in acc.items():
        meta = eligible_lookup.get(pid, {})
        rows.append({
            'player_id': pid,
            'player_name': a['player_name_api'] or meta.get('player_name') or '',
            'pbp_fga_global': int(meta.get('pbp_fga') or 0),
            'tracked_fga': int(a['tracked_fga']),
            'tracked_fgm': int(a['tracked_fgm']),
            'tracked_3pa': int(a['tracked_3pa']),
            'tracked_3pm': int(a['tracked_3pm']),
            'actual_efg_num': float(a['actual_efg_num']),
            'expected_efg_num': float(a['expected_efg_num']),
            'api_official_fga': int(a['api_official_fga']),
            'unlisted_fga': int(a['unlisted_fga']),
            'successful_games': int(a['successful_games']),
            'failed_games': int(a['failed_games']),
        })
    pd.DataFrame(rows).to_csv(out / f'partial_{args.shard_index}.csv', index=False)
    pd.DataFrame(errors).to_csv(out / f'errors_{args.shard_index}.csv', index=False)
    summary = {
        'shard_index': args.shard_index,
        'num_shards': args.num_shards,
        'games': len(shard_games),
        'eligible_players_global': len(eligible_ids),
        'request_pairs': len(manifest),
        'errors': len(errors),
        'elapsed_seconds': time.time() - started,
        'mean_request_seconds': (sum(timings) / len(timings)) if timings else None,
        'max_request_seconds': max(timings) if timings else None,
    }
    (out / f'summary_{args.shard_index}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
