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
import requests

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


def fetch_pair(game_id: str, player_id: int, max_attempts: int = 5):
    params = {'GameID': game_id, 'PlayerID': int(player_id)}
    last = None
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            r = session().get(XFG_URL, params=params, timeout=(6, 30))
            elapsed = time.time() - started
            if r.status_code == 200:
                j = r.json()
                if (isinstance(j, dict)
                        and str(j.get('gameId', '')).zfill(10) == game_id
                        and int(j.get('playerId') or 0) == int(player_id)):
                    return j, {'status': 200, 'attempts': attempt, 'elapsed': elapsed}
                last = {'status': 200, 'attempts': attempt, 'elapsed': elapsed, 'error': 'unexpected_payload'}
            else:
                last = {'status': r.status_code, 'attempts': attempt, 'elapsed': elapsed}
        except Exception as e:
            last = {'status': None, 'attempts': attempt, 'elapsed': time.time() - started, 'error': repr(e)}
        if attempt < max_attempts:
            time.sleep(min(8.0, 0.65 * (2 ** (attempt - 1))))
    return None, last or {'error': 'unknown'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='manifest.csv')
    ap.add_argument('--shard-index', type=int, required=True)
    ap.add_argument('--num-shards', type=int, required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out-dir', default='team_xefg_shards')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.manifest, dtype={'game_id': str})
    m['game_id'] = m['game_id'].str.zfill(10)
    games = sorted(m['game_id'].unique().tolist())
    shard_games = set(games[args.shard_index::args.num_shards])
    sm = m[m['game_id'].isin(shard_games)].copy()

    acc = defaultdict(lambda: {
        'tracked_fga': 0, 'tracked_fgm': 0, 'tracked_3pa': 0, 'tracked_3pm': 0,
        'actual_efg_num': 0.0, 'expected_efg_num': 0.0,
        'api_official_fga': 0, 'unlisted_fga': 0,
        'successful_pairs': 0, 'failed_pairs': 0,
    })
    errors = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_pair, str(r.game_id), int(r.player_id)):
                (str(r.game_id), int(r.player_id), str(r.player_name), str(r.team_abb), int(r.pbp_fga))
            for r in sm.itertuples(index=False)
        }
        total = len(futs)
        for done, fut in enumerate(as_completed(futs), start=1):
            game_id, pid, pname, team, pbp_fga = futs[fut]
            a = acc[team]
            try:
                payload, meta = fut.result()
            except Exception as e:
                payload, meta = None, {'error': repr(e)}
            if payload is None:
                a['failed_pairs'] += 1
                errors.append({'game_id': game_id, 'player_id': pid, 'player_name': pname, 'team_abb': team, 'pbp_fga': pbp_fga, **meta})
            else:
                a['successful_pairs'] += 1
                shot_list = payload.get('shotList') or []
                official = payload.get('shots')
                if isinstance(official, (int, float)) and not isinstance(official, bool):
                    oi = int(official)
                    a['api_official_fga'] += oi
                    a['unlisted_fga'] += max(0, oi - len(shot_list))
                for s in shot_list:
                    try:
                        xfg = float(s.get('shotQuality'))
                    except Exception:
                        continue
                    if math.isnan(xfg):
                        continue
                    made = int(s.get('success') or 0)
                    is3 = str(s.get('shotType') or '').upper().startswith('3PT')
                    w = 1.5 if is3 else 1.0
                    a['tracked_fga'] += 1
                    a['tracked_fgm'] += made
                    a['tracked_3pa'] += int(is3)
                    a['tracked_3pm'] += int(is3 and made)
                    a['actual_efg_num'] += made * w
                    a['expected_efg_num'] += xfg * w
            if done % 500 == 0 or done == total:
                print(f'shard={args.shard_index} done={done}/{total} errors={len(errors)} elapsed={time.time()-started:.1f}s', flush=True)

    pbp_team = sm.groupby('team_abb', as_index=False)['pbp_fga'].sum().rename(columns={'pbp_fga':'pbp_fga_shard'})
    pbp_lookup = dict(zip(pbp_team['team_abb'], pbp_team['pbp_fga_shard']))
    rows = []
    for team in sorted(set(sm['team_abb']) | set(acc.keys())):
        a = acc[team]
        rows.append({'team_abb': team, 'pbp_fga_shard': int(pbp_lookup.get(team, 0)), **a})
    pd.DataFrame(rows).to_csv(out / f'partial_{args.shard_index}.csv', index=False)
    pd.DataFrame(errors).to_csv(out / f'errors_{args.shard_index}.csv', index=False)
    summary = {
        'shard_index': args.shard_index,
        'games': len(shard_games),
        'request_pairs': len(sm),
        'errors': len(errors),
        'elapsed_seconds': time.time() - started,
    }
    (out / f'summary_{args.shard_index}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
