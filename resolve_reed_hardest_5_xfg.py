from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

PLAYER_ID = 1642263
PLAYER_NAME = 'Reed Sheppard'
PBP_URL = 'https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
XFG_URL = 'https://stats.gleague.nba.com/stats/shotqualityvideologs'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}


def norm_gid(v) -> str:
    s = str(v)
    if s.endswith('.0'):
        s = s[:-2]
    return ''.join(ch for ch in s if ch.isdigit()).zfill(10)


def fetch_xfg(session: requests.Session, gid: str) -> dict:
    params = {'GameID': gid, 'PlayerID': PLAYER_ID}
    last = None
    for attempt in range(1, 6):
        try:
            r = session.get(XFG_URL, params=params, timeout=(8, 35))
            if r.status_code == 200:
                j = r.json()
                if str(j.get('gameId') or '').zfill(10) == gid and int(j.get('playerId') or 0) == PLAYER_ID:
                    return j
                last = f'unexpected payload: {r.text[:200]}'
            else:
                last = f'HTTP {r.status_code}: {r.text[:200]}'
        except Exception as exc:
            last = repr(exc)
        if attempt < 5:
            time.sleep(min(6.0, 0.7 * (2 ** (attempt - 1))))
    raise RuntimeError(f'xFG fetch failed for {gid}: {last}')


def main() -> None:
    print('Downloading authoritative full-season 2025-26 PBP release...', flush=True)
    usecols = ['game_id', 'event_num', 'player1_name', 'is_field_goal']
    df = pd.read_csv(PBP_URL, usecols=usecols, low_memory=False)
    if len(df) < 500_000:
        raise RuntimeError(f'PBP coverage sanity check failed: only {len(df)} rows')

    shooter = df['player1_name'].astype(str).str.match(r'^\s*1642263(?:\.0)?\s+', na=False)
    fg = pd.to_numeric(df['is_field_goal'], errors='coerce').fillna(0).eq(1)
    reed = df[shooter & fg].copy()
    if reed.empty:
        raise RuntimeError('No Reed Sheppard FGA rows in authoritative 2025-26 PBP release')
    reed['game_id'] = reed['game_id'].map(norm_gid)
    game_ids = sorted(reed['game_id'].unique().tolist())
    print(f'PBP_ROWS={len(df)} REED_FGA_ROWS={len(reed)} REED_GAMES_WITH_FGA={len(game_ids)}', flush=True)

    s = requests.Session()
    s.headers.update(HEADERS)
    shots = []
    errors = []
    for i, gid in enumerate(game_ids, 1):
        try:
            j = fetch_xfg(s, gid)
            for sh in (j.get('shotList') or []):
                try:
                    xfg = float(sh.get('shotQuality'))
                except Exception:
                    xfg = math.nan
                shots.append({
                    'season': '2025-26',
                    'player_id': PLAYER_ID,
                    'player_name': PLAYER_NAME,
                    'game_id': str(sh.get('gameId') or j.get('gameId') or '').zfill(10),
                    'game_date': j.get('gameDate'),
                    'matchup': j.get('matchup'),
                    'event_id': int(sh.get('eventNum')) if sh.get('eventNum') is not None else None,
                    'period': sh.get('period'),
                    'game_clock': sh.get('gameClock'),
                    'action_type': sh.get('actionType'),
                    'shot_type': sh.get('shotType'),
                    'made': int(sh.get('success') or 0),
                    'xfg': xfg,
                    'xfg_pct': xfg * 100 if not math.isnan(xfg) else None,
                    'loc_x': sh.get('locX'),
                    'loc_y': sh.get('locY'),
                    'guid': sh.get('guid'),
                    'large_hls': sh.get('largeHls'),
                    'playlist_url': sh.get('playListUrl'),
                })
        except Exception as exc:
            errors.append({'game_id': gid, 'error': repr(exc)})
        if i % 10 == 0 or i == len(game_ids):
            print(f'XFG_GAME_PROGRESS={i}/{len(game_ids)} SHOTS={len(shots)} ERRORS={len(errors)}', flush=True)

    if errors:
        Path('reed_xfg_errors.json').write_text(json.dumps(errors, indent=2))
        # Do not silently rank an incomplete season if any Reed player-game request failed.
        raise RuntimeError(f'Official xFG requests failed for {len(errors)} Reed games; see reed_xfg_errors.json')

    allshots = pd.DataFrame(shots)
    if allshots.empty:
        raise RuntimeError('Official xFG returned zero Reed shot rows')
    made = allshots[(allshots['made'] == 1) & pd.to_numeric(allshots['xfg'], errors='coerce').notna() & allshots['event_id'].notna()].copy()
    made['xfg'] = pd.to_numeric(made['xfg'], errors='coerce')
    top = made.sort_values(['xfg', 'game_id', 'event_id'], kind='stable').head(5).copy()
    if len(top) != 5:
        raise RuntimeError(f'Expected 5 Reed makes with official xFG and exact event IDs, got {len(top)}')

    events = []
    for rank, row in enumerate(top.to_dict('records'), 1):
        events.append({
            'rank': rank,
            'season': '2025-26',
            'player_id': PLAYER_ID,
            'player_name': PLAYER_NAME,
            'game_id': row['game_id'],
            'event_id': int(row['event_id']),
            'game_date': row.get('game_date'),
            'matchup': row.get('matchup'),
            'period': row.get('period'),
            'game_clock': row.get('game_clock'),
            'action_type': row.get('action_type'),
            'shot_type': row.get('shot_type'),
            'xfg': float(row['xfg']),
            'xfg_pct': round(float(row['xfg']) * 100, 2),
            'loc_x': row.get('loc_x'),
            'loc_y': row.get('loc_y'),
            'difficulty_definition': 'lowest official NBA shot-level xFG among Reed Sheppard made field goals',
            'source_provenance': 'NBA shotqualityvideologs via stats.gleague.nba.com; Reed game set from authoritative ramirobentes 2025-26 PBP release asset',
        })

    request = {
        'query': 'Reed Sheppard 5 hardest made field goals, 2025-26 regular season, ranked by lowest official NBA shot-level xFG',
        'player_id': PLAYER_ID,
        'season': '2025-26',
        'ranking': {'field': 'xfg', 'direction': 'ascending', 'made_only': True, 'take': 5},
        'pbp_release_url': PBP_URL,
        'xfg_source': XFG_URL,
        'reed_fga_rows': len(reed),
        'reed_games_with_fga': len(game_ids),
        'official_xfg_shot_rows': len(allshots),
        'official_xfg_made_rows': len(made),
        'events': events,
    }
    Path('request.json').write_text(json.dumps(request, indent=2, default=str))
    allshots.to_csv('reed_official_xfg_shots_2025_26.csv', index=False)
    print('TOP5=' + json.dumps(events), flush=True)


if __name__ == '__main__':
    main()
