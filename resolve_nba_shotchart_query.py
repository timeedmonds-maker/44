from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

BASE = 'https://stats.nba.com/stats/shotchartdetail'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36'


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        'User-Agent': UA,
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    q = json.loads(Path(args.request).read_text(encoding='utf-8'))

    player_id = str(q['player_id'])
    season = q.get('season', '2025-26')
    season_type = q.get('season_type', 'Regular Season')
    action_contains = q.get('action_contains', 'Dunk').lower()
    last_n = int(q.get('last_n', 5))

    params = {
        'AheadBehind': '', 'CFID': '', 'CFPARAMS': '', 'ClutchTime': '',
        'ContextFilter': '', 'ContextMeasure': 'FGA', 'DateFrom': '', 'DateTo': '',
        'EndPeriod': '10', 'EndRange': '28800', 'GameID': '', 'GameSegment': '',
        'LastNGames': '0', 'LeagueID': '00', 'Location': '', 'Month': '0',
        'OpponentTeamID': '0', 'Outcome': '', 'Period': '0', 'PlayerID': player_id,
        'PlayerPosition': '', 'PointDiff': '', 'Position': '', 'RangeType': '0',
        'RookieYear': '', 'Season': season, 'SeasonSegment': '', 'SeasonType': season_type,
        'StartPeriod': '1', 'StartRange': '0', 'TeamID': '0', 'VsConference': '', 'VsDivision': ''
    }
    url = BASE + '?' + urllib.parse.urlencode(params)
    data = fetch_json(url)
    rs = (data.get('resultSets') or data.get('resultSet') or [])
    if isinstance(rs, dict):
        rs = [rs]
    shotset = None
    for s in rs:
        headers = s.get('headers') or []
        if 'GAME_ID' in headers and ('GAME_EVENT_ID' in headers or 'EVENT_TYPE' in headers):
            shotset = s
            break
    if shotset is None:
        raise SystemExit('Shot chart result set not found')

    headers = shotset['headers']
    rows = [dict(zip(headers, row)) for row in shotset.get('rowSet', [])]
    matches = []
    for r in rows:
        action = str(r.get('ACTION_TYPE') or '')
        made = int(r.get('SHOT_MADE_FLAG') or 0) == 1
        if made and action_contains in action.lower():
            matches.append(r)

    def key(r):
        return (str(r.get('GAME_DATE') or ''), str(r.get('GAME_ID') or ''), int(r.get('GAME_EVENT_ID') or 0))
    matches.sort(key=key)
    selected = matches[-last_n:]
    events = []
    for rank, r in enumerate(selected, 1):
        events.append({
            'rank': rank,
            'game_date': r.get('GAME_DATE'),
            'game_id': str(r.get('GAME_ID')),
            'event_id': int(r.get('GAME_EVENT_ID')),
            'player_id': int(r.get('PLAYER_ID') or player_id),
            'player_name': r.get('PLAYER_NAME'),
            'action_type': r.get('ACTION_TYPE'),
            'shot_type': r.get('SHOT_TYPE'),
            'shot_distance': r.get('SHOT_DISTANCE'),
            'htm': r.get('HTM'),
            'vtm': r.get('VTM'),
            'loc_x': r.get('LOC_X'),
            'loc_y': r.get('LOC_Y'),
        })

    payload = {
        'source': 'official NBA stats shotchartdetail',
        'request_url': url,
        'query': q,
        'total_player_shots': len(rows),
        'total_matching_made_actions': len(matches),
        'resolved_count': len(events),
        'events': events,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)
    if len(events) != last_n:
        raise SystemExit(f'Expected {last_n} events, got {len(events)}')


if __name__ == '__main__':
    main()
