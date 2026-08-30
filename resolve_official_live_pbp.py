from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.parse
import urllib.request
from pathlib import Path

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36'
STATIC_SCHEDULES = [
    'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json',
    'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json',
]


def get_json(url: str, timeout: int = 30, nba_headers: bool = False) -> dict:
    headers = {'User-Agent': UA, 'Accept': 'application/json, text/plain, */*'}
    if nba_headers:
        headers.update({'Referer': 'https://www.nba.com/', 'Origin': 'https://www.nba.com'})
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def walk_games(obj):
    if isinstance(obj, dict):
        if 'gameId' in obj and ('homeTeam' in obj or 'awayTeam' in obj):
            yield obj
        for v in obj.values():
            yield from walk_games(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk_games(x)


def tricode(team):
    if not isinstance(team, dict):
        return ''
    for k in ('teamTricode', 'teamCode', 'tricode'):
        if team.get(k):
            return str(team[k]).upper()
    return ''


def schedule_games_static(team: str, game_prefix: str):
    diagnostics = []
    for url in STATIC_SCHEDULES:
        try:
            data = get_json(url, 30)
            games = list(walk_games(data))
            prefix_games = [g for g in games if str(g.get('gameId', '')).startswith(game_prefix)]
            selected = [g for g in prefix_games if team in {tricode(g.get('homeTeam')), tricode(g.get('awayTeam'))}]
            diagnostics.append({'url': url, 'all_games': len(games), 'prefix_games': len(prefix_games), 'team_games': len(selected)})
            if selected:
                out = []
                for g in selected:
                    out.append({
                        'game_id': str(g.get('gameId')),
                        'game_date': g.get('gameDate') or g.get('gameDateEst') or g.get('gameDateTimeUTC') or g.get('gameDateTimeEst') or '',
                        'home': tricode(g.get('homeTeam')),
                        'away': tricode(g.get('awayTeam')),
                    })
                return out, diagnostics
        except Exception as e:
            diagnostics.append({'url': url, 'error': repr(e)})
    return [], diagnostics


def schedule_games_stats(team_id: int, season: str, game_prefix: str):
    params = {
        'DateFrom': '', 'DateTo': '', 'LeagueID': '00', 'Season': season,
        'SeasonType': 'Regular Season', 'TeamID': str(team_id)
    }
    url = 'https://stats.nba.com/stats/teamgamelog?' + urllib.parse.urlencode(params)
    data = get_json(url, 25, nba_headers=True)
    rs = data.get('resultSets') or []
    if not rs:
        return [], {'url': url, 'error': 'no_result_sets'}
    headers = rs[0].get('headers') or []
    rows = [dict(zip(headers, r)) for r in rs[0].get('rowSet', [])]
    out = []
    for r in rows:
        gid = str(r.get('Game_ID') or r.get('GAME_ID') or '')
        if gid.startswith(game_prefix):
            out.append({'game_id': gid, 'game_date': r.get('GAME_DATE') or '', 'home': '', 'away': ''})
    return out, {'url': url, 'rows': len(rows), 'team_games': len(out)}


def resolve_game(game, player_id: int, action_contains: str):
    gid = game['game_id']
    url = f'https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json'
    data = get_json(url, 30)
    actions = ((data.get('game') or {}).get('actions') or [])
    found = []
    for a in actions:
        try:
            pid = int(a.get('personId') or 0)
        except Exception:
            pid = 0
        if pid != player_id:
            continue
        desc = str(a.get('description') or '')
        subtype = str(a.get('subType') or '')
        action_type = str(a.get('actionType') or '')
        hay = ' '.join([desc, subtype, action_type]).lower()
        is_fg = a.get('isFieldGoal') in {1, True, '1'}
        made = str(a.get('shotResult') or '').lower() == 'made'
        if is_fg and made and action_contains in hay:
            found.append({
                'game_date': game.get('game_date', ''),
                'game_id': gid,
                'event_id': int(a.get('actionNumber') or a.get('actionId') or 0),
                'period': int(a.get('period') or 0),
                'clock': a.get('clock'),
                'description': desc,
                'player_id': pid,
                'player_name': a.get('playerName'),
                'team': a.get('teamTricode'),
                'action_type': action_type,
                'sub_type': subtype,
                'shot_distance': a.get('shotDistance'),
                'x': a.get('xLegacy'),
                'y': a.get('yLegacy'),
            })
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    q = json.loads(Path(args.request).read_text(encoding='utf-8'))
    team = str(q.get('team', 'LAL')).upper()
    team_id = int(q.get('team_id', 1610612747))
    player_id = int(q['player_id'])
    season = str(q.get('season', '2025-26'))
    prefix = str(q.get('game_prefix', '00225'))
    action_contains = str(q.get('action_contains', 'dunk')).lower()
    last_n = int(q.get('last_n', 5))

    games, diagnostics = schedule_games_static(team, prefix)
    schedule_source = 'official_nba_static_schedule'
    if not games:
        try:
            games, statdiag = schedule_games_stats(team_id, season, prefix)
            diagnostics.append({'stats_teamgamelog': statdiag})
            schedule_source = 'official_nba_stats_teamgamelog'
        except Exception as e:
            diagnostics.append({'stats_teamgamelog_error': repr(e)})

    if not games:
        print(json.dumps({'schedule_diagnostics': diagnostics}, indent=2), flush=True)
        raise SystemExit('No 2025-26 regular-season team games resolved')

    uniq = {}
    for g in games:
        uniq[g['game_id']] = g
    games = list(uniq.values())

    all_matches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(resolve_game, g, player_id, action_contains): g for g in games}
        for fut in concurrent.futures.as_completed(futs):
            g = futs[fut]
            try:
                all_matches.extend(fut.result())
            except Exception as e:
                diagnostics.append({'game_id': g['game_id'], 'pbp_error': repr(e)})

    def key(e):
        return (str(e.get('game_date') or ''), e['game_id'], int(e['event_id']))
    all_matches.sort(key=key)
    selected = all_matches[-last_n:]
    for i, e in enumerate(selected, 1):
        e['rank'] = i

    payload = {
        'source': 'official NBA liveData play-by-play',
        'schedule_source': schedule_source,
        'query': q,
        'schedule_diagnostics': diagnostics,
        'team_games_resolved': len(games),
        'total_matching_actions': len(all_matches),
        'resolved_count': len(selected),
        'events': selected,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)
    if len(selected) != last_n:
        raise SystemExit(f'Expected {last_n}, got {len(selected)}')


if __name__ == '__main__':
    main()
