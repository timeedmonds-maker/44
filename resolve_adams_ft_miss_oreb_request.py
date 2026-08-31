from __future__ import annotations

import concurrent.futures
import json
import urllib.request
from pathlib import Path

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36'
ADAMS_ID = 203500
HOU = 'HOU'
GAME_PREFIX = '00225'
STATIC_SCHEDULES = [
    'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json',
    'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json',
]


def get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={
        'User-Agent': UA,
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def walk_games(obj):
    if isinstance(obj, dict):
        if 'gameId' in obj and ('homeTeam' in obj or 'awayTeam' in obj):
            yield obj
        for value in obj.values():
            yield from walk_games(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_games(value)


def tricode(team) -> str:
    if not isinstance(team, dict):
        return ''
    for key in ('teamTricode', 'teamCode', 'tricode'):
        if team.get(key):
            return str(team[key]).upper()
    return ''


def schedule_games() -> tuple[list[dict], list[dict]]:
    diagnostics = []
    for url in STATIC_SCHEDULES:
        try:
            data = get_json(url)
            games = list(walk_games(data))
            selected = []
            for game in games:
                gid = str(game.get('gameId') or '')
                home = tricode(game.get('homeTeam'))
                away = tricode(game.get('awayTeam'))
                if gid.startswith(GAME_PREFIX) and HOU in {home, away}:
                    selected.append({
                        'game_id': gid,
                        'game_date': game.get('gameDate') or game.get('gameDateEst') or game.get('gameDateTimeUTC') or game.get('gameDateTimeEst') or '',
                        'home': home,
                        'away': away,
                    })
            diagnostics.append({'url': url, 'games_total': len(games), 'hou_2025_26_regular_season_games': len(selected)})
            if selected:
                unique = {g['game_id']: g for g in selected}
                return list(unique.values()), diagnostics
        except Exception as exc:
            diagnostics.append({'url': url, 'error': repr(exc)})
    return [], diagnostics


def person_id(action: dict) -> int:
    try:
        return int(action.get('personId') or 0)
    except Exception:
        return 0


def text(action: dict, key: str) -> str:
    return str(action.get(key) or '').strip()


def is_missed_ft(action: dict) -> bool:
    action_type = text(action, 'actionType').lower().replace(' ', '')
    subtype = text(action, 'subType').lower().replace(' ', '')
    desc = text(action, 'description').lower()
    result = text(action, 'shotResult').lower()
    is_ft = 'freethrow' in action_type or 'freethrow' in subtype or 'free throw' in desc
    missed = result in {'miss', 'missed'} or 'miss' in desc
    return is_ft and missed


def is_adams_offensive_rebound(action: dict) -> bool:
    action_type = text(action, 'actionType').lower()
    subtype = text(action, 'subType').lower()
    desc = text(action, 'description').lower()
    team = text(action, 'teamTricode').upper()
    is_rebound = 'rebound' in action_type or 'rebound' in subtype or 'rebound' in desc
    offensive = 'offensive' in subtype or 'offensive rebound' in desc or '(off:' in desc
    return is_rebound and offensive and person_id(action) == ADAMS_ID and team == HOU


def action_number(action: dict) -> int:
    try:
        return int(action.get('actionNumber') or action.get('actionId') or 0)
    except Exception:
        return 0


def resolve_game(game: dict) -> dict:
    gid = game['game_id']
    url = f'https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json'
    data = get_json(url)
    actions = ((data.get('game') or {}).get('actions') or [])
    events = []
    missed_ft_count = 0

    # LiveData actions are consumed in their delivered chronological sequence.
    # "Immediately after" is strict: the very next action must be Adams' OREB.
    for index in range(len(actions) - 1):
        ft = actions[index]
        rb = actions[index + 1]
        if not is_missed_ft(ft):
            continue
        if text(ft, 'teamTricode').upper() != HOU:
            continue
        if person_id(ft) == ADAMS_ID:
            continue
        missed_ft_count += 1
        if int(ft.get('period') or 0) != int(rb.get('period') or 0):
            continue
        if not is_adams_offensive_rebound(rb):
            continue

        shooter_id = person_id(ft)
        events.append({
            'game_id': gid,
            'event_id': action_number(rb),
            'ft_event_id': action_number(ft),
            'game_date': game.get('game_date', ''),
            'matchup': f"{game.get('away', '')} @ {game.get('home', '')}",
            'period': int(rb.get('period') or 0),
            'clock': rb.get('clock') or ft.get('clock') or '',
            'ft_shooter_id': shooter_id,
            'ft_shooter': ft.get('playerName') or ft.get('playerNameI') or '',
            'ft_description': ft.get('description') or '',
            'rebounder_id': ADAMS_ID,
            'rebounder': rb.get('playerName') or rb.get('playerNameI') or 'Steven Adams',
            'rebound_description': rb.get('description') or '',
            'rebound_subtype': rb.get('subType') or '',
            'source_action_index_ft': index,
            'source_action_index_rebound': index + 1,
            'source': 'official NBA liveData play-by-play',
        })

    return {
        'game_id': gid,
        'missed_teammate_fts': missed_ft_count,
        'qualifying_events': events,
        'action_count': len(actions),
    }


def main() -> None:
    games, schedule_diagnostics = schedule_games()
    if not games:
        print(json.dumps({'schedule_diagnostics': schedule_diagnostics}, indent=2), flush=True)
        raise SystemExit('No Houston 2025-26 regular-season games resolved from official NBA schedule')

    diagnostics = []
    events = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(resolve_game, game): game for game in games}
        for future in concurrent.futures.as_completed(futures):
            game = futures[future]
            try:
                result = future.result()
                diagnostics.append({
                    'game_id': result['game_id'],
                    'missed_teammate_fts': result['missed_teammate_fts'],
                    'qualifying_count': len(result['qualifying_events']),
                    'action_count': result['action_count'],
                })
                events.extend(result['qualifying_events'])
            except Exception as exc:
                diagnostics.append({'game_id': game['game_id'], 'error': repr(exc)})

    events.sort(key=lambda event: (str(event.get('game_date') or ''), event['game_id'], event['source_action_index_rebound']))
    for rank, event in enumerate(events, 1):
        event['rank'] = rank

    payload = {
        'label': 'Steven Adams 2025-26 offensive rebounds immediately after teammate missed free throw - all camera angles UHD',
        'definition': '2025-26 regular season; Houston teammate (non-Adams) misses a free throw; the immediately following official NBA LiveData action is an offensive rebound credited to Steven Adams for Houston',
        'source': 'official NBA liveData play-by-play (cdn.nba.com)',
        'video_anchor': 'Steven Adams offensive rebound actionNumber',
        'expected_count': len(events),
        'workers': 8,
        'schedule_diagnostics': schedule_diagnostics,
        'game_diagnostics': sorted(diagnostics, key=lambda d: d['game_id']),
        'events': events,
    }
    Path('request.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'HOU_GAMES={len(games)} QUALIFYING_EVENTS={len(events)}', flush=True)
    for event in events:
        print(json.dumps(event, ensure_ascii=False), flush=True)
    if not events:
        raise SystemExit('No qualifying Adams FT-miss OREB events resolved from official NBA LiveData')


if __name__ == '__main__':
    main()
