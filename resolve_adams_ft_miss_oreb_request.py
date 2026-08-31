from __future__ import annotations

import concurrent.futures
import csv
import io
import json
import re
import urllib.request
from pathlib import Path

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36'
ADAMS_ID = 203500
HOU = 'HOU'
PBP_INDEX_URL = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.csv'


def get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={
        'User-Agent': UA,
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.nba.com/',
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def norm_gid(value) -> str:
    s = str(value or '').strip()
    try:
        s = str(int(float(s)))
    except Exception:
        s = re.sub(r'\D', '', s)
    return s.zfill(10)


def source_games() -> list[dict]:
    req = urllib.request.Request(PBP_INDEX_URL, headers={'User-Agent': UA})
    games = {}
    with urllib.request.urlopen(req, timeout=180) as response:
        text_stream = io.TextIOWrapper(response, encoding='utf-8-sig', newline='')
        reader = csv.DictReader(text_stream)
        for row in reader:
            gid = norm_gid(row.get('game_id'))
            if not gid.startswith('002'):
                continue
            team = str(row.get('team_abb') or '').strip().upper()
            home = str(row.get('team_home') or '').strip().upper()
            away = str(row.get('team_away') or '').strip().upper()
            if HOU not in {team, home, away}:
                continue
            rec = games.setdefault(gid, {
                'game_id': gid,
                'game_date': str(row.get('game_date') or row.get('date') or '').strip(),
                'home': home,
                'away': away,
            })
            if not rec['game_date']:
                rec['game_date'] = str(row.get('game_date') or row.get('date') or '').strip()
            if not rec['home']:
                rec['home'] = home
            if not rec['away']:
                rec['away'] = away
    return list(games.values())


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
    # LiveData rebound rows identify the player and expose offensive status either
    # in subType or in the running Off/Def rebound totals in the description.
    offensive = 'offensive' in subtype or 'offensive rebound' in desc
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
    adams_rebound_samples = []

    for action in actions:
        if person_id(action) == ADAMS_ID and 'rebound' in (' '.join([
            text(action, 'actionType'), text(action, 'subType'), text(action, 'description')
        ])).lower():
            if len(adams_rebound_samples) < 5:
                adams_rebound_samples.append({
                    'actionNumber': action_number(action),
                    'actionType': action.get('actionType'),
                    'subType': action.get('subType'),
                    'description': action.get('description'),
                    'teamTricode': action.get('teamTricode'),
                    'period': action.get('period'),
                    'clock': action.get('clock'),
                })

    # Official LiveData list order is the chronological event chain. "Immediately
    # after" is strict: the next action itself must be Adams' offensive rebound.
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

        events.append({
            'game_id': gid,
            'event_id': action_number(rb),
            'ft_event_id': action_number(ft),
            'game_date': game.get('game_date', ''),
            'matchup': f"{game.get('away', '')} @ {game.get('home', '')}",
            'period': int(rb.get('period') or 0),
            'clock': rb.get('clock') or ft.get('clock') or '',
            'ft_shooter_id': person_id(ft),
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
        'adams_rebound_samples': adams_rebound_samples,
    }


def main() -> None:
    games = source_games()
    if not games:
        raise SystemExit('No Houston 2025-26 regular-season games resolved from PBP game index')

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
                    'adams_rebound_samples': result['adams_rebound_samples'],
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
        'game_index_source': 'ramirobentes/nba_pbp_data pbp-final-2026/data.csv (game IDs only)',
        'event_source': 'official NBA liveData play-by-play (cdn.nba.com)',
        'video_anchor': 'Steven Adams offensive rebound actionNumber',
        'expected_count': len(events),
        'workers': 8,
        'hou_games_resolved': len(games),
        'game_diagnostics': sorted(diagnostics, key=lambda d: d['game_id']),
        'events': events,
    }
    Path('request.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'HOU_GAMES={len(games)} QUALIFYING_EVENTS={len(events)}', flush=True)
    for event in events:
        print(json.dumps(event, ensure_ascii=False), flush=True)
    if not events:
        print(json.dumps({'diagnostics': payload['game_diagnostics']}, indent=2), flush=True)
        raise SystemExit('No qualifying Adams FT-miss OREB events resolved from official NBA LiveData')


if __name__ == '__main__':
    main()
