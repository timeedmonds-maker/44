from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from pathlib import Path

PBP_URL = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.csv'
UA = 'timeedmonds-maker-44-adams-ft-miss-oreb'
ADAMS_ID = '203500'
ADAMS_NAME = 'STEVEN ADAMS'
HOU = 'HOU'
ACTOR_RE = re.compile(r'^\s*(\d+)\s+(.+?)\s*$')


def scalar(v) -> str:
    return '' if v is None else str(v).strip()


def integer(v):
    try:
        return int(float(v))
    except Exception:
        return None


def actor(v) -> tuple[str, str]:
    s = scalar(v)
    m = ACTOR_RE.match(s)
    return (m.group(1), m.group(2).strip()) if m else ('', s)


def is_adams(v) -> bool:
    pid, name = actor(v)
    return pid == ADAMS_ID or ADAMS_NAME in name.upper()


def norm_gid(v) -> str:
    s = scalar(v)
    try:
        s = str(int(float(s)))
    except Exception:
        s = re.sub(r'\D', '', s)
    return s.zfill(10)


def main() -> None:
    request = urllib.request.Request(PBP_URL, headers={'User-Agent': UA})
    rows = []
    with urllib.request.urlopen(request, timeout=180) as response:
        text = io.TextIOWrapper(response, encoding='utf-8-sig', newline='')
        reader = csv.DictReader(text)
        fields = reader.fieldnames or []
        event_col = next((c for c in ('event_num', 'event_id', 'eventnum', 'event_no', 'event_number') if c in fields), None)
        if not event_col:
            raise SystemExit(f'No event identifier column. fields={fields}')
        for source_row, row in enumerate(reader):
            gid = norm_gid(row.get('game_id'))
            if not gid.startswith('002'):
                continue
            rows.append({
                'source_row': source_row,
                'game_id': gid,
                'event_id': integer(row.get(event_col)),
                'period': integer(row.get('period')),
                'msg_type': integer(row.get('msg_type')),
                'team': scalar(row.get('team_abb')).upper(),
                'player1_name': scalar(row.get('player1_name')),
                'description': scalar(row.get('description')),
                'clock': scalar(row.get('clock')),
                'game_date': scalar(row.get('game_date') or row.get('date')),
                'team_home': scalar(row.get('team_home')),
                'team_away': scalar(row.get('team_away')),
            })

    rows.sort(key=lambda x: (x['game_id'], x['event_id'] if x['event_id'] is not None else 10**9, x['source_row']))

    events = []
    for i in range(len(rows) - 1):
        ft = rows[i]
        rb = rows[i + 1]
        if ft['msg_type'] != 3:
            continue
        if 'MISS' not in ft['description'].upper():
            continue
        if ft['team'] != HOU:
            continue
        if is_adams(ft['player1_name']):
            continue
        if rb['game_id'] != ft['game_id'] or rb['period'] != ft['period']:
            continue
        if rb['msg_type'] != 4:
            continue
        if not is_adams(rb['player1_name']):
            continue
        if rb['team'] != ft['team']:
            continue

        shooter_id, shooter_name = actor(ft['player1_name'])
        rebounder_id, rebounder_name = actor(rb['player1_name'])
        events.append({
            'rank': len(events) + 1,
            'game_id': rb['game_id'],
            'event_id': rb['event_id'],
            'ft_event_id': ft['event_id'],
            'game_date': rb['game_date'],
            'matchup': f"{rb['team_away']} @ {rb['team_home']}" if rb['team_away'] and rb['team_home'] else '',
            'period': rb['period'],
            'clock': rb['clock'],
            'ft_shooter_id': shooter_id,
            'ft_shooter': shooter_name,
            'ft_description': ft['description'],
            'rebounder_id': rebounder_id or ADAMS_ID,
            'rebounder': rebounder_name,
            'rebound_description': rb['description'],
        })

    if not events:
        adams_rows = [r for r in rows if is_adams(r['player1_name'])]
        hou_fts = [r for r in rows if r['msg_type'] == 3 and r['team'] == HOU]
        print(json.dumps({
            'diagnostic': 'no qualifying events',
            'adams_player1_rows': len(adams_rows),
            'hou_ft_rows': len(hou_fts),
            'adams_examples': adams_rows[:10],
            'hou_ft_examples': hou_fts[:10],
        }, indent=2), flush=True)
        raise SystemExit('No qualifying events resolved')

    payload = {
        'label': 'Steven Adams 2025-26 OREB immediately after teammate missed FT - all camera angles UHD',
        'definition': '2025-26 regular season; HOU missed FT by non-Adams teammate; the very next ordered PBP row is a rebound credited to Steven Adams for HOU in the same game and period',
        'source': 'ramirobentes/nba_pbp_data pbp-final-2026/data.csv',
        'expected_count': len(events),
        'workers': 8,
        'events': events,
    }
    Path('request.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'QUALIFYING_EVENTS={len(events)}', flush=True)
    for event in events:
        print(json.dumps(event, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
