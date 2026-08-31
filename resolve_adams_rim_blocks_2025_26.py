from __future__ import annotations

import csv
import json
import math
import urllib.request
from datetime import datetime
from pathlib import Path

PBP_URL = 'https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
ADAMS = '203500 Steven Adams'
ADAMS_ID = 203500
RIM_AREA = 'Restricted Area'
UA = 'Mozilla/5.0'
OUT = Path('request.json')
ALL_OUT = Path('adams_all_blocks_2025_26.json')
RIM_OUT = Path('adams_rim_blocks_2025_26.json')


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def inum(v):
    try:
        return int(float(v))
    except Exception:
        return None


def main() -> None:
    req = urllib.request.Request(PBP_URL, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=180) as r, open('pbp.csv', 'wb') as f:
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            f.write(b)

    rows = []
    with open('pbp.csv', newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        source_rows = list(reader)

    for i, row in enumerate(source_rows):
        gid = (row.get('game_id') or '').strip()
        if not gid.startswith('002'):
            continue
        if (row.get('msg_type') or '').strip() != '2':
            continue
        if (row.get('is_field_goal') or '').strip() != '1':
            continue
        if (row.get('shot_result') or '').strip().lower() != 'missed':
            continue
        if (row.get('player2_name') or '').strip() != ADAMS:
            continue
        desc = (row.get('description') or '').strip()
        if '- blocked' not in desc.lower():
            continue
        eid = inum(row.get('event_num'))
        if eid is None:
            continue
        dist = fnum(row.get('shot_distance'))
        area = (row.get('area') or '').strip()
        block_event_id = None
        block_desc = None
        if i + 1 < len(source_rows):
            nxt = source_rows[i + 1]
            same = (
                (nxt.get('game_id') or '').strip() == gid
                and (nxt.get('period') or '').strip() == (row.get('period') or '').strip()
                and (nxt.get('clock') or '').strip() == (row.get('clock') or '').strip()
            )
            if same and (nxt.get('msg_type') or '').strip() == '92' and (nxt.get('player1_name') or '').strip() == ADAMS:
                block_event_id = inum(nxt.get('event_num'))
                block_desc = (nxt.get('description') or '').strip()

        rec = {
            'game_date': (row.get('game_date') or '').strip(),
            'game_id': gid.zfill(10),
            'event_id': eid,
            'blocked_shot_event_id': eid,
            'block_stat_event_id': block_event_id,
            'period': inum(row.get('period')),
            'clock': (row.get('clock') or '').strip(),
            'team_home': (row.get('team_home') or '').strip(),
            'team_away': (row.get('team_away') or '').strip(),
            'shooter': (row.get('player1_name') or '').strip(),
            'blocker': ADAMS,
            'blocker_id': ADAMS_ID,
            'description': desc,
            'block_description': block_desc,
            'action_type': (row.get('action_type') or '').strip(),
            'sub_type': (row.get('sub_type') or '').strip(),
            'descriptor': (row.get('descriptor') or '').strip(),
            'shot_distance_ft': dist,
            'locX': fnum(row.get('locX')),
            'locY': fnum(row.get('locY')),
            'area': area,
            'area_detail': (row.get('area_detail') or '').strip(),
            'at_rim': area.lower() == RIM_AREA.lower(),
            'rim_definition': 'NBA PBP shot area = Restricted Area',
            'video_anchor': 'blocked shot event_id',
        }
        rows.append(rec)

    if not rows:
        raise SystemExit('No Steven Adams blocks found')
    missing_stat = [r for r in rows if r['block_stat_event_id'] is None]
    if missing_stat:
        raise SystemExit(f'Block-stat pairing QA failed for {len(missing_stat)} rows: {missing_stat}')
    keys = [(r['game_id'], r['event_id']) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit('Duplicate blocked-shot event keys found')

    def dtkey(r):
        try:
            return datetime.strptime(r['game_date'], '%Y-%m-%d')
        except Exception:
            return datetime.max

    rows.sort(key=lambda r: (dtkey(r), r['game_id'], r['event_id']))
    for i, r in enumerate(rows, 1):
        r['all_block_rank'] = i

    rim = [r.copy() for r in rows if r['at_rim']]
    for i, r in enumerate(rim, 1):
        r['rank'] = i

    if not rim:
        raise SystemExit('No Adams blocks classified by NBA PBP as Restricted Area')

    all_payload = {
        'source': PBP_URL,
        'player': 'Steven Adams',
        'player_id': ADAMS_ID,
        'season': '2025-26',
        'filter': 'blocked FGA row: msg_type=2, is_field_goal=1, shot_result=Missed, player2_name=203500 Steven Adams',
        'all_block_count': len(rows),
        'blocks': rows,
    }
    rim_payload = {
        'source': PBP_URL,
        'player': 'Steven Adams',
        'player_id': ADAMS_ID,
        'season': '2025-26',
        'rim_definition': 'NBA PBP shot area = Restricted Area',
        'video_anchor': 'blocked shot event_num',
        'expected_count': len(rim),
        'events': rim,
    }
    ALL_OUT.write_text(json.dumps(all_payload, indent=2, ensure_ascii=False))
    RIM_OUT.write_text(json.dumps(rim_payload, indent=2, ensure_ascii=False))
    OUT.write_text(json.dumps(rim_payload, indent=2, ensure_ascii=False))

    print(f'ALL_ADAMS_BLOCKS={len(rows)} RIM_BLOCKS={len(rim)}')
    print('AREA_VALUES=' + json.dumps(sorted({r['area'] for r in rows})))
    for r in rim:
        print(f"R{r['rank']:02d} {r['game_date']} {r['team_away']} @ {r['team_home']} {r['game_id']}/{r['event_id']} P{r['period']} {r['clock']} {r['shot_distance_ft']}ft {r['area']} :: {r['description']}")


if __name__ == '__main__':
    main()
