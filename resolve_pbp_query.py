from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from collections import deque
from pathlib import Path

DEFAULT_URL = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.csv'
UA = 'Mozilla/5.0'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    req = json.loads(Path(args.request).read_text(encoding='utf-8'))
    source_url = req.get('source_csv_url', DEFAULT_URL)
    player_id = str(req.get('player_id', '')).strip()
    player_name_contains = str(req.get('player_name_contains', '')).strip().lower()
    team_filter = str(req.get('team', '')).strip().upper()
    description_contains = str(req.get('description_contains', '')).strip().lower()
    last_n = int(req.get('last_n', 5))
    require_made_fg = bool(req.get('require_made_fg', True))

    if not player_id and not player_name_contains and not team_filter:
        raise SystemExit('Provide player_id, player_name_contains, or team')

    matched = deque(maxlen=last_n)
    total_matches = 0
    identity_examples = []

    request = urllib.request.Request(source_url, headers={'User-Agent': UA})
    with urllib.request.urlopen(request, timeout=120) as resp:
        text = io.TextIOWrapper(resp, encoding='utf-8-sig', newline='')
        reader = csv.DictReader(text)
        for row in reader:
            p1 = row.get('player1_name') or ''
            desc = row.get('description') or ''
            p1_low = p1.lower()

            if team_filter:
                teams = {str(row.get('team_abb') or '').upper(), str(row.get('team_home') or '').upper(), str(row.get('team_away') or '').upper()}
                if team_filter not in teams:
                    continue

            if player_id or player_name_contains:
                id_ok = bool(player_id) and (
                    p1.startswith(player_id + ' ') or
                    p1 == player_id or
                    (' ' + player_id + ' ') in (' ' + p1 + ' ')
                )
                name_ok = bool(player_name_contains) and player_name_contains in p1_low
                if not (id_ok or name_ok):
                    continue

            if len(identity_examples) < 20 and p1 and p1 not in identity_examples:
                identity_examples.append(p1)

            if description_contains and description_contains not in desc.lower():
                continue
            if require_made_fg and row.get('msg_type') not in {'1', '1.0'}:
                continue
            try:
                shot_pts = int(float(row.get('shot_pts') or 0))
            except ValueError:
                shot_pts = 0
            if require_made_fg and shot_pts <= 0:
                continue

            total_matches += 1
            matched.append({
                'game_date': row.get('game_date'),
                'game_id': row.get('game_id'),
                'event_id': int(float(row.get('event_num') or 0)),
                'period': int(float(row.get('period') or 0)),
                'clock': row.get('clock'),
                'description': desc,
                'player1_name': p1,
                'team': row.get('team_abb'),
                'team_home': row.get('team_home'),
                'team_away': row.get('team_away'),
                'shot_pts': shot_pts,
            })

    events = list(matched)
    for idx, e in enumerate(events, 1):
        e['rank'] = idx

    payload = {
        'source': source_url,
        'query': req,
        'identity_examples': identity_examples,
        'total_matches': total_matches,
        'resolved_count': len(events),
        'events': events,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)
    if len(events) != last_n:
        raise SystemExit(f'Expected {last_n} resolved events, got {len(events)}')


if __name__ == '__main__':
    main()
