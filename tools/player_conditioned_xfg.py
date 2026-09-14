#!/usr/bin/env python3
"""Player-conditioned xFG estimate from a player's own tracked 2025-26 shots.

Definition (default):
- exact target shot resolved by game_id + player_id + event_num from NBA shotqualityvideologs
- comparison pool is the same player's other 2025-26 regular-season tracked shots
- similar distance: +/- 1.0 ft
- similar league-average xFG: +/- 2.5 percentage points
- player-conditioned xFG = empirical FG% of that comparison pool

The target shot is excluded from the comparison pool (leave-one-out). Missing xFG
is never imputed. The tool emits the sample rows used so every displayed value is
auditable and reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
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


def sess() -> requests.Session:
    s = getattr(THREAD_LOCAL, 'session', None)
    if s is None:
        s = requests.Session(); s.headers.update(HEADERS); THREAD_LOCAL.session = s
    return s


def fetch_xfg_pair(game_id: str, player_id: int, attempts: int = 5) -> tuple[dict | None, dict]:
    last: dict[str, Any] = {}
    for n in range(1, attempts + 1):
        started = time.time()
        try:
            r = sess().get(XFG_URL, params={'GameID': game_id, 'PlayerID': int(player_id)}, timeout=(8, 35))
            elapsed = time.time() - started
            if r.status_code == 200:
                j = r.json()
                if (isinstance(j, dict)
                    and str(j.get('gameId') or '').zfill(10) == game_id
                    and int(j.get('playerId') or 0) == int(player_id)):
                    return j, {'status': 200, 'attempts': n, 'elapsed_s': elapsed}
                last = {'status': 200, 'attempts': n, 'elapsed_s': elapsed, 'error': 'unexpected_payload'}
            else:
                last = {'status': r.status_code, 'attempts': n, 'elapsed_s': elapsed, 'body': r.text[:160]}
        except Exception as e:
            last = {'status': None, 'attempts': n, 'elapsed_s': time.time()-started, 'error': repr(e)}
        if n < attempts:
            time.sleep(min(6.0, 0.7 * (2 ** (n - 1))))
    return None, last


def numeric(v: Any) -> float | None:
    try:
        x = float(v)
        if math.isfinite(x): return x
    except Exception:
        pass
    return None


def event_num_from_shot(s: dict) -> int | None:
    for k in ('eventNum','event_num','eventNumber','eventId','eventID','actionNumber'):
        if k in s:
            try: return int(float(s[k]))
            except Exception: pass
    return None


def xfg_pct_from_shot(s: dict) -> float | None:
    q = numeric(s.get('shotQuality'))
    if q is None: return None
    return 100.0*q if q <= 1.5 else q


def direct_distance_ft(s: dict) -> tuple[float | None, str | None]:
    for k in ('shotDistance','shotDistanceFeet','shot_distance','shotDist','distance'):
        if k not in s: continue
        x = numeric(s.get(k))
        if x is None: continue
        # Avoid treating coordinate-like values as feet.
        if 0 <= x <= 45: return x, k
    # NBA shot-chart locX/locY convention is tenths of feet from basket centre.
    x, y = numeric(s.get('locX')), numeric(s.get('locY'))
    if x is not None and y is not None and abs(x) <= 400 and abs(y) <= 500:
        d = math.hypot(x, y) / 10.0
        if 0 <= d <= 45: return d, 'derived_hypot_locX_locY_div10'
    return None, None


def parse_distance_from_text(row: pd.Series) -> tuple[float | None, str | None]:
    preferred = [
        'description','text','event_description','desc','home_description','away_description',
        'description_home','description_away','home_desc','away_desc'
    ]
    cols = preferred + [c for c in row.index if 'desc' in c.lower() or 'text' in c.lower()]
    seen = set()
    for c in cols:
        if c in seen or c not in row.index: continue
        seen.add(c)
        txt = str(row.get(c) or '')
        m = re.search(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*['’]", txt)
        if not m: m = re.search(r'(?<!\d)(\d{1,2}(?:\.\d+)?)\s*(?:ft|feet)\b', txt, re.I)
        if m:
            x = float(m.group(1))
            if 0 <= x <= 45: return x, f'pbp_text:{c}'
    return None, None


def load_pbp(cache: Path) -> pd.DataFrame:
    if not cache.exists():
        r = requests.get(PBP_URL, timeout=180); r.raise_for_status(); cache.write_bytes(r.content)
    df = next(iter(pyreadr.read_r(str(cache)).values()))
    df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    if 'event_num' in df.columns:
        df['event_num'] = pd.to_numeric(df['event_num'], errors='coerce').astype('Int64')
    return df


def pbp_player_fga(df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    p = df['player1_name'].astype(str).str.extract(r'^\s*(\d+)\s+(.*)$')
    pid = pd.to_numeric(p[0], errors='coerce')
    is_fg = pd.to_numeric(df['is_field_goal'], errors='coerce').fillna(0).eq(1)
    reg = df['game_id'].str.startswith('002')
    out = df.loc[is_fg & reg & pid.eq(int(player_id))].copy()
    out['player_id'] = int(player_id)
    out['player_name_parsed'] = p.loc[out.index,1].values
    return out.sort_values(['game_id','event_num'])


def normalize_shots(payload: dict, pbp_game: pd.DataFrame, game_id: str, player_id: int) -> list[dict]:
    shot_list = payload.get('shotList') or []
    pbp_rows = list(pbp_game.sort_values('event_num').iterrows())
    can_order_map = len(shot_list) == len(pbp_rows) and len(shot_list) > 0
    rows = []
    for i, s in enumerate(shot_list):
        ev = event_num_from_shot(s)
        match_method = 'shot_object_event'
        if ev is None and can_order_map:
            ev = int(pbp_rows[i][1]['event_num'])
            match_method = 'chronological_order_exact_count'
        pbp_row = None
        if ev is not None:
            h = pbp_game[pbp_game['event_num'].eq(ev)]
            if len(h) == 1: pbp_row = h.iloc[0]
        d, dsrc = direct_distance_ft(s)
        if d is None and pbp_row is not None:
            # Structured PBP distance aliases first.
            for k in ('shot_distance','shot_distance_ft','shot_dist','distance'):
                if k in pbp_row.index:
                    x = numeric(pbp_row.get(k))
                    if x is not None and 0 <= x <= 45:
                        d, dsrc = x, f'pbp:{k}'; break
            if d is None:
                d, dsrc = parse_distance_from_text(pbp_row)
        xpct = xfg_pct_from_shot(s)
        if xpct is None: continue
        rows.append({
            'game_id': game_id,
            'player_id': int(player_id),
            'event_num': ev,
            'event_match_method': match_method,
            'xfg_pct': xpct,
            'shot_distance_ft': d,
            'distance_source': dsrc,
            'made': int(s.get('success') or 0),
            'shot_type': s.get('shotType'),
            'action_type': s.get('actionType'),
            'period': s.get('period'),
            'game_clock': s.get('gameClock'),
            'loc_x': s.get('locX'),
            'loc_y': s.get('locY'),
            'guid': s.get('guid'),
        })
    return rows


def wilson_interval(made: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0: return None, None
    p = made/n; den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return 100*(center-half), 100*(center+half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--player-id', type=int, required=True)
    ap.add_argument('--game-id', required=True)
    ap.add_argument('--event-num', type=int, required=True)
    ap.add_argument('--distance-window-ft', type=float, default=1.0)
    ap.add_argument('--xfg-window-pp', type=float, default=2.5)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pbp-cache', type=Path, default=Path('artifacts/cache/pbp-final-2026.rds'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.game_id = str(a.game_id).zfill(10)
    a.out.mkdir(parents=True, exist_ok=True); a.pbp_cache.parent.mkdir(parents=True, exist_ok=True)

    pbp = load_pbp(a.pbp_cache)
    fga = pbp_player_fga(pbp, a.player_id)
    if fga.empty: raise RuntimeError(f'No regular-season FGA found for player {a.player_id}')
    player_names = fga['player_name_parsed'].dropna().astype(str)
    player_name = player_names.mode().iloc[0] if not player_names.empty else str(a.player_id)
    games = sorted(fga['game_id'].unique().tolist())

    rows: list[dict] = []
    errors = []
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
        futs = {ex.submit(fetch_xfg_pair,g,a.player_id):(g) for g in games}
        for fut in as_completed(futs):
            g = futs[fut]
            try: payload, meta = fut.result()
            except Exception as e: payload, meta = None, {'error':repr(e)}
            if payload is None:
                errors.append({'game_id':g, **meta}); continue
            pg = fga[fga['game_id'].eq(g)]
            rows.extend(normalize_shots(payload, pg, g, a.player_id))

    shots = pd.DataFrame(rows)
    if shots.empty: raise RuntimeError('No tracked xFG shots returned')
    shots['event_num'] = pd.to_numeric(shots['event_num'], errors='coerce').astype('Int64')
    target = shots[(shots.game_id.eq(a.game_id)) & shots.event_num.eq(a.event_num)]
    if len(target) != 1:
        # Preserve a schema diagnostic so failures are actionable.
        (a.out/'all_tracked_shots.csv').write_text(shots.to_csv(index=False))
        raise RuntimeError(f'Expected exactly one target shot {a.game_id}/{a.event_num}; found {len(target)}')
    tr = target.iloc[0]
    if pd.isna(tr['shot_distance_ft']):
        raise RuntimeError('Target shot has no reproducible shot distance')

    td = float(tr['shot_distance_ft']); tx = float(tr['xfg_pct'])
    comparable = shots[
        shots['shot_distance_ft'].notna()
        & shots['xfg_pct'].notna()
        & (shots['shot_distance_ft'].sub(td).abs() <= a.distance_window_ft + 1e-9)
        & (shots['xfg_pct'].sub(tx).abs() <= a.xfg_window_pp + 1e-9)
        & ~((shots['game_id'].eq(a.game_id)) & shots['event_num'].eq(a.event_num))
    ].copy().sort_values(['game_id','event_num'])
    n = len(comparable); made = int(comparable['made'].sum()) if n else 0
    player_xfg = 100.0*made/n if n else None
    lo, hi = wilson_interval(made,n)

    result = {
        'definition_version': 'player_conditioned_xfg_v1',
        'season': '2025-26', 'season_type': 'Regular Season',
        'player_id': int(a.player_id), 'player_name': player_name,
        'target': {
            'game_id': a.game_id, 'event_num': int(a.event_num),
            'league_xfg_pct': tx, 'shot_distance_ft': td,
            'made': int(tr['made']), 'shot_type': tr.get('shot_type'), 'action_type': tr.get('action_type'),
            'distance_source': tr.get('distance_source'), 'event_match_method': tr.get('event_match_method'),
        },
        'comparison_rule': {
            'same_player': True,
            'tracked_shots_only': True,
            'distance_window_ft': float(a.distance_window_ft),
            'xfg_window_pp': float(a.xfg_window_pp),
            'target_shot_excluded': True,
        },
        'player_conditioned_xfg_pct': player_xfg,
        'sample_fga': int(n), 'sample_fgm': int(made),
        'wilson_95_pct': [lo,hi],
        'tracked_season_shots_with_xfg': int(len(shots)),
        'tracked_season_shots_with_distance': int(shots['shot_distance_ft'].notna().sum()),
        'games_requested': int(len(games)), 'request_errors': int(len(errors)),
        'source': XFG_URL,
        'note': 'Empirical make rate on the player\'s other tracked season shots inside the fixed distance/xFG windows; no missing xFG imputed.',
    }
    shots.to_csv(a.out/'all_tracked_shots.csv', index=False)
    comparable.to_csv(a.out/'comparison_sample.csv', index=False)
    pd.DataFrame(errors).to_csv(a.out/'request_errors.csv', index=False)
    (a.out/'player_conditioned_xfg.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if n:
        print('\nCOMPARISON_SAMPLE')
        print(comparable[['game_id','event_num','shot_distance_ft','xfg_pct','made','shot_type','action_type']].to_string(index=False))

if __name__ == '__main__':
    main()
