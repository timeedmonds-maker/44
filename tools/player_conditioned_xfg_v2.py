#!/usr/bin/env python3
"""Player-conditioned xFG using official shot-level distance as source of truth.

Definition:
- target shot resolved exactly by game_id + player_id + event_num from NBA shotqualityvideologs
- xFG comes from shotqualityvideologs shotQuality
- shot distance comes ONLY from official NBA shotchartdetail SHOT_DISTANCE,
  joined by game_id + event_num + player_id
- comparison pool is the same player's other tracked 2025-26 regular-season shots
- similar distance: +/- 1.0 ft (default)
- similar league xFG: +/- 2.5 percentage points (default)
- player-conditioned xFG = empirical FG% of the comparison pool
- target shot excluded leave-one-out

No coordinate-derived shot distance and no imputation are permitted in v2.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

XFG_URL = 'https://stats.gleague.nba.com/stats/shotqualityvideologs'
SHOTCHART_URL = 'https://stats.nba.com/stats/shotchartdetail'
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
        s = requests.Session()
        s.headers.update(HEADERS)
        THREAD_LOCAL.session = s
    return s


def numeric(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def event_num_from_shot(s: dict) -> int | None:
    for k in ('eventNum','event_num','eventNumber','eventId','eventID','actionNumber'):
        if k in s:
            try:
                return int(float(s[k]))
            except Exception:
                pass
    return None


def xfg_pct_from_shot(s: dict) -> float | None:
    q = numeric(s.get('shotQuality'))
    if q is None:
        return None
    return 100.0*q if q <= 1.5 else q


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


def fetch_official_shotchart(player_id: int, season: str = '2025-26', season_type: str = 'Regular Season', attempts: int = 5) -> list[dict]:
    params = {
        'AheadBehind':'','CFID':'','CFPARAMS':'','ClutchTime':'','ContextFilter':'','ContextMeasure':'FGA',
        'DateFrom':'','DateTo':'','EndPeriod':'10','EndRange':'28800','GameID':'','GameSegment':'',
        'LastNGames':'0','LeagueID':'00','Location':'','Month':'0','OpponentTeamID':'0','Outcome':'',
        'Period':'0','PlayerID':str(int(player_id)),'PlayerPosition':'','PointDiff':'','Position':'',
        'RangeType':'0','RookieYear':'','Season':season,'SeasonSegment':'','SeasonType':season_type,
        'StartPeriod':'1','StartRange':'0','TeamID':'0','VsConference':'','VsDivision':''
    }
    last = None
    for n in range(1, attempts+1):
        try:
            r = sess().get(SHOTCHART_URL, params=params, timeout=(8,45))
            if r.status_code == 200:
                j = r.json()
                sets = j.get('resultSets') or j.get('resultSet') or []
                if isinstance(sets, dict): sets = [sets]
                for rs in sets:
                    h = rs.get('headers') or []
                    if 'GAME_ID' in h and 'GAME_EVENT_ID' in h and 'SHOT_DISTANCE' in h:
                        return [dict(zip(h,row)) for row in rs.get('rowSet',[])]
                last = 'shot result set not found'
            else:
                last = f'HTTP {r.status_code}: {r.text[:160]}'
        except Exception as e:
            last = repr(e)
        if n < attempts:
            time.sleep(min(8.0, 0.9*(2**(n-1))))
    raise RuntimeError(f'official shotchart fetch failed: {last}')


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
    ap.add_argument('--season', default='2025-26')
    ap.add_argument('--season-type', default='Regular Season')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.game_id = str(a.game_id).zfill(10)
    a.out.mkdir(parents=True, exist_ok=True)

    # Authoritative shot-level distance universe.
    chart = pd.DataFrame(fetch_official_shotchart(a.player_id, a.season, a.season_type))
    if chart.empty:
        raise RuntimeError('Official shotchart returned no shots')
    chart['game_id'] = chart['GAME_ID'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    chart['event_num'] = pd.to_numeric(chart['GAME_EVENT_ID'], errors='coerce').astype('Int64')
    chart['shot_distance_ft'] = pd.to_numeric(chart['SHOT_DISTANCE'], errors='coerce')
    chart = chart[chart['game_id'].str.startswith('002')].copy()
    chart = chart.dropna(subset=['event_num'])
    dup = chart.duplicated(['game_id','event_num'], keep=False)
    if dup.any():
        d = chart.loc[dup,['game_id','event_num','SHOT_DISTANCE']]
        raise RuntimeError(f'Non-unique official shotchart keys: {d.head(10).to_dict("records")}')
    distance_lookup = {(r.game_id,int(r.event_num)): float(r.shot_distance_ft)
                       for r in chart.itertuples(index=False) if pd.notna(r.shot_distance_ft)}
    games = sorted(chart['game_id'].unique().tolist())
    player_name = str(chart['PLAYER_NAME'].dropna().mode().iloc[0]) if 'PLAYER_NAME' in chart and not chart['PLAYER_NAME'].dropna().empty else str(a.player_id)

    rows: list[dict] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
        futs = {ex.submit(fetch_xfg_pair,g,a.player_id):g for g in games}
        for fut in as_completed(futs):
            g = futs[fut]
            try: payload, meta = fut.result()
            except Exception as e: payload, meta = None, {'error':repr(e)}
            if payload is None:
                errors.append({'game_id':g, **meta}); continue
            for s in payload.get('shotList') or []:
                ev = event_num_from_shot(s)
                xpct = xfg_pct_from_shot(s)
                if ev is None or xpct is None:
                    continue
                key = (g,int(ev))
                d = distance_lookup.get(key)
                rows.append({
                    'game_id':g,'player_id':int(a.player_id),'event_num':int(ev),
                    'xfg_pct':float(xpct),'shot_distance_ft':d,
                    'distance_source':'official_nba_shotchartdetail.SHOT_DISTANCE' if d is not None else None,
                    'made':int(s.get('success') or 0),'shot_type':s.get('shotType'),'action_type':s.get('actionType'),
                    'period':s.get('period'),'game_clock':s.get('gameClock'),'loc_x':s.get('locX'),'loc_y':s.get('locY'),
                    'guid':s.get('guid'),'event_match_method':'exact_game_id_event_num_player_id'
                })

    shots = pd.DataFrame(rows)
    if shots.empty: raise RuntimeError('No tracked xFG shots returned')
    target = shots[(shots.game_id.eq(a.game_id)) & shots.event_num.eq(int(a.event_num))]
    if len(target) != 1:
        shots.to_csv(a.out/'all_tracked_shots.csv',index=False)
        raise RuntimeError(f'Expected exactly one target shot {a.game_id}/{a.event_num}; found {len(target)}')
    tr = target.iloc[0]
    if pd.isna(tr['shot_distance_ft']):
        raise RuntimeError('Target xFG shot has no exact official SHOT_DISTANCE join')

    td = float(tr['shot_distance_ft']); tx = float(tr['xfg_pct'])
    comparable = shots[
        shots['shot_distance_ft'].notna()
        & shots['xfg_pct'].notna()
        & (shots['shot_distance_ft'].sub(td).abs() <= a.distance_window_ft + 1e-9)
        & (shots['xfg_pct'].sub(tx).abs() <= a.xfg_window_pp + 1e-9)
        & ~((shots['game_id'].eq(a.game_id)) & shots['event_num'].eq(int(a.event_num)))
    ].copy().sort_values(['game_id','event_num'])

    n = len(comparable); made = int(comparable['made'].sum()) if n else 0
    player_xfg = 100.0*made/n if n else None
    lo, hi = wilson_interval(made,n)
    joined = int(shots['shot_distance_ft'].notna().sum())

    result = {
        'definition_version':'player_conditioned_xfg_v2_official_distance',
        'season':a.season,'season_type':a.season_type,
        'player_id':int(a.player_id),'player_name':player_name,
        'target':{
            'game_id':a.game_id,'event_num':int(a.event_num),
            'league_xfg_pct':tx,'shot_distance_ft':td,
            'made':int(tr['made']),'shot_type':tr.get('shot_type'),'action_type':tr.get('action_type'),
            'distance_source':'official_nba_shotchartdetail.SHOT_DISTANCE',
            'event_match_method':'exact game_id + event_num + player_id'
        },
        'comparison_rule':{
            'same_player':True,'tracked_shots_only':True,
            'distance_source_of_truth':'official NBA shotchartdetail SHOT_DISTANCE',
            'coordinate_derived_distance_allowed':False,
            'distance_window_ft':float(a.distance_window_ft),
            'xfg_window_pp':float(a.xfg_window_pp),'target_shot_excluded':True
        },
        'player_conditioned_xfg_pct':player_xfg,
        'sample_fga':int(n),'sample_fgm':int(made),'wilson_95_pct':[lo,hi],
        'tracked_season_shots_with_xfg':int(len(shots)),
        'tracked_season_shots_with_authoritative_distance':joined,
        'distance_join_coverage_pct':100.0*joined/len(shots),
        'official_shotchart_fga':int(len(chart)),
        'games_requested':int(len(games)),'request_errors':int(len(errors)),
        'xfg_source':XFG_URL,'distance_source':SHOTCHART_URL,
        'note':'Empirical make rate on the player\'s other tracked season shots inside the fixed authoritative SHOT_DISTANCE/xFG windows; target excluded; no missing distance or xFG imputed.'
    }
    shots.to_csv(a.out/'all_tracked_shots.csv',index=False)
    comparable.to_csv(a.out/'comparison_sample.csv',index=False)
    chart.to_csv(a.out/'official_shotchart.csv',index=False)
    pd.DataFrame(errors).to_csv(a.out/'request_errors.csv',index=False)
    (a.out/'player_conditioned_xfg.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if n:
        print('\nCOMPARISON_SAMPLE')
        print(comparable[['game_id','event_num','shot_distance_ft','xfg_pct','made','shot_type','action_type']].to_string(index=False))

if __name__ == '__main__':
    main()
