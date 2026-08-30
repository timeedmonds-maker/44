from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

XFG_URL = 'https://stats.gleague.nba.com/stats/shotqualityvideologs'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}
TLS = threading.local()


def session():
    s = getattr(TLS, 's', None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        TLS.s = s
    return s


def fetch_pair(game_id: str, player_id: int, attempts: int = 7):
    last = None
    for a in range(1, attempts + 1):
        started = time.time()
        try:
            r = session().get(XFG_URL, params={'GameID': game_id, 'PlayerID': int(player_id)}, timeout=(7, 35))
            elapsed = time.time() - started
            if r.status_code == 200:
                j = r.json()
                if str(j.get('gameId') or '').zfill(10) == str(game_id).zfill(10) and int(j.get('playerId') or 0) == int(player_id):
                    return j, {'status': 200, 'attempts': a, 'elapsed': elapsed}
                last = {'status': 200, 'attempts': a, 'elapsed': elapsed, 'error': 'unexpected_payload'}
            else:
                last = {'status': r.status_code, 'attempts': a, 'elapsed': elapsed, 'error': (r.text or '')[:120]}
        except Exception as e:
            last = {'status': None, 'attempts': a, 'elapsed': time.time() - started, 'error': repr(e)}
        if a < attempts:
            time.sleep(min(30.0, 1.25 * (2 ** (a - 1))) + random.random())
    return None, last or {'error': 'unknown'}


def payload_contribution(payload: dict):
    tracked_fga = tracked_fgm = tracked_3pa = tracked_3pm = 0
    actual_num = expected_num = 0.0
    shots = payload.get('shotList') or []
    for s in shots:
        try:
            xfg = float(s.get('shotQuality'))
        except Exception:
            continue
        if not math.isfinite(xfg):
            continue
        made = 1 if int(s.get('success') or 0) else 0
        is3 = str(s.get('shotType') or '').upper().startswith('3PT')
        w = 1.5 if is3 else 1.0
        tracked_fga += 1
        tracked_fgm += made
        tracked_3pa += int(is3)
        tracked_3pm += int(is3 and made)
        actual_num += made * w
        expected_num += xfg * w
    official = payload.get('shots')
    official_fga = int(official) if isinstance(official, (int, float)) and not isinstance(official, bool) else 0
    return {
        'tracked_fga': tracked_fga,
        'tracked_fgm': tracked_fgm,
        'tracked_3pa': tracked_3pa,
        'tracked_3pm': tracked_3pm,
        'actual_efg_num': actual_num,
        'expected_efg_num': expected_num,
        'api_official_fga': official_fga,
        'unlisted_fga': max(0, official_fga - len(shots)),
        'successful_games': 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='downloaded')
    ap.add_argument('--out-dir', default='final')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--min-tracked-fga', type=int, default=300)
    args = ap.parse_args()

    root = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    partial_paths = sorted(root.rglob('partial_*.csv'))
    error_paths = sorted(root.rglob('errors_*.csv'))
    summary_paths = sorted(root.rglob('summary_*.json'))
    print(f'partials={len(partial_paths)} errors_files={len(error_paths)} summaries={len(summary_paths)}', flush=True)

    parts=[]
    for p in partial_paths:
        try: d=pd.read_csv(p)
        except pd.errors.EmptyDataError: continue
        if len(d): parts.append(d)
    if not parts:
        raise SystemExit('No partial rows found')
    df=pd.concat(parts, ignore_index=True)
    numeric=['tracked_fga','tracked_fgm','tracked_3pa','tracked_3pm','actual_efg_num','expected_efg_num','api_official_fga','unlisted_fga','successful_games','failed_games']
    for c in numeric:
        df[c]=pd.to_numeric(df.get(c), errors='coerce').fillna(0)
    agg={c:'sum' for c in numeric}
    agg.update({'player_name':'first','pbp_fga_global':'max'})
    g=df.groupby('player_id',as_index=False).agg(agg).set_index('player_id')

    # Consolidate transient request failures from all shards, then retry exactly once at the pair level.
    errs=[]
    for p in error_paths:
        try: e=pd.read_csv(p)
        except pd.errors.EmptyDataError: continue
        if len(e) and {'game_id','player_id'} <= set(e.columns): errs.append(e[['game_id','player_id']])
    if errs:
        e=pd.concat(errs, ignore_index=True).drop_duplicates(['game_id','player_id'])
        e['game_id']=e['game_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
        e['player_id']=pd.to_numeric(e['player_id'],errors='coerce').astype('Int64')
        e=e.dropna(subset=['player_id']).copy()
        pairs=[(str(x.game_id),int(x.player_id)) for x in e.itertuples(index=False)]
    else:
        pairs=[]
    print(f'retry_pairs={len(pairs)}', flush=True)

    unresolved=[]
    retry_success=0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(fetch_pair,gid,pid):(gid,pid) for gid,pid in pairs}
        for n,fut in enumerate(as_completed(futs),1):
            gid,pid=futs[fut]
            payload,meta=fut.result()
            if payload is None:
                unresolved.append({'game_id':gid,'player_id':pid,**meta})
                continue
            retry_success += 1
            c=payload_contribution(payload)
            if pid not in g.index:
                # Extremely defensive; eligible players should already be present in at least one successful shard.
                g.loc[pid,'player_name']=payload.get('playerName') or ''
                g.loc[pid,'pbp_fga_global']=0
                for col in numeric: g.loc[pid,col]=0
            for col,val in c.items():
                g.loc[pid,col]=float(g.loc[pid,col]) + val
            # The original partial counted this pair as failed; remove one failed-game count after successful repair.
            g.loc[pid,'failed_games']=max(0,float(g.loc[pid,'failed_games'])-1)
            if not str(g.loc[pid,'player_name'] or '').strip():
                g.loc[pid,'player_name']=payload.get('playerName') or ''
            if n % 25 == 0 or n == len(pairs):
                print(f'retry_done={n}/{len(pairs)} success={retry_success} unresolved={len(unresolved)}', flush=True)

    g=g.reset_index()
    g['actual_efg_pct']=100*g['actual_efg_num']/g['tracked_fga'].replace(0,pd.NA)
    g['expected_efg_pct']=100*g['expected_efg_num']/g['tracked_fga'].replace(0,pd.NA)
    g['efg_over_expected_pp']=g['actual_efg_pct']-g['expected_efg_pct']
    g['coverage_pct_vs_api_fga']=100*g['tracked_fga']/g['api_official_fga'].replace(0,pd.NA)
    g['coverage_pct_vs_pbp_fga']=100*g['tracked_fga']/g['pbp_fga_global'].replace(0,pd.NA)
    g['api_request_coverage_pct']=100*g['successful_games']/(g['successful_games']+g['failed_games']).replace(0,pd.NA)
    g=g.sort_values(['efg_over_expected_pp','tracked_fga'],ascending=[False,False]).reset_index(drop=True)
    g.insert(0,'rank_all_eligible',range(1,len(g)+1))
    qualified=g.loc[g['tracked_fga']>=args.min_tracked_fga].copy().reset_index(drop=True)
    qualified.insert(0,'rank',range(1,len(qualified)+1))

    g.to_csv(out/'player_efg_over_expected_2025_26_all_eligible.csv',index=False)
    qualified.to_csv(out/'player_efg_over_expected_2025_26.csv',index=False)
    qualified.head(25).to_csv(out/'top25_efg_over_expected_2025_26.csv',index=False)
    pd.DataFrame(unresolved).to_csv(out/'unresolved_retry_pairs.csv',index=False)

    summaries=[]
    for p in summary_paths:
        try: summaries.append(json.loads(p.read_text()))
        except Exception: pass
    initial_pairs=sum(int(x.get('request_pairs',0)) for x in summaries)
    initial_errors=sum(int(x.get('errors',0)) for x in summaries)
    qa={
        'season':'2025-26','season_type':'Regular Season','minimum_tracked_fga':args.min_tracked_fga,
        'source_resource':'shotqualityvideologs','source_host':'stats.gleague.nba.com',
        'definition':{
            'actual_efg':'sum(made * [1.5 if 3PT else 1.0]) / tracked_xfg_FGA',
            'expected_efg':'sum(NBA_xFG * [1.5 if 3PT else 1.0]) / tracked_xfg_FGA',
            'efg_over_expected_pp':'actual eFG% - expected eFG%'
        },
        'partial_files':len(partial_paths),'initial_request_pairs':initial_pairs,'initial_errors':initial_errors,
        'unique_retry_pairs':len(pairs),'retry_successes':retry_success,'unresolved_retry_pairs':len(unresolved),
        'qualified_players':int(len(qualified)),
        'tracked_fga_qualified_total':int(qualified['tracked_fga'].sum()),
        'note':'Expected eFG is exact for publicly exposed NBA xFG shotList rows. Missing individual xFG rows are not imputed.'
    }
    qa['final_request_success_rate_pct']=100*(initial_pairs-len(unresolved))/initial_pairs if initial_pairs else None
    (out/'qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')

    cols=['rank','player_name','tracked_fga','actual_efg_pct','expected_efg_pct','efg_over_expected_pp','coverage_pct_vs_pbp_fga','api_request_coverage_pct']
    print('\nTOP 25 QUALIFIED')
    print(qualified.head(25)[cols].to_string(index=False,formatters={
        'actual_efg_pct':lambda x:f'{x:.3f}',
        'expected_efg_pct':lambda x:f'{x:.3f}',
        'efg_over_expected_pp':lambda x:f'{x:+.3f}',
        'coverage_pct_vs_pbp_fga':lambda x:f'{x:.2f}',
        'api_request_coverage_pct':lambda x:f'{x:.2f}',
    }))
    print(json.dumps(qa,indent=2))


if __name__=='__main__':
    main()
