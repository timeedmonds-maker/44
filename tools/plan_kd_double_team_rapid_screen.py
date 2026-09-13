#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd


def parse_nums(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for x in re.findall(r'(?<!\d)(\d{1,5})(?!\d)', str(v)):
        n=int(x)
        if n not in out: out.append(n)
    return out


def choose_events(r):
    kd=parse_nums(r.get('kd_event_nums'))
    win=parse_nums(r.get('window_event_nums'))
    anchor=int(r.anchor_event_num) if pd.notna(r.get('anchor_event_num')) else None
    chosen=[]
    if kd:
        chosen.append(kd[0])
        if len(kd)>1 and kd[-1]!=kd[0]: chosen.append(kd[-1])
        if anchor is not None and anchor not in chosen and len(chosen)<2: chosen.append(anchor)
    else:
        if win:
            chosen.append(win[0])
            if len(win)>1 and win[-1]!=win[0]: chosen.append(win[-1])
        elif anchor is not None:
            chosen.append(anchor)
    if not chosen and anchor is not None: chosen=[anchor]
    return '|'.join(map(str,chosen[:2]))


def benchmark_sample(df, games=3, per_game=4):
    # Deterministic: force opener if present, then earliest game IDs.
    gids=sorted(df.game_id.astype(str).unique())
    opener='0022500001'
    selected=([opener] if opener in gids else []) + [g for g in gids if g!=opener]
    selected=selected[:games]
    chunks=[]
    for gid in selected:
        g=df[df.game_id.astype(str)==gid].copy()
        picks=[]
        # Always include the validated event-321 possession when available.
        ev321=g[pd.to_numeric(g.anchor_event_num,errors='coerce').eq(321)]
        if len(ev321): picks.append(ev321.iloc[[0]])
        # Then stratify A/B/C so the benchmark exercises every workload type.
        for pri in ('A','B','C'):
            p=g[g.candidate_priority.eq(pri)]
            need=max(0, per_game-sum(len(x) for x in picks))
            if need<=0: break
            if len(p): picks.append(p.head(min(need, max(1, per_game//3))))
        cur=pd.concat(picks,ignore_index=True).drop_duplicates('possession_uid') if picks else g.iloc[0:0]
        if len(cur)<per_game:
            extra=g[~g.possession_uid.isin(set(cur.possession_uid))].head(per_game-len(cur))
            cur=pd.concat([cur,extra],ignore_index=True)
        chunks.append(cur.head(per_game))
    return pd.concat(chunks,ignore_index=True) if chunks else df.iloc[0:0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['benchmark','full'],default='benchmark')
    ap.add_argument('--benchmark-games',type=int,default=3)
    ap.add_argument('--benchmark-per-game',type=int,default=4)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str})
    df['game_id']=df.game_id.str.zfill(10)
    df['screen_event_nums']=df.apply(choose_events,axis=1)
    df['screen_anchor_count']=df.screen_event_nums.map(lambda s: len(parse_nums(s)))
    df['screen_plan_reason']=df.apply(lambda r: 'KD-attributed event anchors' if parse_nums(r.kd_event_nums) else 'early+late possession event anchors',axis=1)
    use=benchmark_sample(df,a.benchmark_games,a.benchmark_per_game) if a.mode=='benchmark' else df.copy()
    use=use[use.screen_anchor_count.gt(0)].copy()
    use.to_csv(a.out/'rapid_screen_manifest.csv',index=False)
    include=[]
    shard_dir=a.out/'shards'; shard_dir.mkdir(exist_ok=True)
    for gid,g in use.groupby('game_id',sort=True):
        p=shard_dir/f'{gid}.csv'; g.to_csv(p,index=False)
        include.append({'game_id':gid,'rows':int(len(g)),'shard':str(p)})
    matrix={'include':include}
    (a.out/'matrix.json').write_text(json.dumps(matrix,separators=(',',':')))
    qa={
        'mode':a.mode,'rows':int(len(use)),'games':int(use.game_id.nunique()),
        'priority_counts':use.candidate_priority.value_counts().to_dict(),
        'screen_anchor_count_distribution':use.screen_anchor_count.value_counts().sort_index().to_dict(),
        'full_universe_rows':int(len(df)),'full_universe_games':int(df.game_id.nunique()),
        'fanout_unit':'one game per matrix job','recommended_max_parallel':20,
        'stage_a_semantics':'conservative candidate sieve only; never a final double-team positive label'
    }
    (a.out/'plan_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps(qa,indent=2))
    print('MATRIX='+json.dumps(matrix,separators=(',',':')))
if __name__=='__main__': main()
