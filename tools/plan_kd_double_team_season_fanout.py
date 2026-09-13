#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import pandas as pd


def parse_nums(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    out=[]
    for x in re.findall(r'(?<!\d)(\d{1,6})(?!\d)',str(v)):
        n=int(x)
        if n not in out: out.append(n)
    return out


def clock_sec(v):
    if pd.isna(v): return math.nan
    m=re.match(r'^(\d+):(\d+(?:\.\d+)?)$',str(v).strip())
    return int(m.group(1))*60+float(m.group(2)) if m else math.nan


def unique(seq):
    out=[]
    for x in seq:
        if x is not None and x not in out: out.append(x)
    return out


def plan_row(r,long_seconds=18.0):
    kd=parse_nums(r.get('kd_event_nums'))
    win=parse_nums(r.get('window_event_nums'))
    anchor=int(r.anchor_event_num) if pd.notna(r.get('anchor_event_num')) else None
    st=clock_sec(r.get('start_time')); en=clock_sec(r.get('end_time'))
    dur=abs(st-en) if math.isfinite(st) and math.isfinite(en) else 0.0
    preferred=[]; reason=''
    if kd:
        if dur>long_seconds and len(kd)>1:
            preferred=[kd[0],kd[-1]]; reason='long possession: first+last KD-attributed event'
        else:
            preferred=[kd[-1]]; reason='last KD-attributed event'
    elif win:
        if dur>long_seconds and len(win)>2:
            preferred=[win[len(win)//3],win[min(len(win)-1,(2*len(win))//3)]]
            reason='long no-KD-PBP possession: one-third+two-thirds event anchors'
        elif dur>long_seconds and len(win)>1:
            preferred=[win[0],win[-1]]; reason='long no-KD-PBP possession: early+late event anchors'
        else:
            preferred=[win[len(win)//2]]; reason='no-KD-PBP possession: midpoint event anchor'
    elif anchor is not None:
        preferred=[anchor]; reason='fallback possession anchor'
    preferred=unique(preferred)[:2]
    # Fallback events remain strictly inside the exact possession window, then the exact anchor.
    fallback=unique([*win,anchor])
    fallback=[x for x in fallback if x not in preferred]
    return pd.Series({
        'possession_duration_s':round(dur,2),
        'screen_event_nums':'|'.join(map(str,preferred)),
        'fallback_event_nums':'|'.join(map(str,fallback)),
        'target_clip_count':len(preferred),
        'screen_plan_reason':reason,
    })


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--shards',type=int,default=20); ap.add_argument('--long-seconds',type=float,default=18.0)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,dtype={'game_id':str}); df['game_id']=df.game_id.str.zfill(10)
    plan=df.apply(lambda r:plan_row(r,a.long_seconds),axis=1)
    for c in plan.columns: df[c]=plan[c]
    if (df.target_clip_count<=0).any():
        bad=df[df.target_clip_count<=0][['game_id','possession_uid']]
        raise SystemExit(f'no clip candidates for {len(bad)} possessions')
    game_cost=df.groupby('game_id').agg(rows=('possession_uid','size'),clips=('target_clip_count','sum')).reset_index()
    game_cost=game_cost.sort_values(['clips','rows','game_id'],ascending=[False,False,True])
    bins=[{'id':i,'clips':0,'rows':0,'games':[]} for i in range(a.shards)]
    for _,g in game_cost.iterrows():
        b=min(bins,key=lambda x:(x['clips'],x['rows'],x['id']))
        b['games'].append(g.game_id); b['clips']+=int(g.clips); b['rows']+=int(g.rows)
    shard_dir=a.out/'shards'; shard_dir.mkdir(exist_ok=True)
    include=[]
    for b in bins:
        if not b['games']: continue
        sdf=df[df.game_id.isin(b['games'])].copy().sort_values(['game_id','period','poss_num_team'])
        sid=f"{b['id']:02d}"; path=shard_dir/f'shard_{sid}.csv'; sdf.to_csv(path,index=False)
        include.append({'shard_id':sid,'rows':int(len(sdf)),'estimated_clips':int(sdf.target_clip_count.sum()),'games':len(b['games'])})
    (a.out/'matrix.json').write_text(json.dumps({'include':include},separators=(',',':')))
    df.to_csv(a.out/'season_screen_manifest.csv.gz',index=False,compression='gzip')
    qa={
        'possessions':int(len(df)),'games':int(df.game_id.nunique()),'fanout_shards':len(include),
        'estimated_primary_clips':int(df.target_clip_count.sum()),'two_clip_possessions':int((df.target_clip_count==2).sum()),
        'one_clip_possessions':int((df.target_clip_count==1).sum()),
        'clip_budget_vs_two_per_possession_reduction':round(1-float(df.target_clip_count.sum())/(2*len(df)),4),
        'max_shard_rows':max(x['rows'] for x in include),'min_shard_rows':min(x['rows'] for x in include),
        'max_shard_estimated_clips':max(x['estimated_clips'] for x in include),'min_shard_estimated_clips':min(x['estimated_clips'] for x in include),
        'long_possession_threshold_s':a.long_seconds,
        'notes':'Games are kept intact and greedily balanced across shards. Fallback events are exact-window events only.'
    }
    (a.out/'season_plan_qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2)); print('MATRIX='+json.dumps({'include':include},separators=(',',':')))
if __name__=='__main__': main()
