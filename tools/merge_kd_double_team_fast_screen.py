#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    files=sorted(a.root.rglob('screen_results.csv'))
    if not files: raise SystemExit('no screen_results.csv files found')
    df=pd.concat([pd.read_csv(p,dtype={'game_id':str}) for p in files],ignore_index=True)
    df=df.drop_duplicates('possession_uid',keep='last').sort_values(['game_id','possession_uid'])
    df.to_csv(a.out/'rapid_screen_results.csv',index=False)
    qfiles=sorted(a.root.rglob('screen_qa.json')); shard_qa=[]
    for p in qfiles:
        try: shard_qa.append(json.loads(p.read_text()))
        except Exception: pass
    qa={
        'rows':int(len(df)),'games':int(df.game_id.nunique()),
        'screen_negative':int((df.screen_decision=='screen_negative').sum()),
        'candidate':int((df.screen_decision=='candidate').sum()),
        'candidate_retention_rate':float((df.screen_decision=='candidate').mean()),
        'error_rows':int(df.errors.fillna('').astype(str).ne('').sum()),
        'mean_runtime_seconds_per_possession':float(pd.to_numeric(df.runtime_seconds,errors='coerce').mean()),
        'mean_observable_fraction':float(pd.to_numeric(df.observable_fraction,errors='coerce').mean()),
        'shards_merged':len(files),
        'semantics':'Stage-A conservative sieve. No row here is a final positive double-team label.'
    }
    (a.out/'rapid_screen_qa.json').write_text(json.dumps(qa,indent=2)); (a.out/'shard_qa.json').write_text(json.dumps(shard_qa,indent=2))
    print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
