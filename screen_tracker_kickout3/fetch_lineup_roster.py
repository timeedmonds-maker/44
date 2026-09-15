#!/usr/bin/env python3
"""Build exact ten-player event-lineup roster JSON from official NBA boxscore.

The PBP event supplies the exact on-court player IDs; NBA live boxscore supplies
names and jersey numbers. Output matches tools/adams_event_track_identity_ocr.py.
"""
from __future__ import annotations
import argparse, json, re, urllib.request
from pathlib import Path
import pandas as pd, pyreadr, requests

PBP_URL='https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds'
ID_RE=re.compile(r'(?<!\d)(\d{6,7})(?!\d)')

def gid(v): return str(v).replace('.0','').zfill(10)
def ids(v): return [int(x) for x in ID_RE.findall(str(v or ''))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--game',required=True); ap.add_argument('--event',type=int,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    game=str(a.game).zfill(10)
    r=requests.get(PBP_URL,timeout=180); r.raise_for_status(); tmp=a.out.with_suffix('.rds'); tmp.write_bytes(r.content); d=next(iter(pyreadr.read_r(str(tmp)).values())); tmp.unlink(missing_ok=True)
    g=d.game_id.astype(str).str.replace('.0','',regex=False).str.zfill(10); ev=pd.to_numeric(d.event_num,errors='coerce'); q=d[(g==game)&(ev==a.event)]
    if len(q)!=1: raise RuntimeError(f'expected one PBP row {game}/{a.event}; got {len(q)}')
    row=q.iloc[0]; exact=set(ids(row.get('lineup_home'))+ids(row.get('lineup_away')))
    if len(exact)!=10: raise RuntimeError(f'expected exact 10-player lineup; got {len(exact)} {sorted(exact)}')
    url=f'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game}.json'; req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.nba.com/'})
    with urllib.request.urlopen(req,timeout=60) as resp: j=json.load(resp)
    teams={}; seen=set(); game_obj=j['game']
    for side in ('homeTeam','awayTeam'):
        t=game_obj[side]; tri=str(t['teamTricode']); players=[]
        for p in t.get('players',[]):
            pid=int(p.get('personId') or 0)
            if pid not in exact: continue
            rec={'id':pid,'name':p.get('name') or f"{p.get('firstName','')} {p.get('familyName','')}".strip(),'jersey':str(p.get('jerseyNum') or '')}
            players.append(rec); seen.add(pid)
        teams[tri]={'players':sorted(players,key=lambda x:(int(x['jersey']) if x['jersey'].isdigit() else 999,x['id']))}
    if seen!=exact: raise RuntimeError(f'official boxscore missing exact lineup ids: {sorted(exact-seen)}')
    out={'game_id':game,'event_num':a.event,'teams':teams,'exact_lineup_ids':sorted(exact),'provenance':{'lineup':'ramirobentes nba_pbp_data exact event lineup fields','names_and_jerseys':url}}
    a.out.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
