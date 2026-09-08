from __future__ import annotations

import tempfile
from collections import Counter

import pandas as pd
import pyreadr
import requests

REG_YEARS = range(2009, 2027)   # 2008-09 through 2025-26
PO_YEARS = range(2010, 2027)    # Westbrook's first playoff season through 2026
BASE = 'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main'


def load_rds(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix='.rds') as f:
        f.write(r.content)
        f.flush()
        obj = pyreadr.read_r(f.name)
    if not obj:
        raise RuntimeError(f'No table in {url}')
    return next(iter(obj.values()))


def norm(s: pd.Series) -> pd.Series:
    return (s.fillna('').astype(str).str.lower()
            .str.replace(r'[^a-z ]', '', regex=True).str.strip())


def process(df: pd.DataFrame, season_type: str, end_year: int):
    need = {'msg_type','description','player1_name','player2_name'}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f'{season_type} {end_year} missing {sorted(missing)}')
    msg = pd.to_numeric(df['msg_type'], errors='coerce')
    p2 = norm(df['player2_name'])
    desc = df['description'].fillna('').astype(str)
    mask = (msg.eq(1)
            & p2.str.contains(r'\brussell westbrook\b|\bwestbrook\b', regex=True)
            & desc.str.contains('DUNK', case=False, na=False))
    hit = df.loc[mask, ['player1_name','player2_name','description']].copy()
    hit['season_type'] = season_type
    hit['end_year'] = end_year
    return hit

hits=[]
coverage=[]
for y in REG_YEARS:
    url=f'{BASE}/pbp-final-{y}/data.rds'
    df=load_rds(url)
    h=process(df,'Regular Season',y)
    hits.append(h)
    coverage.append((y,'Regular Season',len(df),len(h)))

for y in PO_YEARS:
    url=f'{BASE}/pbp-final-playoffs{y}/data.rds'
    r=requests.get(url,timeout=180)
    if r.status_code == 404:
        coverage.append((y,'Playoffs',0,0))
        continue
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix='.rds') as f:
        f.write(r.content); f.flush(); obj=pyreadr.read_r(f.name)
    df=next(iter(obj.values()))
    h=process(df,'Playoffs',y)
    hits.append(h)
    coverage.append((y,'Playoffs',len(df),len(h)))

all_hits=pd.concat(hits,ignore_index=True)
# Canonicalize names conservatively by trimmed string; report aliases if any.
all_hits['scorer']=all_hits['player1_name'].fillna('').astype(str).str.strip()
all_hits=all_hits[all_hits['scorer'].ne('')]

combined=all_hits.groupby('scorer').size().sort_values(ascending=False)
reg=all_hits[all_hits.season_type.eq('Regular Season')].groupby('scorer').size()
po=all_hits[all_hits.season_type.eq('Playoffs')].groupby('scorer').size()

out=pd.DataFrame({'combined':combined})
out['regular_season']=reg
out['playoffs']=po
out=out.fillna(0).astype(int).sort_values(['combined','regular_season'],ascending=False)
out.to_csv('westbrook_dunk_assist_leaders.csv')
all_hits.to_csv('westbrook_dunk_assist_events.csv',index=False)

print('TOTAL_DUNK_ASSISTS',len(all_hits))
print('TOP_30')
print(out.head(30).to_string())
print('\nCOVERAGE')
for row in coverage:
    print(*row)
