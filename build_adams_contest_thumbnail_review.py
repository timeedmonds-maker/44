from __future__ import annotations

import io
import math
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

PLAYER_ID=203500
PBP_URL='https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
XFG_URL='https://stats.gleague.nba.com/stats/shotqualityvideologs'
OUT=Path('contest_review')
HEADERS={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36','Referer':'https://www.nba.com/','Origin':'https://www.nba.com','Accept':'application/json, text/plain, */*'}

def norm_gid(v):
    s=str(v)
    if s.endswith('.0'): s=s[:-2]
    return ''.join(c for c in s if c.isdigit()).zfill(10)

def fetch_xfg(sess,gid):
    last=None
    for a in range(5):
        try:
            r=sess.get(XFG_URL,params={'GameID':gid,'PlayerID':PLAYER_ID},timeout=(8,35)); r.raise_for_status()
            j=r.json()
            if int(j.get('playerId') or 0)==PLAYER_ID: return j
            last='unexpected player'
        except Exception as e: last=repr(e)
        time.sleep(min(5,0.6*(2**a)))
    raise RuntimeError(f'{gid}: {last}')

def main():
    OUT.mkdir(exist_ok=True)
    thumbs=OUT/'thumbs'; thumbs.mkdir(exist_ok=True)
    df=pd.read_csv(PBP_URL,usecols=['game_id','player1_name','is_field_goal'],low_memory=False)
    gids=df['game_id'].map(norm_gid)
    mask=(df['player1_name'].astype(str).str.match(r'^\s*203500(?:\.0)?\s+',na=False)&pd.to_numeric(df['is_field_goal'],errors='coerce').fillna(0).eq(1)&gids.str.startswith('002'))
    games=sorted(gids[mask].unique().tolist())
    s=requests.Session(); s.headers.update(HEADERS)
    rows=[]
    for i,gid in enumerate(games,1):
        j=fetch_xfg(s,gid)
        for sh in j.get('shotList') or []:
            if int(sh.get('success') or 0)!=1: continue
            rows.append({'game_id':gid,'event_id':int(sh['eventNum']),'game_date':j.get('gameDate'),'matchup':j.get('matchup'),'period':sh.get('period'),'game_clock':sh.get('gameClock'),'action_type':sh.get('actionType'),'shot_type':sh.get('shotType'),'xfg':float(sh.get('shotQuality')) if sh.get('shotQuality') is not None else math.nan,'thumbnail':sh.get('largeThumbnail') or sh.get('mediumThumbnail')})
        print(f'games={i}/{len(games)} makes={len(rows)}',flush=True)
    made=pd.DataFrame(rows).sort_values(['game_date','game_id','event_id'],kind='stable').reset_index(drop=True)
    made.insert(0,'review_index',range(1,len(made)+1))
    made.to_csv(OUT/'manifest.csv',index=False)
    font=ImageFont.load_default()
    cells=[]
    for r in made.itertuples(index=False):
        resp=s.get(r.thumbnail,timeout=(8,30)); resp.raise_for_status()
        im=Image.open(io.BytesIO(resp.content)).convert('RGB').resize((640,360))
        canvas=Image.new('RGB',(640,410),'white'); canvas.paste(im,(0,0))
        d=ImageDraw.Draw(canvas)
        label=f"#{r.review_index:02d}  {r.game_date}  {r.matchup}  E{r.event_id}  Q{r.period} {r.game_clock}  xFG {r.xfg*100:.1f}%"
        label2=str(r.action_type)
        d.text((8,366),label,fill='black',font=font); d.text((8,386),label2,fill='black',font=font)
        p=thumbs/f"{r.review_index:02d}_{r.game_id}_{r.event_id}.jpg"; canvas.save(p,quality=92)
        cells.append((r.review_index,canvas.copy()))
    per=8
    pages=[]
    for pi in range(0,len(cells),per):
        group=cells[pi:pi+per]
        page=Image.new('RGB',(1280,1640),'white')
        for j,(_,im) in enumerate(group):
            x=(j%2)*640; y=(j//2)*410; page.paste(im,(x,y))
        pp=OUT/f"overview_{pi//per+1:02d}.jpg"; page.save(pp,quality=90); pages.append(pp)
    print(f'MAKES={len(made)} PAGES={len(pages)}',flush=True)

if __name__=='__main__': main()
