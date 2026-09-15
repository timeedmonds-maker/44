#!/usr/bin/env python3
"""Resolve and audit all official NBA angles for selected Kickout 3 events."""
from __future__ import annotations
import argparse, html as htmlmod, json, re, subprocess, urllib.parse, urllib.request
from pathlib import Path

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*'}


def get(url,timeout=45):
    req=urllib.request.Request(url,headers=H)
    with urllib.request.urlopen(req,timeout=timeout) as r: return r.read()

def signed_join(base,child):
    joined=urllib.parse.urljoin(base,child); bp=urllib.parse.urlsplit(base); cp=urllib.parse.urlsplit(joined)
    if bp.query and not cp.query: joined=urllib.parse.urlunsplit(cp._replace(query=bp.query))
    return joined

def choose_best(master):
    txt=get(master).decode('utf-8','replace'); lines=[x.strip() for x in txt.splitlines() if x.strip()]
    v=[]
    for i,line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF'): continue
        bw=int(re.search(r'BANDWIDTH=(\d+)',line).group(1)) if re.search(r'BANDWIDTH=(\d+)',line) else 0
        rs=re.search(r'RESOLUTION=(\d+)x(\d+)',line); wh=(int(rs.group(1)),int(rs.group(2))) if rs else (None,None)
        child=None
        for q in lines[i+1:i+5]:
            if not q.startswith('#'): child=q; break
        if child: v.append((bw,wh,signed_join(master,child)))
    if not v: return master,{'bandwidth':None,'width':None,'height':None}
    ch=max(v,key=lambda x:x[0]); return ch[2],{'bandwidth':ch[0],'width':ch[1][0],'height':ch[1][1]}

def parse_options(game,event):
    page=f'https://clips.nba.com/?gameNo={game}&eventNum={event}&source=grs'; txt=get(page).decode('utf-8','replace'); opts=[]
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',txt,flags=re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip()); lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower():
            _,q=choose_best(u); opts.append({'label':lab,'master_url':u,**q})
    if not opts: raise RuntimeError(f'No signed lrmedia HLS options for {game}/{event}')
    return page,opts

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    m=json.loads(a.manifest.read_text()); results=[]
    for i,e in enumerate(m['selected_events'],1):
        page,opts=parse_options(str(e['game_id']),int(e['shot_event_num']))
        results.append({'event_index':i,'game_id':str(e['game_id']),'shot_event_num':int(e['shot_event_num']),'clip_page':page,'angle_count':len(opts),'angles':opts})
    out={'tool_id':'SCREEN_TRACKER_KICKOUT_3','stage':'official_angle_audit','events':results}
    (a.out/'angle_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
