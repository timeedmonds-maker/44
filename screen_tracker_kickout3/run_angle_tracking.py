#!/usr/bin/env python3
"""Run the existing Screen Tracker CV stack against one exact official angle.

This wrapper monkeypatches only the *input clip resolver* in the imported
production tracker. Canonical Screen Tracker code remains read-only.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
import requests

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'tools'
sys.path.insert(0,str(TOOLS))
import adams_screen_prod_v3 as prod
from fetch_official_nba_event_clip import parse as resolve_clip, UA

HEADERS={'User-Agent':UA,'Referer':'https://clips.nba.com/'}

def choose_variant(url,target_width=960):
    r=requests.get(url,headers=HEADERS,timeout=30);r.raise_for_status();txt=r.text
    if '#EXT-X-STREAM-INF' not in txt:return url,None
    lines=[x.strip() for x in txt.splitlines() if x.strip()]; variants=[]
    for i,line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF'):continue
        rs=re.search(r'RESOLUTION=(\d+)x(\d+)',line);bw=re.search(r'BANDWIDTH=(\d+)',line);child=None
        for q in lines[i+1:i+5]:
            if not q.startswith('#'):child=q;break
        if not child:continue
        full=urljoin(url,child)
        if not urlsplit(full).query and urlsplit(url).query:
            b=urlsplit(url);f=urlsplit(full);full=urlunsplit((f.scheme,f.netloc,f.path,b.query,f.fragment))
        w=int(rs.group(1)) if rs else 10**9;h=int(rs.group(2)) if rs else None
        variants.append((w,h,int(bw.group(1)) if bw else None,full))
    under=[v for v in variants if v[0]<=target_width];ch=max(under,key=lambda x:x[0]) if under else min(variants,key=lambda x:x[0])
    return ch[3],{'width':ch[0],'height':ch[1],'bandwidth':ch[2]}

def make_fetch(angle_label):
    def fetch(game,event,out,target_width=960):
        page,angles,_=resolve_clip(game,event)
        matches=[x for x in angles if str(x.get('label','')).strip()==angle_label]
        if len(matches)!=1:raise RuntimeError(f'expected one {angle_label!r} angle, got {[x.get("label") for x in angles]}')
        hls,variant=choose_variant(matches[0]['url'],target_width)
        hdr=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
        subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',hdr,'-i',hls,'-c','copy',str(out)],check=True,timeout=180)
        return {'clip_page':page,'angle':angle_label,'angle_count':len(angles),'variant':variant}
    return fetch

def main():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--angle-label',required=True);args,rest=pre.parse_known_args()
    prod.base.fetch_native_clip=make_fetch(args.angle_label)
    sys.argv=[sys.argv[0]]+rest
    prod.main()

if __name__=='__main__':main()
