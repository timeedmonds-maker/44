#!/usr/bin/env python3
import re, urllib.parse
from curl_cffi import requests
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://www.nba.com/stats/events'}
PAGE='https://www.nba.com/stats/events?GameEventID=438&GameID=0021700015&Season=2017-18&flag=1'
TARGETS=['36843','32409','6407','31955','86553','72972','2784','84570','16562']
NEEDLES=['function At','At=','lurl','murl','surl','hdurl','missing.mp4','videoUrls','videoeventsasset']

def extract_module(js, mid):
    for pat in [mid+':(e,n,t)=>',mid+':(e,t,a)=>',mid+':function(',mid+':e=>']:
        st=js.find(pat)
        if st>=0: break
    else: return None
    brace=js.find('{',st)
    if brace<0: return js[st:st+10000]
    depth=0; quote=None; esc=False
    for i in range(brace,len(js)):
        ch=js[i]
        if quote:
            if esc: esc=False
            elif ch=='\\': esc=True
            elif ch==quote: quote=None
            continue
        if ch in "'\"`": quote=ch; continue
        if ch=='{': depth+=1
        elif ch=='}':
            depth-=1
            if depth==0: return js[st:i+1]
    return js[st:st+40000]

def safe(u):
    p=urllib.parse.urlsplit(u); return p.netloc+p.path

s=requests.Session(impersonate='chrome120')
r=s.get('https://www.nba.com/',headers=H,timeout=20); print('PREWARM',r.status_code,len(r.content))
r=s.get(PAGE,headers=H,timeout=20); print('PAGE',r.status_code,len(r.content))
scripts=[urllib.parse.urljoin(r.url,x) for x in re.findall(r'<script[^>]+src=["\']([^"\']+)',r.text,re.I)]
chunks=[]
for u in scripts:
    try:
        rr=s.get(u,headers=H,timeout=20)
        if rr.status_code==200: chunks.append((u,rr.text))
    except Exception: pass
print('CHUNKS',len(chunks))
for mid in TARGETS:
    found=False
    for u,js in chunks:
        mod=extract_module(js,mid)
        if mod:
            found=True
            print(f'\n===== MODULE {mid} {safe(u)} LEN {len(mod)} =====')
            print(mod[:30000])
    if not found: print('\nNOT_FOUND',mid)
for u,js in chunks:
    if 'lurl' not in js and 'videoeventsasset' not in js: continue
    print('\n===== PLAYER_CONTEXT',safe(u),'=====')
    for n in NEEDLES:
        pos=0; emitted=0
        while emitted<6:
            i=js.find(n,pos)
            if i<0: break
            print(f'\n-- {n} @{i} --')
            print(re.sub(r'\s+',' ',js[max(0,i-2500):i+6500])[:9000])
            pos=i+len(n); emitted+=1
