#!/usr/bin/env python3
import re, urllib.parse
from curl_cffi import requests

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://www.nba.com/stats/events'}
PAGE='https://www.nba.com/stats/events?GameEventID=438&GameID=0021700015&Season=2017-18&flag=1'
TARGETS=['36843','91832','25306','18744','50807','47644','16395','27293','23704','38188']
NEEDLES=['function At','const At','At=','function qe','const qe','qe=','lurl','murl','surl','hdurl','videos.nba.com','missing.mp4','resource:','stats.nba.com','api-hub','/stats/','fetch(','axios','transform','parse','videoUrls']

def safe(u):
 p=urllib.parse.urlsplit(u); return p.netloc+p.path

def extract_module(js, mid):
 # webpack modules normally `MID:function(e,t,a){...},NEXT:`. Brace parser is more robust than regex.
 pats=[mid+':function(',mid+':(e,t,a)=>',mid+':e=>']
 starts=[js.find(p) for p in pats if js.find(p)>=0]
 if not starts:return None
 st=min(starts)
 brace=js.find('{',st)
 if brace<0:return js[st:st+8000]
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
   if depth==0:return js[st:i+1]
 return js[st:st+30000]

s=requests.Session(impersonate='chrome120')
r=s.get('https://www.nba.com/',headers=H,timeout=20); print('PREWARM',r.status_code,len(r.content))
r=s.get(PAGE,headers=H,timeout=20); print('PAGE',r.status_code,len(r.content))
scripts=re.findall(r'<script[^>]+src=["\']([^"\']+)',r.text,re.I)
chunks=[]
for src in scripts:
 u=urllib.parse.urljoin(r.url,src)
 try:
  rr=s.get(u,headers=H,timeout=20)
  if rr.status_code==200 and 'javascript' in rr.headers.get('content-type','').lower() or u.endswith('.js'):
   chunks.append((u,rr.text))
 except Exception as e: print('FETCH_ERR',safe(u),type(e).__name__)
print('CHUNKS',len(chunks))

for mid in TARGETS:
 found=False
 for u,js in chunks:
  mod=extract_module(js,mid)
  if mod:
   found=True
   print('\n===== MODULE',mid,'FROM',safe(u),'LEN',len(mod),'=====')
   print(mod[:40000])
 if not found: print('\nMODULE_NOT_FOUND',mid)

for u,js in chunks:
 hits=[]
 for n in NEEDLES:
  pos=0
  while True:
   i=js.find(n,pos)
   if i<0: break
   hits.append((i,n)); pos=i+len(n)
 if hits:
  # Only emit useful chunks, cap contexts.
  useful=[x for x in hits if x[1] in ['function At','const At','At=','function qe','const qe','qe=','lurl','murl','surl','hdurl','stats.nba.com','api-hub','/stats/','videoUrls','missing.mp4']]
  if useful:
   print('\n===== NEEDLE CHUNK',safe(u),'=====')
   for i,n in useful[:40]:
    print('\n--',n,'@',i,'--')
    print(re.sub(r'\s+',' ',js[max(0,i-1800):i+5000])[:7000])
