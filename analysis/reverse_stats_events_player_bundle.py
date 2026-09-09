#!/usr/bin/env python3
import re,json,urllib.parse
from curl_cffi import requests

IMPERSONATE='chrome120'
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
WEB_HEADERS={'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Accept-Language':'en-US,en;q=0.9','User-Agent':UA,'Referer':'https://www.nba.com/'}
EVENTS=[('2017','0021700015','438','2017-18'),('2019','0041800163','215','2018-19'),('2025','0022500375','608','2025-26')]
TERMS=['videoeventsasset','videoevents','videodetailsasset','videodetails','videoUrls','GameEventID','GameID','m3u8','lrmedia','clips.nba.com','videos.nba.com','pmd.cdn.turner','cdn.turner','videoAvailableFlag','Play Video','videojs','video.js','playerOptions','sources','sourceUrl','mediaUrl','playback','playlist','manifest','akamai','mux','brightcove']

def safe(url):
 p=urllib.parse.urlsplit(url); return {'host':p.netloc,'path':p.path,'query_keys':sorted(urllib.parse.parse_qs(p.query).keys())}

def ctx(js,term):
 i=js.lower().find(term.lower())
 if i<0:return None
 return re.sub(r'\s+',' ',js[max(0,i-1000):i+3500])[:5000]

s=requests.Session(impersonate=IMPERSONATE)
try:
 r=s.get('https://www.nba.com/',headers=WEB_HEADERS,timeout=20)
 print('PREWARM',r.status_code,len(r.content),'cookies',sorted(s.cookies.get_dict().keys()))
except Exception as e: print('PREWARM_ERR',e)

all_scripts=[]
for label,gid,eid,season in EVENTS:
 url=f'https://www.nba.com/stats/events?GameEventID={eid}&GameID={gid}&Season={season}&flag=1'
 try:
  r=s.get(url,headers=WEB_HEADERS,timeout=25)
  print('\nPAGE',label,r.status_code,len(r.content),safe(r.url))
  html=r.text
  title=re.search(r'<title[^>]*>(.*?)</title>',html,re.I|re.S)
  print('TITLE',re.sub(r'\s+',' ',title.group(1)).strip() if title else None)
  scripts=re.findall(r'<script[^>]+src=["\']([^"\']+)',html,re.I)
  print('SCRIPT_COUNT',len(scripts))
  for x in scripts:
   su=urllib.parse.urljoin(r.url,x)
   if su not in all_scripts: all_scripts.append(su)
   print('SCRIPT',safe(su))
  for pat in ['__NEXT_DATA__','buildId','assetPrefix','stats/events','video']:
   if pat.lower() in html.lower():
    i=html.lower().find(pat.lower())
    print('HTML_CTX',pat,re.sub(r'\s+',' ',html[max(0,i-700):i+2500])[:3200])
 except Exception as e: print('PAGE_ERR',label,type(e).__name__,str(e)[:250])

print('\nUNIQUE_SCRIPTS',len(all_scripts))
for su in all_scripts:
 try:
  r=s.get(su,headers={'User-Agent':UA,'Referer':'https://www.nba.com/stats/events'},timeout=20)
  js=r.text
  hits=[t for t in TERMS if t.lower() in js.lower()]
  if hits:
   print('\nCHUNK',r.status_code,len(r.content),safe(r.url),'HITS',hits)
   for t in hits:
    c=ctx(js,t)
    if c: print('CTX',t,c)
 except Exception as e: print('SCRIPT_ERR',safe(su),type(e).__name__,str(e)[:180])

# Try Next build manifest routes if advertised in scripts.
for su in list(all_scripts):
 if '_buildManifest.js' in su or '_ssgManifest.js' in su:
  continue
# infer static roots/build ids from script paths and fetch manifests
roots=set()
for su in all_scripts:
 p=urllib.parse.urlsplit(su)
 m=re.search(r'(.+?/_next/static)/([^/]+)/',p.path)
 if m:
  roots.add((p.scheme+'://'+p.netloc+m.group(1),m.group(2)))
for root,bid in sorted(roots):
 for name in ['_buildManifest.js','_ssgManifest.js']:
  u=f'{root}/{bid}/{name}'
  try:
   r=s.get(u,headers={'User-Agent':UA,'Referer':'https://www.nba.com/stats/events'},timeout=15)
   print('\nMANIFEST',r.status_code,len(r.content),safe(u))
   txt=r.text
   if 'events' in txt.lower() or 'stats' in txt.lower():
    for t in ['stats/events','events','video']:
     c=ctx(txt,t)
     if c: print('MANIFEST_CTX',t,c)
  except Exception as e: print('MANIFEST_ERR',safe(u),type(e).__name__,str(e)[:150])
