#!/usr/bin/env python3
import json,hashlib,re,urllib.parse
from curl_cffi import requests

IMPERSONATE='chrome120'
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
STATS_HEADERS={
 'Accept':'application/json, text/plain, */*','Accept-Language':'en-US,en;q=0.9','Cache-Control':'no-cache',
 'Connection':'keep-alive','Origin':'https://www.nba.com','Pragma':'no-cache','Referer':'https://www.nba.com/',
 'Sec-Fetch-Dest':'empty','Sec-Fetch-Mode':'cors','Sec-Fetch-Site':'same-site','User-Agent':UA,
 'sec-ch-ua':'"Google Chrome";v="120", "Chromium";v="120", "Not)A;Brand";v="24"','sec-ch-ua-mobile':'?0','sec-ch-ua-platform':'"Windows"',
 'x-nba-stats-origin':'stats','x-nba-stats-token':'true'}
WEB_HEADERS={'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Accept-Language':'en-US,en;q=0.9','User-Agent':UA,'Referer':'https://www.nba.com/'}
EVENTS=[('2017','0021700015','438','2017-18'),('2019','0041800163','215','2018-19'),('2025','0022500375','608','2025-26')]

def safe(url):
 p=urllib.parse.urlsplit(url); return {'host':p.netloc,'path':p.path,'query_keys':sorted(urllib.parse.parse_qs(p.query).keys())}

def walk_urls(x,out):
 if isinstance(x,dict):
  for k,v in x.items():
   if isinstance(v,str) and v.startswith('http'): out.append((k,v))
   else: walk_urls(v,out)
 elif isinstance(x,list):
  for v in x: walk_urls(v,out)

s=requests.Session(impersonate=IMPERSONATE)
s.headers.update(STATS_HEADERS)
try:
 r=s.get('https://www.nba.com/',headers=WEB_HEADERS,timeout=20)
 print('PREWARM',r.status_code,len(r.content),'cookies',sorted(s.cookies.get_dict().keys()))
except Exception as e: print('PREWARM_ERROR',type(e).__name__,str(e)[:200])

for label,gid,eid,season in EVENTS:
 print('\n===',label,gid,eid,'===')
 for ep in ['videoevents','videoeventsasset']:
  url=f'https://stats.nba.com/stats/{ep}?GameEventID={eid}&GameID={gid}'
  try:
   r=s.get(url,timeout=25)
   print('API',ep,r.status_code,r.headers.get('content-type'),len(r.content),'final',safe(r.url))
   text=r.text
   if r.status_code==200:
    obj=r.json(); print('TOP',list(obj)[:10]); print('SNIP',json.dumps(obj)[:2200])
    urls=[]; walk_urls(obj,urls)
    for key,u in urls:
     print('MEDIA_FIELD',key,safe(u))
     try:
      mr=s.get(u,headers={'User-Agent':UA,'Referer':'https://www.nba.com/','Origin':'https://www.nba.com'},timeout=20,stream=True)
      chunk=next(mr.iter_content(chunk_size=65536),b'')
      print('MEDIA_FETCH',key,mr.status_code,mr.headers.get('content-type'),mr.headers.get('content-length'),'first',len(chunk),'sha256_first',hashlib.sha256(chunk).hexdigest() if chunk else None)
     except Exception as me: print('MEDIA_ERROR',key,type(me).__name__,str(me)[:180])
  except Exception as e: print('API_ERROR',ep,type(e).__name__,str(e)[:240])

 page=f'https://www.nba.com/stats/events?GameEventID={eid}&GameID={gid}&Season={season}&flag=1'
 try:
  r=s.get(page,headers=WEB_HEADERS,timeout=25)
  print('EVENT_PAGE',r.status_code,r.headers.get('content-type'),len(r.content),'final',safe(r.url))
  title=re.search(r'<title[^>]*>(.*?)</title>',r.text,re.I|re.S)
  print('EVENT_TITLE',re.sub(r'\s+',' ',title.group(1)).strip()[:180] if title else None)
  print('EVENT_MARKERS',[x for x in ['videoeventsasset','videoevents','.m3u8','lrmedia','videoAvailableFlag','Play Video'] if x.lower() in r.text.lower()])
 except Exception as e: print('EVENT_PAGE_ERROR',type(e).__name__,str(e)[:240])
