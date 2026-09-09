#!/usr/bin/env python3
import re,json,urllib.parse,hashlib
from curl_cffi import requests
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
PAGE='https://www.nba.com/stats/events?GameEventID=438&GameID=0021700015&Season=2017-18&flag=1'
EVENTS=[('2017','0021700015','438'),('2019','0041800163','215'),('2025','0022500375','608')]

def safe(u):
 p=urllib.parse.urlsplit(u); return {'host':p.netloc,'path':p.path,'query_keys':sorted(urllib.parse.parse_qs(p.query).keys())}

def contexts(js,needle,span=1000):
 out=[]; pos=0
 while len(out)<12:
  i=js.find(needle,pos)
  if i<0: break
  out.append(re.sub(r'\s+',' ',js[max(0,i-span):i+span]))
  pos=i+len(needle)
 return out

s=requests.Session(impersonate='chrome120')
pre=s.get('https://www.nba.com/',headers={'User-Agent':UA},timeout=20)
print('PREWARM',pre.status_code,len(pre.content),'cookies',sorted(s.cookies.get_dict().keys()))
p=s.get(PAGE,headers={'User-Agent':UA,'Referer':'https://www.nba.com/'},timeout=20)
print('PAGE',p.status_code,len(p.content),safe(p.url))
scripts=[urllib.parse.urljoin(p.url,x) for x in re.findall(r'<script[^>]+src=["\']([^"\']+)',p.text,re.I)]
for u in scripts:
 try:
  r=s.get(u,headers={'User-Agent':UA,'Referer':PAGE},timeout=15)
  if r.status_code!=200: continue
  js=r.text
  if 'bL' in js or 'stats.nba.com' in js or 'videoeventsasset' in js:
   for needle in ['bL:', 'bL=', 'stats.nba.com', 'https://stats', '/stats/', 'videoeventsasset']:
    for c in contexts(js,needle,800):
     print('CONST_CTX',safe(u),needle,c[:2200])
 except Exception: pass

variants=[
 ('minimal',{}),
 ('browser',{'User-Agent':UA,'Accept':'*/*','Origin':'https://www.nba.com','Referer':PAGE}),
 ('json',{'User-Agent':UA,'Accept':'application/json, text/plain, */*','Origin':'https://www.nba.com','Referer':PAGE}),
]
for label,gid,eid in EVENTS:
 print('\n===',label,gid,eid,'===')
 url=f'https://stats.nba.com/stats/videoeventsasset?GameID={gid}&GameEventID={eid}'
 for vn,h in variants:
  try:
   r=s.get(url,headers=h,timeout=30)
   print('FETCH',vn,r.status_code,r.headers.get('content-type'),len(r.content),safe(r.url))
   if r.status_code==200:
    try:
     obj=r.json(); print('JSON',json.dumps(obj)[:5000])
     raw=json.dumps(obj)
     urls=re.findall(r'https?://[^"\\\s]+',raw)
     for media in urls[:30]:
      print('URL',safe(media))
      if any(x in media for x in ['.mp4','.m3u8','playlist']):
       try:
        mr=s.get(media,headers={'User-Agent':UA,'Referer':PAGE,'Origin':'https://www.nba.com'},timeout=20,stream=True)
        chunk=next(mr.iter_content(chunk_size=65536),b'')
        print('MEDIA',mr.status_code,mr.headers.get('content-type'),mr.headers.get('content-length'),'first',len(chunk),'sha',hashlib.sha256(chunk).hexdigest() if chunk else None)
       except Exception as me: print('MEDIA_ERR',type(me).__name__,str(me)[:160])
    except Exception as je: print('JSON_ERR',type(je).__name__,r.text[:500])
  except Exception as e: print('FETCH_ERR',vn,type(e).__name__,str(e)[:220])
