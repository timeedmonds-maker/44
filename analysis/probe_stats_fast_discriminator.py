#!/usr/bin/env python3
from curl_cffi import requests
import json,time
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
P='https://www.nba.com/stats/events'
s=requests.Session(impersonate='chrome120')
r=s.get('https://www.nba.com/',headers={'User-Agent':UA},timeout=10);print('PRE',r.status_code,len(r.content))
for label,g,e in [('2017','0021700015','438'),('2025','0022500375','608')]:
 u=f'https://stats.nba.com/stats/videoeventsasset?GameID={g}&GameEventID={e}'
 h={'User-Agent':UA,'Accept':'*/*','Origin':'https://www.nba.com','Referer':P}
 t=time.time()
 try:
  r=s.get(u,headers=h,timeout=8)
  print(label,'STATUS',r.status_code,'SECS',round(time.time()-t,2),'CT',r.headers.get('content-type'),'BYTES',len(r.content))
  if r.status_code==200:
   x=r.json();print(label,'JSON',json.dumps(x)[:7000])
 except Exception as ex:print(label,'ERR',type(ex).__name__,str(ex)[:250],'SECS',round(time.time()-t,2))
