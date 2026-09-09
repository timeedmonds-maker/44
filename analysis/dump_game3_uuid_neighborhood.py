#!/usr/bin/env python3
import csv,json,time,urllib.parse
from curl_cffi import requests
GAME='0041800163'; TARGET=215
s=requests.Session(impersonate='chrome120')
headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131 Safari/537.36','Referer':'https://www.nba.com/'}
rows=[]
for ev in range(TARGET-15,TARGET+16):
    u='https://stats.nba.com/stats/videoeventsasset?'+urllib.parse.urlencode({'GameID':GAME,'GameEventID':ev})
    try:
        r=s.get(u,headers=headers,timeout=8)
        if r.status_code!=200:
            rows.append({'event_num':ev,'status':r.status_code}); continue
        d=r.json().get('resultSets') or {}; pl=d.get('playlist') or []; vu=(d.get('Meta') or {}).get('videoUrls') or []
        if pl and vu:
            rows.append({'event_num':ev,'status':200,'uuid':vu[0].get('uuid'),'description':pl[0].get('dsc'),'period':pl[0].get('p'),'gc':pl[0].get('gc')})
        else: rows.append({'event_num':ev,'status':200})
    except Exception as e: rows.append({'event_num':ev,'error':type(e).__name__})
    time.sleep(.08)
open('game3_uuid_neighborhood.json','w').write(json.dumps(rows,indent=2))
with open('game3_uuid_neighborhood.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['event_num','status','uuid','description','period','gc','error']); w.writeheader()
    for x in rows: w.writerow(x)
for x in rows:
    if x.get('uuid'): print(x['event_num'],x['uuid'],x.get('description',''))
