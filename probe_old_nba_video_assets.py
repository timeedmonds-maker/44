from __future__ import annotations
import json, re, requests, urllib.parse
from pathlib import Path

EVENTS=[
('0021300124',225),('0021300868',209),('0041300157',129),('0041300232',145),('0041300235',300),('0041300235',398),
('0021801220',7),
]
H={
'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36',
'Accept':'application/json, text/plain, */*','Accept-Language':'en-US,en;q=0.9','Origin':'https://www.nba.com','Referer':'https://www.nba.com/','Connection':'keep-alive'
}

def walk(x,path='$'):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items(): out += walk(v,path+'.'+str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x): out += walk(v,path+f'[{i}]')
    elif isinstance(x,str):
        if 'http' in x.lower() or re.fullmatch(r'[0-9a-fA-F-]{32,40}',x or ''):
            out.append({'path':path,'value':x})
    return out

allout=[]
s=requests.Session(); s.headers.update(H)
for gid,eid in EVENTS:
    rec={'game_id':gid,'event_id':eid,'endpoints':{}}
    for endpoint in ('videoevents','videoeventsasset'):
        url='https://stats.nba.com/stats/'+endpoint
        try:
            r=s.get(url,params={'GameID':gid,'GameEventID':eid},timeout=60)
            e={'status':r.status_code,'url':r.url,'content_type':r.headers.get('content-type'),'bytes':len(r.content),'text_head':r.text[:1000]}
            if r.ok:
                try:
                    j=r.json(); e['json']=j; e['interesting']=walk(j)
                except Exception as exc:e['json_error']=repr(exc)
            rec['endpoints'][endpoint]=e
        except Exception as exc:rec['endpoints'][endpoint]={'error':repr(exc)}
    allout.append(rec)
Path('old_nba_video_asset_probe.json').write_text(json.dumps(allout,indent=2))
for rec in allout:
    print('\nEVENT',rec['game_id'],rec['event_id'])
    for ep,e in rec['endpoints'].items():
        print(ep,'status=',e.get('status'),'bytes=',e.get('bytes'),'error=',e.get('error'))
        for x in e.get('interesting',[]): print(' ',x['path'],x['value'])
