#!/usr/bin/env python3
import json, re, requests
from pathlib import Path

GAME_ID='0022100923'; EVENT_ID='352'
OUT=Path('outputs/morant_event352_asset_probe'); OUT.mkdir(parents=True, exist_ok=True)
headers={
 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138 Safari/537.36',
 'Referer':'https://www.nba.com/',
 'Origin':'https://www.nba.com',
 'Accept':'application/json, text/plain, */*',
 'Accept-Language':'en-US,en;q=0.9',
}
urls=[
 f'https://stats.nba.com/stats/videoeventsasset?GameEventID={EVENT_ID}&GameID={GAME_ID}',
 f'https://stats.nba.com/stats/videoeventsasset?GameEventID={EVENT_ID}&GameID={GAME_ID}&PlayerID=0&TeamID=0',
]
results=[]
for u in urls:
    try:
        r=requests.get(u,headers=headers,timeout=30)
        rec={'url':u,'status':r.status_code,'content_type':r.headers.get('content-type'),'text':r.text[:20000]}
        try: rec['json']=r.json()
        except Exception: pass
    except Exception as e:
        rec={'url':u,'error':repr(e)}
    results.append(rec)

# Test a few common direct formulations around the known Broadcast media id.
media='0001254773_1509kbps.mp4'
candidates=[
 f'https://lrmedia.nba.com/{media}',
 f'https://lrmedia.nba.com/CFSec/_definst_/{media}',
 f'https://lrmedia.nba.com/CFSec/_definst_/mp4:{media}',
 f'https://lrmedia.nba.com/ondemand/_definst_/mp4:{media}/playlist.m3u8',
 f'https://lrmedia.nba.com/ondemand/mp4:{media}/playlist.m3u8',
 f'https://lrmedia.nba.com/vod/_definst_/mp4:{media}/playlist.m3u8',
]
for u in candidates:
    try:
        r=requests.get(u,headers=headers,timeout=15,stream=True,allow_redirects=True)
        chunk=next(r.iter_content(256),b'')
        results.append({'candidate':u,'status':r.status_code,'final_url':r.url,'content_type':r.headers.get('content-type'),'content_length':r.headers.get('content-length'),'first_bytes_hex':chunk[:64].hex()})
    except Exception as e:
        results.append({'candidate':u,'error':repr(e)})

(OUT/'asset_probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
for x in results:
    print(json.dumps({k:v for k,v in x.items() if k not in ('text','json')},ensure_ascii=False),flush=True)
    if 'json' in x:
        print('JSON='+json.dumps(x['json'])[:10000],flush=True)
