#!/usr/bin/env python3
import json,re,urllib.request,urllib.parse
from html.parser import HTMLParser

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
HEADERS={'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8','Referer':'https://www.nba.com/'}
TARGETS=[('2017','0021700015','438','2017-18'),('2019','0041800163','215','2018-19'),('2025','0022500375','608','2025-26')]

def safe(url):
    p=urllib.parse.urlsplit(url)
    return {'host':p.netloc,'path':p.path,'query_keys':sorted(urllib.parse.parse_qs(p.query).keys())}

def get(url,timeout=20):
    req=urllib.request.Request(url,headers=HEADERS)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return r.status,r.headers.get('content-type',''),r.geturl(),r.read()

for label,gid,eid,season in TARGETS:
    print('\n===',label,gid,eid,'===')
    urls=[
      f'https://api-hub-uat.nba.com/stats/events?GameEventID={eid}&GameID={gid}&Season={urllib.parse.quote(season)}&flag=1',
      f'https://api-hub-uat.nba.com/stats/videoevents?GameEventID={eid}&GameID={gid}',
      f'https://api-hub-uat.nba.com/stats/videoeventsasset?GameEventID={eid}&GameID={gid}',
    ]
    for u in urls:
        try:
            st,ct,final,body=get(u)
            print('FETCH',safe(u),'->',st,ct,'bytes',len(body),'final',safe(final))
            text=body.decode('utf-8','replace')
            if 'text/html' in ct or '<html' in text[:500].lower():
                title=re.search(r'<title[^>]*>(.*?)</title>',text,re.I|re.S)
                print('TITLE',re.sub(r'\s+',' ',title.group(1)).strip()[:200] if title else None)
                scripts=re.findall(r'<script[^>]+src=["\']([^"\']+)',text,re.I)
                print('SCRIPTS',len(scripts))
                for s in scripts[:80]:
                    print('SCRIPT',safe(urllib.parse.urljoin(final,s)))
                for needle in ['videoeventsasset','videoevents','lrmedia','clips.nba.com','.m3u8','videoAvailableFlag','Watch Replay','?watch']:
                    if needle.lower() in text.lower(): print('HTML_HAS',needle)
            else:
                try:
                    obj=json.loads(text)
                    print('JSON_KEYS',list(obj)[:20])
                    s=json.dumps(obj)
                    print('JSON_SNIP',s[:1800])
                except Exception:
                    print('BODY_SNIP',text[:1000].replace('\n',' '))
        except Exception as e:
            print('ERROR',safe(u),type(e).__name__,str(e)[:300])

# Static client discovery from accessible team page: look for chunks mentioning stats/events or video APIs.
seed='https://www.nba.com/thunder/game/0021700015-knicks-vs-thunder-oklahoma-city-ok-10-19-2017'
try:
    st,ct,final,body=get(seed)
    text=body.decode('utf-8','replace')
    print('\nSEED',st,len(body),safe(final))
    scripts=re.findall(r'<script[^>]+src=["\']([^"\']+)',text,re.I)
    for s in scripts:
        su=urllib.parse.urljoin(final,s)
        try:
            sst,sct,sfinal,sbody=get(su,10)
            js=sbody.decode('utf-8','replace')
            hits=[n for n in ['videoeventsasset','videoevents','stats/events','GameEventID','videoAvailableFlag','m3u8','lrmedia','videoUrls'] if n.lower() in js.lower()]
            if hits:
                print('CHUNK_HITS',safe(su),hits,'bytes',len(sbody))
                for n in hits:
                    i=js.lower().find(n.lower())
                    print('CTX',n,re.sub(r'\s+',' ',js[max(0,i-500):i+1500])[:2200])
        except Exception:
            pass
except Exception as e:
    print('SEED_ERROR',type(e).__name__,str(e)[:300])
