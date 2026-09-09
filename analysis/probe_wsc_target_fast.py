#!/usr/bin/env python3
import concurrent.futures as cf
import json,re,urllib.parse
from curl_cffi import requests

UUID='995096ac-c2a1-b2fa-89ed-a345759b84be'
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131 Safari/537.36'
S=requests.Session(impersonate='chrome120')

def grab(url, timeout=8):
    try:
        r=S.get(url,headers={'User-Agent':UA,'Accept':'*/*'},timeout=timeout,allow_redirects=True)
        return {'url':url,'status':r.status_code,'bytes':len(r.content),'text':r.text[:12000]}
    except Exception as e:
        return {'url':url,'error':f'{type(e).__name__}: {e}'}

def cdx(pattern):
    q=urllib.parse.urlencode({'url':pattern,'output':'json','filter':'statuscode:200','fl':'timestamp,original,statuscode,digest','collapse':'urlkey','limit':'50'},safe='*:/')
    return grab('https://web.archive.org/cdx/search/cdx?'+q)

patterns=[]
for scheme in ('http','https'):
    patterns.append(f'{scheme}://secure.nba.com/video/wsc/league/{UUID}.secure.xml')
for date in ('2019/04/19','2019/04/20'):
    for host in ('nba.cdn.turner.com','ssl.cdn.turner.com','pmd.cdn.turner.com'):
        patterns.append(f'https://{host}/nba/big/nba/wsc/{date}/{UUID}*')

with cf.ThreadPoolExecutor(max_workers=8) as ex:
    wayback=list(ex.map(cdx,patterns))

# Probe current secure resolver too, only as a sanity check; it is expected to be fallback HTML today.
current=[]
for scheme in ('http','https'):
    current.append(grab(f'{scheme}://secure.nba.com/video/wsc/league/{UUID}.secure.xml',8))

# Common Crawl 2019 indexes, exact target URL variants, concurrently.
coll=grab('https://index.commoncrawl.org/collinfo.json',8)
cc=[]
if coll.get('status')==200:
    try:
        cols=[x for x in json.loads(coll['text']) if str(x.get('id','')).startswith('CC-MAIN-2019-')]
    except Exception:
        # collinfo text may be truncated by grab; fetch it once without truncation.
        try:
            rr=S.get('https://index.commoncrawl.org/collinfo.json',headers={'User-Agent':UA},timeout=8)
            cols=[x for x in rr.json() if str(x.get('id','')).startswith('CC-MAIN-2019-')]
        except Exception:
            cols=[]
    cc_tasks=[]
    for c in cols:
        idx=c['id']+'-index'
        for pat in [f'secure.nba.com/video/wsc/league/{UUID}.secure.xml',f'*.cdn.turner.com/nba/big/nba/wsc/2019/04/*/{UUID}*']:
            q=urllib.parse.urlencode({'url':pat,'output':'json'},safe='*:/')
            cc_tasks.append((idx,pat,'https://index.commoncrawl.org/'+idx+'?'+q))
    def one(task):
        idx,pat,url=task; r=grab(url,8); r['index']=idx; r['pattern']=pat; return r
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        cc=list(ex.map(one,cc_tasks))

ids=[]
for group in (wayback,current,cc):
    for r in group:
        ids += [int(x) for x in re.findall(r'\.nba_(\d+)_',r.get('text',''))]

report={'uuid':UUID,'wayback':wayback,'current_secure':current,'commoncrawl':cc,'asset_ids':sorted(set(ids))}
open('wsc_target_fast.json','w').write(json.dumps(report,indent=2))
print(json.dumps({'wayback_queries':len(wayback),'wayback_200':sum(x.get('status')==200 for x in wayback),'cc_queries':len(cc),'cc_200_with_bytes':sum(x.get('status')==200 and x.get('bytes',0)>0 for x in cc),'asset_ids':sorted(set(ids))},indent=2))
