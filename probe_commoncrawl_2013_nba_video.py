from __future__ import annotations
import gzip,json,re
from pathlib import Path
from urllib.parse import quote
import requests

S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 (historical-media-research/1.0)'})
GAME='0021300124'; EVENT='225'
TARGETS=[
 f'www.nba.com/video/games/warriors/2013/11/14/{GAME}-okc-gsw-play{EVENT}.nba/*',
 f'nba.com/video/games/warriors/2013/11/14/{GAME}-okc-gsw-play{EVENT}.nba/*',
 f'secure.nba.com/assets/amp/include/video/iframe.html*{GAME}*play{EVENT}*',
 f'stats.nba.com/stats/videoevents*{GAME}*{EVENT}*',
]

def extract(text):
 urls=[]; uuids=[]
 for u in re.findall(r'https?://[^\s"\'<>\\]+',text):
  if any(x in u.lower() for x in ('wsc','turner','.mp4','.m3u8','secure.nba','videoevents')) and u not in urls: urls.append(u)
 for x in re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',text):
  if x not in uuids: uuids.append(x)
 return {'urls':urls[:200],'uuids':uuids[:200]}

def fetch_record(row):
 fn=row.get('filename'); off=int(row.get('offset',0)); length=int(row.get('length',0))
 if not fn or length<=0:return {'error':'no range metadata'}
 u='https://data.commoncrawl.org/'+fn
 try:
  r=S.get(u,headers={'Range':f'bytes={off}-{off+length-1}'},timeout=30)
  rec={'status':r.status_code,'url':u,'bytes':len(r.content)}
  raw=r.content
  try: raw=gzip.decompress(raw)
  except Exception: pass
  text=raw.decode('utf-8','replace')
  rec['head']=text[:3000]; rec['extract']=extract(text)
  return rec
 except Exception as e:return {'error':repr(e),'url':u}

out={'game':GAME,'event':EVENT,'collections':[]}
try:
 cols=S.get('https://index.commoncrawl.org/collinfo.json',timeout=20).json()
except Exception as e:
 raise SystemExit('collinfo failed '+repr(e))
# Focus on captures around when the page existed, with a few later crawls in case URLs persisted.
chosen=[]
for c in cols:
 cid=c.get('id','')
 m=re.search(r'(20\d{2})',cid)
 yr=int(m.group(1)) if m else 0
 if 2013 <= yr <= 2016: chosen.append(c)
print('COLLECTIONS',[(c.get('id'),c.get('name')) for c in chosen])

for c in chosen:
 crec={'id':c.get('id'),'name':c.get('name'),'queries':[]}
 endpoint=c.get('cdx-api') or c.get('index')
 if not endpoint:
  # Current collinfo commonly exposes cdx-api.
  endpoint='https://index.commoncrawl.org/'+c['id']+'-index'
 for target in TARGETS:
  try:
   r=S.get(endpoint,params={'url':target,'output':'json','filter':'status:200'},timeout=20)
   qr={'target':target,'status':r.status_code,'url':r.url,'text_head':r.text[:1000],'rows':[]}
   if r.ok:
    for line in r.text.splitlines():
     try: qr['rows'].append(json.loads(line))
     except Exception: pass
   print('QUERY',c.get('id'),target,'status',r.status_code,'rows',len(qr['rows']))
   qr['captures']=[]
   for row in qr['rows'][:8]:
    cap={'row':row,'record':fetch_record(row)}; qr['captures'].append(cap)
    ex=cap['record'].get('extract',{})
    if ex.get('uuids') or ex.get('urls'): print(' HIT',row.get('timestamp'),row.get('url'),ex)
   crec['queries'].append(qr)
  except Exception as e:
   crec['queries'].append({'target':target,'error':repr(e)})
 out['collections'].append(crec)
Path('commoncrawl_2013_nba_video_probe.json').write_text(json.dumps(out,indent=2))
