from __future__ import annotations
import json,re,time
from pathlib import Path
from urllib.parse import quote
import requests

UA='Mozilla/5.0 (compatible; historical-media-research/1.0)'
S=requests.Session(); S.headers.update({'User-Agent':UA})
GAME='0021300124'; EVENT='225'
CANDIDATES=[
 f'https://stats.nba.com/stats/videoevents?GameEventID={EVENT}&GameID={GAME}',
 f'https://stats.nba.com/stats/videoevents?GameID={GAME}&GameEventID={EVENT}',
 f'http://stats.nba.com/stats/videoevents?GameEventID={EVENT}&GameID={GAME}',
 f'http://stats.nba.com/stats/videoevents?GameID={GAME}&GameEventID={EVENT}',
 f'https://www.nba.com/video/games/warriors/2013/11/14/{GAME}-okc-gsw-play{EVENT}.nba/index.html',
 f'http://www.nba.com/video/games/warriors/2013/11/14/{GAME}-okc-gsw-play{EVENT}.nba/index.html',
 f'https://www.nba.com/video/games/warriors/2013/11/14/{GAME}-okc-gsw-play{EVENT}.nba/',
 f'http://www.nba.com/video/games/warriors/2013/11/14/{GAME}-okc-gsw-play{EVENT}.nba/',
 f'https://secure.nba.com/assets/amp/include/video/iframe.html?contentId=2013/11/14/{GAME}okcgswplay{EVENT}&team=',
 f'http://secure.nba.com/assets/amp/include/video/iframe.html?contentId=2013/11/14/{GAME}okcgswplay{EVENT}&team=',
]

def get(url,timeout=25):
 try:
  r=S.get(url,timeout=timeout,allow_redirects=True)
  return r,{'status':r.status_code,'final_url':r.url,'content_type':r.headers.get('content-type'),'bytes':len(r.content)}
 except Exception as e:return None,{'error':repr(e)}

def cdx(url):
 params={'url':url,'output':'json','fl':'timestamp,original,statuscode,mimetype,digest,length','filter':'statuscode:200','collapse':'digest'}
 try:
  r=S.get('https://web.archive.org/cdx/search/cdx',params=params,timeout=30)
  rec={'status':r.status_code,'url':r.url,'text_head':r.text[:1000]}
  if r.ok:
   try: rec['rows']=r.json()
   except Exception as e: rec['json_error']=repr(e)
  return rec
 except Exception as e:return {'error':repr(e),'target':url}

def extract(text):
 urls=[]; uuids=[]
 for u in re.findall(r'https?://[^\s"\'<>\\]+',text):
  if any(x in u.lower() for x in ('wsc','turner','.mp4','.m3u8','videoevents','secure.nba')) and u not in urls: urls.append(u)
 for x in re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',text):
  if x not in uuids: uuids.append(x)
 return {'urls':urls[:200],'uuids':uuids[:200]}

out={'game':GAME,'event':EVENT,'queries':[],'wildcards':[]}
for u in CANDIDATES:
 rec={'target':u,'cdx':cdx(u),'captures':[]}
 rows=rec['cdx'].get('rows') or []
 if len(rows)>1:
  for row in rows[1:][-5:]:
   ts,orig=row[0],row[1]
   # id_ asks Wayback for raw archived payload rather than rewritten page.
   wu=f'https://web.archive.org/web/{ts}id_/{orig}'
   r,meta=get(wu,timeout=35); cap={'timestamp':ts,'original':orig,'wayback_url':wu,**meta}
   if r is not None:
    cap['head']=r.text[:3000]; cap['extract']=extract(r.text)
   rec['captures'].append(cap)
 out['queries'].append(rec)
 print('\nTARGET',u,'rows',max(0,len(rows)-1), 'CDX',rec['cdx'].get('status'),rec['cdx'].get('error'))
 for cap in rec['captures']:
  print(' CAP',cap.get('timestamp'),cap.get('status'),cap.get('extract'))

# Broad archive lookups around the exact game/event identifiers.
for target in [
 f'www.nba.com/video/*/{GAME}*play{EVENT}*',
 f'nba.com/video/*/{GAME}*play{EVENT}*',
 f'stats.nba.com/stats/videoevents*{GAME}*{EVENT}*',
 f'secure.nba.com/*{GAME}*play{EVENT}*',
]:
 rec=cdx(target); out['wildcards'].append({'target':target,'cdx':rec})
 print('WILDCARD',target,'status',rec.get('status'),'rows',len(rec.get('rows') or []),'head',rec.get('text_head','')[:300])

Path('wayback_2013_nba_video_probe.json').write_text(json.dumps(out,indent=2))
