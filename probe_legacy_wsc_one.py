from __future__ import annotations
import json, requests, xml.etree.ElementTree as ET

GID='0021300124'; EID=225
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36','Referer':'https://www.nba.com/','Origin':'https://www.nba.com','Accept':'application/json,text/plain,*/*','x-nba-stats-origin':'stats','x-nba-stats-token':'true','Accept-Language':'en-US,en;q=0.9'}
s=requests.Session(); s.headers.update(H)
out={'game_id':GID,'event_id':EID}
try:
 r=s.get('https://stats.nba.com/stats/videoevents',params={'GameEventID':EID,'GameID':GID},timeout=20)
 out['videoevents_status']=r.status_code; out['videoevents_bytes']=len(r.content); out['videoevents_head']=r.text[:500]
 print('videoevents',r.status_code,len(r.content),flush=True)
 if r.ok:
  j=r.json(); out['videoevents']=j
  vu=((j.get('resultSets') or {}).get('Meta') or {}).get('videoUrls') or []
  if not vu: raise RuntimeError('no videoUrls')
  uuid=vu[0].get('uuid'); out['uuid']=uuid; print('uuid',uuid,flush=True)
  x=s.get(f'https://secure.nba.com/video/wsc/league/{uuid}.secure.xml',timeout=20)
  out['xml_status']=x.status_code; out['xml_bytes']=len(x.content); out['xml_text']=x.text
  print('xml',x.status_code,len(x.content),flush=True)
  if x.ok:
   root=ET.fromstring(x.content)
   files=[(el.text or '').strip() for el in root.iter('file') if (el.text or '').strip()]
   out['files']=files
   for f in files: print('FILE',f,flush=True)
except Exception as e:
 out['error']=repr(e); print('ERROR',repr(e),flush=True)
open('legacy_wsc_one.json','w').write(json.dumps(out,indent=2))
