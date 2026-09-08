from __future__ import annotations
import json
from pathlib import Path
import requests

EVENTS=[
 {'gid':'0021300124','eid':225,'date':'2013/11/14','away':'okc','home':'gsw','home_slug':'warriors'},
 {'gid':'0041300232','eid':145,'date':'2014/05/07','away':'lac','home':'okc','home_slug':'thunder'},
 {'gid':'0021801220','eid':7,'date':'2019/04/10','away':'ind','home':'atl','home_slug':'hawks'},
]
URL='https://neulionscnbav2-a.akamaihd.net/solr/nbad_program/usersearch'
H={'User-Agent':'Mozilla/5.0','Accept':'application/json,*/*'}
S=requests.Session(); S.headers.update(H)

def q(query):
 try:
  r=S.get(URL,params={'fl':'description,image,name,pid,releaseDate,runtime,tags,seoName','q':query,'rows':100,'wt':'json'},timeout=8)
  x={'status':r.status_code,'url':r.url,'bytes':len(r.content),'head':r.text[:500]}
  if r.ok:
   j=r.json(); x['numFound']=j.get('response',{}).get('numFound'); x['docs']=j.get('response',{}).get('docs',[])
  return x
 except Exception as e:return {'error':repr(e),'query':query}

out=[]
for e in EVENTS:
 gid,eid,date,away,home=e['gid'],e['eid'],e['date'],e['away'],e['home']
 stem=f'{gid}-{away}-{home}-play{eid}.nba'
 seos=[
  f'games/{e["home_slug"]}/{date}/{stem}',
  f'{date}/{stem}',
  f'{date}/{gid}{away}{home}play{eid}',
  f'{date}/{gid}-{away}-{home}-play{eid}',
  f'games/{e["home_slug"]}/{date}/{gid}-{away}-{home}-play{eid}',
 ]
 rec={'event':e,'exact':[],'broad':[]}
 for seo in seos: rec['exact'].append({'seo':seo,'result':q('seoName:"'+seo+'"')})
 for term in [gid,f'{gid}-{away}-{home}-play{eid}',f'{gid}{away}{home}play{eid}',f'"{gid}" AND "play{eid}"']:
  rec['broad'].append({'term':term,'result':q(term)})
 out.append(rec)
 print('\nEVENT',gid,eid)
 for x in rec['exact']: print('EXACT',x['seo'],x['result'].get('numFound'),x['result'].get('error'))
 for x in rec['broad']: print('BROAD',x['term'],x['result'].get('numFound'),x['result'].get('error'))
 for bucket in ('exact','broad'):
  for x in rec[bucket]:
   for d in x['result'].get('docs',[])[:10]: print(' DOC',d.get('pid'),d.get('seoName'),d.get('name'))
Path('legacy_nba_watch_probe.json').write_text(json.dumps(out,indent=2))
