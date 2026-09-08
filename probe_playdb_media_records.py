from __future__ import annotations
import json, requests
from pathlib import Path

BASE='https://www.nbaplaydb.com'
H={'User-Agent':'Mozilla/5.0','Accept':'application/json,*/*','Content-Type':'application/json','Origin':BASE,'Referer':BASE+'/search?actionplayer=Steven%20Adams&assistpersonname=Russell%20Westbrook&query=dunk'}
s=requests.Session(); s.headers.update(H)

payload={
 'requests':[{'indexName':'nba-plays','params':{
   'facetFilters':[['actionplayer:Steven Adams'],['assistpersonname:Russell Westbrook']],
   'facets':['*'],'hitsPerPage':500,'maxValuesPerFacet':100,'page':0,'query':'dunk'
 }}],
 'method':'search'
}
r=s.post(BASE+'/api/search',json=payload,timeout=30); r.raise_for_status(); j=r.json(); hits=j['results'][0]['hits']
hits=sorted(hits,key=lambda x:(x.get('gamedate',''),x.get('gameid',''),int(x.get('actionnumber') or 0)))
print('COUNT',len(hits),'OLDEST',hits[0].get('gamedate'),hits[0].get('description'),'NEWEST',hits[-1].get('gamedate'))

# Oldest 20 plus known modern controls and exact first PBP event if present.
sel=hits[:20]+hits[-5:]
ids=[x['id'] for x in sel]
for chunk_start in range(0,len(ids),20):
    chunk=ids[chunk_start:chunk_start+20]
    u=BASE+'/api/getPlaysByIds'
    rr=s.get(u,params={'playIDs':','.join(chunk),'source':'standard','league':'nba'},timeout=30)
    print('GETPLAYS',rr.status_code,rr.url)
    print(rr.text[:5000])
    if rr.ok:
        data=rr.json()
        for p in data.get('plays',[]):
            print('PLAY',json.dumps({k:p.get(k) for k in ['id','gameid','actionnumber','gamedate','description','videourl','hasVideo','source','league']},sort_keys=True))
    else:
        data={'status':rr.status_code,'text':rr.text}
    for p in data.get('plays',[]) if isinstance(data,dict) else []:
        # Inspect candidate media URLs without downloading full files.
        vu=p.get('videourl')
        if vu:
            try:
                hr=s.get(vu,headers={'User-Agent':'Mozilla/5.0','Referer':BASE+'/'},stream=True,timeout=20,allow_redirects=True)
                print('MEDIAHEAD',p.get('id'),hr.status_code,hr.url,hr.headers.get('content-type'),hr.headers.get('content-length'))
                hr.close()
            except Exception as e: print('MEDIAERR',p.get('id'),repr(e))

out={'count':len(hits),'oldest':hits[:40],'newest':hits[-10:]}
# Get all selected records in a second compact pass for artifact.
records=[]
for chunk_start in range(0,len(ids),20):
    chunk=ids[chunk_start:chunk_start+20]
    rr=s.get(BASE+'/api/getPlaysByIds',params={'playIDs':','.join(chunk),'source':'standard','league':'nba'},timeout=30)
    if rr.ok: records.extend(rr.json().get('plays',[]))
out['selected_records']=records
Path('playdb_media_records.json').write_text(json.dumps(out,indent=2))
