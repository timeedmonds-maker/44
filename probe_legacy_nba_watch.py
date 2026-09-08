from __future__ import annotations
import json,re,subprocess
from pathlib import Path
from urllib.parse import quote
import requests

EVENTS=[
 {'gid':'0021300124','eid':225,'date':'2013/11/14','away':'okc','home':'gsw','home_slug':'warriors'},
 {'gid':'0041300232','eid':145,'date':'2014/05/07','away':'lac','home':'okc','home_slug':'thunder'},
]
UA='Mozilla/5.0 (iPhone; CPU iPhone OS 11_0_1 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A402 Safari/604.1'
S=requests.Session(); S.headers.update({'User-Agent':UA,'Accept':'application/json,text/plain,*/*'})

def get(url,params=None,headers=None,timeout=30):
 try:
  r=S.get(url,params=params,headers=headers,timeout=timeout,allow_redirects=True)
  rec={'status':r.status_code,'url':r.url,'final_url':r.url,'content_type':r.headers.get('content-type'),'bytes':len(r.content),'text_head':r.text[:1000]}
  if r.ok:
   try: rec['json']=r.json()
   except Exception: pass
  return rec
 except Exception as e:return {'error':repr(e),'url':url,'params':params}

def media_urls(x):
 out=[]
 def walk(v,path='$'):
  if isinstance(v,dict):
   for k,z in v.items():walk(z,path+'.'+str(k))
  elif isinstance(v,list):
   for i,z in enumerate(v):walk(z,path+f'[{i}]')
  elif isinstance(v,str) and (v.startswith('http') or '.m3u8' in v or '.mp4' in v or '.xml' in v):
   out.append({'path':path,'value':v})
 walk(x); return out

def validate(url,out):
 try:
  p=subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-rw_timeout','30000000','-headers',f'User-Agent: {UA}\r\nReferer: https://www.nba.com/\r\n','-i',url,'-t','12','-map','0:v:0','-map','0:a:0?','-c','copy',str(out)],capture_output=True,text=True,timeout=60)
  if p.returncode:return {'ok':False,'ffmpeg':p.stderr[-1200:]}
  q=subprocess.run(['ffprobe','-v','error','-show_entries','stream=width,height:format=duration','-of','json',str(out)],capture_output=True,text=True,timeout=15)
  j=json.loads(q.stdout); streams=j.get('streams') or [{}]; dur=float((j.get('format') or {}).get('duration') or 0)
  return {'ok':q.returncode==0 and dur>=2,'duration':dur,'width':streams[0].get('width'),'height':streams[0].get('height'),'bytes':out.stat().st_size}
 except Exception as e:return {'ok':False,'error':repr(e)}

Path('legacy_watch_media').mkdir(exist_ok=True)
allout=[]
for e in EVENTS:
 gid,eid,date,away,home=e['gid'],e['eid'],e['date'],e['away'],e['home']
 stem=f'{gid}-{away}-{home}-play{eid}.nba'
 variants=[
  f'games/{e["home_slug"]}/{date}/{stem}',
  f'{date}/{stem}',
  f'{date}/{gid}{away}{home}play{eid}',
  f'{date}/{gid}-{away}-{home}-play{eid}',
 ]
 rec={'event':e,'variants':[]}
 for seo in variants:
  qrec=get('https://neulionscnbav2-a.akamaihd.net/solr/nbad_program/usersearch',params={'fl':'description,image,name,pid,releaseDate,runtime,tags,seoName','q':'seoName:'+seo,'wt':'json'})
  vr={'seo':seo,'solr':qrec,'docs':[]}
  try: docs=qrec['json']['response']['docs']
  except Exception: docs=[]
  for doc in docs:
   d={'doc':doc,'publishpoints':[]}
   pid=doc.get('pid')
   if pid is not None:
    for pp in [
      'https://watch.nba.com/service/publishpoint',
      'http://watch.nba.com/service/publishpoint',
    ]:
     pr=get(pp,params={'type':'video','format':'json','id':pid},headers={'User-Agent':UA},timeout=30)
     d['publishpoints'].append(pr)
   vr['docs'].append(d)
  rec['variants'].append(vr)
 # broader searches by game id and play token
 rec['broad']=[]
 for term in [gid,f'{gid}-{away}-{home}-play{eid}',f'{gid}{away}{home}play{eid}',f'play{eid}']:
  rec['broad'].append({'term':term,'result':get('https://neulionscnbav2-a.akamaihd.net/solr/nbad_program/usersearch',params={'fl':'description,image,name,pid,releaseDate,runtime,tags,seoName','q':term,'rows':50,'wt':'json'})})
 # old NBA page / watch / embed direct probes
 rec['pages']=[]
 urls=[
  f'https://www.nba.com/video/games/{e["home_slug"]}/{date}/{stem}/',
  f'https://watch.nba.com/video/games/{e["home_slug"]}/{date}/{stem}',
  f'https://watch.nba.com/video/{date}/{gid}{away}{home}play{eid}',
  f'https://secure.nba.com/assets/amp/include/video/iframe.html?contentId={quote(date+"/"+gid+away+home+"play"+str(eid),safe="/")}&team=',
 ]
 for u in urls: rec['pages'].append(get(u,timeout=30))
 # team API content-id variants
 rec['team_api']=[]
 token='internal|bb88df6b4c2244e78822812cecf1ee1b'
 for team in [e['home_slug']]:
  for cid in [date+'/'+gid+away+home+'play'+str(eid), 'games/'+e['home_slug']+'/'+date+'/'+stem]:
   ar=get(f'https://api.nba.net/2/{team}/video,imported_video,wsc/',params={'videoid':cid},headers={'accessToken':token,'User-Agent':UA},timeout=30)
   rec['team_api'].append({'team':team,'content_id':cid,'result':ar})
 # validate any media URL found in all structures
 candidates=[]
 for x in media_urls(rec):
  u=x['value']
  if any(z in u.lower() for z in ('.m3u8','.mp4')) and u not in candidates:candidates.append(u)
 rec['candidates']=candidates
 rec['validated']=[]
 for i,u in enumerate(candidates[:20]):
  v=validate(u,Path('legacy_watch_media')/f'{gid}_{eid}_{i}.mp4'); v['url']=u; rec['validated'].append(v)
  if v.get('ok'):break
 allout.append(rec)
 print('\nEVENT',gid,eid,'candidates',len(candidates))
 for vr in rec['variants']:
  try:n=vr['solr']['json']['response']['numFound']
  except Exception:n=None
  print(' SEO',vr['seo'],'found',n)
 print('GOOD',[x for x in rec['validated'] if x.get('ok')][:1])
Path('legacy_nba_watch_probe.json').write_text(json.dumps(allout,indent=2))
