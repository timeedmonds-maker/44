from __future__ import annotations
import hashlib, json, re, subprocess, tempfile, urllib.error, urllib.parse, urllib.request
from pathlib import Path

EVENTS=[('0021700015',438,'2017'),('0041800163',215,'2019'),('0022500375',608,'2025-control')]
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://www.nba.com/','Origin':'https://www.nba.com','Accept':'application/json, text/plain, */*'}
MH={'User-Agent':UA,'Referer':'https://www.nba.com/','Accept':'*/*'}
PLACEHOLDER=['0000180039fc39fc38c839fc39fc10000000','0000180019fc39fc38c819fc39fc18000000','0000100011fc11fc18ec11fc19fc10000000']

def get(url,headers=MH,timeout=15,limit=None):
 req=urllib.request.Request(url,headers=headers)
 with urllib.request.urlopen(req,timeout=timeout) as r:
  data=r.read() if limit is None else r.read(limit)
  return r.status,r.headers.get('content-type',''),r.geturl(),data

def safe(url):
 p=urllib.parse.urlsplit(url);return {'host':p.hostname,'path':p.path,'query':bool(p.query)}

def resolve(gid,eid):
 q=urllib.parse.urlencode({'GameID':gid,'GameEventID':eid,'LeagueID':'00'})
 _,_,_,raw=get('https://stats.gleague.nba.com/stats/videoeventsasset?'+q,H,20)
 j=json.loads(raw);v=j['resultSets']['Meta']['videoUrls'][0]
 d=j['resultSets']['playlist'][0]
 return v,d

def ahash(path,t):
 p=subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',str(path),'-vf','scale=16:9,format=gray','-frames:v','1','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True)
 if p.returncode or len(p.stdout)!=144:return ''
 vals=list(p.stdout);avg=sum(vals)/len(vals);bits=''.join('1' if x>=avg else '0' for x in vals);return f'{int(bits,2):036x}'

def ham(a,b):
 return (int(a,16)^int(b,16)).bit_count() if a and b and len(a)==len(b) else 999

def qa_mp4(url):
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'x.mp4'
  try:
   _,ct,final,data=get(url,MH,25)
   p.write_bytes(data)
   pr=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height','-select_streams','v:0','-of','json',str(p)],capture_output=True,text=True)
   if pr.returncode:return {'ok':False,'reason':'ffprobe','bytes':len(data),'content_type':ct,'final':safe(final)}
   j=json.loads(pr.stdout);dur=float(j.get('format',{}).get('duration') or 0);s=j.get('streams',[{}])[0]
   fps=[ahash(p,max(.25,dur*f)) for f in (.25,.5,.75)];dist=[ham(a,b) for a,b in zip(fps,PLACEHOLDER)]
   ph=all(x<=12 for x in dist)
   return {'ok':dur>=2 and all(fps) and not ph,'duration':dur,'width':s.get('width'),'height':s.get('height'),'bytes':len(data),'content_type':ct,'final':safe(final),'placeholder_like':ph,'placeholder_distances':dist,'sha256':hashlib.sha256(data).hexdigest()}
  except Exception as e:return {'ok':False,'error':type(e).__name__+': '+str(e)[:160]}

def probe_candidate(label,url):
 try:
  status,ct,final,data=get(url,MH,12,256*1024)
  text=data.decode('utf-8','replace')
  return {'label':label,'status':status,'content_type':ct,'final':safe(final),'bytes_read':len(data),'is_m3u8':text.lstrip().startswith('#EXTM3U'),'prefix':text[:32].replace('\n','\\n') if 'text' in ct or 'mpegurl' in ct else None}
 except urllib.error.HTTPError as e:return {'label':label,'http_error':e.code}
 except Exception as e:return {'label':label,'error':type(e).__name__+': '+str(e)[:140]}

for gid,eid,label in EVENTS:
 print('\nEVENT',label,gid,eid,flush=True)
 try:v,d=resolve(gid,eid)
 except Exception as e:print('RESOLVE_FAIL',type(e).__name__,str(e)[:180],flush=True);continue
 uuid=v.get('uuid');base=v.get('lurl') or v.get('murl') or v.get('surl')
 print('IDENTITY',json.dumps({'uuid':uuid,'description':d.get('dsc'),'base':safe(base) if base else None},sort_keys=True),flush=True)
 if base:
  print('ARCHIVE_QA',json.dumps(qa_mp4(base),sort_keys=True),flush=True)
  p=urllib.parse.urlsplit(base);stem=re.sub(r'_(1280x720|960x540|320x180)\.mp4$','',p.path)
  directory=p.path.rsplit('/',1)[0]
  candidates=[
   ('uuid.m3u8',urllib.parse.urlunsplit((p.scheme,p.netloc,stem+'.m3u8','',''))),
   ('uuid_1280x720.m3u8',urllib.parse.urlunsplit((p.scheme,p.netloc,stem+'_1280x720.m3u8','',''))),
   ('uuid_master.m3u8',urllib.parse.urlunsplit((p.scheme,p.netloc,stem+'_master.m3u8','',''))),
   ('event_master.m3u8',urllib.parse.urlunsplit((p.scheme,p.netloc,directory+'/master.m3u8','',''))),
   ('event_playlist.m3u8',urllib.parse.urlunsplit((p.scheme,p.netloc,directory+'/playlist.m3u8','',''))),
   ('uuid_dir_master.m3u8',urllib.parse.urlunsplit((p.scheme,p.netloc,directory+'/'+uuid+'/master.m3u8','',''))),
  ]
  for name,u in candidates:print('HLS_CANDIDATE',json.dumps(probe_candidate(name,u),sort_keys=True),flush=True)
 # Unsigned Wowza UUID naming check; useful only to discover whether the object exists.
 if uuid:
  for suffix in (f'mp4:{uuid}_1280x720.mp4/playlist.m3u8',f'mp4:{uuid}_960x540.mp4/playlist.m3u8'):
   u='https://lrmedia.nba.com/CFSec/_definst_/'+suffix
   print('LR_UUID',json.dumps(probe_candidate(suffix.split('/')[0],u),sort_keys=True),flush=True)
