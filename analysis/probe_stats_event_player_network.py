from __future__ import annotations
import json, re, urllib.parse
from playwright.sync_api import sync_playwright

EVENTS=[
 ('0021300124',225,'2013-14','2013'),
 ('0021700015',438,'2017-18','2017'),
 ('0041800163',215,'2018-19','2019'),
 ('0022500375',608,'2025-26','2025-control'),
]
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36'
KEYWORDS=('m3u8','.mp4','videoevents','videoeventsasset','videodetails','clips.nba.com','lrmedia','wsc','media')

def safe_url(url):
 p=urllib.parse.urlsplit(url)
 path=p.path
 # preserve endpoint/path identity, never query values/tokens
 return {'scheme':p.scheme,'host':p.hostname,'path':path,'has_query':bool(p.query)}

def interesting(url,headers=None):
 u=url.lower();ct=((headers or {}).get('content-type') or '').lower()
 return any(k in u for k in KEYWORDS) or ct.startswith('video/') or 'mpegurl' in ct or 'application/vnd.apple.mpegurl' in ct

with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=[
  '--disable-blink-features=AutomationControlled',
  '--autoplay-policy=no-user-gesture-required',
  '--disable-features=IsolateOrigins,site-per-process',
 ])
 ctx=browser.new_context(user_agent=UA,viewport={'width':1440,'height':1000},locale='en-US')
 for gid,eid,season,label in EVENTS:
  page=ctx.new_page();seen=[]
  def add(kind,url,status=None,headers=None,method=None):
   if not interesting(url,headers):return
   rec={'kind':kind,'url':safe_url(url)}
   if status is not None:rec['status']=status
   if method:rec['method']=method
   ct=((headers or {}).get('content-type') or '')
   if ct:rec['content_type']=ct
   key=json.dumps(rec,sort_keys=True)
   if key not in {json.dumps(x,sort_keys=True) for x in seen}:seen.append(rec)
  page.on('request',lambda req:add('request',req.url,method=req.method))
  def onresp(resp):
   try:add('response',resp.url,status=resp.status,headers=resp.headers)
   except Exception:pass
  page.on('response',onresp)
  url=f'https://www.nba.com/stats/events?CFID=&CFPARAMS=&GameEventID={eid}&GameID={gid}&Season={season}&flag=1'
  result={'label':label,'game_id':gid,'event_id':eid,'season':season,'target':safe_url(url),'navigation':None,'title':None,'media':[],'dom_video':[]}
  try:
   resp=page.goto(url,wait_until='domcontentloaded',timeout=90000)
   result['navigation']={'status':resp.status if resp else None,'url':safe_url(page.url)}
   page.wait_for_timeout(7000)
   # click likely play controls / video surface, then wait for media requests
   selectors=['button[aria-label*="play" i]','.vjs-big-play-button','button:has-text("Play")','video']
   for sel in selectors:
    try:
     loc=page.locator(sel).first
     if loc.count() and loc.is_visible():
      loc.click(timeout=2500,force=True);page.wait_for_timeout(6000);break
    except Exception:pass
   try:
    result['title']=page.title()
   except Exception:pass
   try:
    vals=page.eval_on_selector_all('video,video source','els=>els.map(e=>({src:e.src||null,currentSrc:e.currentSrc||null,type:e.type||null})).filter(x=>x.src||x.currentSrc)')
    result['dom_video']=[{'src':safe_url(x['src']) if x.get('src') else None,'currentSrc':safe_url(x['currentSrc']) if x.get('currentSrc') else None,'type':x.get('type')} for x in vals]
   except Exception:pass
  except Exception as e:
   result['error']=type(e).__name__+': '+str(e)[:240]
  result['media']=seen
  print('EVENT_NETWORK',json.dumps(result,sort_keys=True),flush=True)
  page.close()
 ctx.close();browser.close()
