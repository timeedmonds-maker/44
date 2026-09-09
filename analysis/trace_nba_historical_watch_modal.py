from __future__ import annotations
import asyncio,json,re,urllib.parse
from playwright.async_api import async_playwright

PAGES=[
 ('2017','https://www.nba.com/nuggets/game/0021601198-thunder-vs-nuggets-denver-co-04-09-2017'),
 ('2019','https://www.nba.com/lakers/game/0021800562')]
TERMS=('watch','replay','video','stream','media','playback','entitle','bam','m3u8','mpd','dash','league','program','content-api','api-hub','nba-cdn','akamai','brightcove')
SENSITIVE=('token','auth','authorization','jwt','sig','signature','key','cookie','session','access_token','refresh_token')

def safe(url):
 p=urllib.parse.urlsplit(url)
 keys=[]
 try:keys=sorted({k for k,v in urllib.parse.parse_qsl(p.query,keep_blank_values=True) if k.lower() not in SENSITIVE})
 except Exception:pass
 return {'host':p.hostname,'path':p.path,'query_keys':keys}
def interesting(url):return any(t in url.lower() for t in TERMS)
def clean_post(post):
 if not post:return None
 s=post[:1000]
 for k in SENSITIVE:
  s=re.sub(r'(?i)(["\']?'+re.escape(k)+r'["\']?\s*[:=]\s*)[^,&}\s]+',r'\1[MASKED]',s)
 return s

async def trace_one(context,label,url):
 page=await context.new_page(); events=[]
 page.on('request',lambda req: events.append(('REQ',req.method,req.resource_type,req.url,req.post_data)))
 async def resp(r):
  if interesting(r.url):
   try:ct=(await r.all_headers()).get('content-type')
   except Exception:ct=None
   events.append(('RESP',r.status,r.request.resource_type,r.url,ct))
 page.on('response',resp)
 print('\n===',label,'BASE ===',flush=True)
 try:
  nav=await page.goto(url,wait_until='domcontentloaded',timeout=45000)
  print('NAV',nav.status if nav else None,'title',await page.title(),flush=True)
  await page.wait_for_timeout(4500)
 except Exception as e:print('NAV_ERR',type(e).__name__,str(e)[:160],flush=True)
 # Capture current replay href if present.
 for selector in ('[data-testid="watch-replay-cta"]','a:has-text("Watch Replay")','button:has-text("Watch Replay")'):
  try:
   loc=page.locator(selector);n=await loc.count()
   if n:
    el=loc.first
    print('REPLAY_ELEMENT',selector,'href',await el.get_attribute('href'),'visible',await el.is_visible(),flush=True)
    before=len(events)
    if await el.is_visible():
     try:
      await el.click(timeout=8000)
      await page.wait_for_timeout(8000)
      print('AFTER_CLICK_URL',safe(page.url),'title',await page.title(),flush=True)
     except Exception as e:print('CLICK_ERR',type(e).__name__,str(e)[:160],flush=True)
    for ev in events[before:]:
     if interesting(ev[3]) or (ev[0]=='REQ' and ev[1]!='GET'):
      if ev[0]=='REQ':print('AFTER_REQ',json.dumps({'method':ev[1],'type':ev[2],**safe(ev[3]),'post':clean_post(ev[4])},sort_keys=True),flush=True)
      else:print('AFTER_RESP',json.dumps({'status':ev[1],'type':ev[2],**safe(ev[3]),'ct':ev[4]},sort_keys=True),flush=True)
    break
  except Exception as e:pass
 # Try explicit ?watch entry as NBA's component does.
 explicit=url+('&' if '?' in url else '?')+'watch'
 p2=await context.new_page(); ev2=[]
 p2.on('request',lambda req: ev2.append(('REQ',req.method,req.resource_type,req.url,req.post_data)))
 async def resp2(r):
  if interesting(r.url):
   try:ct=(await r.all_headers()).get('content-type')
   except Exception:ct=None
   ev2.append(('RESP',r.status,r.request.resource_type,r.url,ct))
 p2.on('response',resp2)
 print('EXPLICIT',explicit,flush=True)
 try:
  nav=await p2.goto(explicit,wait_until='domcontentloaded',timeout=45000)
  await p2.wait_for_timeout(9000)
  print('EXPLICIT_NAV',nav.status if nav else None,'url',safe(p2.url),'title',await p2.title(),flush=True)
  # show likely modal/video DOM only
  for sel in ('video','source','[role="dialog"]','[data-testid*="watch"]','[data-testid*="video"]'):
   try:
    loc=p2.locator(sel);n=min(await loc.count(),20)
    vals=[]
    for i in range(n):
     el=loc.nth(i); rec={'sel':sel}
     for a in ('src','href','data-src','data-testid','aria-label'):
      v=await el.get_attribute(a)
      if v: rec[a]=v if not v.startswith('http') else safe(v)
     try:
      txt=re.sub(r'\s+',' ',(await el.inner_text()).strip())[:220]
      if txt:rec['text']=txt
     except Exception:pass
     vals.append(rec)
    if vals:print('DOM',json.dumps(vals,ensure_ascii=False),flush=True)
   except Exception:pass
 except Exception as e:print('EXPLICIT_ERR',type(e).__name__,str(e)[:160],flush=True)
 seen=set()
 for ev in ev2:
  if ev[3] in seen:continue
  seen.add(ev[3])
  if interesting(ev[3]) or (ev[0]=='REQ' and ev[1]!='GET'):
   if ev[0]=='REQ':print('WATCH_REQ',json.dumps({'method':ev[1],'type':ev[2],**safe(ev[3]),'post':clean_post(ev[4])},sort_keys=True),flush=True)
   else:print('WATCH_RESP',json.dumps({'status':ev[1],'type':ev[2],**safe(ev[3]),'ct':ev[4]},sort_keys=True),flush=True)
 await p2.close();await page.close()

async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled'])
  context=await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152.0.0.0 Safari/537.36',viewport={'width':1440,'height':1000},locale='en-US')
  for label,url in PAGES: await trace_one(context,label,url)
  await browser.close()
asyncio.run(main())
