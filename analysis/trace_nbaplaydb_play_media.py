from __future__ import annotations
import asyncio, json, re, urllib.parse
from playwright.async_api import async_playwright

PLAYS=[
 ('2017','https://www.nbaplaydb.com/plays/VhbPgkcUui3/2017-04-09-thunder-vs-nuggets-steven-adams-2pt-video'),
 ('2019','https://www.nbaplaydb.com/plays/vday_Lm0axe/2019-01-02-thunder-vs-lakers-steven-adams-2pt-video'),
]

def safe(url):
 p=urllib.parse.urlsplit(url)
 q=urllib.parse.parse_qsl(p.query,keep_blank_values=True)
 return {'host':p.hostname,'path':p.path,'query_keys':sorted({k for k,v in q})}

def interesting(url):
 u=url.lower()
 return any(x in u for x in ('nba.com','video','clip','media','m3u8','mp4','merge','download','stream','playurl','videourl','video-url','pbp'))

async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled'])
  context=await browser.new_context(
   user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36',
   viewport={'width':1440,'height':1100},
   locale='en-US'
  )
  for label,url in PLAYS:
   page=await context.new_page()
   reqs=[]; resps=[]
   page.on('request',lambda req: reqs.append((req.method,req.resource_type,req.url,req.post_data)))
   async def on_response(resp):
    try:
     if interesting(resp.url):
      resps.append((resp.status,resp.request.resource_type,resp.url,resp.headers.get('content-type')))
    except Exception:pass
   page.on('response',on_response)
   print('\nPLAY',label,url,flush=True)
   try:
    nav=await page.goto(url,wait_until='domcontentloaded',timeout=45000)
    print('NAV',nav.status if nav else None,'title',await page.title(),flush=True)
    await page.wait_for_timeout(6000)
   except Exception as e:print('NAV_ERROR',type(e).__name__,str(e)[:180],flush=True)
   # DOM inspection
   try:
    text=(await page.locator('body').inner_text())[:12000]
    print('BODY',re.sub(r'\s+',' ',text)[:5000],flush=True)
   except Exception:pass
   for sel in ('video','source','a','button'):
    try:
     els=page.locator(sel); n=min(await els.count(),80)
     vals=[]
     for i in range(n):
      el=els.nth(i)
      rec={'tag':sel}
      for attr in ('src','href','data-src','data-url','data-video','download','aria-label','title'):
       v=await el.get_attribute(attr)
       if v:rec[attr]=v
      try:
       t=(await el.inner_text()).strip()
       if t:rec['text']=re.sub(r'\s+',' ',t)[:160]
      except Exception:pass
      blob=' '.join(str(x) for x in rec.values()).lower()
      if sel in ('video','source') or any(x in blob for x in ('play','download','clip','video','watch')):vals.append(rec)
     if vals: print('DOM_'+sel.upper(),json.dumps(vals[:40],ensure_ascii=False),flush=True)
    except Exception as e:pass
   # Click likely play controls/buttons one at a time, recording new traffic.
   candidates=[]
   for sel in ('button','a'):
    try:
     n=min(await page.locator(sel).count(),100)
     for i in range(n):
      el=page.locator(sel).nth(i)
      try:t=(await el.inner_text()).strip()
      except Exception:t=''
      aria=(await el.get_attribute('aria-label')) or ''
      title=(await el.get_attribute('title')) or ''
      href=(await el.get_attribute('href')) or ''
      blob=' '.join((t,aria,title,href)).lower()
      if any(k in blob for k in ('play','watch','download','video','clip')):
       candidates.append((sel,i,t[:80],aria[:80],title[:80],href[:180]))
    except Exception:pass
   print('CLICK_CANDIDATES',json.dumps(candidates[:30],ensure_ascii=False),flush=True)
   for sel,i,t,aria,title,href in candidates[:8]:
    before=len(reqs)
    try:
     el=page.locator(sel).nth(i)
     if await el.is_visible():
      await el.click(timeout=4000)
      await page.wait_for_timeout(4000)
      print('CLICKED',json.dumps({'sel':sel,'i':i,'text':t,'href':href}),flush=True)
      if len(reqs)>before:
       new=reqs[before:]
       for method,typ,u,post in new:
        if interesting(u) or method!='GET':
         print(' AFTER_REQ',json.dumps({'method':method,'type':typ,**safe(u),'post':post[:300] if post else None},sort_keys=True),flush=True)
    except Exception as e:print('CLICK_ERR',sel,i,type(e).__name__,str(e)[:100],flush=True)
   # Dump all interesting initial traffic, sanitized.
   seen=set()
   for method,typ,u,post in reqs:
    key=(method,u)
    if key in seen:continue
    seen.add(key)
    if interesting(u) or method!='GET':
     print('REQ',json.dumps({'method':method,'type':typ,**safe(u),'post':post[:400] if post else None},sort_keys=True),flush=True)
   seen=set()
   for st,typ,u,ct in resps:
    if u in seen:continue
    seen.add(u)
    print('RESP',json.dumps({'status':st,'type':typ,'ct':ct,**safe(u)},sort_keys=True),flush=True)
   # Scan rendered HTML for media/API literals.
   try:
    content=await page.content()
    for pat in ('videourl','videoUrl','m3u8','mp4','nba.com','/api/'):
     if pat.lower() in content.lower():
      print('HTML_HAS',pat,flush=True)
    literals=set(re.findall(r'https?://[^\"\'<>\\s]+',content))
    for u in list(literals):
     if interesting(u):print('HTML_URL',json.dumps(safe(u),sort_keys=True),flush=True)
   except Exception:pass
   await page.close()
  await browser.close()

asyncio.run(main())
