from __future__ import annotations
import json,re
from pathlib import Path
from playwright.sync_api import sync_playwright

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131 Safari/537.36'
PAGES=[
 ('game2013','https://www.nbaplaydb.com/games/20131114-OKCGSW'),
 ('search','https://www.nbaplaydb.com/search?actionplayer=Steven+Adams&query=dunk'),
 ('player','https://www.nbaplaydb.com/players/steven-adams'),
 ('game2016','https://www.nbaplaydb.com/games/20160522-OKCGSW'),
]

def main():
 out=[]
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled'])
  c=b.new_context(user_agent=UA,viewport={'width':1440,'height':1200},locale='en-US')
  for label,url in PAGES:
   pg=c.new_page(); hits=[]; apis=[]
   def resp(r):
    u=r.url; ct=(r.headers.get('content-type') or '').lower()
    if any(x in u.lower() for x in ('.m3u8','.mp4','.mpd','video','clip','playback','media')):
     hits.append({'url':u,'status':r.status,'content_type':ct})
    if any(x in u.lower() for x in ('api','search','graphql','elastic','play')):
     apis.append({'url':u,'status':r.status,'content_type':ct})
   pg.on('response',resp)
   rec={'label':label,'url':url,'hits':hits,'apis':apis}
   try:
    pg.goto(url,wait_until='domcontentloaded',timeout=90000)
    pg.wait_for_timeout(8000)
    rec['title']=pg.title(); rec['final_url']=pg.url
    body=pg.locator('body').inner_text(timeout=10000)
    rec['body_head']=body[:8000]
    rec['contains_target']=('Adams' in body and 'Westbrook' in body and 'Dunk' in body)
    rec['links']=pg.eval_on_selector_all('a','els=>els.map(e=>({t:(e.innerText||e.textContent||"").trim(),h:e.href})).filter(x=>x.h)')
    rec['video_nodes']=pg.eval_on_selector_all('video,video source','els=>els.map(e=>({tag:e.tagName,src:e.src,currentSrc:e.currentSrc,poster:e.poster}))')
    # Extract inline identifiers/URLs
    html=pg.content(); vals=[]
    for m in re.finditer(r'https?://[^"\'<>\s]+',html,re.I):
     v=m.group(0).replace('&amp;','&')
     if any(x in v.lower() for x in ('video','clip','.m3u8','.mp4','.mpd','api','cloudfront','amazonaws')) and v not in vals: vals.append(v)
    rec['inline_urls']=vals[:300]
    # Click any target play links/buttons visible
    targets=pg.get_by_text(re.compile('Adams.*Dunk|Dunk.*Adams',re.I))
    rec['target_count']=targets.count()
    for i in range(min(targets.count(),3)):
     try:
      el=targets.nth(i); rec.setdefault('target_texts',[]).append(el.inner_text()[:500])
      el.click(timeout=3000); pg.wait_for_timeout(5000)
      rec.setdefault('after_click_urls',[]).append(pg.url)
      rec['video_nodes_after']=pg.eval_on_selector_all('video,video source','els=>els.map(e=>({tag:e.tagName,src:e.src,currentSrc:e.currentSrc,poster:e.poster}))')
      break
     except Exception as e: rec.setdefault('click_errors',[]).append(repr(e))
   except Exception as e: rec['error']=repr(e)
   out.append(rec); pg.close()
  c.close(); b.close()
 Path('nbaplaydb_2013_probe.json').write_text(json.dumps(out,indent=2))
 for r in out:
  print('\n===',r['label'],'===')
  print(r.get('title'),r.get('final_url'),'target?',r.get('contains_target'),'count',r.get('target_count'))
  print('BODY',r.get('body_head','')[:2000].replace('\n',' | '))
  print('APIS')
  for x in r.get('apis',[]): print(x)
  print('HITS')
  for x in r.get('hits',[]): print(x)
  print('TARGET LINKS')
  for x in r.get('links',[]):
   if 'play/' in x.get('h','') or 'Adams' in x.get('t','') or 'dunk' in x.get('t','').lower(): print(x)

if __name__=='__main__': main()
