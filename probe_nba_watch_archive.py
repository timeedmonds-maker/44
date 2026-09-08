from __future__ import annotations
import json,re
from pathlib import Path
from playwright.sync_api import sync_playwright

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131 Safari/537.36'
PAGES=[
 ('full_a','https://www.nba.com/watch/video/thunder-warriors-11-14-2013-a'),
 ('condensed','https://www.nba.com/watch/video/thunder-warriors-11-14-2013-condensed-fullgame'),
 ('game','https://www.nba.com/game/okc-vs-gsw-0021300124'),
]

def main():
 out=[]
 with sync_playwright() as p:
  b=p.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled'])
  c=b.new_context(user_agent=UA,viewport={'width':1440,'height':1000},locale='en-US')
  for label,url in PAGES:
   pg=c.new_page(); hits=[]; allreq=[]
   def resp(r):
    u=r.url; ct=(r.headers.get('content-type') or '').lower()
    if any(x in u.lower() for x in ('.m3u8','.mp4','.mpd','manifest','playback','stream','video')) or 'mpegurl' in ct or 'dash+xml' in ct:
     hits.append({'url':u,'status':r.status,'content_type':ct})
    if any(x in u.lower() for x in ('graphql','api.nba','watch','content')):
     allreq.append({'url':u,'status':r.status,'content_type':ct})
   pg.on('response',resp)
   rec={'label':label,'url':url,'hits':hits,'api':allreq}
   try:
    pg.goto(url,wait_until='domcontentloaded',timeout=90000)
    pg.wait_for_timeout(9000)
    rec['title']=pg.title(); rec['final_url']=pg.url
    rec['body_head']=pg.locator('body').inner_text(timeout=5000)[:3000]
    rec['html_head']=pg.content()[:5000]
    rec['video_nodes']=pg.eval_on_selector_all('video,video source','els=>els.map(e=>({tag:e.tagName,src:e.src,currentSrc:e.currentSrc,poster:e.poster}))')
    # click plausible playback controls, including generic center controls
    for sel in ['button[aria-label*="play" i]','.vjs-big-play-button','button:has-text("Play")','video']:
     try:
      loc=pg.locator(sel).first
      if loc.count() and loc.is_visible():
       loc.click(timeout=2500); pg.wait_for_timeout(7000)
     except Exception: pass
    rec['video_nodes_after']=pg.eval_on_selector_all('video,video source','els=>els.map(e=>({tag:e.tagName,src:e.src,currentSrc:e.currentSrc,poster:e.poster}))')
    # pull inline JSON-ish strings containing media/player IDs
    text=pg.content()
    vals=[]
    for pat in [r'https?://[^"\'<> ]+',r'"(?:videoId|contentId|assetId|pid|playbackId|mediaId)"\s*:\s*"?([^",}<]+)']:
     for m in re.finditer(pat,text,re.I):
      v=m.group(0)[:1000]
      if v not in vals: vals.append(v)
    rec['inline_candidates']=vals[:200]
   except Exception as e: rec['error']=repr(e)
   out.append(rec); pg.close()
  c.close(); b.close()
 Path('nba_watch_archive_probe.json').write_text(json.dumps(out,indent=2))
 for r in out:
  print('\n',r['label'],r.get('title'),r.get('final_url'))
  print('hits',len(r['hits']))
  for x in r['hits'][-30:]: print(x)
  print('video',r.get('video_nodes_after'))

if __name__=='__main__': main()
