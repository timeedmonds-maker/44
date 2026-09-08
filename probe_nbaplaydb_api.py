from __future__ import annotations
import asyncio, json, re
from pathlib import Path
from playwright.async_api import async_playwright

URL='https://www.nbaplaydb.com/search?actionplayer=Steven%20Adams&assistpersonname=Russell%20Westbrook&query=dunk'

async def main():
    out={'url':URL,'api_requests':[],'api_responses':[],'target_links':[]}
    async with async_playwright() as p:
        b=await p.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled'])
        c=await b.new_context(viewport={'width':1440,'height':1200},locale='en-US',user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36')
        pg=await c.new_page()

        async def on_request(req):
            if '/api/search' in req.url:
                out['api_requests'].append({'method':req.method,'url':req.url,'post_data':req.post_data,'headers':await req.all_headers()})
        async def on_response(resp):
            if '/api/search' in resp.url:
                rec={'status':resp.status,'url':resp.url,'headers':await resp.all_headers()}
                try:
                    txt=await resp.text(); rec['text']=txt
                    try: rec['json']=json.loads(txt)
                    except Exception: pass
                except Exception as e: rec['body_error']=repr(e)
                out['api_responses'].append(rec)
        pg.on('request',on_request); pg.on('response',on_response)

        await pg.goto(URL,wait_until='domcontentloaded',timeout=90000)
        await pg.wait_for_timeout(10000)
        out['title']=await pg.title(); out['final_url']=pg.url
        body=await pg.locator('body').inner_text()
        out['body_head']=body[:12000]
        out['result_count_text']=[x.strip() for x in body.splitlines() if 'result' in x.lower()][:20]
        links=await pg.locator('a[href*="/plays/"]').evaluate_all("els=>els.map(a=>({text:(a.innerText||a.textContent||'').trim(),href:a.href}))")
        out['target_links']=links[:200]
        # Inspect app JS for API schema / filter parameter names.
        scripts=await pg.locator('script[src]').evaluate_all('els=>els.map(s=>s.src)')
        out['scripts']=scripts
        js_hits=[]
        for src in scripts:
            if '/_next/static/chunks/' not in src: continue
            try:
                r=await c.request.get(src,timeout=20000)
                if not r.ok: continue
                txt=await r.text()
                if '/api/search' in txt or 'assistpersonname' in txt or 'actionplayer' in txt:
                    snippets=[]
                    for pat in ['/api/search','assistpersonname','actionplayer','seasonsegment','gamedate']:
                        start=0
                        while True:
                            i=txt.find(pat,start)
                            if i<0: break
                            snippets.append(txt[max(0,i-800):min(len(txt),i+1600)])
                            start=i+len(pat)
                            if len(snippets)>=20: break
                    js_hits.append({'src':src,'snippets':snippets[:20]})
            except Exception as e:
                pass
        out['js_hits']=js_hits
        Path('nbaplaydb_api_probe.json').write_text(json.dumps(out,indent=2))
        print('TITLE',out.get('title'))
        print('RESULT TEXT',out.get('result_count_text'))
        print('API REQS',json.dumps(out['api_requests'],indent=2)[:12000])
        print('API RESPONSES')
        for r in out['api_responses']:
            j=r.get('json')
            print(' status',r['status'],'url',r['url'],'type',type(j).__name__,'keys',list(j.keys())[:30] if isinstance(j,dict) else None)
            print((r.get('text') or '')[:20000])
        print('PLAY LINKS',len(links))
        for x in links[:50]: print(x)
        print('JS HIT FILES',len(js_hits))
        for h in js_hits[:8]:
            print('JS',h['src'])
            for s in h['snippets'][:4]: print(s[:3000])
        await b.close()

if __name__=='__main__': asyncio.run(main())
