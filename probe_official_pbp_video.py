import asyncio, json, re
from pathlib import Path
from playwright.async_api import async_playwright

GAME_ID='0021300124'
EVENT_NUM='225'
URL=f'https://www.nba.com/game/okc-vs-gsw-{GAME_ID}/play-by-play'
TARGET_RE=re.compile(r"Adams.*(?:Slam|Dunk).*Westbrook", re.I)
KEY_RE=re.compile(r'video|media|clip|wsc|event|playbyplay|play-by-play|pbp|akamai|turner|lrmedia|clips\.nba|secure\.nba|watch', re.I)

async def main():
    out={'game_id':GAME_ID,'event_num':EVENT_NUM,'url':URL,'requests':[],'responses':[],'console':[]}
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True)
        ctx=await browser.new_context(viewport={'width':1600,'height':1200}, user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36')
        page=await ctx.new_page()
        page.on('console', lambda m: out['console'].append({'type':m.type,'text':m.text}) if KEY_RE.search(m.text or '') else None)
        page.on('request', lambda r: out['requests'].append({'method':r.method,'url':r.url,'resource_type':r.resource_type}) if KEY_RE.search(r.url) else None)
        page.on('response', lambda r: out['responses'].append({'status':r.status,'url':r.url,'content_type':r.headers.get('content-type','')}) if KEY_RE.search(r.url) else None)
        try:
            await page.goto(URL, wait_until='domcontentloaded', timeout=90000)
            await page.wait_for_timeout(8000)
        except Exception as e:
            out['goto_error']=repr(e)

        # Consent / privacy overlays, if present.
        for label in ['Accept All','I Accept','Accept','Agree']:
            try:
                b=page.get_by_role('button', name=re.compile(f'^{re.escape(label)}$', re.I))
                if await b.count():
                    await b.first.click(timeout=1500)
                    await page.wait_for_timeout(1000)
                    break
            except Exception:
                pass

        # Ensure every play is rendered.
        try:
            allbtn=page.get_by_role('button', name=re.compile(r'^ALL$',re.I))
            if await allbtn.count():
                await allbtn.first.click(timeout=5000)
                await page.wait_for_timeout(5000)
        except Exception as e:
            out['all_button_error']=repr(e)

        out['title']=await page.title()
        body=(await page.locator('body').inner_text())
        out['target_text_present']=bool(TARGET_RE.search(body))
        out['adams_lines']=[x.strip() for x in body.splitlines() if 'Adams' in x and ('Dunk' in x or 'Slam' in x)][:50]

        # Inspect NEXT payload: this often exposes the data/API route even when hrefs are generated in React.
        try:
            nxt=await page.locator('#__NEXT_DATA__').text_content()
            out['next_data_len']=len(nxt or '')
            if nxt:
                Path('next_data.json').write_text(nxt, encoding='utf-8')
                nd=json.loads(nxt)
                # keep strings containing our event, video/media terms
                hits=[]
                def walk(v,path=''):
                    if isinstance(v,dict):
                        for k,x in v.items(): walk(x,f'{path}.{k}' if path else k)
                    elif isinstance(v,list):
                        for i,x in enumerate(v): walk(x,f'{path}[{i}]')
                    elif isinstance(v,(str,int,float,bool)):
                        s=str(v)
                        if EVENT_NUM==s or GAME_ID in s or KEY_RE.search(s):
                            hits.append({'path':path,'value':s[:1000]})
                walk(nd)
                out['next_data_hits']=hits[:500]
        except Exception as e:
            out['next_data_error']=repr(e)

        # Find the Adams dunk text and climb to the event container.
        candidates=page.get_by_text(TARGET_RE)
        out['candidate_count']=await candidates.count()
        if not await candidates.count():
            candidates=page.get_by_text(re.compile(r'Adams.*Dunk',re.I))
            out['fallback_candidate_count']=await candidates.count()

        if await candidates.count():
            node=candidates.first
            try:
                await node.scroll_into_view_if_needed(timeout=5000)
            except Exception: pass
            out['candidate_text']=await node.inner_text()
            ancestor=node
            ancestors=[]
            for depth in range(9):
                try:
                    html=await ancestor.evaluate('(e)=>e.outerHTML')
                    text=await ancestor.inner_text()
                    anchors=await ancestor.locator('a').evaluate_all("els=>els.map(a=>({href:a.href,text:a.innerText,aria:a.getAttribute('aria-label')}))")
                    buttons=await ancestor.locator('button').evaluate_all("els=>els.map(b=>({text:b.innerText,aria:b.getAttribute('aria-label'),title:b.getAttribute('title'),data:[...b.attributes].filter(x=>x.name.startsWith('data-')).map(x=>[x.name,x.value])}))")
                    ancestors.append({'depth':depth,'text':text[:1500],'html':html[:10000],'anchors':anchors,'buttons':buttons})
                    if anchors or buttons or 'video' in html.lower() or 'event' in html.lower():
                        out['event_container']=ancestors[-1]
                    ancestor=ancestor.locator('..')
                except Exception:
                    break
            out['ancestors']=ancestors

            # First try clicking the play description itself: NBA's PBP rows commonly make video-enabled events interactive.
            before=len(out['requests'])
            try:
                await node.click(timeout=5000, force=True)
                out['clicked_text']=True
                await page.wait_for_timeout(8000)
            except Exception as e:
                out['click_text_error']=repr(e)

            # If that did not generate media/network traffic, click any button/link in the nearest event-ish ancestor.
            generated=[x for x in out['requests'][before:] if KEY_RE.search(x['url'])]
            if not generated:
                chosen=None
                for depth in range(min(5,len(ancestors))):
                    anc=node
                    for _ in range(depth): anc=anc.locator('..')
                    for sel in ['button','a']:
                        loc=anc.locator(sel)
                        n=min(await loc.count(),6)
                        for i in range(n):
                            el=loc.nth(i)
                            try:
                                meta=((await el.get_attribute('aria-label')) or '')+' '+((await el.get_attribute('title')) or '')+' '+((await el.inner_text()) or '')
                                href=(await el.get_attribute('href')) or ''
                                if re.search(r'video|watch|play', meta+' '+href, re.I):
                                    chosen={'depth':depth,'selector':sel,'index':i,'meta':meta,'href':href}
                                    await el.click(timeout=4000, force=True)
                                    await page.wait_for_timeout(8000)
                                    raise StopAsyncIteration
                            except StopAsyncIteration:
                                raise
                            except Exception:
                                pass
                out['chosen_control']=chosen

        # Extract whatever player/source is now mounted.
        out['final_url']=page.url
        out['videos']=await page.locator('video').evaluate_all("els=>els.map(v=>({src:v.src,currentSrc:v.currentSrc,poster:v.poster,html:v.outerHTML}))")
        out['sources']=await page.locator('source').evaluate_all("els=>els.map(s=>({src:s.src,type:s.type}))")
        out['all_media_hrefs']=await page.locator('a').evaluate_all("els=>els.map(a=>a.href).filter(h=>/video|media|clip|event|watch|wsc|akamai|turner/i.test(h))")
        try:
            out['performance_media']=await page.evaluate("performance.getEntriesByType('resource').map(x=>x.name).filter(x=>/video|media|clip|event|watch|wsc|akamai|turner|m3u8|mp4/i.test(x))")
        except Exception as e:
            out['performance_error']=repr(e)
        await page.screenshot(path='official_pbp_probe.png', full_page=True)
        await browser.close()

    Path('official_pbp_probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['ancestors','next_data_hits','requests','responses','console']},indent=2))
    print('\n=== REQUESTS ===')
    for x in out['requests'][-120:]: print(x)
    print('\n=== RESPONSES ===')
    for x in out['responses'][-120:]: print(x)

if __name__=='__main__':
    try:
        asyncio.run(main())
    except StopAsyncIteration:
        # Should not escape, but retain artifacts if a nested control matched.
        pass
