from __future__ import annotations
import asyncio, json, math, re
from pathlib import Path
from playwright.async_api import async_playwright

URL='https://www.nbaplaydb.com/search?actionplayer=Steven%20Adams&assistpersonname=Russell%20Westbrook&query=dunk'

async def main():
    out={'url':URL,'pages':[],'detail_probes':[]}
    async with async_playwright() as p:
        b=await p.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled'])
        c=await b.new_context(viewport={'width':1440,'height':1200},locale='en-US',user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36')
        pg=await c.new_page()
        await pg.goto(URL,wait_until='domcontentloaded',timeout=90000)
        await pg.wait_for_timeout(6000)

        async def search_page(page_no:int):
            payload={
              'requests':[{'indexName':'nba-plays','params':{
                'facetFilters':[['actionplayer:Steven Adams'],['assistpersonname:Russell Westbrook']],
                'facets':['*'],'highlightPostTag':'__/ais-highlight__','highlightPreTag':'__ais-highlight__',
                'hitsPerPage':27,'maxValuesPerFacet':100,'page':page_no,'query':'dunk'
              }}],
              'method':'search'
            }
            return await pg.evaluate("""async (payload)=>{
              const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
              return {status:r.status,text:await r.text()};
            }""",payload)

        first=await search_page(0)
        if first['status']!=200: raise RuntimeError('page0 '+str(first))
        j0=json.loads(first['text']); res0=j0['results'][0]
        nb=int(res0['nbHits']); pages=int(res0['nbPages'])
        allhits=list(res0['hits']); out['pages'].append({'page':0,'n':len(res0['hits'])})
        print('NBHITS',nb,'NBPAGES',pages)
        for page_no in range(1,pages):
            rr=await search_page(page_no)
            print('PAGE',page_no,'STATUS',rr['status'])
            if rr['status']!=200: raise RuntimeError('page '+str(page_no)+' '+rr['text'][:500])
            jj=json.loads(rr['text']); hits=jj['results'][0]['hits']; allhits.extend(hits); out['pages'].append({'page':page_no,'n':len(hits)})

        # Dedupe and sort chronologically.
        byid={h['id']:h for h in allhits}
        hits=list(byid.values())
        hits.sort(key=lambda x:(x.get('gamedate',''),x.get('gameid',''),int(x.get('actionnumber') or 0)))
        out['count']=len(hits); out['oldest_hits']=hits[:50]; out['newest_hits']=hits[-10:]
        print('UNIQUE',len(hits))
        for h in hits[:30]: print('OLD',h.get('gamedate'),h.get('gameid'),h.get('actionnumber'),h.get('id'),h.get('description'), 'video=',h.get('videourl'))

        # Specifically identify exact 2013 PBP event 0021300124 / 225 if indexed.
        exact=[h for h in hits if h.get('gameid')=='0021300124' and int(h.get('actionnumber') or -1)==225]
        print('EXACT_2013',json.dumps(exact,indent=2)[:6000])
        out['exact_2013']=exact

        selected=hits[:25] + exact + hits[-3:]
        selected_ids=[]
        for h in selected:
            if h['id'] not in selected_ids:selected_ids.append(h['id'])

        records=[]
        for i in range(0,len(selected_ids),15):
            ids=selected_ids[i:i+15]
            q=','.join(ids)
            rec=await pg.evaluate("""async (q)=>{
              const r=await fetch('/api/getPlaysByIds?playIDs='+encodeURIComponent(q)+'&source=standard&league=nba');
              return {status:r.status,text:await r.text()};
            }""",q)
            print('GETPLAYS',i,rec['status'],rec['text'][:2000])
            if rec['status']==200:
                jj=json.loads(rec['text']); records.extend(jj.get('plays',[]))
        out['records']=records
        for r in records:
            print('MEDIAREC',json.dumps({k:r.get(k) for k in ['id','gameid','actionnumber','gamedate','description','videourl','hasVideo']},sort_keys=True))

        # Open up to 3 oldest play detail pages and capture the actual media/API calls.
        old_records={r['id']:r for r in records}
        for h in hits[:3] + exact[:1]:
            rid=h['id']; slug=(h.get('gamedate','')[:4]+'-'+h.get('gamedate','')[4:6]+'-'+h.get('gamedate','')[6:8]+'-video')
            # Use search-page href when possible by reconstructing only the ID route; Next will canonicalize.
            detail_url='https://www.nbaplaydb.com/plays/'+rid+'/video'
            dp=await c.new_page(); traffic=[]
            def onresp(resp):
                u=resp.url; ct=(resp.headers.get('content-type') or '').lower()
                if any(x in u.lower() for x in ('.m3u8','.mp4','.mpd','video','media','clip','getplaysbyids','nba.com')) or 'mpegurl' in ct:
                    traffic.append({'status':resp.status,'url':u,'content_type':ct})
            dp.on('response',onresp)
            probe={'id':rid,'url':detail_url,'traffic':traffic}
            try:
                await dp.goto(detail_url,wait_until='domcontentloaded',timeout=90000); await dp.wait_for_timeout(8000)
                probe['title']=await dp.title(); probe['final_url']=dp.url
                probe['body_head']=(await dp.locator('body').inner_text())[:4000]
                probe['videos']=await dp.locator('video,video source').evaluate_all("els=>els.map(e=>({tag:e.tagName,src:e.src,currentSrc:e.currentSrc,poster:e.poster}))")
                # Click plausible player controls.
                for sel in ['video','button[aria-label*="play" i]','.vjs-big-play-button','button:has-text("Play")']:
                    try:
                        loc=dp.locator(sel).first
                        if await loc.count() and await loc.is_visible():
                            await loc.click(timeout=2500,force=True); await dp.wait_for_timeout(5000); break
                    except Exception: pass
                probe['videos_after']=await dp.locator('video,video source').evaluate_all("els=>els.map(e=>({tag:e.tagName,src:e.src,currentSrc:e.currentSrc,poster:e.poster}))")
                probe['performance']=await dp.evaluate("performance.getEntriesByType('resource').map(x=>x.name).filter(x=>/m3u8|mp4|mpd|video|media|clip|nba/i.test(x))")
            except Exception as e:probe['error']=repr(e)
            out['detail_probes'].append(probe)
            print('DETAIL',json.dumps(probe,indent=2)[:12000])
            await dp.close()

        Path('nbaplaydb_api_probe.json').write_text(json.dumps(out,indent=2))
        await b.close()

if __name__=='__main__': asyncio.run(main())
