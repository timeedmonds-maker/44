#!/usr/bin/env python3
import asyncio,json,re,hashlib,urllib.parse
from curl_cffi import requests as crequests
from playwright.async_api import async_playwright

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
EVENTS=[('2017','0021700015','438','2017-18'),('2019','0041800163','215','2018-19'),('2025','0022500375','608','2025-26')]

def safe(u):
 p=urllib.parse.urlsplit(u); return {'host':p.netloc,'path':p.path,'query_keys':sorted(urllib.parse.parse_qs(p.query).keys())}

async def one(browser,label,gid,eid,season):
    url=f'https://www.nba.com/stats/events?GameEventID={eid}&GameID={gid}&Season={season}&flag=1'
    s=crequests.Session(impersonate='chrome120')
    pre=s.get('https://www.nba.com/',headers={'User-Agent':UA},timeout=20)
    doc=s.get(url,headers={'User-Agent':UA,'Referer':'https://www.nba.com/'},timeout=20)
    print('\n===',label,gid,eid,'DOC',doc.status_code,len(doc.content),'===')
    context=await browser.new_context(user_agent=UA)
    # transfer harmless NBA cookies from the successful curl_cffi session
    cookies=[]
    for k,v in s.cookies.get_dict().items():
        cookies.append({'name':k,'value':v,'domain':'.nba.com','path':'/'})
    if cookies:
        try: await context.add_cookies(cookies)
        except Exception as e: print('COOKIE_ERR',type(e).__name__,str(e)[:120])
    page=await context.new_page()
    captures=[]
    async def route_handler(route,request):
        if request.is_navigation_request() and request.url.startswith('https://www.nba.com/stats/events'):
            await route.fulfill(status=200,body=doc.content,headers={'content-type':'text/html; charset=utf-8'})
        else:
            await route.continue_()
    await page.route('**/*',route_handler)
    async def on_response(resp):
        u=resp.url
        if 'stats.nba.com/stats/' in u or 'videoevents' in u.lower() or any(x in u.lower() for x in ['.mp4','.m3u8','lrmedia']):
            rec={'status':resp.status,**safe(u)}
            print('RESP',json.dumps(rec))
            if 'videoeventsasset' in u.lower():
                try:
                    body=await resp.body()
                    print('VIDEOEVENTS_BODY_BYTES',len(body),'SHA',hashlib.sha256(body).hexdigest())
                    txt=body.decode('utf-8','replace')
                    print('VIDEOEVENTS_BODY',txt[:10000])
                    captures.append(txt)
                except Exception as e: print('BODY_ERR',type(e).__name__,str(e)[:160])
    page.on('response',on_response)
    try:
        await page.goto(url,wait_until='domcontentloaded',timeout=30000)
        print('TITLE',await page.title())
        await page.wait_for_timeout(18000)
        # emit visible video source/state if browser created a video element
        vals=await page.evaluate('''() => Array.from(document.querySelectorAll('video')).map(v => ({src:v.currentSrc||v.src,duration:v.duration,readyState:v.readyState,networkState:v.networkState,error:v.error&&v.error.code}))''')
        for v in vals:
            if v.get('src'): v['src']=safe(v['src'])
            print('VIDEO_ELEMENT',json.dumps(v))
    except Exception as e: print('PAGE_ERR',type(e).__name__,str(e)[:240])
    await context.close()

async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True)
        for args in EVENTS:
            await one(browser,*args)
        await browser.close()
asyncio.run(main())
