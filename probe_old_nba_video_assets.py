from __future__ import annotations

import json, re, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin

import requests

EVENTS=[
    ('0021300124',225,'2013-11-14'),
    ('0041300232',145,'2014-05-27'),
    ('0021801220',7,'2019-04-10'),
]
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:105.0) Gecko/20100101 Firefox/105.0'
NBA_H={
    'Host':'stats.nba.com','Referer':'https://www.nba.com/','Origin':'https://stats.nba.com/',
    'Connection':'keep-alive','x-nba-stats-origin':'stats','x-nba-stats-token':'true',
    'Cache-Control':'max-age=0','Upgrade-Insecure-Requests':'1','User-Agent':UA,
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding':'gzip, deflate, br','Accept-Language':'en-US,en;q=0.9',
}
GEN_H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*','Accept-Language':'en-US,en;q=0.9'}


def safe_get(session,url,headers=None,params=None,timeout=25):
    t=time.time()
    try:
        r=session.get(url,headers=headers,params=params,timeout=timeout,allow_redirects=True)
        return r, {'elapsed':round(time.time()-t,3),'status':r.status_code,'bytes':len(r.content),'final_url':r.url,'content_type':r.headers.get('content-type')}
    except Exception as e:
        return None, {'elapsed':round(time.time()-t,3),'error':repr(e)}


def uuid_from_json(j):
    out=[]
    try:
        rs=j.get('resultSets') or {}
        if isinstance(rs,dict):
            meta=rs.get('Meta') or {}
            for v in meta.get('videoUrls') or []:
                u=v.get('uuid')
                if u and u not in out: out.append(u)
    except Exception: pass
    return out


def urls_from_xml(text):
    out=[]
    try:
        root=ET.fromstring(text)
        for node in root.iter():
            if node.tag.lower().endswith('file') and node.text and node.text.strip().startswith('http'):
                u=node.text.strip()
                if u not in out: out.append(u)
    except Exception:
        for u in re.findall(r'https?://[^<\s\"]+',text):
            if u not in out: out.append(u)
    return out


def clips_static(session,gid,eid):
    url=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
    r,meta=safe_get(session,url,headers=GEN_H,timeout=35)
    out={'url':url,**meta,'options':[],'media_literals':[]}
    if not r:return out
    txt=r.text
    out['title']=(re.search(r'<title>(.*?)</title>',txt,re.I|re.S).group(1).strip() if re.search(r'<title>(.*?)</title>',txt,re.I|re.S) else '')
    for m in re.finditer(r'<option\s+value=[\"\']([^\"\']+)[\"\']([^>]*)>(.*?)</option>',txt,re.I|re.S):
        val=m.group(1).replace('&amp;','&'); lab=re.sub(r'<[^>]+>','',m.group(3)).strip()
        out['options'].append({'label':lab,'url':val,'selected':'selected' in m.group(2).lower()})
    lits=re.findall(r'https?://[^\"\'<>\s]+',txt)
    out['media_literals']=[u.replace('&amp;','&') for u in lits if any(x in u.lower() for x in ('.m3u8','.mp4','lrmedia','wsc'))][:100]
    out['html_head']=txt[:600]
    return out


def legacy_stats(session,gid,eid):
    endpoints=[]; uuids=[]
    for host in ('https://stats.nba.com','https://stats.gleague.nba.com'):
        for ep in ('videoevents','videoeventsasset'):
            h=dict(NBA_H); h['Host']=host.split('//',1)[1]
            r,meta=safe_get(session,host+'/stats/'+ep,headers=h,params={'GameID':gid,'GameEventID':eid},timeout=22)
            rec={'host':host,'endpoint':ep,**meta}
            if r is not None:
                rec['text_head']=r.text[:500]
                if r.ok:
                    try:
                        j=r.json(); rec['uuids']=uuid_from_json(j); uuids += rec['uuids']
                    except Exception as e: rec['json_error']=repr(e)
            endpoints.append(rec)
    uuids=list(dict.fromkeys(uuids))
    wsc=[]
    for uid in uuids:
        for base in ('https://secure.nba.com/video/wsc/league/','http://secure.nba.com/video/wsc/league/'):
            r,meta=safe_get(session,base+uid+'.secure.xml',headers={'User-Agent':UA,'Referer':'https://www.nba.com/'},timeout=20)
            rec={'uuid':uid,'url':base+uid+'.secure.xml',**meta}
            if r is not None:
                rec['files']=urls_from_xml(r.text)
                rec['xml_head']=r.text[:500]
            wsc.append(rec)
    return {'endpoints':endpoints,'uuids':uuids,'wsc':wsc}


def browser_capture(gid,eid):
    from playwright.sync_api import sync_playwright
    targets=[
        ('clips',f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'),
        ('stats_event',f'https://www.nba.com/stats/events?GameEventID={eid}&GameID={gid}&flag=1'),
    ]
    out=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled'])
        ctx=browser.new_context(user_agent=UA,viewport={'width':1440,'height':1000},locale='en-US')
        for label,url in targets:
            pg=ctx.new_page(); media=[]; responses=[]
            def onresp(resp):
                u=resp.url
                if any(x in u.lower() for x in ('.m3u8','.mp4','lrmedia','/video/wsc/','videoevents')):
                    if u not in media: media.append(u)
                if 'clips.nba.com' in u or 'stats.nba.com/stats/video' in u:
                    responses.append({'url':u,'status':resp.status,'content_type':resp.headers.get('content-type')})
            pg.on('response',onresp)
            rec={'label':label,'url':url,'media':media,'responses':responses}
            try:
                pg.goto(url,wait_until='domcontentloaded',timeout=90000)
                pg.wait_for_timeout(12000)
                rec['title']=pg.title(); rec['final_url']=pg.url
                rec['options']=pg.eval_on_selector_all('option','els=>els.map(e=>({label:e.textContent.trim(),url:e.value,selected:e.selected}))')
                rec['video_srcs']=pg.eval_on_selector_all('video,video source','els=>els.map(e=>e.currentSrc||e.src).filter(Boolean)')
                rec['body_head']=pg.locator('body').inner_text(timeout=5000)[:1200]
                for sel in ['button[aria-label*="play" i]','.vjs-big-play-button','button:has-text("Play")']:
                    try:
                        loc=pg.locator(sel).first
                        if loc.count() and loc.is_visible(): loc.click(timeout=2000); pg.wait_for_timeout(4000)
                    except Exception: pass
            except Exception as e: rec['error']=repr(e)
            out.append(rec); pg.close()
        ctx.close(); browser.close()
    return out


def validate_candidate(url,outfile):
    try:
        hdr='User-Agent: '+UA+'\r\nReferer: https://www.nba.com/\r\n'
        p=subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-rw_timeout','30000000','-headers',hdr,'-i',url,'-t','20','-map','0:v:0','-map','0:a:0?','-c','copy',str(outfile)],capture_output=True,text=True,timeout=70)
        if p.returncode:return {'ok':False,'ffmpeg_error':p.stderr[-1000:]}
        q=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height:format=duration','-of','json',str(outfile)],capture_output=True,text=True,timeout=20)
        j=json.loads(q.stdout); s=(j.get('streams') or [{}])[0]; dur=float((j.get('format') or {}).get('duration') or 0)
        return {'ok':q.returncode==0 and dur>=2,'duration':dur,'width':s.get('width'),'height':s.get('height'),'bytes':outfile.stat().st_size}
    except Exception as e:return {'ok':False,'error':repr(e)}


def main():
    s=requests.Session(); results=[]; Path('probe_media').mkdir(exist_ok=True)
    for gid,eid,date in EVENTS:
        rec={'game_id':gid,'event_id':eid,'date':date}
        rec['clips_static']=clips_static(s,gid,eid)
        rec['legacy']=legacy_stats(s,gid,eid)
        try: rec['browser']=browser_capture(gid,eid)
        except Exception as e: rec['browser_error']=repr(e); rec['browser']=[]
        candidates=[]
        for o in rec['clips_static'].get('options',[]):
            if any(x in o['url'].lower() for x in ('.m3u8','.mp4')): candidates.append(('clips_static',o['url']))
        candidates += [('clips_literal',u) for u in rec['clips_static'].get('media_literals',[])]
        for w in rec['legacy'].get('wsc',[]): candidates += [('legacy_wsc',u) for u in w.get('files',[])]
        for b in rec['browser']:
            candidates += [('browser',u) for u in b.get('media',[])] + [('browser_dom',u) for u in b.get('video_srcs',[])]
            candidates += [('browser_option',o.get('url')) for o in b.get('options',[]) if o.get('url') and any(x in o.get('url','').lower() for x in ('.m3u8','.mp4'))]
        ded=[]; seen=set()
        for src,u in candidates:
            if not u or u in seen: continue
            seen.add(u); ded.append((src,u))
        rec['candidates']=[{'source':src,'url':u} for src,u in ded]
        rec['validated']=[]
        for i,(src,u) in enumerate(ded[:12]):
            vr=validate_candidate(u,Path('probe_media')/f'{gid}_{eid}_{i}.mp4'); vr.update({'source':src,'url':u}); rec['validated'].append(vr)
            if vr.get('ok'): break
        results.append(rec)
        print('\nEVENT',gid,eid)
        print('static options',len(rec['clips_static'].get('options',[])),'legacy uuids',rec['legacy'].get('uuids'))
        print('browser media',sum(len(x.get('media',[])) for x in rec['browser']),'candidates',len(ded))
        good=[x for x in rec['validated'] if x.get('ok')]
        print('VALID',good[:1])
    Path('old_nba_video_asset_probe.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
