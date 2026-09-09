import datetime, html, json, re, time, urllib.parse, urllib.request, urllib.error
from pathlib import Path

EVENTS=[
    ('0021300124','225','2013-11-14'),
    ('0021700015','438','2017-10-19'),
    ('0041800163','215','2019-04-19'),
    ('0022500375','608','2025-26-control'),
]
H={'User-Agent':'Mozilla/5.0','Referer':'https://clips.nba.com/'}

def read(url, timeout=20):
    req=urllib.request.Request(url,headers=H)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return r.read(), r.geturl(), r.status

def highest_variant(url, text):
    lines=text.splitlines(); candidates=[]
    for i,line in enumerate(lines):
        if line.startswith('#EXT-X-STREAM-INF'):
            m=re.search(r'BANDWIDTH=(\d+)',line)
            bw=int(m.group(1)) if m else 0
            nxt=next((x for x in lines[i+1:] if x and not x.startswith('#')),None)
            if nxt: candidates.append((bw,urllib.parse.urljoin(url,nxt)))
    return max(candidates)[1] if candidates else None

def token_meta(u):
    p=urllib.parse.urlsplit(u)
    q=dict(urllib.parse.parse_qsl(p.query,keep_blank_values=True))
    v=q.get('wowzatokenendtime')
    out={'playlist_host':p.hostname,'playlist_path':p.path,'token_endtime':v}
    if v:
        try:
            iv=int(v)
            out['token_endtime_utc']=datetime.datetime.fromtimestamp(iv,datetime.timezone.utc).isoformat()
            out['token_expired']=iv < int(time.time())
        except Exception:
            out['token_endtime_parse']='non_epoch'
    return out

def probe(gid,eid,date):
    out={'game_id':gid,'event_num':eid,'date':date,'status':'unresolved','angles':[]}
    page=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
    try:
        raw,_,_=read(page); s=raw.decode('utf-8','replace')
        title=re.findall(r'<title>(.*?)</title>',s,re.S)
        out['title']=html.unescape(title[0].strip()) if title else None
        opts=[]
        for u,a,t in re.findall(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',s,re.S):
            if '.m3u8' not in u: continue
            opts.append({'url':html.unescape(u),'label':re.sub('<[^>]+>','',t).strip(),'selected':'selected' in a})
        out['angle_count']=len(opts)
        if not opts:
            out['status']='no_hls_advertised'; return out
        ordered=[o for o in opts if o['selected']]+[o for o in opts if not o['selected']]
        for o in ordered:
            a={'label':o['label'],**token_meta(o['url'])}
            try:
                u=o['url']; b,final,code=read(u); m=b.decode('utf-8','replace')
                a['playlist_http']=code
                if not m.startswith('#EXTM3U'): raise ValueError('not m3u8')
                for _ in range(3):
                    v=highest_variant(final,m)
                    if not v: break
                    b,final,code=read(v); m=b.decode('utf-8','replace'); a['variant_http']=code
                seg=next((x for x in m.splitlines() if x and not x.startswith('#')),None)
                if not seg: raise ValueError('no media segment')
                sb,_,scode=read(urllib.parse.urljoin(final,seg))
                a['segment_http']=scode; a['segment_bytes']=len(sb)
                if len(sb)<188: raise ValueError('tiny media segment')
                out['status']='hls_segment_retrieved'; out['working_angle']=o['label']; out['angles'].append(a); return out
            except urllib.error.HTTPError as e:
                a['http_error']=e.code
            except Exception as e:
                a['error']=type(e).__name__+': '+str(e)[:180]
            out['angles'].append(a)
        out['status']='hls_advertised_but_not_retrieved'
    except Exception as e:
        out['error']=type(e).__name__+': '+str(e)[:200]
    return out

rows=[]
for x in EVENTS:
    r=probe(*x); rows.append(r); print(json.dumps(r),flush=True)
Path('adams_westbrook_hls_probe.json').write_text(json.dumps(rows,indent=2))
