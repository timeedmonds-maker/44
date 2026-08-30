from __future__ import annotations
import argparse, html as htmlmod, json, re, urllib.request, urllib.parse

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*'}

def get(url):
    req=urllib.request.Request(url,headers=H)
    with urllib.request.urlopen(req,timeout=30) as r:
        return r.read().decode('utf-8',errors='replace')

def join(base,child):
    u=urllib.parse.urljoin(base,child)
    bp=urllib.parse.urlsplit(base); cp=urllib.parse.urlsplit(u)
    if bp.query and not cp.query:
        cp=cp._replace(query=bp.query); u=urllib.parse.urlunsplit(cp)
    return u

def inspect_playlist(url):
    text=get(url)
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    variants=[]
    for i,line in enumerate(lines):
        if line.startswith('#EXT-X-STREAM-INF:'):
            bw=re.search(r'BANDWIDTH=(\d+)',line)
            avg=re.search(r'AVERAGE-BANDWIDTH=(\d+)',line)
            res=re.search(r'RESOLUTION=(\d+x\d+)',line)
            codecs=re.search(r'CODECS="([^"]+)"',line)
            child=''
            for nxt in lines[i+1:]:
                if not nxt.startswith('#'):
                    child=join(url,nxt); break
            variants.append({
                'bandwidth':int(bw.group(1)) if bw else None,
                'average_bandwidth':int(avg.group(1)) if avg else None,
                'resolution':res.group(1) if res else None,
                'codecs':codecs.group(1) if codecs else None,
                'child_path': urllib.parse.urlsplit(child).path if child else None,
            })
    return variants

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--game',required=True); ap.add_argument('--event',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    page=f'https://clips.nba.com/?gameNo={a.game}&eventNum={a.event}&source=grs'
    text=get(page)
    title_m=re.search(r'<title>(.*?)</title>',text,re.I|re.S)
    title=htmlmod.unescape(title_m.group(1).strip()) if title_m else ''
    options=[]
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',text,re.I|re.S):
        url=htmlmod.unescape(m.group(1).strip()); attrs=m.group(2).lower(); label=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in url.lower() and 'lrmedia.nba.com' in url.lower():
            variants=inspect_playlist(url)
            options.append({'label':label,'selected':'selected' in attrs,'playlist_path':urllib.parse.urlsplit(url).path,'variants':variants})
    payload={'game_id':a.game,'event_id':int(a.event),'title':title,'option_count':len(options),'options':options}
    open(a.out,'w',encoding='utf-8').write(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
