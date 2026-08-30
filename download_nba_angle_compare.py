from __future__ import annotations
import argparse, html as htmlmod, json, os, re, subprocess, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*'}
FFMPEG=os.environ.get('FFMPEG_BINARY','ffmpeg')

def get(url, timeout=45):
    req=urllib.request.Request(url,headers=H)
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def signed_join(base, child):
    joined=urllib.parse.urljoin(base,child); bp=urllib.parse.urlsplit(base); cp=urllib.parse.urlsplit(joined)
    if bp.query and not cp.query: joined=urllib.parse.urlunsplit(cp._replace(query=bp.query))
    return joined

def parse_media(url, depth=0):
    if depth>3: raise RuntimeError('playlist nesting')
    text=get(url).decode('utf-8','replace'); lines=[x.strip() for x in text.splitlines() if x.strip()]
    variants=[]
    for i,line in enumerate(lines):
        if line.startswith('#EXT-X-STREAM-INF:'):
            bw=int(re.search(r'BANDWIDTH=(\d+)',line).group(1)) if re.search(r'BANDWIDTH=(\d+)',line) else 0
            for child in lines[i+1:]:
                if not child.startswith('#'):
                    variants.append((bw,signed_join(url,child))); break
    if variants:return parse_media(max(variants,key=lambda x:x[0])[1],depth+1)
    init=None; seg=[]
    for line in lines:
        if line.startswith('#EXT-X-MAP:'):
            m=re.search(r'URI="([^"]+)"',line)
            if m:init=signed_join(url,m.group(1))
        elif not line.startswith('#'):seg.append(signed_join(url,line))
    if not seg: raise RuntimeError('no segments')
    return init,seg

def download_hls(url,out):
    init,segs=parse_media(url); urls=([init] if init else [])+segs; parts=[None]*len(urls)
    def one(i,u): return i,get(u)
    with ThreadPoolExecutor(max_workers=min(8,len(urls))) as pool:
        fs=[pool.submit(one,i,u) for i,u in enumerate(urls)]
        for f in as_completed(fs):i,b=f.result();parts[i]=b
    part=out.with_suffix('.hls.part')
    with part.open('wb') as fh:
        for b in parts: fh.write(b)
    subprocess.run([FFMPEG,'-nostdin','-y','-v','error','-i',str(part),'-map','0:v:0','-map','0:a:0?','-c','copy','-movflags','+faststart',str(out)],check=True)
    part.unlink(missing_ok=True)

def probe(path):
    p=subprocess.run([FFMPEG,'-hide_banner','-i',str(path)],capture_output=True,text=True)
    t=p.stderr
    m=re.search(r'Video:.*?\b(\d{2,5})x(\d{2,5})\b',t)
    d=re.search(r'Duration:\s*(\d+):(\d+):([0-9.]+)',t)
    return {'width':int(m.group(1)) if m else None,'height':int(m.group(2)) if m else None,'duration':(int(d.group(1))*3600+int(d.group(2))*60+float(d.group(3))) if d else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    req=json.loads(Path(a.request).read_text()); gid=str(req['game_id']); eid=int(req['event_id']); labels=req['angle_labels']
    page=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'; txt=get(page).decode('utf-8','replace')
    opts={}
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',txt,flags=re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip()); lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower():opts[lab]=u
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True); report=[]
    for i,lab in enumerate(labels,1):
        if lab not in opts: raise RuntimeError(f'missing angle {lab}; have {list(opts)}')
        safe=re.sub(r'[^A-Za-z0-9]+','_',lab).strip('_'); dst=out/f'{i:02d}_{safe}_1920x1080_SOURCE.mp4'; download_hls(opts[lab],dst); q=probe(dst)
        if (q['width'],q['height'])!=(1920,1080): raise RuntimeError(f'{lab} was {q}')
        report.append({'rank':i,'label':lab,'file':dst.name,**q})
    (out/'qa.json').write_text(json.dumps({'game_id':gid,'event_id':eid,'angles':report},indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
