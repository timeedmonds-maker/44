#!/usr/bin/env python3
import argparse, html, json, re, subprocess
from pathlib import Path
import requests

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}

def parse(game,event):
    page=f'https://clips.nba.com/?gameNo={game}&eventNum={event}&source=grs'
    r=requests.get(page,headers=H,timeout=45); r.raise_for_status(); text=r.text
    opts=[]
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',text,flags=re.I|re.S):
        url=html.unescape(m.group(1).strip()); attrs=m.group(2).lower(); label=re.sub(r'<[^>]+>','',html.unescape(m.group(3))).strip()
        if '.m3u8' in url.lower() and 'lrmedia.nba.com' in url.lower(): opts.append({'url':url,'selected':'selected' in attrs,'label':label})
    if not opts: raise RuntimeError('no signed lrmedia HLS')
    # Prefer Broadcast explicitly for stable court calibration.
    chosen=next((o for o in opts if o['label'].strip().lower()=='broadcast'),None) or next((o for o in opts if o['selected']),opts[0])
    return page,opts,chosen

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--game',required=True); ap.add_argument('--event',type=int,required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    page,opts,ch=parse(a.game,a.event)
    headers=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',headers,'-i',ch['url'],'-map','0:v:0','-map','0:a:0?','-c','copy','-movflags','+faststart',str(out)],check=True)
    p=subprocess.run(['ffprobe','-v','error','-show_entries','stream=width,height,avg_frame_rate:format=duration','-of','json',str(out)],capture_output=True,text=True,check=True)
    meta={'game_id':a.game,'event_num':a.event,'page_url':page,'chosen_angle':ch['label'],'angle_count':len(opts),'probe':json.loads(p.stdout)}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
