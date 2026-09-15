#!/usr/bin/env python3
from __future__ import annotations
import argparse, html as htmlmod, json, re, subprocess, urllib.request
from pathlib import Path
import cv2, numpy as np

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'
HEAD=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
CANDIDATES=['Broadcast','High Tight','In Arena','Left Slash','Right Slash','Left HandHeld','Right HandHeld']

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':UA,'Referer':'https://clips.nba.com/'})
    with urllib.request.urlopen(req,timeout=45) as r:return r.read().decode('utf-8','replace')

def options(game,event):
    txt=get(f'https://clips.nba.com/?gameNo={game}&eventNum={event}&source=grs'); out={}
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',txt,flags=re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip()); lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower():out[lab]=u
    return out

def download(url,path):
    subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',HEAD,'-i',url,'-map','0:v:0','-map','0:a:0?','-c','copy',str(path)],check=True,timeout=240)

def frame(path,t):
    c=cv2.VideoCapture(str(path)); c.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,im=c.read(); c.release(); return im if ok else None

def montage(rows,times,out):
    thumbs=[]
    for lab,path in rows:
        ims=[]
        for t in times:
            im=frame(path,t)
            if im is None: im=np.zeros((540,960,3),np.uint8)
            cv2.putText(im,f'{lab}  {t:.1f}s',(16,28),cv2.FONT_HERSHEY_SIMPLEX,.70,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(im,f'{lab}  {t:.1f}s',(16,28),cv2.FONT_HERSHEY_SIMPLEX,.70,(255,255,255),2,cv2.LINE_AA)
            ims.append(cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA))
        thumbs.append(np.hstack(ims))
    cv2.imwrite(str(out),np.vstack(thumbs),[cv2.IMWRITE_JPEG_QUALITY,94])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    m=json.loads(a.manifest.read_text()); report=[]; times=[6.0,8.0,10.0,12.0,14.0]
    for ei,e in enumerate(m['selected_events'],1):
        d=a.out/f'event_{ei}';d.mkdir(exist_ok=True);opts=options(e['game_id'],e['shot_event_num']);rows=[]
        for lab in CANDIDATES:
            if lab not in opts:continue
            p=d/(re.sub('[^A-Za-z0-9]+','_',lab)+'.mp4');download(opts[lab],p);rows.append((lab,p))
        montage(rows,times,d/'angle_montage.jpg')
        report.append({'event_index':ei,'game_id':e['game_id'],'event_num':e['shot_event_num'],'labels':[x[0] for x in rows],'times_s':times})
    (a.out/'preview_qa.json').write_text(json.dumps({'events':report},indent=2))
if __name__=='__main__':main()
