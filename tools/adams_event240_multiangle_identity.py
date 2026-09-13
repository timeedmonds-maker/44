#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys,re
from pathlib import Path
from urllib.parse import urljoin,urlsplit,urlunsplit
import cv2,numpy as np,requests
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from fetch_official_nba_event_clip import parse,UA
H={'User-Agent':UA,'Referer':'https://clips.nba.com/'}

def variant(url,target=960):
 r=requests.get(url,headers=H,timeout=30); r.raise_for_status(); txt=r.text
 if '#EXT-X-STREAM-INF' not in txt:return url
 lines=[x.strip() for x in txt.splitlines() if x.strip()]; vs=[]
 for i,line in enumerate(lines):
  if not line.startswith('#EXT-X-STREAM-INF'):continue
  m=re.search(r'RESOLUTION=(\d+)x(\d+)',line); uri=next((lines[j] for j in range(i+1,min(i+4,len(lines))) if not lines[j].startswith('#')),None)
  if not uri:continue
  full=urljoin(url,uri)
  if not urlsplit(full).query and urlsplit(url).query:
   q=urlsplit(url); f=urlsplit(full); full=urlunsplit((f.scheme,f.netloc,f.path,q.query,f.fragment))
  vs.append((int(m.group(1)) if m else 10**9,full))
 under=[v for v in vs if v[0]<=target]
 return (max(under,key=lambda x:x[0]) if under else min(vs,key=lambda x:x[0]))[1] if vs else url

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--game',default='0022500001'); ap.add_argument('--event',type=int,default=240); ap.add_argument('--out',type=Path,default=Path('artifacts/adams_event240_multiangle')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 page,angles,_=parse(a.game,a.event); rows=[]; recs=[]; hdr=f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
 for i,ang in enumerate(angles):
  try:
   mp4=a.out/f'angle_{i:02d}.mp4'; subprocess.run(['ffmpeg','-y','-v','error','-rw_timeout','30000000','-headers',hdr,'-i',variant(ang['url']),'-c','copy',str(mp4)],check=True,timeout=240)
   cap=cv2.VideoCapture(str(mp4)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); imgs=[]
   for frac in (.36,.46,.56,.66):
    fi=max(0,min(n-1,int(round((n-1)*frac)))) if n else 0; cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ok,fr=cap.read()
    if not ok:continue
    fr=cv2.resize(fr,(480,270),interpolation=cv2.INTER_AREA); cv2.rectangle(fr,(0,0),(480,31),(0,0,0),-1); cv2.putText(fr,f'{i:02d} {ang["label"][:26]} {fi/fps:.2f}s',(7,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA); imgs.append(fr)
   cap.release()
   if imgs: rows.append(np.hstack(imgs))
   recs.append({'index':i,'label':ang['label'],'frames':n,'fps':fps,'status':'ok'})
  except Exception as e: recs.append({'index':i,'label':ang.get('label',''),'status':'error','error':str(e)})
 if rows: cv2.imwrite(str(a.out/'multiangle_identity_sheet.jpg'),np.vstack(rows))
 (a.out/'angles.json').write_text(json.dumps({'game_id':a.game,'event_num':a.event,'page':page,'angles':recs},indent=2)); print(json.dumps({'angle_count':len(angles),'ok':sum(x['status']=='ok' for x in recs)},indent=2))
if __name__=='__main__':main()
