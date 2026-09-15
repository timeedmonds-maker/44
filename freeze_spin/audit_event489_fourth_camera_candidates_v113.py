from __future__ import annotations

"""Exact-event native-raster fourth-camera candidate audit.

Discovery only.  Downloads official HLS for candidate labels at one exact NBA event,
keeps every source at its native raster, extracts deterministic frames, and ranks only
for visible fixed-geometry information.  A rank can never promote a camera.
"""
import argparse, json, math, re, subprocess
from pathlib import Path
import cv2, numpy as np
from freeze_spin.scan_same_game_camera_priors import inventory_camera, safe_label
import nba_video_worker as w

CANDIDATES=[
 'Left HandHeld','Right HandHeld','High Tight','In Arena','Left Slash','Right Slash',
 'Left Above Rim','Right Above Rim','Broadcast','Other Broadcast','Mobile Broadcast','Play by Play'
]

def extract_frames(video:Path,out:Path,n:int=11):
 q=w.probe_video(video)
 if not q.get('ok'): raise RuntimeError(q)
 dur=float(q['duration']);out.mkdir(parents=True,exist_ok=True);rows=[]
 for i,frac in enumerate(np.linspace(.14,.86,n)):
  t=max(.05,min(dur-.05,dur*float(frac)));p=out/f'f{i:02d}.png'
  subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-ss',f'{t:.5f}','-i',str(video),'-frames:v','1',str(p)],check=True)
  rows.append(p)
 return rows

def metrics(path:Path):
 im=cv2.imread(str(path));h,wid=im.shape[:2];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var());edges=cv2.Canny(gray,70,160)
 lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(35,int(wid*.065)),minLineLength=max(35,int(wid*.075)),maxLineGap=max(5,int(wid*.01)))
 nline=0 if lines is None else int(len(lines));long=0
 if lines is not None:
  for x1,y1,x2,y2 in lines[:,0]:
   if math.hypot(float(x2-x1),float(y2-y1))>=wid*.16: long+=1
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);hh,ss,vv=cv2.split(hsv);Y,X=np.indices(hh.shape)
 # Orange/red basket hardware signal in upper 70%; diagnostic only.
 orange=(((hh<=25)|(hh>=170))&(ss>=85)&(vv>=65)&(Y<h*.72)).astype(np.uint8)
 n,lab,stats,_=cv2.connectedComponentsWithStats(orange,8);orange_max=int(np.max(stats[1:,cv2.CC_STAT_AREA])) if n>1 else 0
 # White/gray rectangular basket/board signal in upper 65%; diagnostic only.
 white=((ss<=75)&(vv>=135)&(Y<h*.65)).astype(np.uint8)
 n,lab,stats,_=cv2.connectedComponentsWithStats(white,8);board_like=0
 for i in range(1,n):
  x,y,bw,bh,a=map(int,stats[i,:5])
  if bw>=wid*.06 and bh>=h*.035 and .7<=bw/max(bh,1)<=4.0: board_like=max(board_like,a)
 floor_edges=float(np.mean(edges[int(h*.56):]>0))
 # Score is selection convenience only; no metric meaning.
 score=.004*sharp+1.5*long+.004*board_like+.025*orange_max+250*floor_edges
 return {'sharpness':sharp,'hough_lines':nline,'long_hough_lines':long,'orange_component_max_area':orange_max,'white_board_like_max_area':board_like,'floor_edge_fraction':floor_edges,'selection_score':float(score),'width':wid,'height':h}

def thumb(path:Path,label:str,m:dict):
 im=cv2.imread(str(path));t=cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA)
 cv2.rectangle(t,(0,0),(480,48),(0,0,0),-1);cv2.putText(t,label,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
 cv2.putText(t,f"native {m['width']}x{m['height']} score {m['selection_score']:.1f}",(8,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
 return t

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--game-id',required=True);ap.add_argument('--event-id',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[];cells=[]
 for label in CANDIDATES:
  slug=safe_label(label);rec={'label':label,'status':'failed'}
  try:
   url,title=inventory_camera(a.game_id,a.event_id,label);clip=a.out/f'{slug}_SOURCE.mp4';w.download_hls_source(url,clip);q=w.probe_video(clip)
   if not q.get('ok'): raise RuntimeError(q)
   fs=extract_frames(clip,a.out/f'{slug}_frames');ms=[metrics(p) for p in fs];j=max(range(len(fs)),key=lambda i:ms[i]['selection_score'])
   rec.update({'status':'ok','title':title,'probe':q,'frames':[p.name for p in fs],'selected_frame':str(fs[j].relative_to(a.out)),'selected_metrics':ms[j],'all_frame_metrics':ms})
   cells.append(thumb(fs[j],label,ms[j]));clip.unlink(missing_ok=True)
  except Exception as exc:rec['error']=repr(exc)
  rows.append(rec);print('CANDIDATE',label,rec['status'],rec.get('selected_metrics'),rec.get('error'),flush=True)
 if cells:
  cols=3;rr=[]
  for i in range(0,len(cells),cols):
   r=cells[i:i+cols]
   while len(r)<cols:r.append(np.zeros_like(cells[0]))
   rr.append(np.hstack(r))
  cv2.imwrite(str(a.out/'event489_fourth_camera_candidates_montage.png'),np.vstack(rr))
 ranked=sorted([r for r in rows if r['status']=='ok'],key=lambda r:r['selected_metrics']['selection_score'],reverse=True)
 payload={'schema_version':1,'status':'DISCOVERY_ONLY_NO_PROMOTION','game_id':a.game_id,'event_id':a.event_id,'purpose':'Native-raster exact-event audit for a clean independent fourth metric camera. Selection score is not calibration evidence.','candidates':rows,'diagnostic_ranking':[{'label':r['label'],'selected_frame':r['selected_frame'],'metrics':r['selected_metrics']} for r in ranked],'permissions':{'physical_camera_identity_allowed':False,'metric_camera_allowed':False,'replay_render_allowed':False}}
 (a.out/'event489_fourth_camera_candidates_v113.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps({'status':payload['status'],'ok_labels':[r['label'] for r in ranked],'ranking':[r['label'] for r in ranked]},indent=2))
 if len(ranked)<4: raise SystemExit('fewer than four real candidate feeds')
if __name__=='__main__':main()
