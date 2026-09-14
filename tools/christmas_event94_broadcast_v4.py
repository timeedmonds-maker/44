#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

PLAYERS={9:('DURANT','HOU'),6:('ADAMS','HOU'),1:('LARAVIA','LAL'),11:('AYTON','LAL')}
TEAM={'HOU':{'ring':(45,55,235),'plate':(41,48,170)},'LAL':{'ring':(35,190,250),'plate':(110,50,105)}}
RELEASE=10.56; XFG=42.6; FREEZE_S=1.20; START=7.20; END=13.00
FONT_REG='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf'; FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'
def font(sz,bold=False): return ImageFont.truetype(FONT_BOLD if bold else FONT_REG,sz)
def alpha_poly(frame,pts,color,alpha):
    ov=frame.copy(); cv2.fillPoly(ov,[np.array(pts,np.int32)],color); cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
def alpha_rect(frame,p1,p2,color,alpha):
    ov=frame.copy(); cv2.rectangle(ov,p1,p2,color,-1); cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
def pil_text(frame,xy,text,size,fill=(255,255,255),bold=False,anchor=None):
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(im); d.text(xy,text,font=font(size,bold),fill=fill,anchor=anchor); return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
def draw_tile(frame,x,y,w,h,kicker,value,accent=(190,190,190),value_size=28):
    pts=[(x,y),(x+w-10,y),(x+w,y+10),(x+w,y+h),(x,y+h)]; alpha_poly(frame,pts,(15,20,27),.84); cv2.polylines(frame,[np.array(pts,np.int32)],True,(98,104,112),1,cv2.LINE_AA); cv2.line(frame,(x+10,y+7),(x+42,y+7),accent,2,cv2.LINE_AA); frame=pil_text(frame,(x+11,y+14),kicker,11,(208,213,218),True); return pil_text(frame,(x+11,y+29),value,value_size,(250,250,250),True)
def draw_contact(frame,x,y,w,elapsed):
    h=34; pts=[(x,y),(x+w-7,y),(x+w,y+7),(x+w,y+h),(x,y+h)]; alpha_poly(frame,pts,(15,20,27),.84); cv2.polylines(frame,[np.array(pts,np.int32)],True,(90,96,104),1,cv2.LINE_AA); cv2.line(frame,(x+8,y+h-5),(x+42,y+h-5),(48,57,205),2,cv2.LINE_AA); frame=pil_text(frame,(x+10,y+8),'SCREEN CONTACT',10,(196,202,208),True); return pil_text(frame,(x+w-12,y+8),f'{elapsed:.1f} s',15,(255,255,255),True,'ra')
def draw_name(frame,cx,head_y,name,team):
    fs=13; f=font(fs,True); d=ImageDraw.Draw(Image.new('RGB',(1,1))); bb=d.textbbox((0,0),name,font=f); w=bb[2]-bb[0]+18; h=23; x=int(round(cx-w/2)); y=int(round(head_y-31)); x=max(2,min(frame.shape[1]-w-2,x)); y=max(2,min(frame.shape[0]-h-2,y)); alpha_rect(frame,(x,y),(x+w,y+h),TEAM[team]['plate'],.92); cv2.line(frame,(x+4,y+h-2),(x+w-4,y+h-2),(235,235,235),1,cv2.LINE_AA); return pil_text(frame,(x+w/2,y+h/2),name,fs,(255,255,255),True,'mm')
def draw_ring(frame,cx,cy,team):
    # Exactly identical ring axes for both teams. Feet sit at the ring centre.
    rx,ry=35,13; cx=int(round(cx)); cy=int(round(cy)); dark=(18,20,24); col=TEAM[team]['ring']
    # Rear black loop remains visible on either side of the legs; centre gap represents player occlusion.
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,188,253,dark,5,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,287,352,dark,5,cv2.LINE_AA)
    # Front coloured arc with broadcast keyline.
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,dark,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,col,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,dark,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,col,4,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,dark,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,col,4,cv2.LINE_AA)
def dotted(frame,p1,p2):
    p1=np.array(p1,float); p2=np.array(p2,float); dist=np.linalg.norm(p2-p1); n=max(2,int(dist/11))
    for a in np.linspace(0,1,n):
        p=(1-a)*p1+a*p2; q=(int(round(p[0])),int(round(p[1]))); cv2.circle(frame,q,3,(20,22,26),-1,cv2.LINE_AA); cv2.circle(frame,q,2,(245,245,245),-1,cv2.LINE_AA)
def interpolators(df):
    out={}
    for tid in PLAYERS:
        g=df[df.track_id==tid].sort_values('time_s')
        if g.empty: continue
        t=g.time_s.to_numpy(float); v=g[['x1','y1','x2','y2']].to_numpy(float); sm=v.copy(); k=np.array([1,2,3,2,1],float); k/=k.sum()
        for j in range(4): sm[:,j]=np.convolve(np.pad(v[:,j],(2,2),mode='edge'),k,mode='valid')
        out[tid]=(t,sm)
    return out
def box_at(it,tid,t):
    if tid not in it:return None
    tt,v=it[tid]
    if t<tt[0]-.08 or t>tt[-1]+.08:return None
    return tuple(float(np.interp(t,tt,v[:,j])) for j in range(4))
def contact_times(df):
    # First sustained Adams-LaRavia body overlap, then first sustained >12px separation.
    raw={}
    for tid in (6,1):
        g=df[df.track_id==tid].sort_values('time_s'); raw[tid]=(g.time_s.to_numpy(float),g[['x1','y1','x2','y2']].to_numpy(float))
    t0=max(raw[6][0][0],raw[1][0][0]); t1=min(raw[6][0][-1],raw[1][0][-1]); ts=np.arange(t0,t1+1e-6,1/60)
    def b(tid,t):
        tt,v=raw[tid]; return [float(np.interp(t,tt,v[:,j])) for j in range(4)]
    g=[]
    for t in ts:
        a=b(6,t); l=b(1,t); gap=max(l[0]-a[2],a[0]-l[2],0); yov=max(0,min(a[3],l[3])-max(a[1],l[1])); g.append((t,gap,yov))
    st=None
    for i in range(len(g)-2):
        if all(z[1]<=1.5 and z[2]>35 for z in g[i:i+3]): st=g[i][0]; break
    en=None
    if st is not None:
        si=next(i for i,z in enumerate(g) if z[0]>=st)
        for i in range(si,len(g)-3):
            if all(z[1]>12 for z in g[i:i+4]): en=g[i][0]; break
    return float(st or 8.735),float(en or 11.738)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--tracks',required=True); ap.add_argument('--primary-json',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.tracks); it=interpolators(df); cs,ce=contact_times(df); pr=json.loads(Path(a.primary_json).read_text()); primary=pr['primary_defender']; dist=float(primary['distance_ft']); defp=tuple(primary['anchor'])
    cap=cv2.VideoCapture(a.source); fps=float(cap.get(cv2.CAP_PROP_FPS) or 29.97); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); sf=int(round(START*fps)); ef=int(round(END*fps)); cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    tmp=out/'v4_silent.mp4'; wr=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); freeze_n=int(round(FREEZE_S*fps)); release_written=False; fi=sf
    previews=[7.20,8.74,9.56,10.56,11.74,12.56]; saved=set()
    while fi<ef:
        ok,orig=cap.read();
        if not ok:break
        t=fi/fps; fr=orig.copy(); cur={}
        for tid,(nm,tm) in PLAYERS.items():
            b=box_at(it,tid,t)
            if b is None:continue
            cur[tid]=b; cx=(b[0]+b[2])/2; cy=b[3]-4; draw_ring(fr,cx,cy,tm); fr=draw_name(fr,cx,b[1],nm,tm)
        fr=draw_tile(fr,W-222,20,202,61,'SCREEN COVERAGE','DEEP DROP',(192,196,202),25)
        if cs<=t: fr=draw_contact(fr,W-222,87,202,max(0,min(t,ce)-cs))
        if RELEASE-1<=t<=RELEASE+2: fr=draw_tile(fr,20,20,194,66,'SHOT QUALITY',f'xFG  {XFG:.1f}%',TEAM['HOU']['ring'],27)
        isrel=abs(t-RELEASE)<=.5/fps
        if isrel and not release_written:
            db=cur.get(9); dp=((db[0]+db[2])/2,db[3]-4) if db else tuple(pr['durant']['anchor']); dotted(fr,dp,defp); mx=(dp[0]+defp[0])/2; my=(dp[1]+defp[1])/2; tw,th=152,38; x=int(max(4,min(W-tw-4,mx-tw/2))); y=int(max(4,min(H-th-4,my-50))); pts=[(x,y),(x+tw-6,y),(x+tw,y+6),(x+tw,y+th),(x,y+th)]; alpha_poly(fr,pts,(15,20,27),.87); cv2.polylines(fr,[np.array(pts,np.int32)],True,(110,116,124),1,cv2.LINE_AA); fr=pil_text(fr,(x+8,y+6),'PRIMARY DEFENDER',9,(202,207,212),True); fr=pil_text(fr,(x+8,y+17),f'{dist:.1f} FT',17,(255,255,255),True); wr.write(fr); [wr.write(fr) for _ in range(freeze_n)]; cv2.imwrite(str(out/'release_freeze.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,97]); release_written=True
        else: wr.write(fr)
        for pt in previews:
            if pt not in saved and abs(t-pt)<=.5/fps: cv2.imwrite(str(out/f'preview_{pt:.2f}.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,95]); saved.add(pt)
        fi+=1
    cap.release(); wr.release()
    native=out/'durant_adams_christmas_broadcast_v4_native.mp4'; fc=(f"[0:a]atrim=start={START:.3f}:end={RELEASE:.3f},asetpts=PTS-STARTPTS[a0];anullsrc=r=48000:cl=stereo,atrim=duration={FREEZE_S:.3f}[sil];[0:a]atrim=start={RELEASE:.3f}:end={END:.3f},asetpts=PTS-STARTPTS[a1];[a0][sil][a1]concat=n=3:v=0:a=1[a]")
    subprocess.run(['ffmpeg','-y','-v','error','-i',a.source,'-i',str(tmp),'-filter_complex',fc,'-map','1:v:0','-map','[a]','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',str(native)],check=True)
    uhd=out/'durant_adams_christmas_broadcast_v4_UHD.mp4'; vf='hqdn3d=1.0:0.8:2.0:1.6,scale=3840:2160:flags=lanczos,cas=strength=0.22,fps=30'; subprocess.run(['ffmpeg','-y','-v','error','-i',str(native),'-vf',vf,'-c:v','libx264','-profile:v','high','-crf','16','-preset','medium','-maxrate','36M','-bufsize','72M','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(uhd)],check=True)
    qa={'source_start_s':START,'source_end_s':END,'release_s':RELEASE,'freeze_s':FREEZE_S,'xfg_pct':XFG,'contact_start_s':cs,'contact_end_s':ce,'contact_duration_s':ce-cs,'distance_ft':dist,'primary_defender_track':primary.get('track_id'),'rings_fixed_axes_native_px':[35,13],'deterministic_only':True,'native':str(native),'uhd':str(uhd)}; (out/'qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__':main()
