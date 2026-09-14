#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess, sys
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from render_deterministic_master import render as render_presentation

PLAYERS={9:('DURANT','HOU'),6:('ADAMS','HOU'),1:('LARAVIA','LAL'),11:('AYTON','LAL')}
TEAM={
 'HOU': {'ring':(45,55,235),'plate':(41,48,170)},
 'LAL': {'ring':(35,190,250),'plate':(110,50,105)},
}
RELEASE=10.56
XFG=42.6
CONTACT_START=8.735
FREEZE_S=1.20
START=7.20
END=13.00
FONT_REG='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'

# Deterministic release anchors from the validated pose probe.
DURANT_RELEASE=(381.65057373046875,390.4165802001953)
LARAVIA_RELEASE=(676.9086303710938,385.4612731933594)
# Four visible paint corners at the release frame. The NBA paint is 19 ft x 16 ft.
PAINT_IMG=np.array([[263,293],[165,366],[575,313],[512,387]],dtype=np.float32)
PAINT_COURT=np.array([[0,518],[0,1006],[579,518],[579,1006]],dtype=np.float32)


def font(sz,bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG,sz)

def alpha_poly(frame, pts, color, alpha):
    ov=frame.copy(); cv2.fillPoly(ov,[np.array(pts,np.int32)],color); cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

def alpha_rect(frame,p1,p2,color,alpha):
    ov=frame.copy(); cv2.rectangle(ov,p1,p2,color,-1); cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

def pil_text(frame, xy, text, size, fill=(255,255,255), bold=False, anchor=None):
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(im); d.text(xy,text,font=font(size,bold),fill=fill,anchor=anchor); return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)

def draw_broadcast_tile(frame, x,y,w,h,kicker,value, accent=(190,190,190), value_size=28):
    pts=[(x,y),(x+w-10,y),(x+w,y+10),(x+w,y+h),(x,y+h)]
    alpha_poly(frame,pts,(15,20,27),0.84)
    cv2.polylines(frame,[np.array(pts,np.int32)],True,(98,104,112),1,cv2.LINE_AA)
    cv2.line(frame,(x+10,y+7),(x+42,y+7),accent,2,cv2.LINE_AA)
    frame=pil_text(frame,(x+11,y+14),kicker,11,(208,213,218),True)
    frame=pil_text(frame,(x+11,y+29),value,value_size,(250,250,250),True)
    return frame

def draw_contact_strip(frame, x,y,w,elapsed):
    h=34; pts=[(x,y),(x+w-7,y),(x+w,y+7),(x+w,y+h),(x,y+h)]
    alpha_poly(frame,pts,(15,20,27),0.84); cv2.polylines(frame,[np.array(pts,np.int32)],True,(90,96,104),1,cv2.LINE_AA)
    cv2.line(frame,(x+8,y+h-5),(x+42,y+h-5),(48,57,205),2,cv2.LINE_AA)
    frame=pil_text(frame,(x+10,y+8),'SCREEN CONTACT',10,(196,202,208),True)
    frame=pil_text(frame,(x+w-12,y+8),f'{elapsed:.1f} s',15,(255,255,255),True,'ra')
    return frame

def draw_name(frame,cx,head_y,name,team):
    fs=13; f=font(fs,True); dummy=Image.new('RGB',(1,1)); d=ImageDraw.Draw(dummy); bb=d.textbbox((0,0),name,font=f); tw=bb[2]-bb[0]
    w=tw+18; h=23; x=int(round(cx-w/2)); y=int(round(head_y-31));
    x=max(2,min(frame.shape[1]-w-2,x)); y=max(2,min(frame.shape[0]-h-2,y))
    col=TEAM[team]['plate']; alpha_rect(frame,(x,y),(x+w,y+h),col,.92); cv2.line(frame,(x+4,y+h-2),(x+w-4,y+h-2),(235,235,235),1,cv2.LINE_AA)
    return pil_text(frame,(x+w/2,y+h/2),name,fs,(255,255,255),True,'mm')

def draw_ring(frame,cx,cy,team):
    # Equal geometry for both teams. No solid black anywhere.
    # The rear arc uses a darker team tint, broken at centre so it sits behind the legs.
    rx,ry=35,13; cx=int(round(cx)); cy=int(round(cy)); col=TEAM[team]['ring']
    if team=='HOU': rear=(35,45,145); outer=(60,70,205)
    else: rear=(65,125,175); outer=(80,165,220)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,188,253,rear,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,287,352,rear,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,outer,7,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,col,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,outer,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,col,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,outer,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,col,4,cv2.LINE_AA)

def dotted_line(frame,p1,p2,color=(245,245,245),radius=3,gap=11):
    p1=np.array(p1,float); p2=np.array(p2,float); dist=np.linalg.norm(p2-p1)
    if dist<1:return
    n=max(2,int(dist/gap))
    for a in np.linspace(0,1,n):
        p=(1-a)*p1+a*p2
        cv2.circle(frame,(int(round(p[0])),int(round(p[1]))),radius,(150,154,160),-1,cv2.LINE_AA)
        cv2.circle(frame,(int(round(p[0])),int(round(p[1]))),max(1,radius-1),color,-1,cv2.LINE_AA)

def build_track_interpolators(df):
    out={}
    for tid in PLAYERS:
        g=df[df.track_id==tid].sort_values('time_s')
        if g.empty: continue
        t=g.time_s.to_numpy(float); vals=g[['x1','y1','x2','y2']].to_numpy(float)
        sm=vals.copy(); kernel=np.array([1,2,3,2,1],float); kernel/=kernel.sum()
        for j in range(4):
            pad=np.pad(vals[:,j],(2,2),mode='edge'); sm[:,j]=np.convolve(pad,kernel,mode='valid')
        out[tid]=(t,sm)
    return out

def box_at(interp,tid,t):
    if tid not in interp:return None
    tt,v=interp[tid]
    if t<tt[0]-0.08 or t>tt[-1]+0.08:return None
    return tuple(float(np.interp(t,tt,v[:,j])) for j in range(4))

def local_distance_ft():
    H=cv2.getPerspectiveTransform(PAINT_IMG,PAINT_COURT)
    def court(p):
        v=H@np.array([p[0],p[1],1.0],dtype=np.float64); return v[:2]/v[2]
    d=court(DURANT_RELEASE); l=court(LARAVIA_RELEASE)
    return float(np.linalg.norm(d-l)/30.48),d,l,H

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--tracks',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(a.tracks); interp=build_track_interpolators(df)
    # User-validated contact onset; timer stops at Durant release, not physical separation.
    cstart=CONTACT_START; cend=RELEASE
    laravia_distance_ft,durant_cm,laravia_cm,Hlocal=local_distance_ft()

    cap=cv2.VideoCapture(a.source); fps=float(cap.get(cv2.CAP_PROP_FPS) or 29.97); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
    sf=int(round(START*fps)); ef=int(round(END*fps)); cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    tmp=out/'v5_silent.mp4'; wr=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
    freeze_n=int(round(FREEZE_S*fps)); release_written=False; frame_idx=sf; rendered=0
    preview_times=[7.20,8.74,9.56,10.56,11.74,12.56]; saved=set()
    while frame_idx<ef:
        ok,orig=cap.read();
        if not ok: break
        t=frame_idx/fps; fr=orig.copy()
        for tid,(name,tm) in PLAYERS.items():
            b=box_at(interp,tid,t)
            if b is None: continue
            cx=(b[0]+b[2])/2; cy=b[3]-4.0
            draw_ring(fr,cx,cy,tm); fr=draw_name(fr,cx,b[1],name,tm)
        fr=draw_broadcast_tile(fr,W-222,20,202,61,'SCREEN COVERAGE','DEEP DROP',(192,196,202),25)
        if cstart<=t:
            elapsed=max(0.0,min(t,cend)-cstart); fr=draw_contact_strip(fr,W-222,87,202,elapsed)
        if RELEASE-1.0<=t<=RELEASE+2.0:
            fr=draw_broadcast_tile(fr,20,20,194,66,'SHOT QUALITY',f'xFG  {XFG:.1f}%',TEAM['HOU']['ring'],27)
        is_release_frame=abs(t-RELEASE)<=0.5/fps
        if is_release_frame and not release_written:
            dotted_line(fr,DURANT_RELEASE,LARAVIA_RELEASE)
            mx=(DURANT_RELEASE[0]+LARAVIA_RELEASE[0])/2; my=(DURANT_RELEASE[1]+LARAVIA_RELEASE[1])/2
            tagw,tagh=152,38; x=int(max(4,min(W-tagw-4,mx-tagw/2))); y=int(max(4,min(H-tagh-4,my-50)))
            pts=[(x,y),(x+tagw-6,y),(x+tagw,y+6),(x+tagw,y+tagh),(x,y+tagh)]
            alpha_poly(fr,pts,(15,20,27),.87); cv2.polylines(fr,[np.array(pts,np.int32)],True,(110,116,124),1,cv2.LINE_AA)
            fr=pil_text(fr,(x+8,y+6),'LARAVIA GAP',9,(202,207,212),True); fr=pil_text(fr,(x+8,y+17),f'{laravia_distance_ft:.1f} FT',17,(255,255,255),True)
            wr.write(fr); rendered+=1
            for _ in range(freeze_n): wr.write(fr); rendered+=1
            cv2.imwrite(str(out/'release_freeze.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,97]); release_written=True
        else:
            wr.write(fr); rendered+=1
        for pt in preview_times:
            if pt not in saved and abs(t-pt)<=0.5/fps:
                cv2.imwrite(str(out/f'preview_{pt:.2f}.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,95]); saved.add(pt)
        frame_idx+=1
    cap.release(); wr.release()

    native=out/'durant_adams_christmas_broadcast_v5_native.mp4'
    fc=(f"[0:a]atrim=start={START:.3f}:end={RELEASE:.3f},asetpts=PTS-STARTPTS[a0];"
        f"anullsrc=r=48000:cl=stereo,atrim=duration={FREEZE_S:.3f}[sil];"
        f"[0:a]atrim=start={RELEASE:.3f}:end={END:.3f},asetpts=PTS-STARTPTS[a1];"
        f"[a0][sil][a1]concat=n=3:v=0:a=1[a]")
    subprocess.run(['ffmpeg','-y','-v','error','-i',a.source,'-i',str(tmp),'-filter_complex',fc,'-map','1:v:0','-map','[a]','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',str(native)],check=True)

    uhd=out/'durant_adams_christmas_broadcast_v5_UHD.mp4'
    presentation_qa=render_presentation(native,uhd,'uhd',preset='medium')
    qa={
      'source_start_s':START,'source_end_s':END,'release_s':RELEASE,'freeze_s':FREEZE_S,
      'xfg_pct':XFG,'xfg_source_window':[RELEASE-1,RELEASE+2],
      'contact_start_s':cstart,'contact_end_s':cend,'contact_duration_s':cend-cstart,
      'distance_target':'LaRavia (initial defender)','distance_ft':laravia_distance_ft,
      'durant_release_anchor':DURANT_RELEASE,'laravia_release_anchor':LARAVIA_RELEASE,
      'paint_corners_img':PAINT_IMG.tolist(),'paint_corners_court_cm':PAINT_COURT.tolist(),
      'durant_court_cm':durant_cm.tolist(),'laravia_court_cm':laravia_cm.tolist(),
      'local_homography':Hlocal.tolist(),'rings_fixed_axes_native_px':[35,13],
      'ring_black_pixels_intentionally_drawn':False,'participants':PLAYERS,
      'native':str(native),'uhd':str(uhd),'presentation_qa':presentation_qa,
      'deterministic_only':True,
      'method':'official NBA source + deterministic tracking/pose anchors + local court homography + OpenCV/Pillow/FFmpeg; no generated video pixels'
    }
    (out/'qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
