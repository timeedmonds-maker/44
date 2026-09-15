#!/usr/bin/env python3
"""LOCKED_BROADCAST_SCREEN_V2

Deterministic screen-analysis renderer derived from LOCKED_BROADCAST_SCREEN_V1.
V2 preserves the locked Christmas-V7 visual primitives while adding:
- cumulative screen-contact time across explicitly validated disjoint contact phases;
- a second shot-quality tile for a player/season similar-shot xFG comparator.

No generated imagery. No AI super-resolution. Real source pixels + deterministic
OpenCV/Pillow overlays only.
"""
from pathlib import Path
import argparse, json, re, subprocess
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

TOOL_ID='LOCKED_BROADCAST_SCREEN_V2'
PANEL_FILL=(70,70,72)
PANEL_BORDER=(164,164,168)
KICKER_TEXT=(210,210,214)
VALUE_TEXT=(252,252,252)
PANEL_ALPHA=0.66
FONT_REG='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'

def font(sz,bold=False): return ImageFont.truetype(FONT_BOLD if bold else FONT_REG,sz)
def rgb_to_bgr(rgb): return (int(rgb[2]),int(rgb[1]),int(rgb[0]))
def tint(bgr,factor): return tuple(max(0,min(255,int(round(c*factor)))) for c in bgr)
def alpha_poly(frame,pts,color=PANEL_FILL,alpha=PANEL_ALPHA):
    ov=frame.copy(); cv2.fillPoly(ov,[np.array(pts,np.int32)],color); cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
def alpha_rect(frame,p1,p2,color,alpha):
    ov=frame.copy(); cv2.rectangle(ov,p1,p2,color,-1); cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
def pil_text(frame,xy,text,size,fill=(255,255,255),bold=False,anchor=None):
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(im); d.text(xy,text,font=font(size,bold),fill=fill,anchor=anchor); return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
def draw_tile(frame,x,y,w,h,kicker,value,value_size=26):
    pts=[(x,y),(x+w-8,y),(x+w,y+8),(x+w,y+h),(x,y+h)]
    alpha_poly(frame,pts); cv2.polylines(frame,[np.array(pts,np.int32)],True,PANEL_BORDER,1,cv2.LINE_AA)
    frame=pil_text(frame,(x+10,y+12),kicker,10,KICKER_TEXT,True)
    frame=pil_text(frame,(x+10,y+28),value,value_size,VALUE_TEXT,True)
    return frame
def draw_contact(frame,x,y,w,elapsed):
    h=34; pts=[(x,y),(x+w-7,y),(x+w,y+7),(x+w,y+h),(x,y+h)]
    alpha_poly(frame,pts); cv2.polylines(frame,[np.array(pts,np.int32)],True,PANEL_BORDER,1,cv2.LINE_AA)
    frame=pil_text(frame,(x+10,y+8),'SCREEN CONTACT',10,KICKER_TEXT,True)
    frame=pil_text(frame,(x+w-12,y+8),f'{elapsed:.1f} s',15,VALUE_TEXT,True,'ra')
    return frame
def draw_name(frame,cx,head_y,name,bgr,dx=0,dy=0):
    fs=13; f=font(fs,True); d=ImageDraw.Draw(Image.new('RGB',(1,1))); bb=d.textbbox((0,0),name,font=f); tw=bb[2]-bb[0]
    w=tw+18; h=23; x=int(round(cx+dx-w/2)); y=int(round(head_y-31+dy)); x=max(2,min(frame.shape[1]-w-2,x)); y=max(2,min(frame.shape[0]-h-2,y))
    alpha_rect(frame,(x,y),(x+w,y+h),bgr,.92); cv2.line(frame,(x+4,y+h-2),(x+w-4,y+h-2),(235,235,235),1,cv2.LINE_AA)
    return pil_text(frame,(x+w/2,y+h/2),name,fs,(255,255,255),True,'mm')
def draw_ring(frame,cx,cy,bgr):
    rx,ry=35,13; cx=int(round(cx)); cy=int(round(cy)); outer=tint(bgr,1.18); rear=tint(bgr,.63)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,188,253,rear,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,287,352,rear,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,outer,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,bgr,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,outer,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,bgr,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,outer,7,cv2.LINE_AA); cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,bgr,4,cv2.LINE_AA)
def dotted_line(frame,p1,p2,radius=4,gap=14):
    p1=np.array(p1,float); p2=np.array(p2,float); dist=float(np.linalg.norm(p2-p1))
    if dist<1:return
    n=max(2,int(dist/gap))
    for a in np.linspace(0,1,n):
        p=(1-a)*p1+a*p2; xy=(int(round(p[0])),int(round(p[1])))
        cv2.circle(frame,xy,radius+2,(20,23,28),-1,cv2.LINE_AA); cv2.circle(frame,xy,radius,(255,255,255),-1,cv2.LINE_AA)
def draw_distance_tag(frame,cx,head_y,label):
    f=font(17,True); d=ImageDraw.Draw(Image.new('RGB',(1,1))); bb=d.textbbox((0,0),label,font=f); tw=bb[2]-bb[0]
    w=max(72,tw+22); h=34; x=int(round(cx-w/2)); y=int(round(head_y-70)); x=max(4,min(frame.shape[1]-w-4,x)); y=max(4,min(frame.shape[0]-h-4,y))
    pts=[(x,y),(x+w-7,y),(x+w,y+7),(x+w,y+h),(x,y+h)]
    alpha_poly(frame,pts); cv2.polylines(frame,[np.array(pts,np.int32)],True,PANEL_BORDER,1,cv2.LINE_AA)
    return pil_text(frame,(x+w/2,y+h/2),label,17,VALUE_TEXT,True,'mm')
def parse_distance(desc):
    m=re.search(r"(\d+(?:\.\d+)?)\s*['’]",str(desc))
    if not m: raise ValueError(f'No shot distance found in description: {desc!r}')
    v=float(m.group(1)); return int(v) if abs(v-round(v))<1e-9 else v

def build_interp(df,tids):
    out={}
    for tid in tids:
        g=df[df.track_id==tid].sort_values('time_s')
        if g.empty: continue
        t=g.time_s.to_numpy(float); vals=g[['x1','y1','x2','y2']].to_numpy(float); sm=vals.copy(); k=np.array([1,2,3,2,1],float); k/=k.sum()
        for j in range(4): sm[:,j]=np.convolve(np.pad(vals[:,j],(2,2),mode='edge'),k,mode='valid')
        out[int(tid)]=(t,sm)
    return out
def box_at(interp,tid,t):
    if tid not in interp:return None
    tt,v=interp[tid]
    if t<tt[0]-.10 or t>tt[-1]+.10:return None
    return tuple(float(np.interp(t,tt,v[:,j])) for j in range(4))
def center(box): return ((box[0]+box[2])/2,(box[1]+box[3])/2)
def cumulative_contact(t,phases):
    total=0.0
    for s,e in phases:
        if t<=s: continue
        total += max(0.0,min(t,e)-s)
    return total

def mux_with_freeze_audio(source,silent_video,native,start,release,end,freeze):
    fc=(f"[0:a]atrim=start={start:.3f}:end={release:.3f},asetpts=PTS-STARTPTS[a0];" f"anullsrc=r=48000:cl=stereo,atrim=duration={freeze:.3f}[sil];" f"[0:a]atrim=start={release:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a1];" f"[a0][sil][a1]concat=n=3:v=0:a=1[a]")
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','a','-show_entries','stream=index','-of','csv=p=0',str(source)],capture_output=True,text=True)
    if probe.stdout.strip():
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(source),'-i',str(silent_video),'-filter_complex',fc,'-map','1:v:0','-map','[a]','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-shortest',str(native)],check=True)
    else:
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(silent_video),'-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(native)],check=True)

def render(source,tracks,config,out):
    cfg=json.load(open(config)); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    start=float(cfg['timing']['start_s']); end=float(cfg['timing']['end_s']); release=float(cfg['timing']['release_s']); freeze=float(cfg['timing'].get('freeze_s',1.2))
    shot_desc=cfg['shot']['description']; dist=parse_distance(shot_desc); dist_label=f'{dist:g} FT'
    team_bgr={k:rgb_to_bgr(v['primary_rgb']) for k,v in cfg['teams'].items()}
    players={int(p['track_id']):p for p in cfg['players']}; shooter=int(cfg['roles']['shooter_track_id']); defender=int(cfg['roles']['primary_defender_track_id'])
    if cfg['timing'].get('contact_phases'):
        phases=[[float(s),float(e)] for s,e in cfg['timing']['contact_phases']]
    else:
        phases=[[float(cfg['timing'].get('contact_start_s',release)),float(cfg['timing'].get('contact_end_s',release))]]
    assert phases and all(e>=s for s,e in phases), phases
    phases=sorted(phases); cstart=phases[0][0]; contact_total=sum(e-s for s,e in phases)
    similar=cfg['shot'].get('similar_xfg') or None
    df=pd.read_csv(tracks); interp=build_interp(df,players.keys())
    cap=cv2.VideoCapture(source); fps=float(cap.get(cv2.CAP_PROP_FPS) or 29.97); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); sf=int(round(start*fps)); ef=int(round(end*fps)); cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    silent=out/'locked_screen_silent.mp4'; wr=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); freeze_n=int(round(freeze*fps)); release_written=False; frame_idx=sf
    screen_value=cfg['screen'].get('value','SCREEN ACTION'); preview_times=[float(cfg['timing'].get('preview_screen_s',cstart+.2)),release]; saved=set()
    while frame_idx<ef:
        ok,orig=cap.read()
        if not ok:break
        t=frame_idx/fps; fr=orig.copy(); boxes={}
        for tid,p in players.items():
            b=box_at(interp,tid,t); boxes[tid]=b
            if b is None: continue
            cx=(b[0]+b[2])/2; cy=b[3]-4; col=team_bgr[p['team']]; draw_ring(fr,cx,cy,col); fr=draw_name(fr,cx,b[1],p['label'],col,float(p.get('label_dx',0)),float(p.get('label_dy',0)))
        fr=draw_tile(fr,W-222,20,202,61,cfg['screen'].get('kicker','SCREEN ACTION'),screen_value,22)
        if t>=cstart:
            fr=draw_contact(fr,W-222,87,202,cumulative_contact(t,phases))
        show_quality=release-1.0<=t<=release+2.0
        if cfg['shot'].get('league_xfg_pct') is not None and show_quality:
            fr=draw_tile(fr,20,20,194,66,'SHOT xFG',f"{float(cfg['shot']['league_xfg_pct']):.1f}%",27)
        if similar is not None and show_quality:
            n=int(similar.get('n',0)); kicker=f"SHEPPARD 25-26 SIMILAR  n={n}" if n else 'SHEPPARD 25-26 SIMILAR'
            fr=draw_tile(fr,20,92,238,66,kicker,f"xFG  {float(similar['xfg_pct']):.1f}%",25)
        is_release=abs(t-release)<=.5/fps
        if is_release and not release_written:
            bs=boxes.get(shooter); bd=boxes.get(defender)
            if bs and bd: dotted_line(fr,center(bs),center(bd))
            if bs:
                sp=players[shooter]; fr=draw_distance_tag(fr,(bs[0]+bs[2])/2+float(sp.get('distance_dx',0)),bs[1]+float(sp.get('distance_dy',0)),dist_label)
            cv2.imwrite(str(out/'release_freeze.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,97]); wr.write(fr)
            for _ in range(freeze_n): wr.write(fr)
            release_written=True
        else: wr.write(fr)
        for pt in preview_times:
            if pt not in saved and abs(t-pt)<=.5/fps:
                cv2.imwrite(str(out/f'preview_{pt:.2f}.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,95]); saved.add(pt)
        frame_idx+=1
    cap.release(); wr.release()
    native=out/f"{cfg['output_basename']}_native.mp4"; mux_with_freeze_audio(Path(source),silent,native,start,release,end,freeze)
    uhd=out/f"{cfg['output_basename']}_UHD.mp4"
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-i',str(native),'-vf','hqdn3d=0.6:0.6:2.0:2.0,scale=3840:2160:flags=lanczos,cas=0.22,fps=30','-c:v','libx264','-profile:v','high','-preset','medium','-crf','16','-maxrate','36M','-bufsize','72M','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(uhd)],check=True)
    qa={'tool_id':TOOL_ID,'parent_tool_id':'LOCKED_BROADCAST_SCREEN_V1','visual_lineage':{'repo':'timeedmonds-maker/44','branch':'codex/adams-screen-okc-opener','run_id':34826047210,'commit':'40d0ed3974ecbd0fcd433670cddd462c26f9e397','baseline':'Christmas event 94 V7'},'v2_changes':['cumulative_disjoint_contact_phases','player_season_similar_shot_xfg_tile'],'deterministic_only':True,'ai_image_generation':False,'ai_super_resolution':False,'shot_description':shot_desc,'shot_distance_tag':dist_label,'shot_distance_source':'shot_description_text_regex','event_xfg_pct':cfg['shot'].get('league_xfg_pct'),'similar_xfg':similar,'contact_phases':phases,'screen_contact_total_s':contact_total,'team_primary_rgb':{k:v['primary_rgb'] for k,v in cfg['teams'].items()},'panel_style':'neutral_translucent_grey_v1','native':str(native),'uhd':str(uhd),'release_s':release,'freeze_s':freeze}
    (out/'qa.json').write_text(json.dumps(qa,indent=2)); (out/'LOCKED_TOOL_ID.txt').write_text(TOOL_ID+'\n'); print(json.dumps(qa,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--tracks',required=True); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); render(a.source,a.tracks,a.config,a.out)
if __name__=='__main__': main()
