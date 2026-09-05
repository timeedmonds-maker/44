from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

CSV = Path('outputs/cavs_player_xefg_2025_26.csv')
OUT = Path('outputs/cavs_player_xefg_quadrant_2025_26_final.png')
HEADSHOT_URLS = [
    'https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png',
    'https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png',
]
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36'


def font(size, bold=False, italic=False):
    if bold and italic:
        p = '/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf'
    elif bold:
        p = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    elif italic:
        p = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'
    else:
        p = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(p, size)


def fetch_headshot(pid):
    s = requests.Session()
    s.headers.update({'User-Agent': UA, 'Referer': 'https://www.nba.com/'})
    for tmpl in HEADSHOT_URLS:
        try:
            r = s.get(tmpl.format(player_id=int(pid)), timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                return process(Image.open(io.BytesIO(r.content)).convert('RGBA'))
        except Exception:
            pass
    return None


def process(im):
    bb = im.getchannel('A').getbbox()
    if bb:
        im = im.crop(bb)
    # Same accepted deterministic headshot treatment: retain 70% of foreground height.
    w, h = im.size
    im = im.crop((0, 0, w, max(1, int(round(h * 0.70)))))
    bb = im.getchannel('A').getbbox()
    if bb:
        im = im.crop(bb)
    return im


def initials(name, size=180):
    im = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((4, 4, size - 4, size - 4), fill=(111, 38, 61, 255), outline=(253, 187, 48, 255), width=6)
    parts = [x for x in name.replace("'", '').replace('.', '').split() if x]
    txt = ''.join(x[0].upper() for x in parts[:2]) or '?'
    f = font(54, bold=True)
    bb = d.textbbox((0, 0), txt, font=f)
    d.text(((size - (bb[2]-bb[0]))/2, (size-(bb[3]-bb[1]))/2-5), txt, font=f, fill='white')
    return im


def normalize(im, w, h):
    bb = im.getchannel('A').getbbox()
    if bb:
        im = im.crop(bb)
    ratio = min(w / im.width, h / im.height)
    nw, nh = max(1, round(im.width * ratio)), max(1, round(im.height * ratio))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    canvas.alpha_composite(im, ((w-nw)//2, h-nh))
    return canvas


def centered(d, text, cx, y, f, fill=(0,0,0,255)):
    bb = d.textbbox((0, 0), text, font=f)
    d.text((cx - (bb[2]-bb[0])/2, y), text, font=f, fill=fill)


def draw_player(img, d, hs, cx, cy, row, name_font, stat_font, small_font, leader_y=None):
    if leader_y is not None and abs(cy - leader_y) > 14:
        d.line((cx, leader_y, cx, cy), fill=(180,180,180,255), width=2)
        d.ellipse((cx-5, leader_y-5, cx+5, leader_y+5), fill=(90,90,90,255))
    x0 = int(round(cx - hs.width/2))
    y0 = int(round(cy - hs.height/2 - 34))
    img.alpha_composite(hs, (x0, y0))
    centered(d, str(row.player_name), cx, y0 + hs.height + 2, name_font)
    if row.tracked_fga > 0:
        centered(d, f'xEFG {row.xEFG_pct:.1f}% | eFG {row.tracked_actual_eFG_pct:.1f}%', cx, y0 + hs.height + 37, stat_font, (45,45,45,255))
        centered(d, f'{row.eFG_minus_xEFG_pp:+.1f} pp | n={int(row.tracked_fga):,}', cx, y0 + hs.height + 65, small_font, (70,70,70,255))
    else:
        centered(d, 'xEFG — | eFG —', cx, y0 + hs.height + 37, stat_font, (45,45,45,255))
        centered(d, f'n=0 tracked | PBP FGA {int(row.pbp_fga)}', cx, y0 + hs.height + 65, small_font, (70,70,70,255))


def main():
    df = pd.read_csv(CSV)
    # Main chart uses meaningful tracked samples; every other CLE player still appears in the low-sample strip.
    main = df[df.tracked_fga >= 50].copy().reset_index(drop=True)
    small = df[df.tracked_fga < 50].copy().sort_values(['tracked_fga','player_name'], ascending=[False,True]).reset_index(drop=True)

    # Download all official NBA CDN headshots on GitHub Actions.
    shots = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_headshot, int(r.player_id)): (int(r.player_id), str(r.player_name)) for r in df.itertuples(index=False)}
        for fut in as_completed(futs):
            pid, name = futs[fut]
            im = fut.result()
            shots[pid] = im if im is not None else initials(name)

    W, H = 3200, 2400
    bg = (250,250,250,255)
    img = Image.new('RGBA', (W,H), bg)
    d = ImageDraw.Draw(img)

    L, T, R, B = 245, 335, 3070, 1815
    PW, PH = R-L, B-T

    x_min, x_max = 47.0, 65.0
    y_min, y_max = -15.5, 14.5
    team_x = 100 * df.expected_efg_num.sum() / df.tracked_fga.sum()
    xp = lambda x: L + (x-x_min)/(x_max-x_min)*PW
    yp = lambda y: B - (y-y_min)/(y_max-y_min)*PH

    f_title = font(78, bold=True)
    f_sub = font(38)
    f_axis = font(35, bold=True)
    f_tick = font(25)
    f_quad = font(27, italic=True)
    f_name = font(27, bold=True)
    f_stat = font(20)
    f_small = font(18)
    f_strip_title = font(28, bold=True)

    centered(d, 'Shot Quality & Shot Making', W/2, 42, f_title)
    centered(d, '2025-26 Cleveland Cavaliers players | NBA shot-level tracking data', W/2, 140, f_sub, (35,35,35,255))

    axis = (45,45,45,255)
    ref = (155,155,155,255)
    d.line((L,T,L,B), fill=axis, width=3)
    d.line((L,B,R,B), fill=axis, width=3)
    vx = xp(team_x)
    hy = yp(0)
    for y in range(T,B,28):
        d.line((vx,y,vx,min(y+16,B)), fill=ref, width=3)
    for x in range(L,R,28):
        d.line((x,hy,min(x+16,R),hy), fill=ref, width=3)

    for xt in range(48,65,2):
        px = xp(xt)
        d.line((px,B,px,B+12), fill=axis, width=3)
        centered(d, str(xt), px, B+18, f_tick, axis)
    for yt in range(-14,15,2):
        py = yp(yt)
        d.line((L-12,py,L,py), fill=axis, width=3)
        lab = str(yt)
        bb = d.textbbox((0,0),lab,font=f_tick)
        d.text((L-20-(bb[2]-bb[0]), py-(bb[3]-bb[1])/2), lab, font=f_tick, fill=axis)

    centered(d, 'Expected eFG% (shot quality)', (L+R)/2, 1905, f_axis)
    ylab = 'Actual eFG% vs expected (percentage points)'
    tmp = Image.new('RGBA',(1100,80),(255,255,255,0))
    td = ImageDraw.Draw(tmp)
    td.text((0,0), ylab, font=f_axis, fill=(0,0,0,255))
    tmp = tmp.rotate(90, expand=True)
    img.alpha_composite(tmp, (36, int((T+B-tmp.height)/2)))

    q = (82,82,82,255)
    d.multiline_text((L+55,T+28),'Worse shot quality /\nBetter than expected shot making',font=f_quad,fill=q,spacing=2)
    tr = 'Better shot quality /\nBetter than expected shot making'
    bb=d.multiline_textbbox((0,0),tr,font=f_quad,spacing=2)
    d.multiline_text((R-30-(bb[2]-bb[0]),T+28),tr,font=f_quad,fill=q,spacing=2)
    d.multiline_text((L+55,B-82),'Worse shot quality /\nWorse than expected shot making',font=f_quad,fill=q,spacing=2)
    br='Better shot quality /\nWorse than expected shot making'
    bb=d.multiline_textbbox((0,0),br,font=f_quad,spacing=2)
    d.multiline_text((R-30-(bb[2]-bb[0]),B-82),br,font=f_quad,fill=q,spacing=2)

    # Exact x locked; vertical-only deterministic relaxation for headshot/label spacing.
    anchors={int(r.player_id):(xp(float(r.xEFG_pct)),yp(float(r.eFG_minus_xEFG_pp))) for r in main.itertuples(index=False)}
    display_y={pid:p[1] for pid,p in anchors.items()}
    pids=list(anchors)
    min_sep=185.0
    max_disp=145.0
    for _ in range(900):
        moved=False
        for i in range(len(pids)):
            for j in range(i+1,len(pids)):
                a,b=pids[i],pids[j]
                if abs(anchors[a][0]-anchors[b][0])>175:
                    continue
                dy=display_y[b]-display_y[a]
                if abs(dy)<min_sep:
                    moved=True
                    sign=1 if dy>=0 else -1
                    if abs(dy)<1e-6:
                        sign=1 if anchors[b][1]>=anchors[a][1] else -1
                    push=(min_sep-abs(dy))*0.52
                    display_y[a]-=sign*push
                    display_y[b]+=sign*push
        for pid in pids:
            display_y[pid]+=(anchors[pid][1]-display_y[pid])*0.02
            delta=display_y[pid]-anchors[pid][1]
            if abs(delta)>max_disp:
                display_y[pid]=anchors[pid][1]+(max_disp if delta>0 else -max_disp)
        if not moved:
            break

    for r in main.sort_values('tracked_fga',ascending=True).itertuples(index=False):
        pid=int(r.player_id)
        hs=normalize(shots[pid],150,112)
        draw_player(img,d,hs,anchors[pid][0],display_y[pid],r,f_name,f_stat,f_small,leader_y=anchors[pid][1])

    # Low-sample strip keeps every Cavaliers player visible without letting tiny samples destroy the main axis scale.
    strip_top=2010
    d.rounded_rectangle((110,strip_top,3090,2360), radius=24, outline=(185,185,185,255), width=2, fill=(246,246,246,255))
    d.text((145,strip_top+24),'Low tracked-xFG sample (<50 FGA)',font=f_strip_title,fill=(0,0,0,255))
    d.text((145,strip_top+66),'Shown for completeness; not plotted in the main quadrant.',font=f_small,fill=(75,75,75,255))
    if len(small):
        x0=520
        avail=2480
        step=avail/max(1,len(small))
        for i,r in enumerate(small.itertuples(index=False)):
            cx=x0+step*(i+0.5)
            hs=normalize(shots[int(r.player_id)],126,92)
            img.alpha_composite(hs,(int(cx-hs.width/2),strip_top+110))
            centered(d,str(r.player_name),cx,strip_top+206,font(20,bold=True))
            if int(r.tracked_fga)>0:
                centered(d,f'xEFG {r.xEFG_pct:.1f}% | eFG {r.tracked_actual_eFG_pct:.1f}%',cx,strip_top+236,font(15),(50,50,50,255))
                centered(d,f'{r.eFG_minus_xEFG_pp:+.1f} pp | n={int(r.tracked_fga)}',cx,strip_top+260,font(14),(80,80,80,255))
            else:
                centered(d,'xEFG — | eFG —',cx,strip_top+236,font(15),(50,50,50,255))
                centered(d,f'n=0 tracked | PBP {int(r.pbp_fga)}',cx,strip_top+260,font(14),(80,80,80,255))

    img.convert('RGB').save(OUT,optimize=True,compress_level=9)
    print(f'Wrote {OUT} with {len(main)} plotted + {len(small)} low-sample = {len(df)} total CLE players')


if __name__=='__main__':
    main()
