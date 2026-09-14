#!/usr/bin/env python3
"""Deterministic Christmas event 94 broadcast V6.

V6 keeps the validated V5 geometry/timing/measurement and changes presentation only:
- higher-contrast Durant->LaRavia dotted release line
- ESPN-inspired graphite/red/white tactical panels (no ChatGPT-style colour bands)
- full-duration UHD QA is enforced by the workflow
No generated imagery or generative video processing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import christmas_event94_broadcast_v5 as base


def draw_broadcast_tile(frame, x, y, w, h, kicker, value, accent=(190,190,190), value_size=28):
    # ESPN-inspired tactical slab: sharp graphite card, white hierarchy, thin scorebug-like rule.
    pts=[(x,y),(x+w-8,y),(x+w,y+8),(x+w,y+h),(x,y+h)]
    base.alpha_poly(frame,pts,(10,12,16),0.90)
    cv2.polylines(frame,[np.array(pts,np.int32)],True,(68,72,78),1,cv2.LINE_AA)
    # Red rule for shot/contact information; neutral-white rule for coverage.
    is_neutral = kicker == 'SCREEN COVERAGE'
    rule = (220,220,220) if is_neutral else (42,48,220)
    cv2.line(frame,(x+8,y+7),(x+w-16,y+7),rule,2,cv2.LINE_AA)
    frame=base.pil_text(frame,(x+10,y+13),kicker,10,(188,193,199),True)
    frame=base.pil_text(frame,(x+10,y+28),value,value_size,(250,250,250),True)
    return frame


def draw_contact_strip(frame, x, y, w, elapsed):
    h=34
    pts=[(x,y),(x+w-7,y),(x+w,y+7),(x+w,y+h),(x,y+h)]
    base.alpha_poly(frame,pts,(10,12,16),0.90)
    cv2.polylines(frame,[np.array(pts,np.int32)],True,(68,72,78),1,cv2.LINE_AA)
    cv2.line(frame,(x+8,y+5),(x+w-16,y+5),(42,48,220),2,cv2.LINE_AA)
    frame=base.pil_text(frame,(x+10,y+9),'SCREEN CONTACT',10,(188,193,199),True)
    frame=base.pil_text(frame,(x+w-12,y+9),f'{elapsed:.1f} s',15,(255,255,255),True,'ra')
    return frame


def dotted_line(frame, p1, p2, color=(255,255,255), radius=4, gap=14):
    # Broadcast legibility: larger white dots with a dark keyline/shadow, still a true dotted line.
    p1=np.array(p1,float); p2=np.array(p2,float); dist=float(np.linalg.norm(p2-p1))
    if dist < 1:
        return
    n=max(2,int(dist/gap))
    for a in np.linspace(0,1,n):
        p=(1-a)*p1+a*p2
        xy=(int(round(p[0])),int(round(p[1])))
        cv2.circle(frame,xy,radius+2,(20,23,28),-1,cv2.LINE_AA)
        cv2.circle(frame,xy,radius,(255,255,255),-1,cv2.LINE_AA)


base.draw_broadcast_tile = draw_broadcast_tile
base.draw_contact_strip = draw_contact_strip
base.dotted_line = dotted_line

if __name__ == '__main__':
    base.main()
