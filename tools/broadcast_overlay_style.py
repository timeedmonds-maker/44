#!/usr/bin/env python3
"""Reusable deterministic broadcast-overlay styling primitives.

Locked presentation default for analytics information panels:
- semi-transparent neutral grey fill
- neutral grey border
- no coloured accent bands/rules
- white/grey broadcast typography

This module draws only deterministic OpenCV/Pillow overlays on real source video.
It does not generate or synthesize imagery.
"""
from __future__ import annotations

import cv2
import numpy as np

PANEL_FILL = (70, 70, 72)       # BGR neutral grey
PANEL_BORDER = (164, 164, 168)  # BGR neutral grey
KICKER_TEXT = (210, 210, 214)
VALUE_TEXT = (252, 252, 252)
PANEL_ALPHA = 0.66


def gray_alpha_poly(frame, pts, color=None, alpha=None):
    """Draw a neutral translucent broadcast panel.

    ``color`` and ``alpha`` are accepted for drop-in compatibility with older
    clip renderers but deliberately ignored so analytics boxes cannot silently
    reintroduce coloured accents or opaque black slabs.
    """
    ov = frame.copy()
    poly = np.array(pts, np.int32)
    cv2.fillPoly(ov, [poly], PANEL_FILL)
    cv2.addWeighted(ov, PANEL_ALPHA, frame, 1.0 - PANEL_ALPHA, 0, frame)


def panel_outline(frame, pts):
    cv2.polylines(frame, [np.array(pts, np.int32)], True, PANEL_BORDER, 1, cv2.LINE_AA)


def draw_info_tile(base, frame, x, y, w, h, kicker, value, value_size=28):
    """Standard clipped-corner analytics tile, no colour accent."""
    pts = [(x, y), (x + w - 8, y), (x + w, y + 8), (x + w, y + h), (x, y + h)]
    gray_alpha_poly(frame, pts)
    panel_outline(frame, pts)
    frame = base.pil_text(frame, (x + 10, y + 12), kicker, 10, KICKER_TEXT, True)
    frame = base.pil_text(frame, (x + 10, y + 28), value, value_size, VALUE_TEXT, True)
    return frame


def draw_contact_strip(base, frame, x, y, w, elapsed):
    """Compact neutral contact-time strip, no colour accent."""
    h = 34
    pts = [(x, y), (x + w - 7, y), (x + w, y + 7), (x + w, y + h), (x, y + h)]
    gray_alpha_poly(frame, pts)
    panel_outline(frame, pts)
    frame = base.pil_text(frame, (x + 10, y + 8), 'SCREEN CONTACT', 10, KICKER_TEXT, True)
    frame = base.pil_text(frame, (x + w - 12, y + 8), f'{elapsed:.1f} s', 15, VALUE_TEXT, True, 'ra')
    return frame
