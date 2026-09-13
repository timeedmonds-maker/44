#!/usr/bin/env python3
"""Clean broadcast presentation wrapper for Adams screen V3.

Keeps the V3 detector/tracker/geometry pipeline unchanged, but replaces the
QA-heavy presentation with unobtrusive broadcast-style player rings and
team-colour name plates. The large top analytics panel and bottom mini-court
are intentionally suppressed.
"""
from __future__ import annotations

import cv2

import adams_screen_prod_poc as base
import adams_screen_prod_v3 as v3


def _mix(c, target=(255, 255, 255), amount=0.28):
    return tuple(int(round((1.0 - amount) * c[i] + amount * target[i])) for i in range(3))


def broadcast_ring(frame, box, color, thick=3):
    """Layered anti-aliased ellipse with a subtle broadcast-style halo."""
    x1, y1, x2, y2 = box
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round(y2 - 1))
    bw = max(24, int(round((x2 - x1) * 0.82)))
    bh = max(10, int(round(bw * 0.30)))
    axes = (bw // 2, bh // 2)

    # Soft dark keyline improves legibility on both hardwood and paint.
    shadow = frame.copy()
    cv2.ellipse(shadow, (cx, cy + 1), axes, 0, 0, 360, (12, 12, 12), max(6, thick + 3), cv2.LINE_AA)
    cv2.addWeighted(shadow, 0.32, frame, 0.68, 0, frame)

    # Bright team-colour ring with a restrained highlight edge. A tiny rear
    # gap keeps the graphic from feeling like a detection ellipse.
    cv2.ellipse(frame, (cx, cy), axes, 0, 12, 168, color, max(3, thick), cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), axes, 0, 192, 348, color, max(3, thick), cv2.LINE_AA)
    hi = _mix(color, amount=0.42)
    cv2.ellipse(frame, (cx, cy), (max(2, axes[0] - 2), max(2, axes[1] - 1)), 0, 18, 162, hi, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (max(2, axes[0] - 2), max(2, axes[1] - 1)), 0, 198, 342, hi, 1, cv2.LINE_AA)


def _rounded_rect(img, x1, y1, x2, y2, color, radius=5):
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)


def broadcast_label(frame, xy, text, color, scale=.46):
    """Solid team-colour name tag. Suppress anonymous tracker IDs."""
    if not text or (text.startswith('T') and text[1:].isdigit()):
        return
    x, y = map(int, xy)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.43, float(scale))
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 8, 5
    w, h = tw + 2 * pad_x, th + baseline + 2 * pad_y
    x = max(4, min(frame.shape[1] - w - 4, x))
    y2 = max(h + 4, min(frame.shape[0] - 4, y))
    y1 = y2 - h

    # Small shadow plus saturated colour plate, mirroring TV telestration.
    _rounded_rect(frame, x + 2, y1 + 2, x + w + 2, y2 + 2, (18, 18, 18), 5)
    _rounded_rect(frame, x, y1, x + w, y2, color, 5)
    cv2.putText(frame, text, (x + pad_x, y2 - pad_y - baseline), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def broadcast_color(team, role='other'):
    # BGR. Keep role highlights strong while ordinary tracks use two clean
    # broadcast colours rather than pale QA greys.
    if role == 'adams':
        return (38, 82, 236)       # Rockets red
    if role == 'ballhandler':
        return (242, 170, 44)      # electric blue/cyan
    if role in ('screened_defender', 'adams_defender'):
        return (242, 170, 44)
    if team is None:
        return (220, 220, 220)
    return (38, 82, 236) if int(team) == 0 else (242, 170, 44)


def main():
    # Presentation-only monkey patches. V3 tracking/QA/data logic is untouched.
    base.ring = broadcast_ring
    base.label = broadcast_label
    base.color_for = broadcast_color
    base.draw_panel = lambda frame, ctx, metrics: None
    base.mini_court = lambda frame, court_xy, roles, origin, size=(290, 155): None
    v3.main()


if __name__ == '__main__':
    main()
