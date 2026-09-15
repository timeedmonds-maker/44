#!/usr/bin/env python3
"""Universal Screen Tracker camera-angle policy.

New Screen Tracker applications are two-angle by default. The canonical primary
is Broadcast. The secondary is the highest-priority distinct official angle,
with High Tight preferred when available. Never duplicate Broadcast to satisfy
the two-angle contract; fail instead.
"""
from __future__ import annotations

PRIMARY_LABEL = "Broadcast"
SECONDARY_PRIORITY = [
    "High Tight",
    "Other Broadcast",
    "Mobile Broadcast",
    "Play by Play",
    "Left Slash",
    "Right Slash",
    "Left HandHeld",
    "Right HandHeld",
    "In Arena",
]
DEFAULT_MODE = "two_angle_default"
LEGACY_MODE = "legacy_single_angle"


def _norm(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def select_two_angles(angles):
    if len(angles) < 2:
        raise RuntimeError("Screen Tracker requires two distinct official camera angles")
    primary = next((a for a in angles if _norm(a.get("label")) == _norm(PRIMARY_LABEL)), None)
    if primary is None:
        primary = next((a for a in angles if a.get("selected")), None) or angles[0]
    secondary = None
    for label in SECONDARY_PRIORITY:
        secondary = next(
            (
                a for a in angles
                if _norm(a.get("label")) == _norm(label)
                and _norm(a.get("label")) != _norm(primary.get("label"))
            ),
            None,
        )
        if secondary is not None:
            break
    if secondary is None:
        secondary = next(
            (a for a in angles if _norm(a.get("label")) != _norm(primary.get("label"))),
            None,
        )
    if secondary is None:
        raise RuntimeError("No distinct second official camera angle available")
    return primary, secondary


def mode_for_application(app):
    explicit = (app.get("angle_policy") or {}).get("mode")
    if explicit:
        return explicit
    # 1.0.0 applications predate the two-angle contract and remain reproducible.
    # 1.1.0+ applications default to two angles even if angle_policy is omitted.
    return LEGACY_MODE if app.get("screen_tracker_version") == "1.0.0" else DEFAULT_MODE
