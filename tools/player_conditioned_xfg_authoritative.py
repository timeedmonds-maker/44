#!/usr/bin/env python3
"""Canonical player-conditioned xFG entry point using authoritative SHOT_DISTANCE.

The G League stats host serves the same official NBA shotchartdetail result set
reliably in Actions and includes SHOT_DISTANCE. We deliberately do not derive
shot distance from locX/locY in this path.
"""
from __future__ import annotations

import player_conditioned_xfg_v2 as base

base.SHOTCHART_URL = 'https://stats.gleague.nba.com/stats/shotchartdetail'

if __name__ == '__main__':
    base.main()
