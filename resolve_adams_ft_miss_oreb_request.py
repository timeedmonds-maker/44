from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyreadr
import requests

BASE = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main"
PBP_URL = f"{BASE}/pbp-final-2026/data.rds"
PBP_META_URL = f"{BASE}/pbp-final-2026/data.txt"
UA = "nba-video-worker-adams-ft-oreb-rds"
ADAMS_ID = "203500"
HOU = "HOU"
ACTOR_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def scalar(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def integer(v):
    try:
        if pd.isna(v) or v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def actor(v) -> tuple[str, str]:
    s = scalar(v)
    if not s:
        return "", ""
    m = ACTOR_RE.match(s)
    if m:
        return m.group(1), m.group(2).strip()
    return "", s


def norm_gid(v) -> str:
    s = scalar(v)
    if not s:
        return ""
    try:
        s = str(int(float(s)))
    except Exception:
        s = re.sub(r"\D", "", s)
    return s.zfill(10)


def event_col(columns: list[str]) -> str:
    for c in ("event_num", "event_id", "eventnum", "event_no", "event_number"):
        if c in columns:
            return c
    raise RuntimeError(f"No event identifier column found. Columns={columns}")


def expected_rows() -> int:
    r = requests.get(PBP_META_URL, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    m = re.search(r"(?m)^rows:\s*(\d+)\s*$", r.text)
    if not m:
        raise RuntimeError("pbp-final-2026: no rows metadata")
    return int(m.group(1))


def download(url: str, dest: Path) -> None:
    with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=240) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    tmp = Path("pbp2026.rds")
    download(PBP_URL, tmp)
    pbp = next(iter(pyreadr.read_r(str(tmp)).values())).reset_index(drop=True)
    exp = expected_rows()
    if len(pbp) != exp:
        raise SystemExit(f"PBP row mismatch {len(pbp)}/{exp}")

    cols = list(pbp.columns)
    evc = event_col(cols)
    required = {"game_id", "period", "msg_type", "team_abb", "player1_name", "description", evc}
    missing = sorted(required - set(cols))
    if missing:
        raise SystemExit(f"Missing required PBP columns: {missing}")

    # Critical: RDS source-row order is chronological. event_num is retained only
    # as the NBA video address and MUST NOT be used to sort the flattened data.
    q = pbp.copy().reset_index(drop=False).rename(columns={"index": "source_row"})
    q["_gid"] = q.game_id.map(norm_gid)
    q["_event"] = pd.to_numeric(q[evc], errors="coerce")
    q["_period"] = pd.to_numeric(q.period, errors="coerce")
    q["_msg"] = pd.to_numeric(q.msg_type, errors="coerce")
    q["_team"] = q.team_abb.fillna("").astype(str).str.strip().str.upper()
    q["_desc"] = q.description.fillna("").astype(str)
    q = q[q._gid.str.startswith("002")].copy().reset_index(drop=True)

    events = []
    hou_missed_teammate_fts = 0
    adams_rebound_rows = 0
    for i in range(len(q) - 1):
        ft = q.iloc[i]
        rb = q.iloc[i + 1]
        if integer(ft._msg) != 3 or "MISS" not in str(ft._desc).upper() or ft._team != HOU:
            continue
        shooter_id, shooter_name = actor(ft.player1_name)
        if shooter_id == ADAMS_ID:
            continue
        hou_missed_teammate_fts += 1

        if rb._gid != ft._gid or integer(rb._period) != integer(ft._period):
            continue
        if integer(rb._msg) != 4:
            continue
        rebounder_id, rebounder_name = actor(rb.player1_name)
        if rebounder_id != ADAMS_ID or rb._team != HOU:
            continue
        adams_rebound_rows += 1

        rec = {
            "rank": len(events) + 1,
            "season": "2025-26",
            "game_id": rb._gid,
            "event_id": integer(rb._event),
            "ft_event_id": integer(ft._event),
            "period": integer(rb._period),
            "clock": scalar(rb.get("clock", "")),
            "ft_shooter_id": shooter_id,
            "ft_shooter": shooter_name,
            "ft_description": scalar(ft.description),
            "rebounder_id": rebounder_id,
            "rebounder": rebounder_name,
            "rebound_description": scalar(rb.description),
            "team": HOU,
            "source_row_ft": integer(ft.source_row),
            "source_row_rebound": integer(rb.source_row),
            "source": "ramirobentes/nba_pbp_data pbp-final-2026/data.rds",
            "event_anchor": "Steven Adams rebound event immediately following teammate missed free throw",
        }
        for c in ("team_home", "team_away", "date", "game_date"):
            if c in q.columns:
                rec[c] = scalar(rb.get(c, ""))
        events.append(rec)

    payload = {
        "label": "Steven Adams 2025-26 offensive rebounds immediately after teammate missed free throw - all camera angles UHD",
        "definition": "2025-26 regular season; Houston teammate (non-Adams) misses a free throw; the immediately following chronological PBP row is a rebound credited to Steven Adams for Houston",
        "source": "ramirobentes/nba_pbp_data pbp-final-2026/data.rds",
        "ordering": "native RDS source-row chronology; NBA event_num retained only as video address",
        "video_anchor": "Steven Adams rebound event_id",
        "pbp_rows": len(pbp),
        "expected_pbp_rows": exp,
        "hou_missed_teammate_fts": hou_missed_teammate_fts,
        "expected_count": len(events),
        "workers": 8,
        "events": events,
    }
    Path("request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"PBP_ROWS={len(pbp)} HOU_MISSED_TEAMMATE_FTS={hou_missed_teammate_fts} QUALIFYING_EVENTS={len(events)}", flush=True)
    for event in events:
        print(json.dumps(event, ensure_ascii=False), flush=True)
    if adams_rebound_rows != len(events):
        raise SystemExit("Internal event-count mismatch")
    if not events:
        raise SystemExit("No qualifying Adams FT-miss OREB events resolved from chronological RDS PBP")


if __name__ == "__main__":
    main()
