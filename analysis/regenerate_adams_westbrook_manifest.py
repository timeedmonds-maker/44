from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pyreadr
import requests

BASE = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main"
YEARS = range(2014, 2020)  # season ending year: 2014 == 2013-14
ADAMS = "203500 Steven Adams"
WESTBROOK = "201566 Russell Westbrook"
EXPECTED = 381
OUT = Path("public_manifest")
OUT.mkdir(exist_ok=True)


def get(url: str, timeout: int = 180) -> bytes:
    print("GET", url, flush=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    print(" bytes", len(r.content), flush=True)
    return r.content


def regular(year: int) -> pd.DataFrame:
    url = f"{BASE}/pbp-final-{year}/data.rds"
    p = OUT / f"pbp_{year}.rds"
    p.write_bytes(get(url))
    objects = pyreadr.read_r(str(p))
    if not objects:
        raise RuntimeError(f"No data frame in {p}")
    df = next(iter(objects.values()))
    p.unlink(missing_ok=True)
    return df


def playoffs(year: int) -> pd.DataFrame:
    url = f"{BASE}/pbp-final-playoffs{year}/data.csv"
    b = get(url)
    return pd.read_csv(io.BytesIO(b), low_memory=False)


def filt(df: pd.DataFrame, year: int, season_type: str) -> pd.DataFrame:
    needed = {"game_date", "game_id", "event_num", "period", "clock", "team_away", "team_home", "description", "player1_name", "player2_name", "msg_type"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{year} {season_type} missing columns: {sorted(missing)}")

    p1 = df["player1_name"].fillna("").astype(str)
    p2 = df["player2_name"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str)
    msg = pd.to_numeric(df["msg_type"], errors="coerce")
    mask = (
        p1.eq(ADAMS)
        & p2.eq(WESTBROOK)
        & msg.eq(1)
        & desc.str.contains("dunk", case=False, regex=False)
    )
    x = df.loc[mask, ["game_date", "game_id", "event_num", "period", "clock", "team_away", "team_home", "description"]].copy()
    x["season"] = f"{year-1}-{str(year)[-2:]}"
    x["season_type"] = season_type
    x["scorer"] = ADAMS
    x["assister"] = WESTBROOK
    x["game_id"] = x["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    x["event_num"] = pd.to_numeric(x["event_num"], errors="raise").astype(int)
    x["game_date"] = pd.to_datetime(x["game_date"], errors="raise").dt.strftime("%Y-%m-%d")
    print(year, season_type, "matches", len(x), flush=True)
    return x


frames = []
for year in YEARS:
    frames.append(filt(regular(year), year, "Regular Season"))
    try:
        frames.append(filt(playoffs(year), year, "Playoffs"))
    except requests.HTTPError as exc:
        print(year, "Playoffs unavailable", exc, flush=True)

m = pd.concat(frames, ignore_index=True)
m = m.drop_duplicates(subset=["game_id", "event_num"]).copy()
m = m.sort_values(["game_date", "game_id", "period", "event_num"], kind="stable").reset_index(drop=True)
m.insert(0, "rank", range(1, len(m) + 1))

cols = ["rank", "season", "season_type", "game_date", "game_id", "event_num", "period", "clock", "team_away", "team_home", "scorer", "assister", "description"]
m = m[cols]
path = OUT / "adams_westbrook_dunk_event_links_public.csv"
m.to_csv(path, index=False)

print("TOTAL_UNIQUE", len(m), flush=True)
print("FIRST", m.iloc[0].to_dict() if len(m) else None, flush=True)
print("LAST", m.iloc[-1].to_dict() if len(m) else None, flush=True)
print(m.groupby(["season", "season_type"]).size().to_string(), flush=True)

if len(m) != EXPECTED:
    print(f"COUNT_MISMATCH expected={EXPECTED} actual={len(m)}", file=sys.stderr, flush=True)
    sys.exit(2)
