from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyreadr
import requests

PBP_URL = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds"
XFG_URL = "https://stats.gleague.nba.com/stats/shotqualityvideologs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    s = getattr(THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        THREAD_LOCAL.session = s
    return s


def norm_game_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def parse_player(s: pd.Series) -> pd.DataFrame:
    x = s.astype(str).str.extract(r"^\s*(\d+)\s+(.*)$")
    x.columns = ["player_id", "player_name"]
    x["player_id"] = pd.to_numeric(x["player_id"], errors="coerce").astype("Int64")
    return x


def fetch_pair(game_id: str, player_id: int, max_attempts: int = 4) -> tuple[dict | None, dict]:
    params = {"GameID": game_id, "PlayerID": int(player_id)}
    last: dict = {}
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            r = session().get(XFG_URL, params=params, timeout=(6, 25))
            elapsed = time.time() - started
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and str(j.get("gameId", "")).zfill(10) == game_id and int(j.get("playerId") or 0) == int(player_id):
                    return j, {"status": 200, "attempts": attempt, "elapsed": elapsed}
                last = {"status": 200, "attempts": attempt, "elapsed": elapsed, "error": "unexpected_payload"}
            else:
                last = {"status": r.status_code, "attempts": attempt, "elapsed": elapsed}
        except Exception as e:
            last = {"status": None, "attempts": attempt, "elapsed": time.time() - started, "error": repr(e)}
        if attempt < max_attempts:
            time.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
    return None, last


def shot_is_three(shot: dict) -> bool:
    return str(shot.get("shotType") or "").upper().startswith("3PT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-fga", type=int, default=300)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-dir", default="xfg_efg_output")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Match the validated long-rebound-era extraction source and player-ID parsing.
    rds = out / "pbp_source.rds"
    r = requests.get(PBP_URL, timeout=180)
    r.raise_for_status()
    rds.write_bytes(r.content)
    pbp = next(iter(pyreadr.read_r(str(rds)).values()))
    rds.unlink(missing_ok=True)

    pbp["game_id"] = norm_game_id(pbp["game_id"])
    is_fg = pd.to_numeric(pbp["is_field_goal"], errors="coerce").fillna(0).eq(1)
    reg = pbp["game_id"].str.startswith("002")
    fga = pbp.loc[is_fg & reg, ["game_id", "event_num", "player1_name", "shot_result"]].copy()
    parsed = parse_player(fga["player1_name"])
    fga["player_id"] = parsed["player_id"]
    fga["player_name"] = parsed["player_name"]
    fga = fga.dropna(subset=["player_id"]).copy()
    fga["player_id"] = fga["player_id"].astype(int)

    pbp_player = (
        fga.groupby(["player_id", "player_name"], as_index=False)
        .agg(pbp_fga=("event_num", "size"))
    )
    eligible = pbp_player.loc[pbp_player["pbp_fga"] >= args.min_fga].copy()
    eligible_ids = set(eligible["player_id"].astype(int).tolist())

    manifest = (
        fga.loc[fga["player_id"].isin(eligible_ids), ["game_id", "player_id", "player_name"]]
        .drop_duplicates(["game_id", "player_id"])
        .sort_values(["game_id", "player_id"])
        .reset_index(drop=True)
    )

    acc: dict[int, dict] = defaultdict(lambda: {
        "tracked_fga": 0,
        "tracked_fgm": 0,
        "tracked_3pa": 0,
        "tracked_3pm": 0,
        "actual_efg_num": 0.0,
        "expected_efg_num": 0.0,
        "api_official_fga": 0,
        "unlisted_fga": 0,
        "successful_games": 0,
        "failed_games": 0,
        "player_name_api": None,
    })
    errors: list[dict] = []
    started = time.time()

    def handle_payload(pid: int, payload: dict) -> None:
        a = acc[pid]
        shot_list = payload.get("shotList") or []
        official_fga = payload.get("shots")
        if isinstance(official_fga, (int, float)) and not isinstance(official_fga, bool):
            a["api_official_fga"] += int(official_fga)
            a["unlisted_fga"] += max(0, int(official_fga) - len(shot_list))
        a["successful_games"] += 1
        a["player_name_api"] = payload.get("playerName") or a["player_name_api"]

        for s in shot_list:
            try:
                xfg = float(s.get("shotQuality"))
            except Exception:
                continue
            if math.isnan(xfg):
                continue
            made = int(s.get("success") or 0)
            is3 = shot_is_three(s)
            efg_weight = 1.5 if is3 else 1.0
            a["tracked_fga"] += 1
            a["tracked_fgm"] += made
            a["tracked_3pa"] += int(is3)
            a["tracked_3pm"] += int(is3 and made)
            a["actual_efg_num"] += made * efg_weight
            a["expected_efg_num"] += xfg * efg_weight

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(fetch_pair, str(row.game_id), int(row.player_id)): (str(row.game_id), int(row.player_id), str(row.player_name))
            for row in manifest.itertuples(index=False)
        }
        total = len(futures)
        for done, fut in enumerate(as_completed(futures), start=1):
            game_id, pid, pname = futures[fut]
            try:
                payload, meta = fut.result()
            except Exception as e:
                payload, meta = None, {"error": repr(e)}
            if payload is None:
                acc[pid]["failed_games"] += 1
                errors.append({"game_id": game_id, "player_id": pid, "player_name": pname, **meta})
            else:
                handle_payload(pid, payload)
            if done % 500 == 0 or done == total:
                print(f"done={done}/{total} elapsed={time.time()-started:.1f}s errors={len(errors)}", flush=True)

    eligible_lookup = eligible.set_index("player_id").to_dict("index")
    rows = []
    for pid in sorted(eligible_ids):
        a = acc[pid]
        tracked = int(a["tracked_fga"])
        if tracked <= 0:
            continue
        actual_efg = a["actual_efg_num"] / tracked
        expected_efg = a["expected_efg_num"] / tracked
        pbp_fga = int(eligible_lookup[pid]["pbp_fga"])
        api_official_fga = int(a["api_official_fga"])
        rows.append({
            "player_id": pid,
            "player_name": a["player_name_api"] or eligible_lookup[pid].get("player_name") or "",
            "pbp_fga": pbp_fga,
            "tracked_xfg_fga": tracked,
            "tracked_fgm": int(a["tracked_fgm"]),
            "tracked_3pa": int(a["tracked_3pa"]),
            "tracked_3pm": int(a["tracked_3pm"]),
            "actual_efg_pct": 100.0 * actual_efg,
            "expected_efg_pct": 100.0 * expected_efg,
            "efg_over_expected_pp": 100.0 * (actual_efg - expected_efg),
            "api_official_fga": api_official_fga,
            "xfg_shotlist_coverage_pct_vs_api_fga": (100.0 * tracked / api_official_fga) if api_official_fga else None,
            "xfg_shotlist_coverage_pct_vs_pbp_fga": 100.0 * tracked / pbp_fga,
            "unlisted_fga_api": int(a["unlisted_fga"]),
            "successful_player_games": int(a["successful_games"]),
            "failed_player_games": int(a["failed_games"]),
        })

    leaderboard = pd.DataFrame(rows).sort_values(
        ["efg_over_expected_pp", "tracked_xfg_fga"], ascending=[False, False]
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    leaderboard.to_csv(out / "player_efg_over_expected_2025_26.csv", index=False)
    leaderboard.head(25).to_csv(out / "top25_efg_over_expected_2025_26.csv", index=False)

    qa = {
        "season": "2025-26",
        "season_type": "Regular Season",
        "source_resource": "shotqualityvideologs",
        "source_host": "stats.gleague.nba.com",
        "definition": {
            "actual_efg": "sum(made * [1.5 if 3PT else 1.0]) / tracked_xfg_fga",
            "expected_efg": "sum(xFG * [1.5 if 3PT else 1.0]) / tracked_xfg_fga",
            "efg_over_expected_pp": "100 * (actual_eFG - expected_eFG)",
        },
        "minimum_pbp_fga": args.min_fga,
        "eligible_players": int(len(eligible_ids)),
        "requested_player_games": int(len(manifest)),
        "failed_player_games": int(len(errors)),
        "success_rate_pct": 100.0 * (len(manifest) - len(errors)) / len(manifest) if len(manifest) else None,
        "elapsed_seconds": time.time() - started,
        "workers": args.workers,
        "important_note": "Expected eFG is computed exactly on publicly listed shotList rows with per-shot xFG. NBA can expose fewer shotList rows than official FGA; coverage is reported per player and missing xFG is not imputed.",
    }
    (out / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    pd.DataFrame(errors).to_csv(out / "errors.csv", index=False)

    print("\nTOP 10")
    cols = ["rank", "player_name", "pbp_fga", "tracked_xfg_fga", "actual_efg_pct", "expected_efg_pct", "efg_over_expected_pp", "xfg_shotlist_coverage_pct_vs_pbp_fga"]
    print(leaderboard.head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nQA")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
