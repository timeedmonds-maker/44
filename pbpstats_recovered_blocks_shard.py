from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

TOTALS_URL = "https://api.pbpstats.com/get-totals/nba"
ON_OFF_TEAM_URL = "https://api.pbpstats.com/get-on-off/nba/team"
SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2000, 2026)]
TEAM_IDS = [
    1610612737, 1610612738, 1610612739, 1610612740, 1610612741,
    1610612742, 1610612743, 1610612744, 1610612745, 1610612746,
    1610612747, 1610612748, 1610612749, 1610612750, 1610612751,
    1610612752, 1610612753, 1610612754, 1610612755, 1610612756,
    1610612757, 1610612758, 1610612759, 1610612760, 1610612761,
    1610612762, 1610612763, 1610612764, 1610612765, 1610612766,
]
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.pbpstats.com",
    "Referer": "https://www.pbpstats.com/",
    "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/131 Safari/537.36",
}


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).casefold().strip()


def player_identity(row: dict[str, Any]) -> tuple[str, str]:
    player_id = str(row.get("EntityId") or row.get("RowId") or row.get("PlayerId") or "").strip()
    player_name = str(row.get("Name") or row.get("ShortName") or "").strip()
    return player_id, player_name


def is_recovered_block_name(value: str) -> bool:
    key = "".join(ch for ch in value.casefold() if ch.isalnum())
    return "block" in key and "recover" in key


def request_json(url: str, params: dict[str, str], attempts: int = 6) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update(HEADERS)
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=(10, 90))
            if response.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:160]!r}")
            if response.status_code == 400:
                return {"ok": False, "absent": True, "status_code": 400, "url": response.url}
            response.raise_for_status()
            return {"ok": True, "status_code": response.status_code, "url": response.url, "payload": response.json()}
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc!r}")
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 12) + random.random())
    return {"ok": False, "absent": False, "errors": errors}


def totals_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("multi_row_table_data")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def results_map(payload: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        return {}
    return {
        str(metric): [row for row in rows if isinstance(row, dict)]
        for metric, rows in payload["results"].items()
        if isinstance(rows, list)
    }


def task(season: str, team_id: int) -> dict[str, Any]:
    common = {"Season": season, "SeasonType": "Regular Season", "TeamId": str(team_id)}
    totals_result = request_json(TOTALS_URL, {**common, "Type": "Player"})
    if totals_result.get("absent"):
        return {"season": season, "team_id": team_id, "absent": True, "totals": [], "on_off": [], "errors": []}
    if not totals_result.get("ok"):
        return {"season": season, "team_id": team_id, "absent": False, "totals": [], "on_off": [], "errors": [f"totals: {totals_result.get('errors')}"]}

    players = totals_rows(totals_result.get("payload"))
    if not players:
        return {"season": season, "team_id": team_id, "absent": True, "totals": [], "on_off": [], "errors": []}

    total_fields = sorted({
        key for row in players for key in row.keys()
        if is_recovered_block_name(str(key))
    })
    block_fields = sorted({
        key for row in players for key in row.keys()
        if "block" in "".join(ch for ch in str(key).casefold() if ch.isalnum())
    })
    total_output: list[dict[str, Any]] = []
    for row in players:
        player_id, player_name = player_identity(row)
        for field in total_fields:
            total_output.append({
                "season": season,
                "team_id": team_id,
                "player_id": player_id,
                "player_name": player_name,
                "seconds_played": row.get("SecondsPlayed"),
                "source_field": field,
                "recovered_blocks": row.get(field),
            })

    name_index: dict[str, tuple[str, str]] = {}
    duplicate_names: set[str] = set()
    for row in players:
        player_id, player_name = player_identity(row)
        key = normalize(player_name)
        if not player_id or not key:
            continue
        if key in name_index:
            duplicate_names.add(key)
        else:
            name_index[key] = (player_id, player_name)
    for key in duplicate_names:
        name_index.pop(key, None)

    onoff_result = request_json(ON_OFF_TEAM_URL, common)
    onoff_output: list[dict[str, Any]] = []
    onoff_metrics: list[str] = []
    all_block_metrics: list[str] = []
    errors: list[str] = []
    if onoff_result.get("ok"):
        result_map = results_map(onoff_result.get("payload"))
        onoff_metrics = sorted(metric for metric in result_map if is_recovered_block_name(metric))
        all_block_metrics = sorted(
            metric for metric in result_map
            if "block" in "".join(ch for ch in metric.casefold() if ch.isalnum())
        )
        for metric in onoff_metrics:
            for row in result_map[metric]:
                subject_name = str(row.get("Name") or "").strip()
                subject_id, canonical_name = name_index.get(normalize(subject_name), ("", subject_name))
                onoff_output.append({
                    "season": season,
                    "team_id": team_id,
                    "player_id": subject_id,
                    "player_name": canonical_name or subject_name,
                    "metric": metric,
                    "minutes_on": row.get("MinutesOn"),
                    "minutes_off": row.get("MinutesOff"),
                    "on": row.get("On"),
                    "off": row.get("Off"),
                    "on_off": row.get("On-Off"),
                })
    elif not onoff_result.get("absent"):
        errors.append(f"on_off: {onoff_result.get('errors')}")

    return {
        "season": season,
        "team_id": team_id,
        "absent": False,
        "totals": total_output,
        "on_off": onoff_output,
        "total_fields": total_fields,
        "onoff_metrics": onoff_metrics,
        "block_fields": block_fields,
        "block_metrics": all_block_metrics,
        "players": len(players),
        "errors": errors,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out-dir", default="pbpstats_recovered_blocks_shards")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("invalid shard index")

    all_tasks = [(season, team_id) for season in SEASONS for team_id in TEAM_IDS]
    selected = [item for i, item in enumerate(all_tasks) if i % args.num_shards == args.shard_index]
    total_rows: list[dict[str, Any]] = []
    onoff_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_fields: Counter[str] = Counter()
    onoff_metrics: Counter[str] = Counter()
    block_fields: Counter[str] = Counter()
    block_metrics: Counter[str] = Counter()
    absent = 0
    players = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(task, season, team_id): (season, team_id) for season, team_id in selected}
        for future in as_completed(futures):
            season, team_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append({"season": season, "team_id": team_id, "error": repr(exc)})
                continue
            if result.get("absent"):
                absent += 1
            players += int(result.get("players", 0) or 0)
            total_rows.extend(result.get("totals", []))
            onoff_rows.extend(result.get("on_off", []))
            total_fields.update(result.get("total_fields", []))
            onoff_metrics.update(result.get("onoff_metrics", []))
            block_fields.update(result.get("block_fields", []))
            block_metrics.update(result.get("block_metrics", []))
            for error in result.get("errors", []):
                errors.append({"season": season, "team_id": team_id, "error": error})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_rows.sort(key=lambda r: (r["season"], int(r["team_id"]), str(r["player_id"]), str(r["source_field"])))
    onoff_rows.sort(key=lambda r: (r["season"], int(r["team_id"]), str(r["player_id"]), str(r["metric"])))
    write_csv(out / f"player_totals_{args.shard_index}.csv", total_rows,
              ["season", "team_id", "player_id", "player_name", "seconds_played", "source_field", "recovered_blocks"])
    write_csv(out / f"team_on_off_{args.shard_index}.csv", onoff_rows,
              ["season", "team_id", "player_id", "player_name", "metric", "minutes_on", "minutes_off", "on", "off", "on_off"])
    write_csv(out / f"errors_{args.shard_index}.csv", errors, ["season", "team_id", "error"])
    summary = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "team_seasons_attempted": len(selected),
        "absent_team_seasons": absent,
        "player_rows_seen": players,
        "player_total_rows": len(total_rows),
        "team_on_off_rows": len(onoff_rows),
        "recovered_block_total_fields": dict(total_fields),
        "recovered_block_on_off_metrics": dict(onoff_metrics),
        "all_block_total_fields": dict(block_fields),
        "all_block_on_off_metrics": dict(block_metrics),
        "errors": len(errors),
    }
    (out / f"summary_{args.shard_index}.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
