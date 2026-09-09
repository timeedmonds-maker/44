#!/usr/bin/env python3
import csv
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

from curl_cffi import requests

GAME_ID = "0041800163"
TARGET_EVENT = 215
TARGET_UUID = "995096ac-c2a1-b2fa-89ed-a345759b84be"
DATE_PATHS = ["2019/04/19", "2019/04/20"]
EVENTS = list(range(TARGET_EVENT - 10, TARGET_EVENT + 11))
TURNER_HOSTS = ["nba.cdn.turner.com", "ssl.cdn.turner.com", "pmd.cdn.turner.com"]
OUT = Path("wsc_same_game_archive")
OUT.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131 Safari/537.36"

s = requests.Session(impersonate="chrome120")

def get(url, timeout=20):
    last = None
    for attempt in range(3):
        try:
            r = s.get(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=timeout, allow_redirects=True)
            return r
        except Exception as exc:
            last = exc
            time.sleep(0.7 * (attempt + 1))
    raise last


def stats_uuid(event_num):
    url = "https://stats.nba.com/stats/videoeventsasset?" + urllib.parse.urlencode({
        "GameID": GAME_ID,
        "GameEventID": event_num,
    })
    try:
        r = get(url, 12)
        if r.status_code != 200:
            return {"event_num": event_num, "stats_status": r.status_code}
        data = r.json()
        rs = data.get("resultSets") or {}
        playlist = rs.get("playlist") or []
        meta = (rs.get("Meta") or {}).get("videoUrls") or []
        if not playlist or not meta:
            return {"event_num": event_num, "stats_status": 200, "empty": True}
        p = playlist[0]
        m = meta[0]
        return {
            "event_num": event_num,
            "stats_status": 200,
            "uuid": m.get("uuid"),
            "description": p.get("dsc"),
            "period": p.get("p"),
            "game_clock": p.get("gc"),
            "lth": m.get("lth") or m.get("ltp"),
        }
    except Exception as exc:
        return {"event_num": event_num, "stats_error": f"{type(exc).__name__}: {exc}"}


def cdx(url_pattern, limit=20):
    q = {
        "url": url_pattern,
        "output": "json",
        "filter": "statuscode:200",
        "collapse": "urlkey",
        "fl": "timestamp,original,statuscode,digest",
        "limit": str(limit),
    }
    url = "https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(q, safe="*:/")
    try:
        r = get(url, 25)
        result = {"query": url_pattern, "status": r.status_code, "bytes": len(r.content), "rows": []}
        if r.status_code == 200 and r.text.strip():
            try:
                arr = r.json()
                if arr and isinstance(arr[0], list):
                    hdr = arr[0]
                    for row in arr[1:]:
                        result["rows"].append(dict(zip(hdr, row)))
            except Exception:
                result["text_head"] = r.text[:500]
        return result
    except Exception as exc:
        return {"query": url_pattern, "error": f"{type(exc).__name__}: {exc}", "rows": []}


def archived_xml(capture):
    ts = capture.get("timestamp")
    original = capture.get("original")
    if not ts or not original:
        return None
    replay = f"https://web.archive.org/web/{ts}id_/{original}"
    try:
        r = get(replay, 25)
        urls = re.findall(r'https?://[^\s<\"\']+', r.text)
        turner = [u for u in urls if "turner.com/nba/big/nba/wsc/" in u and ".mp4" in u]
        return {"replay": replay, "status": r.status_code, "bytes": len(r.content), "turner_urls": turner[:30], "text_head": r.text[:400]}
    except Exception as exc:
        return {"replay": replay, "error": f"{type(exc).__name__}: {exc}"}


def extract_asset_ids(text):
    if not text:
        return []
    return sorted(set(int(x) for x in re.findall(r'\.nba_(\d+)_', text)))


def commoncrawl_target_queries():
    out = []
    try:
        r = get("https://index.commoncrawl.org/collinfo.json", 20)
        if r.status_code != 200:
            return [{"collinfo_status": r.status_code}]
        cols = [x for x in r.json() if str(x.get("id", "")).startswith("CC-MAIN-2019-")]
        for col in cols:
            idx = col["id"] + "-index"
            patterns = [
                f"secure.nba.com/video/wsc/league/{TARGET_UUID}.secure.xml",
                f"nba.cdn.turner.com/nba/big/nba/wsc/2019/04/*/{TARGET_UUID}*",
                f"ssl.cdn.turner.com/nba/big/nba/wsc/2019/04/*/{TARGET_UUID}*",
                f"pmd.cdn.turner.com/nba/big/nba/wsc/2019/04/*/{TARGET_UUID}*",
            ]
            for pat in patterns:
                qurl = f"https://index.commoncrawl.org/{idx}?" + urllib.parse.urlencode({"url": pat, "output": "json"}, safe="*:/")
                try:
                    rr = get(qurl, 20)
                    lines = [ln for ln in rr.text.splitlines() if ln.strip()]
                    if rr.status_code == 200 and lines:
                        out.append({"index": idx, "query": pat, "status": rr.status_code, "rows": [json.loads(x) for x in lines[:50]]})
                except Exception as exc:
                    out.append({"index": idx, "query": pat, "error": f"{type(exc).__name__}: {exc}"})
                time.sleep(0.15)
    except Exception as exc:
        out.append({"collinfo_error": f"{type(exc).__name__}: {exc}"})
    return out


def main():
    print("Fetching same-game UUID neighborhood from canonical Stats endpoint on macOS...", flush=True)
    events = []
    for ev in EVENTS:
        row = stats_uuid(ev)
        events.append(row)
        print("EVENT", ev, row.get("uuid"), row.get("description"), row.get("stats_status"), flush=True)
        time.sleep(0.12)

    # Force the independently verified target UUID even if the endpoint hiccups.
    target = next((x for x in events if x["event_num"] == TARGET_EVENT), None)
    if target is not None:
        target["verified_target_uuid"] = TARGET_UUID
        if not target.get("uuid"):
            target["uuid"] = TARGET_UUID

    archive = []
    candidates = [x for x in events if x.get("uuid")]
    print(f"Archive probing {len(candidates)} UUIDs...", flush=True)
    for i, row in enumerate(candidates, 1):
        uid = row["uuid"]
        rec = {"event_num": row["event_num"], "uuid": uid, "description": row.get("description"), "queries": [], "asset_ids": []}
        # Historical secure XML resolver, both schemes.
        for scheme in ("http", "https"):
            res = cdx(f"{scheme}://secure.nba.com/video/wsc/league/{uid}.secure.xml", 10)
            rec["queries"].append(res)
            for cap in res.get("rows", [])[:2]:
                ax = archived_xml(cap)
                if ax:
                    rec.setdefault("archived_xml", []).append(ax)
                    for u in ax.get("turner_urls", []):
                        rec["asset_ids"].extend(extract_asset_ids(u))
        # Direct Turner paths for the game date and possible UTC-next-day path.
        for date_path in DATE_PATHS:
            for host in TURNER_HOSTS:
                res = cdx(f"https://{host}/nba/big/nba/wsc/{date_path}/{uid}*", 20)
                rec["queries"].append(res)
                for cap in res.get("rows", []):
                    rec["asset_ids"].extend(extract_asset_ids(cap.get("original", "")))
        rec["asset_ids"] = sorted(set(rec["asset_ids"]))
        archive.append(rec)
        print("ARCHIVE", i, "/", len(candidates), "event", row["event_num"], "ids", rec["asset_ids"], flush=True)
        time.sleep(0.15)

    print("Querying 2019 Common Crawl indexes for exact target UUID...", flush=True)
    cc = commoncrawl_target_queries()

    # Flatten any target IDs from all evidence.
    all_ids = []
    target_ids = []
    for rec in archive:
        all_ids.extend(rec.get("asset_ids", []))
        if rec["event_num"] == TARGET_EVENT:
            target_ids.extend(rec.get("asset_ids", []))
    for rec in cc:
        for row in rec.get("rows", []):
            all_ids.extend(extract_asset_ids(row.get("url", "")))
            if TARGET_UUID in row.get("url", ""):
                target_ids.extend(extract_asset_ids(row.get("url", "")))

    report = {
        "game_id": GAME_ID,
        "target_event": TARGET_EVENT,
        "target_uuid": TARGET_UUID,
        "events": events,
        "archive": archive,
        "commoncrawl": cc,
        "all_asset_ids": sorted(set(all_ids)),
        "target_asset_ids": sorted(set(target_ids)),
    }
    if all_ids:
        report["observed_asset_id_min"] = min(all_ids)
        report["observed_asset_id_max"] = max(all_ids)

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "neighbors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_num", "uuid", "description", "asset_ids", "is_target"])
        amap = {x["event_num"]: x for x in archive}
        for row in events:
            ar = amap.get(row["event_num"], {})
            w.writerow([row["event_num"], row.get("uuid", ""), row.get("description", ""), ";".join(map(str, ar.get("asset_ids", []))), int(row["event_num"] == TARGET_EVENT)])

    print("SUMMARY", json.dumps({
        "events_with_uuid": len(candidates),
        "events_with_asset_ids": sum(bool(x.get("asset_ids")) for x in archive),
        "target_asset_ids": sorted(set(target_ids)),
        "observed_min": min(all_ids) if all_ids else None,
        "observed_max": max(all_ids) if all_ids else None,
        "commoncrawl_hits": sum(len(x.get("rows", [])) for x in cc),
    }), flush=True)

if __name__ == "__main__":
    main()
