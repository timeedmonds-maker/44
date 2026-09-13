#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import html as htmlmod
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

GAME_ID = "0022100923"
EVENT_ID = 352
OUT = Path("outputs/morant_adams_2022_event352_all_native_angles")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0 Safari/537.36"
PAGE_HEADERS = {
    "User-Agent": UA,
    "Referer": "https://clips.nba.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
KNOWN_PLACEHOLDER_AHASH = [
    "0000180039fc39fc38c839fc39fc10000000",
    "0000180019fc39fc38c819fc39fc18000000",
    "0000100011fc11fc18ec11fc19fc10000000",
]


def sh(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def safe_name(s: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()).strip("_")
    return x[:80] or "angle"


def parse_clips_page() -> dict:
    page_url = f"https://clips.nba.com/?gameNo={GAME_ID}&eventNum={EVENT_ID}&source=grs"
    r = requests.get(page_url, headers=PAGE_HEADERS, timeout=45)
    r.raise_for_status()
    text = r.text
    (OUT / "clips_page.html").write_text(text, encoding="utf-8")
    title_match = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
    title = htmlmod.unescape(title_match.group(1).strip()) if title_match else ""

    options = []
    seen_url = set()
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', text, flags=re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        attrs = m.group(2).lower()
        label = re.sub(r"<[^>]+>", "", htmlmod.unescape(m.group(3))).strip() or "angle"
        if ".m3u8" not in url.lower() or "lrmedia.nba.com" not in url.lower():
            continue
        if url in seen_url:
            continue
        seen_url.add(url)
        options.append({"url": url, "selected": "selected" in attrs, "label": label})

    if not options:
        urls = re.findall(r'https://lrmedia\.nba\.com/[^"\'<>\\\s]+?\.m3u8[^"\'<>\\\s]*', text, flags=re.I)
        for i, raw in enumerate(urls, 1):
            url = htmlmod.unescape(raw)
            if url in seen_url:
                continue
            seen_url.add(url)
            options.append({"url": url, "selected": i == 1, "label": f"angle_{i:02d}"})

    if not options:
        raise RuntimeError(f"No signed lrmedia HLS found; page title={title!r}; bytes={len(text)}")
    return {"page_url": page_url, "title": title, "angles": options}


def download_hls(url: str, out: Path) -> None:
    headers = f"User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\nOrigin: https://clips.nba.com\r\nAccept: */*\r\n"
    sh([
        "ffmpeg", "-y", "-v", "error", "-rw_timeout", "30000000", "-headers", headers,
        "-i", url, "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", str(out),
    ])


def ahash_frame(path: Path, t: float) -> str:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path),
         "-vf", "scale=16:9,format=gray", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    if p.returncode or len(p.stdout) != 144:
        return ""
    vals = list(p.stdout)
    avg = sum(vals) / len(vals)
    bits = "".join("1" if x >= avg else "0" for x in vals)
    return f"{int(bits, 2):036x}"


def hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    return (int(a, 16) ^ int(b, 16)).bit_count()


def probe_video(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,codec_name:format=duration,bit_rate",
         "-of", "json", str(path)], capture_output=True, text=True,
    )
    if p.returncode:
        return {"ok": False, "reason": "ffprobe_failed", "stderr": p.stderr[-1000:]}
    try:
        j = json.loads(p.stdout)
        s = j["streams"][0]
        duration = float((j.get("format") or {}).get("duration") or 0)
    except Exception as exc:
        return {"ok": False, "reason": "ffprobe_parse", "error": repr(exc)}
    times = [min(max(duration * f, 0.25), max(duration - 0.25, 0.25)) for f in (0.25, 0.5, 0.75)]
    hashes = [ahash_frame(path, t) for t in times]
    distances = [hamming_hex(a, b) for a, b in zip(hashes, KNOWN_PLACEHOLDER_AHASH)]
    placeholder_like = all(d <= 12 for d in distances)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ok = duration >= 2.0 and all(hashes) and not placeholder_like
    return {
        "ok": ok,
        "reason": None if ok else ("known_nba_video_not_available_placeholder" if placeholder_like else "invalid_media"),
        "duration": duration,
        "width": s.get("width"),
        "height": s.get("height"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "codec_name": s.get("codec_name"),
        "bit_rate": (j.get("format") or {}).get("bit_rate"),
        "sha256": sha,
        "visual_fingerprints": hashes,
        "placeholder_hamming_distances": distances,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    clips_dir = OUT / "native_all_angles"
    clips_dir.mkdir(parents=True, exist_ok=True)

    page = parse_clips_page()
    print("CLIPS_PAGE_TITLE=" + page["title"], flush=True)
    print("ANGLE_MENU=" + json.dumps([{k:a[k] for k in ("label","selected")} for a in page["angles"]]), flush=True)

    qa = []
    kept: list[Path] = []
    seen_sha: dict[str, str] = {}

    for i, angle in enumerate(page["angles"], 1):
        rec = {
            "game_id": GAME_ID,
            "event_id": EVENT_ID,
            "clips_page_title": page["title"],
            "angle_index": i,
            "angle_count_on_page": len(page["angles"]),
            "angle_label": angle["label"],
            "selected_on_page": angle["selected"],
            "status": "failed",
        }
        path = clips_dir / f"{i:02d}_{safe_name(angle['label'])}.mp4"
        try:
            download_hls(angle["url"], path)
            probe = probe_video(path)
            rec["probe"] = probe
            if not probe["ok"]:
                raise RuntimeError(probe["reason"])
            sha = probe["sha256"]
            if sha in seen_sha:
                rec["status"] = "duplicate_angle_excluded"
                rec["duplicate_of"] = seen_sha[sha]
                path.unlink(missing_ok=True)
            else:
                seen_sha[sha] = path.name
                rec["status"] = "ok"
                rec["native_path"] = str(path)
                kept.append(path)
        except Exception as exc:
            rec["error"] = repr(exc)
            path.unlink(missing_ok=True)
        qa.append(rec)

    failed = [x for x in qa if x["status"] == "failed"]
    payload = {
        "query": "Ja Morant 19-foot Q2 buzzer-beater vs San Antonio, 2022-02-28, assisted by Steven Adams",
        "game_id": GAME_ID,
        "event_id": EVENT_ID,
        "play_description": "Morant 19' Fadeaway Jumper (29 PTS) (Adams 4 AST)",
        "clips_page_url": page["page_url"],
        "clips_page_title": page["title"],
        "video_source": "clips.nba.com exact game/event page -> every distinct signed lrmedia.nba.com HLS option",
        "rendering": "native source stream-copy via ffmpeg -c copy; no upscale; no video re-encode",
        "angle_options_on_page": len(page["angles"]),
        "valid_distinct_angle_files": len(kept),
        "duplicate_angle_options_excluded": sum(1 for x in qa if x["status"] == "duplicate_angle_excluded"),
        "failed_angle_records": len(failed),
        "angles": qa,
    }
    (OUT / "video_qa.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (OUT / "video_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["filename","angle_label","width","height","avg_frame_rate","duration","bytes","sha256"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in qa:
            if x["status"] != "ok":
                continue
            p = Path(x["native_path"])
            pr = x["probe"]
            w.writerow({
                "filename": p.name,
                "angle_label": x["angle_label"],
                "width": pr.get("width"),
                "height": pr.get("height"),
                "avg_frame_rate": pr.get("avg_frame_rate"),
                "duration": pr.get("duration"),
                "bytes": pr.get("bytes"),
                "sha256": pr.get("sha256"),
            })

    manifest = {
        "game_id": GAME_ID,
        "event_id": EVENT_ID,
        "date_us": "2022-02-28",
        "date_nz": "2022-03-01",
        "matchup": "SAS @ MEM",
        "period": 2,
        "clock": "0:00",
        "description": "Morant 19' Fadeaway Jumper (29 PTS) (Adams 4 AST)",
        "score_after": "SAS 58 - MEM 68",
        "source": "NBA play-by-play event 352",
    }
    (OUT / "event_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zpath = OUT / "morant_adams_2022_event352_all_native_angles.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for p in kept:
            z.write(p, arcname=f"videos/{p.name}")
        for meta in ("event_manifest.json", "video_qa.json", "video_inventory.csv"):
            z.write(OUT / meta, arcname=meta)

    print(f"ANGLE_OPTIONS={len(page['angles'])}", flush=True)
    print(f"VALID_DISTINCT_ANGLES={len(kept)}", flush=True)
    print(f"DUPLICATE_OPTIONS_EXCLUDED={sum(1 for x in qa if x['status'] == 'duplicate_angle_excluded')}", flush=True)
    print(f"FAILED_ANGLES={len(failed)}", flush=True)
    print("VALID_LABELS=" + json.dumps([x["angle_label"] for x in qa if x["status"] == "ok"]), flush=True)
    print(f"ZIP={zpath} BYTES={zpath.stat().st_size}", flush=True)
    if failed or not kept:
        sys.exit(2)


if __name__ == "__main__":
    main()
