from __future__ import annotations

import io
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyreadr
import requests
from PIL import Image, ImageDraw, ImageFont

SEASON = "2025-26"
TEAM = "CLE"
PBP_URL = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds"
XFG_URL = "https://stats.gleague.nba.com/stats/shotqualityvideologs"
HEADSHOT_URLS = [
    "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png",
    "https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def get_font(size: int, bold: bool = False, italic: bool = False):
    candidates = []
    if bold and italic:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"]
    elif bold:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    elif italic:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"]
    candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def build_manifest() -> tuple[pd.DataFrame, dict]:
    r = requests.get(PBP_URL, timeout=180)
    r.raise_for_status()
    Path("data.rds").write_bytes(r.content)
    df = next(iter(pyreadr.read_r("data.rds").values()))
    df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    is_fg = pd.to_numeric(df["is_field_goal"], errors="coerce").fillna(0).eq(1)
    reg = df["game_id"].str.startswith("002")
    raw = df.loc[is_fg & reg & df["team_abb"].eq(TEAM), ["game_id", "event_num", "player1_name", "team_abb"]].copy()
    p = raw["player1_name"].astype(str).str.extract(r"^\s*(\d+)\s+(.*)$")
    raw["player_id"] = pd.to_numeric(p[0], errors="coerce").astype("Int64")
    raw["player_name"] = p[1]
    raw = raw.dropna(subset=["player_id", "player_name"]).copy()
    raw["player_id"] = raw["player_id"].astype(int)
    manifest = (
        raw.groupby(["game_id", "player_id", "player_name"], as_index=False)
        .agg(pbp_fga=("event_num", "size"))
        .sort_values(["game_id", "player_id"])
    )
    qa = {
        "team": TEAM,
        "season": SEASON,
        "pbp_fga": int(manifest["pbp_fga"].sum()),
        "player_game_pairs": int(len(manifest)),
        "players": int(manifest["player_id"].nunique()),
        "games": int(manifest["game_id"].nunique()),
    }
    return manifest, qa


def fetch_pair(game_id: str, player_id: int, attempts: int = 4):
    s = session()
    last = None
    for a in range(1, attempts + 1):
        try:
            r = s.get(XFG_URL, params={"GameID": game_id, "PlayerID": int(player_id)}, timeout=(6, 30))
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and int(j.get("playerId") or 0) == int(player_id):
                    return j, None
                last = "unexpected_payload"
            else:
                last = f"http_{r.status_code}"
        except Exception as e:
            last = repr(e)
        if a < attempts:
            time.sleep(min(6, 0.7 * 2 ** (a - 1)))
    return None, last


def aggregate_players(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    acc = defaultdict(lambda: {
        "pbp_fga": 0,
        "tracked_fga": 0,
        "tracked_fgm": 0,
        "tracked_3pa": 0,
        "tracked_3pm": 0,
        "actual_efg_num": 0.0,
        "expected_efg_num": 0.0,
        "successful_pairs": 0,
        "failed_pairs": 0,
    })
    names = {}
    for r in manifest.itertuples(index=False):
        acc[int(r.player_id)]["pbp_fga"] += int(r.pbp_fga)
        names[int(r.player_id)] = str(r.player_name)
    errors = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(fetch_pair, str(r.game_id), int(r.player_id)): (str(r.game_id), int(r.player_id), str(r.player_name), int(r.pbp_fga))
            for r in manifest.itertuples(index=False)
        }
        for n, fut in enumerate(as_completed(futs), start=1):
            game_id, pid, pname, pbp_fga = futs[fut]
            payload, err = fut.result()
            a = acc[pid]
            if payload is None:
                a["failed_pairs"] += 1
                errors.append({"game_id": game_id, "player_id": pid, "player_name": pname, "pbp_fga": pbp_fga, "error": err})
            else:
                a["successful_pairs"] += 1
                for shot in payload.get("shotList") or []:
                    try:
                        xfg = float(shot.get("shotQuality"))
                    except Exception:
                        continue
                    if math.isnan(xfg):
                        continue
                    made = int(shot.get("success") or 0)
                    is3 = str(shot.get("shotType") or "").upper().startswith("3PT")
                    w = 1.5 if is3 else 1.0
                    a["tracked_fga"] += 1
                    a["tracked_fgm"] += made
                    a["tracked_3pa"] += int(is3)
                    a["tracked_3pm"] += int(is3 and made)
                    a["actual_efg_num"] += made * w
                    a["expected_efg_num"] += xfg * w
            if n % 100 == 0 or n == len(futs):
                print(f"xFG pairs {n}/{len(futs)} errors={len(errors)}", flush=True)
    rows = []
    for pid in sorted(acc):
        a = acc[pid]
        tracked = a["tracked_fga"]
        xefg = 100 * a["expected_efg_num"] / tracked if tracked else math.nan
        efg = 100 * a["actual_efg_num"] / tracked if tracked else math.nan
        rows.append({
            "player_id": pid,
            "player_name": names.get(pid, str(pid)),
            **a,
            "xEFG_pct": xefg,
            "tracked_actual_eFG_pct": efg,
            "eFG_minus_xEFG_pp": efg - xefg if tracked else math.nan,
            "xfg_coverage_pct_vs_pbp_fga": 100 * tracked / a["pbp_fga"] if a["pbp_fga"] else math.nan,
        })
    out = pd.DataFrame(rows).sort_values(["tracked_fga", "player_name"], ascending=[False, True]).reset_index(drop=True)
    return out, errors


def fetch_headshot(player_id: int) -> Image.Image | None:
    s = requests.Session()
    s.headers.update({"User-Agent": HEADERS["User-Agent"], "Referer": "https://www.nba.com/"})
    for tmpl in HEADSHOT_URLS:
        try:
            r = s.get(tmpl.format(player_id=player_id), timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                im = Image.open(io.BytesIO(r.content)).convert("RGBA")
                return process_headshot(im)
        except Exception:
            pass
    return None


def process_headshot(im: Image.Image) -> Image.Image:
    alpha = im.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        im = im.crop(bbox)
    # Retain ~70% of foreground height (prior accepted lineup-chart method), centered on head/upper torso.
    w, h = im.size
    crop_h = max(1, int(round(h * 0.70)))
    top = 0
    im = im.crop((0, top, w, min(h, crop_h)))
    alpha = im.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        im = im.crop(bbox)
    return im


def initials_placeholder(name: str, size: int = 160) -> Image.Image:
    im = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    dr = ImageDraw.Draw(im)
    dr.ellipse((4, 4, size - 4, size - 4), fill=(111, 38, 61, 255), outline=(253, 187, 48, 255), width=5)
    parts = [p for p in name.replace("'", "").replace(".", "").split() if p]
    initials = "".join(p[0].upper() for p in parts[:2]) or "?"
    font = get_font(48, bold=True)
    bb = dr.textbbox((0, 0), initials, font=font)
    dr.text(((size - (bb[2] - bb[0])) / 2, (size - (bb[3] - bb[1])) / 2 - 4), initials, font=font, fill="white")
    return im


def normalize_headshot(im: Image.Image, box_w: int = 156, box_h: int = 118) -> Image.Image:
    # Standardize apparent player size by foreground bbox height rather than source canvas dimensions.
    alpha = im.getchannel("A")
    bb = alpha.getbbox()
    if bb:
        im = im.crop(bb)
    ratio = min(box_w / im.width, box_h / im.height)
    nw = max(1, int(round(im.width * ratio)))
    nh = max(1, int(round(im.height * ratio)))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (box_w, box_h), (255, 255, 255, 0))
    canvas.alpha_composite(im, ((box_w - nw) // 2, box_h - nh))
    return canvas


def render(df: pd.DataFrame, manifest_qa: dict, errors: list[dict]):
    plotted = df[df["tracked_fga"] > 0].copy().reset_index(drop=True)
    if plotted.empty:
        raise RuntimeError("No CLE player xFG shots returned")

    # Fetch official NBA CDN headshots concurrently.
    headshots = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_headshot, int(r.player_id)): (int(r.player_id), str(r.player_name)) for r in plotted.itertuples(index=False)}
        for fut in as_completed(futs):
            pid, name = futs[fut]
            hs = fut.result()
            headshots[pid] = normalize_headshot(hs if hs is not None else initials_placeholder(name))

    W, H = 3200, 2250
    bg = (248, 248, 248, 255)
    im = Image.new("RGBA", (W, H), bg)
    dr = ImageDraw.Draw(im)

    # Layout intentionally mirrors the accepted team chart proportions.
    L, T, R, B = 260, 350, 3090, 1910
    PW, PH = R - L, B - T

    xvals = plotted["xEFG_pct"].astype(float)
    yvals = plotted["eFG_minus_xEFG_pp"].astype(float)
    xmin = math.floor((xvals.min() - 2.0) / 2.0) * 2.0
    xmax = math.ceil((xvals.max() + 2.0) / 2.0) * 2.0
    ymin = math.floor((yvals.min() - 1.5) / 2.0) * 2.0
    ymax = math.ceil((yvals.max() + 1.5) / 2.0) * 2.0
    if xmax - xmin < 12:
        pad = (12 - (xmax - xmin)) / 2
        xmin -= pad; xmax += pad
    if ymax - ymin < 12:
        pad = (12 - (ymax - ymin)) / 2
        ymin -= pad; ymax += pad

    league_x = float(plotted["xEFG_pct"].mean())

    xp = lambda x: L + (x - xmin) / (xmax - xmin) * PW
    yp = lambda y: B - (y - ymin) / (ymax - ymin) * PH

    f_title = get_font(78, bold=True)
    f_sub = get_font(38)
    f_axis = get_font(36, bold=True)
    f_tick = get_font(26)
    f_quad = get_font(28, italic=True)
    f_name = get_font(28, bold=True)
    f_stat = get_font(22)
    f_small = get_font(18)

    title = "Shot Quality & Shot Making"
    subtitle = "2025-26 Cleveland Cavaliers players | NBA shot-level tracking data"
    bb = dr.textbbox((0, 0), title, font=f_title)
    dr.text(((W - (bb[2] - bb[0])) / 2, 44), title, font=f_title, fill=(0, 0, 0, 255))
    bb = dr.textbbox((0, 0), subtitle, font=f_sub)
    dr.text(((W - (bb[2] - bb[0])) / 2, 142), subtitle, font=f_sub, fill=(35, 35, 35, 255))

    axis = (45, 45, 45, 255)
    ref = (155, 155, 155, 255)
    dr.line((L, T, L, B), fill=axis, width=3)
    dr.line((L, B, R, B), fill=axis, width=3)

    # Dashed league-x and zero-y lines.
    vx = xp(league_x)
    hy = yp(0.0)
    for y in range(T, B, 28):
        dr.line((vx, y, vx, min(y + 16, B)), fill=ref, width=3)
    if ymin <= 0 <= ymax:
        for x in range(L, R, 28):
            dr.line((x, hy, min(x + 16, R), hy), fill=ref, width=3)

    # Ticks use clean regular increments without odd edge labels.
    xtick_start = math.ceil(xmin / 2.0) * 2.0
    x = xtick_start
    while x <= xmax + 1e-9:
        px = xp(x)
        dr.line((px, B, px, B + 12), fill=axis, width=3)
        lab = f"{x:.0f}"
        bb = dr.textbbox((0, 0), lab, font=f_tick)
        dr.text((px - (bb[2] - bb[0]) / 2, B + 18), lab, font=f_tick, fill=axis)
        x += 2.0
    ytick_start = math.ceil(ymin / 2.0) * 2.0
    y = ytick_start
    while y <= ymax + 1e-9:
        py = yp(y)
        dr.line((L - 12, py, L, py), fill=axis, width=3)
        lab = f"{y:.0f}"
        bb = dr.textbbox((0, 0), lab, font=f_tick)
        dr.text((L - 20 - (bb[2] - bb[0]), py - (bb[3] - bb[1]) / 2), lab, font=f_tick, fill=axis)
        y += 2.0

    xlab = "Expected eFG% (shot quality)"
    bb = dr.textbbox((0, 0), xlab, font=f_axis)
    dr.text(((L + R - (bb[2] - bb[0])) / 2, 2070), xlab, font=f_axis, fill=(0, 0, 0, 255))

    ylab = "Actual eFG% vs expected (percentage points)"
    ytmp = Image.new("RGBA", (1100, 80), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ytmp)
    yd.text((0, 0), ylab, font=f_axis, fill=(0, 0, 0, 255))
    ytmp = ytmp.rotate(90, expand=True)
    im.alpha_composite(ytmp, (38, int((T + B - ytmp.height) / 2)))

    qfill = (80, 80, 80, 255)
    q1 = "Worse shot quality /\nBetter than expected shot making"
    q2 = "Better shot quality /\nBetter than expected shot making"
    q3 = "Worse shot quality /\nWorse than expected shot making"
    q4 = "Better shot quality /\nWorse than expected shot making"
    dr.multiline_text((L + 55, T + 28), q1, font=f_quad, fill=qfill, spacing=2)
    bb = dr.multiline_textbbox((0, 0), q2, font=f_quad, spacing=2)
    dr.multiline_text((R - 30 - (bb[2] - bb[0]), T + 28), q2, font=f_quad, fill=qfill, spacing=2)
    dr.multiline_text((L + 55, B - 86), q3, font=f_quad, fill=qfill, spacing=2)
    bb = dr.multiline_textbbox((0, 0), q4, font=f_quad, spacing=2)
    dr.multiline_text((R - 30 - (bb[2] - bb[0]), B - 86), q4, font=f_quad, fill=qfill, spacing=2)

    # Exact x coordinates remain locked. Only displayed y can move to resolve collisions.
    anchor = {int(r.player_id): (xp(float(r.xEFG_pct)), yp(float(r.eFG_minus_xEFG_pp))) for r in plotted.itertuples(index=False)}
    disp_y = {pid: p[1] for pid, p in anchor.items()}
    pids = list(anchor)
    min_v = 190.0
    max_disp = 155.0
    for _ in range(900):
        moved = False
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                a, b = pids[i], pids[j]
                dx = abs(anchor[a][0] - anchor[b][0])
                if dx > 180:
                    continue
                dy = disp_y[b] - disp_y[a]
                if abs(dy) < min_v:
                    moved = True
                    sign = 1 if dy >= 0 else -1
                    if abs(dy) < 1e-6:
                        sign = 1 if anchor[b][1] >= anchor[a][1] else -1
                    push = (min_v - abs(dy)) * 0.52
                    disp_y[a] -= sign * push
                    disp_y[b] += sign * push
        for pid in pids:
            disp_y[pid] += (anchor[pid][1] - disp_y[pid]) * 0.018
            delta = disp_y[pid] - anchor[pid][1]
            if abs(delta) > max_disp:
                disp_y[pid] = anchor[pid][1] + (max_disp if delta > 0 else -max_disp)
        if not moved:
            break

    # Draw anchor dot/leader then headshot and figures directly below.
    for r in plotted.sort_values("tracked_fga", ascending=True).itertuples(index=False):
        pid = int(r.player_id)
        axx, ayy = anchor[pid]
        cy = disp_y[pid]
        if abs(cy - ayy) > 18:
            dr.line((axx, ayy, axx, cy), fill=(185, 185, 185, 255), width=2)
            dr.ellipse((axx - 5, ayy - 5, axx + 5, ayy + 5), fill=(90, 90, 90, 255))
        hs = headshots[pid]
        hx = int(round(axx - hs.width / 2))
        hy = int(round(cy - hs.height / 2 - 32))
        im.alpha_composite(hs, (hx, hy))
        name = str(r.player_name)
        stat = f"xEFG {r.xEFG_pct:.1f}% | eFG {r.tracked_actual_eFG_pct:.1f}% | {r.eFG_minus_xEFG_pp:+.1f} pp"
        bb = dr.textbbox((0, 0), name, font=f_name)
        dr.text((axx - (bb[2] - bb[0]) / 2, hy + hs.height + 2), name, font=f_name, fill=(0, 0, 0, 255))
        bb2 = dr.textbbox((0, 0), stat, font=f_stat)
        dr.text((axx - (bb2[2] - bb2[0]) / 2, hy + hs.height + 35), stat, font=f_stat, fill=(45, 45, 45, 255))

    note = "Headshots: official NBA CDN · xFG: NBA shotqualityvideologs · all CLE players with a tracked 2025-26 regular-season FGA"
    bb = dr.textbbox((0, 0), note, font=f_small)
    dr.text((W - 34 - (bb[2] - bb[0]), H - 46), note, font=f_small, fill=(90, 90, 90, 255))

    png = OUT / "cavs_player_xefg_quadrant_2025_26.png"
    im.convert("RGB").save(png, optimize=True, compress_level=9)
    return png, int(len(headshots))


def main():
    manifest, manifest_qa = build_manifest()
    manifest.to_csv(OUT / "cavs_player_xefg_manifest_2025_26.csv", index=False)
    df, errors = aggregate_players(manifest)
    df.to_csv(OUT / "cavs_player_xefg_2025_26.csv", index=False)
    png, headshot_count = render(df, manifest_qa, errors)
    qa = {
        **manifest_qa,
        "source_resource": "NBA shotqualityvideologs",
        "headshot_source": "official NBA CDN 1040x760/260x190",
        "players_with_tracked_xfg": int((df["tracked_fga"] > 0).sum()),
        "tracked_xfg_fga": int(df["tracked_fga"].sum()),
        "request_errors": int(len(errors)),
        "headshots_or_placeholders_rendered": headshot_count,
        "note": "No missing xFG imputed. Player eFG and xEFG use identical tracked-shot denominators. Headshots are downloaded on GitHub Actions and composited deterministically with Pillow; no generative imagery.",
    }
    (OUT / "cavs_player_xefg_qa_2025_26.json").write_text(json.dumps(qa, indent=2))
    pd.DataFrame(errors).to_csv(OUT / "cavs_player_xefg_errors_2025_26.csv", index=False)
    print(df[["player_id","player_name","pbp_fga","tracked_fga","tracked_actual_eFG_pct","xEFG_pct","eFG_minus_xEFG_pp","xfg_coverage_pct_vs_pbp_fga"]].to_string(index=False))
    print(json.dumps(qa, indent=2))
    print(png)


if __name__ == "__main__":
    main()
