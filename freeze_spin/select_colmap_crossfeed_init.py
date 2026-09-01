from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

MAX_IMAGE_ID = 2147483647
PREFERRED = {
    "Broadcast", "Mobile Broadcast", "In Arena", "Left Slash", "Right Slash",
    "Left HandHeld", "Right HandHeld",
}


def decode_pair_id(pair_id: int):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    meta = {m["image"]: m for m in payload["views"]}

    con = sqlite3.connect(str(args.database))
    images = {int(i): name for i, name in con.execute("SELECT image_id, name FROM images")}
    candidates = []
    for pair_id, rows, config in con.execute("SELECT pair_id, rows, config FROM two_view_geometries WHERE rows > 0"):
        i1, i2 = decode_pair_id(int(pair_id))
        n1, n2 = images.get(i1), images.get(i2)
        if not n1 or not n2 or n1 not in meta or n2 not in meta:
            continue
        m1, m2 = meta[n1], meta[n2]
        l1, l2 = m1["label"], m2["label"]
        if l1 == l2:
            continue
        preferred = l1 in PREFERRED and l2 in PREFERRED
        anchor_bonus = int(bool(m1.get("impact_anchor"))) + int(bool(m2.get("impact_anchor")))
        same_time = int(m1.get("frame_index") == m2.get("frame_index"))
        # Verified inliers dominate. Preferred replay cameras, exact impact frames,
        # and same-time pairs only break otherwise similar candidates.
        score = (int(rows), int(preferred), anchor_bonus, same_time)
        candidates.append({
            "image_id1": i1, "image_id2": i2,
            "image1": n1, "image2": n2,
            "label1": l1, "label2": l2,
            "verified_inliers": int(rows),
            "config": int(config),
            "preferred_pair": preferred,
            "impact_anchor_count": anchor_bonus,
            "same_frame_index": bool(same_time),
            "score": list(score),
        })
    con.close()

    if not candidates:
        raise RuntimeError("No geometrically verified cross-feed image pair exists")

    preferred = [c for c in candidates if c["preferred_pair"]]
    pool = preferred if preferred else candidates
    best = max(pool, key=lambda c: tuple(c["score"]))
    report = {
        "selected": best,
        "candidate_count": len(candidates),
        "preferred_candidate_count": len(preferred),
        "top_cross_feed_pairs": sorted(candidates, key=lambda c: tuple(c["score"]), reverse=True)[:30],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{best['image_id1']} {best['image_id2']}")


if __name__ == "__main__":
    main()
