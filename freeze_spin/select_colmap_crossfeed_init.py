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

# COLMAP two-view geometry configuration IDs. For initialization we explicitly
# reject planar/panoramic/degenerate geometries because basketball broadcast
# views share huge planar court/arena regions that can produce thousands of
# verified SIFT matches without useful 3D camera baseline.
CONFIG_NAMES = {
    0: "UNDEFINED",
    1: "DEGENERATE",
    2: "CALIBRATED",
    3: "UNCALIBRATED",
    4: "PLANAR",
    5: "PANORAMIC",
    6: "PLANAR_OR_PANORAMIC",
    7: "WATERMARK",
    8: "MULTIPLE",
}
NONPLANAR_CONFIGS = {2, 3, 8}


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
        config = int(config)
        nonplanar = config in NONPLANAR_CONFIGS
        score = (int(rows), anchor_bonus, same_time)
        candidates.append({
            "image_id1": i1,
            "image_id2": i2,
            "image1": n1,
            "image2": n2,
            "label1": l1,
            "label2": l2,
            "verified_inliers": int(rows),
            "config": config,
            "config_name": CONFIG_NAMES.get(config, f"UNKNOWN_{config}"),
            "nonplanar": nonplanar,
            "preferred_pair": preferred,
            "impact_anchor_count": anchor_bonus,
            "same_frame_index": bool(same_time),
            "score": list(score),
        })
    con.close()

    if not candidates:
        raise RuntimeError("No geometrically verified cross-feed image pair exists")

    # Keep only one strongest temporal-frame candidate per physical feed pair.
    # This prevents five near-identical frame combinations from the same two
    # cameras consuming every diagnostic initialization attempt.
    per_feed_pair = {}
    for c in candidates:
        if not c["preferred_pair"] or not c["nonplanar"]:
            continue
        key = tuple(sorted((c["label1"], c["label2"])))
        prev = per_feed_pair.get(key)
        if prev is None or tuple(c["score"]) > tuple(prev["score"]):
            per_feed_pair[key] = c

    nonplanar_feed_pairs = sorted(
        per_feed_pair.values(), key=lambda c: tuple(c["score"]), reverse=True
    )
    if not nonplanar_feed_pairs:
        raise RuntimeError(
            "No non-planar geometrically verified pair exists among the meaningful physical replay feeds"
        )

    best = nonplanar_feed_pairs[0]
    report = {
        "selected": best,
        "candidate_count": len(candidates),
        "preferred_candidate_count": sum(1 for c in candidates if c["preferred_pair"]),
        "nonplanar_preferred_candidate_count": sum(
            1 for c in candidates if c["preferred_pair"] and c["nonplanar"]
        ),
        "distinct_nonplanar_feed_pair_count": len(nonplanar_feed_pairs),
        "top_nonplanar_feed_pairs": nonplanar_feed_pairs[:12],
        "top_cross_feed_pairs_all_geometry": sorted(
            candidates, key=lambda c: tuple(c["score"]), reverse=True
        )[:30],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{best['image_id1']} {best['image_id2']}")


if __name__ == "__main__":
    main()
