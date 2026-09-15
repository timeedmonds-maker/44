#!/usr/bin/env python3
"""Public per-play runner for the durable Screen Tracker tool.

The private canonical contracts live in timeedmonds-maker/long-rebound-era.
This public runner consumes a self-contained application manifest and invokes a
validated deterministic backend. It never reads the private repository at
runtime.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TOOL_NAME = "screen tracker"
TOOL_ID = "SCREEN_TRACKER"
VERSION = "1.0.0"
# Keep the public engine contract stable while routing every application through
# the V3 compatibility backend, which preserves universal-v2 role resolution and
# upgrades only the floor-ring renderer to camera-perspective normalization.
SUPPORTED_ENGINES = {"drop_coverage_universal_v2": "tools/drop_coverage_universal_v3.py"}


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--application", required=True)
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    app_path = Path(args.application)
    app = load_json(app_path)
    if app.get("screen_tracker_version") != VERSION:
        raise SystemExit(f"Unsupported Screen Tracker version: {app.get('screen_tracker_version')!r}")

    backend = app["backend"]
    engine = backend["engine"]
    if engine not in SUPPORTED_ENGINES:
        raise SystemExit(f"Unsupported Screen Tracker backend: {engine}")

    artifact_root = Path(args.artifact_root)
    source = artifact_root / app["input_artifact"]["source_path"]
    tracks = artifact_root / app["input_artifact"]["tracks_path"]
    source_qa = artifact_root / app["input_artifact"]["source_qa_path"]
    config = Path(backend["config_path"])
    role_manifest = Path(backend["role_manifest_path"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for p in (source, tracks, source_qa, config, role_manifest):
        if not p.exists():
            raise SystemExit(f"Required Screen Tracker input missing: {p}")

    cfg = load_json(config)
    roles = load_json(role_manifest)
    src_qa = load_json(source_qa)
    event = app["event"]

    game_id = str(event["game_id"])
    event_num = int(event["event_num"])
    assert str(cfg["event"]["game_id"]) == game_id
    assert int(cfg["event"]["event_num"]) == event_num
    assert str(roles["game_id"]) == game_id and int(roles["event_num"]) == event_num
    assert str(src_qa["game_id"]) == game_id and int(src_qa["event_num"]) == event_num
    if event.get("coverage_label"):
        assert cfg["screen"]["value"] == event["coverage_label"], (cfg["screen"], event)

    backend_script = Path(SUPPORTED_ENGINES[engine])
    subprocess.run([
        sys.executable, str(backend_script),
        "--source", str(source),
        "--tracks", str(tracks),
        "--source-qa", str(source_qa),
        "--config", str(config),
        "--role-manifest", str(role_manifest),
        "--out", str(out),
    ], check=True)

    qa_path = out / "qa.json"
    qa = load_json(qa_path)
    backend_tool_id = qa.get("tool_id")

    qa["backend_tool_id"] = backend_tool_id
    qa["tool_id"] = TOOL_ID
    qa["tool_name"] = TOOL_NAME
    qa["screen_tracker_version"] = VERSION
    qa["application_id"] = app["application_id"]
    qa["application_manifest"] = str(app_path)
    qa["application_event"] = {"game_id": game_id, "event_num": event_num}
    qa["application_contracts"] = {
        "data": app.get("data_contract", {}),
        "tracking": app.get("tracking_contract", {}),
        "graphics": app.get("graphics_contract", {}),
        "output": app.get("output", {}),
    }
    qa["private_canonical_home"] = "timeedmonds-maker/long-rebound-era:screen_tracker/"
    qa["public_application_home"] = "timeedmonds-maker/44:screen-tracker"
    qa["deterministic_only"] = True
    qa["ai_image_generation"] = False
    qa["ai_super_resolution"] = False

    assert qa.get("universal_role_resolution_v2") is True
    assert qa.get("visual_lock", {}).get("drop_defender_name_bar") is True
    assert qa.get("visual_lock", {}).get("drop_defender_floor_ring") is True
    ring = qa.get("floor_ring_policy_v2", {})
    assert ring.get("mode") == "camera_scale_at_screen_action_constant_within_angle", ring
    assert ring.get("never_smaller_than_canonical_broadcast") is True, ring
    assert float(ring.get("resolved_scale", 0)) >= 1.0, ring
    colour = qa.get("team_colour_resolution", {})
    assert colour.get("rule") == "defense_keeps_primary_offense_switches_secondary_when_primaries_similar", colour
    assert float(colour.get("rgb_distance_threshold", -1)) == 80.0, colour

    qa_path.write_text(json.dumps(qa, indent=2))
    (out / "SCREEN_TRACKER_TOOL_ID.txt").write_text(f"{TOOL_ID}\n{VERSION}\n{app['application_id']}\n")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
