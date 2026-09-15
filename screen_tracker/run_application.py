#!/usr/bin/env python3
"""Public per-play runner for the durable Screen Tracker tool.

Screen Tracker 1.1 makes two official camera angles the default application
contract. Each angle is rendered and QA'd independently, then assembled with the
full-length two-angle combiner. Version 1.0 applications remain reproducible as
explicit legacy single-angle applications.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from angle_policy import DEFAULT_MODE, LEGACY_MODE, mode_for_application

TOOL_NAME = "screen tracker"
TOOL_ID = "SCREEN_TRACKER"
VERSION = "1.1.0"
SUPPORTED_VERSIONS = {"1.0.0", VERSION}
# Keep the public engine key stable while routing through the perspective-
# normalized V3 compatibility backend.
SUPPORTED_ENGINES = {"drop_coverage_universal_v2": "tools/drop_coverage_universal_v3.py"}


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_angle_qa(qa):
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


def run_angle(spec, artifact_root: Path, out: Path, event):
    backend = spec["backend"]
    engine = backend["engine"]
    if engine not in SUPPORTED_ENGINES:
        raise SystemExit(f"Unsupported Screen Tracker backend: {engine}")

    ia = spec["input_artifact"]
    source = artifact_root / ia["source_path"]
    tracks = artifact_root / ia["tracks_path"]
    source_qa = artifact_root / ia["source_qa_path"]
    config = Path(backend["config_path"])
    role_manifest = Path(backend["role_manifest_path"])
    for path in (source, tracks, source_qa, config, role_manifest):
        if not path.exists():
            raise SystemExit(f"Required Screen Tracker input missing: {path}")

    cfg = load_json(config)
    roles = load_json(role_manifest)
    src_qa = load_json(source_qa)
    game_id = str(event["game_id"])
    event_num = int(event["event_num"])
    assert str(cfg["event"]["game_id"]) == game_id
    assert int(cfg["event"]["event_num"]) == event_num
    assert str(roles["game_id"]) == game_id and int(roles["event_num"]) == event_num
    assert str(src_qa["game_id"]) == game_id and int(src_qa["event_num"]) == event_num
    if event.get("coverage_label"):
        assert cfg["screen"]["value"] == event["coverage_label"], (cfg["screen"], event)

    expected_label = spec.get("label")
    actual_label = (src_qa.get("source") or {}).get("angle")
    if expected_label and actual_label:
        assert expected_label.strip().lower() == actual_label.strip().lower(), (expected_label, actual_label)

    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, SUPPORTED_ENGINES[engine],
        "--source", str(source),
        "--tracks", str(tracks),
        "--source-qa", str(source_qa),
        "--config", str(config),
        "--role-manifest", str(role_manifest),
        "--out", str(out),
    ], check=True)
    qa = load_json(out / "qa.json")
    validate_angle_qa(qa)
    qa["camera_label"] = expected_label or actual_label or "Unknown"
    (out / "qa.json").write_text(json.dumps(qa, indent=2))
    return qa


def add_common_qa(qa, app, app_path, version):
    event = app["event"]
    qa["tool_id"] = TOOL_ID
    qa["tool_name"] = TOOL_NAME
    qa["screen_tracker_version"] = version
    qa["application_id"] = app["application_id"]
    qa["application_manifest"] = str(app_path)
    qa["application_event"] = {
        "game_id": str(event["game_id"]),
        "event_num": int(event["event_num"]),
    }
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
    return qa


def run_legacy(app, app_path, artifact_root, out):
    spec = {
        "label": (app.get("angle_policy") or {}).get("primary_label", "Broadcast"),
        "input_artifact": app["input_artifact"],
        "backend": app["backend"],
    }
    qa = run_angle(spec, artifact_root, out, app["event"])
    qa["angle_policy"] = {
        "mode": LEGACY_MODE,
        "angle_count": 1,
        "legacy_reference_only": True,
    }
    add_common_qa(qa, app, app_path, app["screen_tracker_version"])
    (out / "qa.json").write_text(json.dumps(qa, indent=2))
    (out / "SCREEN_TRACKER_TOOL_ID.txt").write_text(
        f"{TOOL_ID}\n{app['screen_tracker_version']}\n{app['application_id']}\n"
    )
    return qa


def run_two_angle(app, app_path, artifact_root, out):
    angles = app.get("angles") or []
    if len(angles) != 2:
        raise SystemExit("Screen Tracker 1.1 default requires exactly two angle specifications")
    labels = [str(a.get("label", "")).strip() for a in angles]
    if not all(labels) or labels[0].lower() == labels[1].lower():
        raise SystemExit(f"Two distinct named camera angles are required: {labels}")

    angle_qas = []
    for i, spec in enumerate(angles):
        angle_out = out / f"angle_{i}"
        qa = run_angle(spec, artifact_root / f"angle_{i}", angle_out, app["event"])
        angle_qas.append(qa)

    combine = HERE / "combine_two_angles.py"
    cmd = [
        sys.executable, str(combine),
        "--out", str(out),
        "--basename", app["output"]["basename"],
    ]
    for i, (label, qa) in enumerate(zip(labels, angle_qas)):
        angle_out = out / f"angle_{i}"
        cmd += [
            "--angle", label,
            str(qa["native"]),
            str(qa["uhd"]),
            str(angle_out / "qa.json"),
        ]
    subprocess.run(cmd, check=True)

    qa = load_json(out / "qa.json")
    assert qa["angle_policy"]["mode"] == DEFAULT_MODE
    assert qa["angle_policy"]["angle_count"] == 2
    assert qa["length_qa"]["second_angle_full_length"] is True
    add_common_qa(qa, app, app_path, VERSION)
    (out / "qa.json").write_text(json.dumps(qa, indent=2))
    (out / "SCREEN_TRACKER_TOOL_ID.txt").write_text(
        f"{TOOL_ID}\n{VERSION}\n{app['application_id']}\nTWO_ANGLE_DEFAULT\n"
    )
    return qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--application", required=True)
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    app_path = Path(args.application)
    app = load_json(app_path)
    version = app.get("screen_tracker_version")
    if version not in SUPPORTED_VERSIONS:
        raise SystemExit(f"Unsupported Screen Tracker version: {version!r}")

    artifact_root = Path(args.artifact_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = mode_for_application(app)
    if mode == LEGACY_MODE:
        qa = run_legacy(app, app_path, artifact_root, out)
    elif mode == DEFAULT_MODE:
        if version != VERSION:
            raise SystemExit("two_angle_default applications must use screen_tracker_version 1.1.0")
        qa = run_two_angle(app, app_path, artifact_root, out)
    else:
        raise SystemExit(f"Unsupported Screen Tracker angle mode: {mode!r}")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
