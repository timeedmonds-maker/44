from __future__ import annotations

"""Build the fail-closed observation contract for v32i player reconstruction.

This does not infer masks, meshes, pixels or geometry.  It converts the accepted
v32 scene manifest into an explicit three-camera / 13-frame contract that later
observation stages (SAM2, SAM 3D Body, MAEM-like association, or deterministic
alternatives) must satisfy.

The contract protects provenance:
- exactly Left Above Rim, Broadcast, Right Above Rim;
- exactly the real ±6-frame burst around the chosen freeze;
- native 960x540 source only;
- accepted metric camera matrices copied from v32;
- learned human outputs are OBSERVATIONS/INITIALIZERS, never final geometry;
- texture authority remains the official NBA RGB frames;
- holdout modes explicitly forbid target-camera RGB texture while allowing the
  already solved target camera K/extrinsics for evaluation.

No RGB is generated, resized, warped, interpolated or upscaled here.
"""

import argparse
import hashlib
import json
from pathlib import Path

CAMERAS = ("Left Above Rim", "Broadcast", "Right Above Rim")
CAMERA_IDS = {name: i for i, name in enumerate(CAMERAS)}
NATIVE_RESOLUTION = (960, 540)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-manifest", type=Path, required=True)
    ap.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="optional extracted stage_a root; if supplied every burst file is hash-verified",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scene = json.loads(args.scene_manifest.read_text())

    resolution = tuple(int(x) for x in scene.get("resolution", []))
    if resolution != NATIVE_RESOLUTION:
        raise RuntimeError(f"v32i requires native {NATIVE_RESOLUTION}, got {resolution}")

    scene_cameras = tuple(scene.get("cameras", {}).keys())
    if set(scene_cameras) != set(CAMERAS) or len(scene_cameras) != 3:
        raise RuntimeError(f"v32i is locked to exactly {CAMERAS}; got {scene_cameras}")

    chosen = scene.get("freeze", {}).get("chosen_frame_indices", {})
    if set(chosen) != set(CAMERAS):
        raise RuntimeError("missing chosen freeze index for one or more v32i cameras")

    burst = scene.get("burst", {})
    if set(burst) != set(CAMERAS):
        raise RuntimeError("burst does not contain exactly the three v32i cameras")

    frames = []
    source_hashes = {}
    for camera in CAMERAS:
        rows = burst[camera]
        rels = [int(r["relative_frame"]) for r in rows]
        if rels != list(range(-6, 7)):
            raise RuntimeError(f"{camera}: expected rel frames -6..+6, got {rels}")
        freeze_rows = [r for r in rows if int(r["relative_frame"]) == 0]
        if len(freeze_rows) != 1:
            raise RuntimeError(f"{camera}: expected exactly one t+00 burst row")
        if int(freeze_rows[0]["source_frame_index"]) != int(chosen[camera]):
            raise RuntimeError(f"{camera}: t+00 does not equal chosen v32 freeze frame")

        cam = scene["cameras"][camera]
        camera_record = {
            "camera_id": CAMERA_IDS[camera],
            "camera": camera,
            "K_px": cam["K_px"],
            "R_world_to_camera": cam["R_world_to_camera"],
            "C_world_cm": cam["C_world_cm"],
            "extrinsic_world_to_camera_3x4": cam["extrinsic_world_to_camera_3x4"],
        }

        for row in rows:
            rel = int(row["relative_frame"])
            file_rel = str(row["file"])
            src_index = int(row["source_frame_index"])
            rec = {
                **camera_record,
                "relative_frame": rel,
                "is_exact_freeze": rel == 0,
                "source_frame_index": src_index,
                "source_rgb": file_rel,
                "source_rgb_role": "AUTHORITATIVE_OFFICIAL_NBA_PIXELS",
                "requested_observations": {
                    "player_video_mask": {
                        "role": "OBSERVATION_ONLY",
                        "confidence_required": True,
                        "may_define_final_geometry": False,
                    },
                    "human_mesh_proposal": {
                        "role": "INITIALIZER_ONLY",
                        "may_define_final_world_geometry": False,
                        "learned_appearance_allowed": False,
                    },
                    "keypoints_2d": {
                        "role": "ROBUST_MULTI_VIEW_CONSTRAINT",
                        "per_joint_confidence_required": True,
                    },
                    "dense_mesh_projection_2d": {
                        "role": "OPTIONAL_ASSOCIATION_CONSTRAINT",
                        "per_vertex_confidence_preferred": True,
                    },
                },
            }
            if args.source_root is not None:
                p = args.source_root / file_rel
                if not p.exists():
                    raise RuntimeError(f"missing real source frame: {p}")
                rec["source_rgb_sha256"] = sha256(p)
                source_hashes[f"{camera}:{rel:+d}"] = rec["source_rgb_sha256"]
            frames.append(rec)

    modes = {}
    for holdout in CAMERAS:
        train = [c for c in CAMERAS if c != holdout]
        modes[f"holdout_{holdout.replace(' ', '_').lower()}"] = {
            "geometry_observation_cameras": train,
            "appearance_texture_cameras": train,
            "evaluation_camera": holdout,
            "evaluation_camera_pose_known": True,
            "evaluation_rgb_forbidden_during_fit": True,
            "purpose": "leave-one-view-out geometry/appearance diagnostic",
        }

    modes["production_all_three"] = {
        "geometry_observation_cameras": list(CAMERAS),
        "appearance_texture_cameras": list(CAMERAS),
        "evaluation_camera": None,
        "evaluation_rgb_forbidden_during_fit": False,
        "purpose": "final restricted 0-25 degree native orbit after diagnostics pass",
    }

    contract = {
        "version": "v32i_three_camera_mesh_first_observation_contract",
        "event": scene.get("event", {}),
        "source_scene_version": scene.get("version"),
        "native_resolution": list(NATIVE_RESOLUTION),
        "camera_count": 3,
        "cameras": list(CAMERAS),
        "freeze_frame_indices": {k: int(v) for k, v in chosen.items()},
        "burst_relative_frames": list(range(-6, 7)),
        "real_source_frame_count": len(frames),
        "frames": frames,
        "validation_modes": modes,
        "foreground_policy": {
            "final_representation": "ONE_SHARED_ARTICULATED_MESH_PER_PLAYER",
            "single_view_mesh_prediction_is_final_geometry": False,
            "capsule_or_blob_body_allowed": False,
            "ray_cloud_body_allowed": False,
            "billboard_body_allowed": False,
            "generic_gaussians_define_primary_anatomy": False,
            "unsupported_player_surface_policy": "UNFILLED_NOT_HALLUCINATED",
        },
        "texture_policy": {
            "source": "OFFICIAL_NBA_RGB_ONLY",
            "view_dependent_projection": True,
            "z_buffer_visibility_required": True,
            "dominant_source_camera_required": True,
            "second_source_blend_only_if_consistent": True,
            "learned_or_generated_player_texture": False,
            "inpainting": False,
        },
        "camera_policy": {
            "camera_geometry_prior": "V32E_ACCEPTED_THREE_CAMERA_CALIBRATION",
            "player_pixels_may_refine_camera": False,
            "micro_refinement_static_geometry_only": True,
        },
        "output_policy": {
            "render_resolution": list(NATIVE_RESOLUTION),
            "orbit_deg": [0, 5, 10, 15, 20, 25],
            "upscale": False,
            "generated_rgb": False,
        },
        "source_hashes_sha256": source_hashes,
        "status": "PASS_V32I_OBSERVATION_CONTRACT" if len(frames) == 39 else "FAIL",
    }

    if contract["status"] != "PASS_V32I_OBSERVATION_CONTRACT":
        raise RuntimeError(f"expected 39 real burst frames, got {len(frames)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contract, indent=2))
    print(json.dumps({
        "status": contract["status"],
        "camera_count": contract["camera_count"],
        "real_source_frame_count": contract["real_source_frame_count"],
        "native_resolution": contract["native_resolution"],
        "validation_modes": list(contract["validation_modes"]),
    }, indent=2))


if __name__ == "__main__":
    main()
