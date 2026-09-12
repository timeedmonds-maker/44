from __future__ import annotations

"""v32n stage 0: prove that the public MHR anatomical body rig is usable on the CPU runner.

This is intentionally independent of SAM 3D Body weights.  It loads Meta's MHR
parametric character/mesh through PyMomentum, inventories joints/parameters and
exports a neutral native body mesh plus metadata.  No NBA image pixels are used
here and this does not claim an Adams fit; it is a capability/provenance probe for
the surface representation that will replace capsules/blobs.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lod', type=int, default=1)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    import pymomentum.geometry as geo
    from mhr.io import get_mhr_fbx_path, get_mhr_model_path

    assets = args.work / 'assets'
    fbx = Path(get_mhr_fbx_path(assets, args.lod))
    model_path = Path(get_mhr_model_path(assets))
    if not fbx.exists() or not model_path.exists():
        subprocess.run(['mhr-download-assets', '--dest', str(args.work)], check=True)
    assert fbx.exists(), fbx
    assert model_path.exists(), model_path

    character = geo.Character.load_fbx(
        str(fbx), str(model_path), load_blendshapes=True
    )
    skel = character.skeleton
    pt = character.parameter_transform
    names = list(skel.joint_names)
    parents = np.asarray(skel.joint_parents, dtype=np.int32)
    pnames = list(pt.names)
    zeros = np.zeros((len(pnames),), dtype=np.float32)
    state = geo.model_parameters_to_skeleton_state(character, zeros)
    verts = np.asarray(character.skin_points(state), dtype=np.float32)
    faces = np.asarray(character.mesh.faces, dtype=np.int32)

    # Write an ASCII PLY without trimesh dependency assumptions downstream.
    ply = args.out / 'mhr_lod1_neutral_probe.ply'
    with ply.open('w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n')
        f.write(f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v in verts:
            f.write(f'{float(v[0])} {float(v[1])} {float(v[2])}\n')
        for tri in faces:
            f.write(f'3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n')

    wanted = ['head','neck','shoulder','arm','forearm','wrist','spine','upleg','leg','foot','ankle']
    body_joints = [
        {'index': i, 'name': n, 'parent': int(parents[i])}
        for i, n in enumerate(names)
        if any(tok in n.lower() for tok in wanted)
    ]
    sets = {k: int(np.sum(np.asarray(v, bool))) for k, v in pt.parameter_sets.items()}
    qa = {
        'version': 'v32n_mhr_cpu_probe',
        'status': 'PASS_V32N_MHR_CPU_PROBE',
        'purpose': 'Anatomical parametric surface capability only; not an Adams reconstruction.',
        'lod': args.lod,
        'joint_count': len(names),
        'model_parameter_count': len(pnames),
        'vertex_count': int(len(verts)),
        'face_count': int(len(faces)),
        'joint_names': names,
        'body_joint_inventory': body_joints,
        'parameter_names': pnames,
        'parameter_sets': sets,
        'pose_parameter_count': int(np.sum(np.asarray(pt.pose_parameters, bool))),
        'rigid_parameter_count': int(np.sum(np.asarray(pt.rigid_parameters, bool))),
        'scaling_parameter_count': int(np.sum(np.asarray(pt.scaling_parameters, bool))),
        'blend_shape_parameter_count': int(np.sum(np.asarray(pt.blend_shape_parameters, bool))),
        'neutral_bounds_cm': {
            'min': np.min(verts, axis=0).tolist(),
            'max': np.max(verts, axis=0).tolist(),
        },
        'generated_rgb': False,
        'nba_geometry_used': False,
        'learned_image_to_mesh_checkpoint_used': False,
        'surface_type': 'MHR articulated skinned triangle mesh',
    }
    (args.out / 'v32n_mhr_cpu_probe.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps({k: qa[k] for k in ['status','joint_count','model_parameter_count','vertex_count','face_count','pose_parameter_count','rigid_parameter_count','scaling_parameter_count','neutral_bounds_cm']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
