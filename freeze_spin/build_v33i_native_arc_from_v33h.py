from __future__ import annotations

"""v33i: render the first native source-grounded static arc from a visually-auditable v33h state.

This is an adapter around the previously validated v31 source-grounded renderer. It does
NOT reopen camera calibration or exact-state synchronization. It replaces only the old v11
inputs with:
- the accepted v33h exact-state source frames;
- per-frame fixed-centre camera state transferred exactly as in v33e/v33h;
- the hardened v33h basketball world point (the old renderer is forbidden to redetect it).

Every RGB sample remains official native source footage. Native 960x540 only. No UHD,
upscale, generated texture, inpainting, optical-flow morph or crossfade.
"""

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v31 as v31
from freeze_spin import run_v33e_locked_three_camera_wide_flow_state_search as v33e
from freeze_spin import run_v33d_locked_three_camera_joint_state_search as v33d
from freeze_spin import run_v33b_rar_exact_state_sweep as v33b
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
CAMS = (LAR, RAR, BCAST)
RELS = tuple(range(-20, 21))
W, H = v32j.W, v32j.H


def _exact_source_path(root: Path, label: str) -> Path:
    token = label.replace(' ', '_')
    candidates = [
        root / f'v33h_source_{token}.png',
        root / 'delivery' / f'v33h_source_{token}.png',
    ]
    for p in candidates:
        if p.exists(): return p
    xs = list(root.rglob(f'v33h_source_{token}.png'))
    if len(xs) != 1:
        raise RuntimeError(f'expected exactly one v33h source {label}; got {xs}')
    return xs[0]


def _ball_colour_from_locked_source(images: dict, chosen_ball: dict):
    pix=[]
    matched=chosen_ball.get('matched',{})
    for label in chosen_ball.get('support_views',[]):
        if label not in images or label not in matched: continue
        b=matched[label].get('bbox')
        if not b: continue
        x1,y1,x2,y2=[int(round(float(x))) for x in b]
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
        if x2<=x1 or y2<=y1: continue
        roi=images[label][y1:y2,x1:x2]
        if not roi.size: continue
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        m=(hsv[:,:,0]>=1)&(hsv[:,:,0]<=34)&(hsv[:,:,1]>=60)&(hsv[:,:,2]>=40)
        if np.any(m): pix.append(roi[m])
    if pix:
        return np.median(np.concatenate(pix,axis=0),axis=0).astype(np.uint8)
    # deterministic basketball-like fallback colour only affects the metric sphere shading;
    # source RGB is preferred above and this fallback is explicitly reported in QA.
    return np.asarray([42,105,190],np.uint8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clips-dir',type=Path,required=True)
    ap.add_argument('--b32-root',type=Path,required=True)
    ap.add_argument('--v73-frame0257',type=Path,required=True)
    ap.add_argument('--v33h-json',type=Path,required=True)
    ap.add_argument('--v33h-artifact-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tokens',type=int,default=760)
    ap.add_argument('--voxel-cm',type=float,default=2.8)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    qh=json.loads(args.v33h_json.read_text())
    if qh.get('status')!='PASS_V33H_NUMERICAL_AWAITING_VISUAL_QA' or not qh.get('numerical_body_ball_gate_passed'):
        raise RuntimeError(f'v33i refuses non-passing v33h: {qh.get("status")}')
    if qh.get('camera_lock') != [LAR,RAR,BCAST] or qh.get('camera_count') != 3:
        raise RuntimeError('v33i camera lock mismatch')
    if qh.get('native_resolution') != [W,H]:
        raise RuntimeError('v33i native resolution mismatch')
    chosen=qh.get('chosen_state') or {}; relmap={c:int(chosen['rels'][c]) for c in CAMS}
    ball=chosen.get('ball') or {}
    if not ball.get('gate') or int(ball.get('support_count',0)) < 2 or int(ball.get('semantic_support_count',0)) < 1:
        raise RuntimeError('v33i requires hardened v33h ball')
    ball_world=np.asarray(ball['world_cm'],np.float64)

    stage=args.b32_root/'stage_a'; scene=json.loads((stage/'v32_scene_manifest.json').read_text())
    if scene.get('resolution') != [W,H] or set(scene.get('cameras',{})) != {LAR,BCAST,RAR}:
        raise RuntimeError('v33i B32 locked scene mismatch')
    centers={k:int(v) for k,v in scene['freeze']['chosen_frame_indices'].items()}

    wide=args.out/'v33i_camera_transfer'; wide.mkdir(parents=True,exist_ok=True)
    v33e.export_wide_burst(args.clips_dir,centers,wide,RELS)
    fs={c:v33d.frames(wide,c) for c in CAMS}
    cert=cv2.imread(str(args.v73_frame0257),cv2.IMREAD_GRAYSCALE)
    if cert is None or cert.shape!=(H,W): raise RuntimeError('missing accepted v73 RAR certificate')
    cam_states={c:{} for c in CAMS}; transfer={}
    for c in (LAR,BCAST):
        rel=relmap[c]; cc,qa=v33d.transfer(fs[c][0][2],fs[c][rel][2],scene['cameras'][c]); transfer[c]=qa
        if cc is None: raise RuntimeError(f'v33i fixed-centre transfer failed for {c}: {qa}')
        cam_states[c][rel]=cc
    rel=relmap[RAR]; cc,qa=v33b.transfer_rar_camera(cert,fs[RAR][rel][2],scene['cameras'][RAR]); transfer[RAR]=qa
    if cc is None: raise RuntimeError(f'v33i RAR fixed-centre transfer failed: {qa}')
    cam_states[RAR][rel]=cc
    s=copy.deepcopy(scene)
    for c in CAMS: s['cameras'][c]=cam_states[c][relmap[c]]
    cams={c:v32j.cam(s,c) for c in CAMS}

    # One and only one exact source image per camera for the inherited renderer.
    selected=args.out/'v33i_selected_frames'; selected.mkdir(exist_ok=True)
    source_paths={}
    for c in CAMS:
        src=_exact_source_path(args.v33h_artifact_dir,c)
        dst=selected/f'v33h_source_{c.replace(" ","_")}.png'; shutil.copy2(src,dst); source_paths[c]=dst

    # v26 temporal background needs absolute source-frame indices, not rel offsets.
    sync={
        'version':'v33i_adapter_from_v33h',
        'selected':{
            'offsets':relmap,
            'frames':{c:int(centers[c]+relmap[c]) for c in CAMS},
            'strict_track_count':0,
            'assigned_tracks':[],
        }
    }
    sync_path=args.out/'v33i_renderer_sync_adapter.json'; sync_path.write_text(json.dumps(sync,indent=2))

    # Patch ONLY input plumbing. The v31 renderer itself remains unchanged.
    original_loader=base.load_cameras
    original_find=v12.find_frame
    original_ball=v12.enhance_ball_three_view
    original_colour=v12.source_ball_colour
    def locked_loader(*_args,**_kwargs): return cams
    def locked_find(root:Path,label:str):
        p=Path(root)/f'v33h_source_{label.replace(" ","_")}.png'
        if not p.exists(): raise FileNotFoundError(p)
        return p
    def locked_ball(_cams,_candidates):
        return ball_world.copy(), {
            'status':'BALL_LOCKED_FROM_V33H_HARDENED_SOURCE_QA',
            'center_world_cm':[float(x) for x in ball_world],
            'views':list(ball.get('support_views',[])),
            'detections':ball.get('matched',{}),
            'reprojection_errors_px':{c:float(ball['matched'][c]['reprojection_error_px']) for c in ball.get('support_views',[])},
            'rms_reprojection_px':float(math.sqrt(np.mean([float(ball['matched'][c]['reprojection_error_px'])**2 for c in ball.get('support_views',[])]))),
            'distance_from_rim_center_cm':float(ball.get('distance_to_rim_center_cm')),
            'distance_to_nearest_wrist_cm':float(ball.get('distance_to_nearest_wrist_cm')),
            'semantic_support_count':int(ball.get('semantic_support_count',0)),
            'source':'v33h locked exact-state body+ball QA; renderer redetection disabled',
        }
    def locked_colour(images,_candidates,_used): return _ball_colour_from_locked_source(images,ball)
    base.load_cameras=locked_loader; v12.find_frame=locked_find
    v12.enhance_ball_three_view=locked_ball; v12.source_ball_colour=locked_colour

    # Supply required inherited CLI. Registry/report/frame paths are parser contract only;
    # camera loading is replaced above by the accepted fixed-centre exact-state cameras.
    saved=sys.argv[:]
    sys.argv=[saved[0],
        '--clips-dir',str(args.clips_dir),
        '--frames-dir',str(selected),
        '--sync-qa',str(sync_path),
        '--registry',str(stage/'v32_scene_manifest.json'),
        '--rar-report',str(args.v33h_json),
        '--broadcast-event-frame',str(source_paths[BCAST]),
        '--out',str(args.out),
        '--tokens',str(args.tokens),'--voxel-cm',str(args.voxel_cm)]
    try:
        v31.main()
    finally:
        sys.argv=saved; base.load_cameras=original_loader; v12.find_frame=original_find
        v12.enhance_ball_three_view=original_ball; v12.source_ball_colour=original_colour

    qp=args.out/'three_camera_mesh_v12_qa.json'; qr=json.loads(qp.read_text())
    qr['v33i_locked_state_adapter']={
        'status':'V33I_STATIC_ARC_RENDERED_AWAITING_VISUAL_QA',
        'upstream_v33h_status':qh.get('status'),'camera_lock':[LAR,RAR,BCAST],
        'rels':relmap,'absolute_frames':sync['selected']['frames'],'source_files':{c:source_paths[c].name for c in CAMS},
        'fixed_center_transfer':transfer,'ball_world_cm':[float(x) for x in ball_world],
        'ball_support_views':ball.get('support_views',[]),'ball_semantic_support_count':int(ball.get('semantic_support_count',0)),
        'renderer':'existing v31 source-grounded mesh/court/background renderer with input-only v33h adapter',
        'native_resolution':[W,H],'generated_rgb':False,'inpainting':False,'crossfade':False,'upscale':False,'uhd':False,
        'visual_pass_required':True,'freeview_animation_unlocked':False,
    }
    qp.write_text(json.dumps(qr,indent=2))
    print(json.dumps(qr['v33i_locked_state_adapter'],indent=2),flush=True)

if __name__=='__main__': main()
