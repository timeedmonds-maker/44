from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from build_portable_moge_pnp_freeview_v12 import (
    W, H, detect_dynamic_and_ball, moge_infer,
    solve_target_from_reference, scaled_world_cloud, reference_cloud,
    angle_between,
)
from build_portable_moge_true_orbit_v16 import true_orbit_pose, project_point


def project(P, X):
    h = np.column_stack([X, np.ones(len(X))])
    q = (P @ h.T).T
    ok = q[:, 2] > 1e-6
    uv = np.zeros((len(X), 2), np.float64)
    uv[ok] = q[ok, :2] / q[ok, 2:3]
    return uv, ok


def bounded_shift(arr: np.ndarray, dx: int, dy: int, fill=0) -> np.ndarray:
    out = np.full_like(arr, fill)
    h, w = arr.shape[:2]
    sx0 = max(0, -dx); sx1 = min(w, w - dx)
    sy0 = max(0, -dy); sy1 = min(h, h - dy)
    dx0 = sx0 + dx; dx1 = sx1 + dx
    dy0 = sy0 + dy; dy1 = sy1 + dy
    if sx1 > sx0 and sy1 > sy0:
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def raster_cloud_bounded(cloud, K, R, C, radius=1):
    X, col, dyn = cloud
    t = -R @ C
    P = K @ np.hstack([R, t.reshape(3, 1)])
    uv, ok = project(P, X.astype(np.float64))
    z = (R @ (X.astype(np.float64) - C).T).T[:, 2]
    u = np.rint(uv[:, 0]).astype(int); v = np.rint(uv[:, 1]).astype(int)
    ok &= (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ids = np.where(ok)[0]
    image = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    dynamic = np.zeros((H, W), np.uint8)
    zbuf = np.full(H * W, np.inf, np.float32)
    if len(ids):
        pix = v[ids] * W + u[ids]
        np.minimum.at(zbuf, pix, z[ids].astype(np.float32))
        win = ids[z[ids] <= zbuf[pix] + 1e-4]
        image[v[win], u[win]] = col[win]
        mask[v[win], u[win]] = 255
        dynamic[v[win], u[win]] = np.where(dyn[win], 255, 0).astype(np.uint8)
    if radius:
        for _ in range(radius):
            base_img = image.copy(); base_mask = mask.copy(); base_dyn = dynamic.copy()
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                simg = bounded_shift(base_img, dx, dy, 0)
                sm = bounded_shift(base_mask, dx, dy, 0)
                sd = bounded_shift(base_dyn, dx, dy, 0)
                take = (mask == 0) & (sm > 0)
                image[take] = simg[take]
                mask[take] = 255
                dynamic[take] = sd[take]
    return image, mask, dynamic


def safe_static_fill(base_img, base_mask, cand_img, cand_mask, cand_dynamic):
    unresolved = base_mask == 0
    candidate = (cand_mask > 0) & (cand_dynamic == 0) & unresolved
    inv = (base_mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    w = (base_mask > 0).astype(np.float32)
    wf = cv2.GaussianBlur(w, (0,0), 4.0)
    mean = np.zeros_like(base_img, np.float32)
    for c in range(3):
        mean[:,:,c] = cv2.GaussianBlur(base_img[:,:,c].astype(np.float32) * w, (0,0), 4.0) / np.maximum(wf, 1e-5)
    lab_c = cv2.cvtColor(cand_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_m = cv2.cvtColor(np.clip(mean,0,255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.linalg.norm(lab_c - lab_m, axis=2)
    take = candidate & (dist <= 12.0) & (wf > 0.05) & (delta <= 36.0)
    return take, {
        "static_safe_fill": int(take.sum()),
        "static_rejected": int((candidate & ~take).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--locked-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', default='In Arena')
    ap.add_argument('--tokens', type=int, default=1600)
    ap.add_argument('--max-degree', type=float, default=5.0)
    ap.add_argument('--frames', type=int, default=31)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    images = {}
    for p in sorted(args.locked_dir.glob('*_apex.png')):
        label = p.stem.replace('_apex','').replace('_',' ')
        im = cv2.imread(str(p))
        if im is not None: images[label] = im
    if args.reference not in images:
        raise RuntimeError(f'Reference {args.reference} unavailable: {list(images)}')

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    detector = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, progress=True).eval()
    moge = MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval()

    views = {}
    for label, image in images.items():
        if image.shape[:2] != (H, W): raise RuntimeError(f'{label} not native 960x540')
        dyn, balls = detect_dynamic_and_ball(detector, image)
        depth, points, valid, K, Kn = moge_infer(moge, image, args.tokens)
        views[label] = {'label':label,'image':image,'dynamic':dyn,'balls':balls,'depth':depth,'points':points,'valid':valid,'K':K,'Kn':Kn}

    ref = views[args.reference]
    solves = {}
    for label, view in views.items():
        if label == args.reference: continue
        s = solve_target_from_reference(ref, view); solves[label] = s
        print(label, {k:v for k,v in s.items() if k not in ('R','t','C')}, flush=True)
    passed = [(lab,s) for lab,s in solves.items() if s.get('passed')]
    if not passed: raise RuntimeError('No secondary camera passed calibration')
    if not ref['balls']: raise RuntimeError('No basketball detected in reference corrected-apex frame')

    b = ref['balls'][0]
    bx = int(np.clip(round(b['cx']),0,W-1)); by = int(np.clip(round(b['cy']),0,H-1))
    target = ref['points'][by,bx].astype(np.float64)
    if not np.all(np.isfinite(target)) or target[2] <= 0.25:
        raise RuntimeError('No valid MoGe depth at detected basketball')

    options=[]
    for lab,s in passed:
        C=s['C'].astype(np.float64); ang=angle_between(-target,C-target)
        if ang >= 3.0: options.append((ang,lab,s))
    if not options: raise RuntimeError('No distinct solved camera supports >=3 degree orbit')
    options.sort(key=lambda x:x[0])
    baseline_angle,target_label,target_solve=options[0]
    render_max=min(float(args.max_degree),float(baseline_angle))
    C_target=target_solve['C'].astype(np.float64)

    clouds={args.reference:reference_cloud(ref)}
    for lab,s in passed: clouds[lab]=scaled_world_cloud(views[lab],s)

    # Dynamic appearance is permitted only from cameras physically near the requested small arc.
    baseline_by_camera={lab:float(angle_between(-target,s['C'].astype(np.float64)-target)) for lab,s in passed}
    dynamic_eligible=[lab for lab in baseline_by_camera if baseline_by_camera[lab] <= render_max + 2.0]
    if target_label not in dynamic_eligible: dynamic_eligible.insert(0,target_label)

    radius0=float(np.linalg.norm(target))
    pivot0=project_point(ref['K'],np.eye(3),np.zeros(3),target)
    focal0=[float(ref['K'][0,0]),float(ref['K'][1,1])]
    pose_qa=[]

    def pose(deg):
        R,C,_=true_orbit_pose(target,C_target,deg)
        rad=float(np.linalg.norm(C-target)); piv=project_point(ref['K'],R,C,target)
        pose_qa.append({'degree':float(deg),'radius':rad,'radius_drift':float(rad-radius0),'pivot_pixel':piv.tolist(),'pivot_drift_px':float(np.linalg.norm(piv-pivot0))})
        return R,C

    def render(deg):
        if deg <= 1e-9:
            return ref['image'].copy(), np.full((H,W),255,np.uint8), {'fills':[]}
        R,C=pose(deg)
        base,mask,dyn=raster_cloud_bounded(clouds[args.reference],ref['K'],R,C,radius=1)
        # Action ROI follows the projected reference people. Dynamic source pixels outside this region are forbidden.
        action_roi=cv2.dilate((dyn>0).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(71,71)),iterations=1)>0
        reports=[]
        for lab,s in passed:
            im,cm,cd=raster_cloud_bounded(clouds[lab],ref['K'],R,C,radius=1)
            dynamic_take=np.zeros((H,W),bool)
            if lab in dynamic_eligible:
                dynamic_take=(mask==0)&(cm>0)&(cd>0)&action_roi
                base[dynamic_take]=im[dynamic_take]; mask[dynamic_take]=255
            static_take,stats=safe_static_fill(base,mask,im,cm,cd)
            base[static_take]=im[static_take]; mask[static_take]=255
            reports.append({'label':lab,'physical_baseline_deg':baseline_by_camera[lab],
                            'dynamic_eligible':lab in dynamic_eligible,'dynamic_fill':int(dynamic_take.sum()),
                            **stats,'accepted_total':int(dynamic_take.sum()+static_take.sum())})
        return base,mask,{'fills':reports}

    still=[]
    for deg in [0,1,2,3,5]:
        actual=min(float(deg),render_max)
        frame,mask,r=render(actual)
        cv2.imwrite(str(args.out/f'true_orbit_{deg:02d}deg_native.png'),frame)
        cv2.imwrite(str(args.out/f'true_orbit_unresolved_{deg:02d}deg.png'),(mask==0).astype(np.uint8)*255)
        still.append({'degree':deg,'actual_degree':actual,'resolved_fraction':float((mask>0).mean()),'unresolved_pixels':int((mask==0).sum()),'fills':r['fills']})
    for i in range(args.frames):
        phase=i/max(args.frames-1,1); deg=render_max*math.sin(math.pi*phase)
        frame,_,_=render(deg); cv2.imwrite(str(args.out/f'motion_{i:03d}.png'),frame)

    max_radius=max(abs(r['radius_drift']) for r in pose_qa) if pose_qa else 0.0
    max_pivot=max(r['pivot_drift_px'] for r in pose_qa) if pose_qa else 0.0
    if max_radius>1e-6: raise RuntimeError(f'Orbit radius drift {max_radius}')
    if max_pivot>0.05: raise RuntimeError(f'Pivot drift {max_pivot}')

    serial={lab:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()} for lab,s in solves.items()}
    qa={'prototype':'portable_moge_true_orbit_v18','event':{'game_id':'0022500301','event_id':489},
        'source_resolution':[W,H],'render_resolution':[W,H],'resolution_policy':'native only',
        'reference':args.reference,'detected_ball_ref':b,'target_world_m':target.tolist(),
        'orbit_method':'constant-radius rigid camera rotation about detected 3D basketball; fixed reference intrinsics/FOV',
        'edge_splat_policy':'explicit bounded shifts only; np.roll/wraparound forbidden',
        'dynamic_fill_policy':'only physically near-baseline solved cameras; only inside dilated projected reference-action ROI',
        'static_fill_policy':'all solved cameras allowed only in narrow unresolved support with stricter local Lab agreement',
        'target_direction_camera':target_label,'real_baseline_angle_deg':baseline_angle,'render_max_degree':render_max,
        'camera_baselines_deg':baseline_by_camera,'dynamic_eligible_cameras':dynamic_eligible,
        'reference_radius':radius0,'reference_focal_px':focal0,'max_radius_drift':max_radius,'max_pivot_drift_px':max_pivot,
        'camera_solves':serial,'pose_qa':pose_qa,'stills':still,
        'generation_policy':'no generated appearance, diffusion, optical-flow morph, or zoom; every output colour is an NBA source pixel'}
    (args.out/'portable_moge_true_orbit_qa_v18.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({'target_direction_camera':target_label,'baseline_angle_deg':baseline_angle,'dynamic_eligible':dynamic_eligible,'max_radius_drift':max_radius,'max_pivot_drift_px':max_pivot,'stills':still},indent=2),flush=True)

if __name__=='__main__': main()
