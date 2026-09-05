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
from build_portable_moge_true_orbit_v18 import raster_cloud_bounded, safe_static_fill


def nearest_verified_ball(apex: dict) -> tuple[float,float,dict]:
    local=float(apex["apex_right_slash_local_time"])
    track=apex.get("diagnostics",{}).get("track",[])
    if not track: raise RuntimeError("No verified Right Slash track in apex JSON")
    q=min(track,key=lambda r:abs(float(r["time"])-local))
    if abs(float(q["time"])-local)>0.09:
        raise RuntimeError(f"Nearest verified ball observation too far from apex: {q['time']} vs {local}")
    return float(q["cx"]),float(q["cy"]),q


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--locked-dir",type=Path,required=True)
    ap.add_argument("--apex-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--reference",default="Right Slash")
    ap.add_argument("--tokens",type=int,default=1600)
    ap.add_argument("--max-degree",type=float,default=5.0)
    ap.add_argument("--frames",type=int,default=31)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    apex=json.load(open(args.apex_json))
    if not apex.get("passed"): raise RuntimeError("Apex JSON did not pass")
    bx_f,by_f,ball_obs=nearest_verified_ball(apex)

    images={}
    for p in sorted(args.locked_dir.glob("*_apex.png")):
        label=p.stem.replace("_apex","").replace("_"," ")
        im=cv2.imread(str(p))
        if im is not None: images[label]=im
    if args.reference not in images: raise RuntimeError(f"Reference {args.reference} unavailable: {list(images)}")
    if len(images)<6: raise RuntimeError(f"Only {len(images)} locked views")

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    detector=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,progress=True).eval()
    moge=MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()

    views={}
    for label,image in images.items():
        if image.shape[:2]!=(H,W): raise RuntimeError(f"{label} not native 960x540")
        dyn,balls=detect_dynamic_and_ball(detector,image)
        depth,points,valid,K,Kn=moge_infer(moge,image,args.tokens)
        views[label]={"label":label,"image":image,"dynamic":dyn,"balls":balls,"depth":depth,"points":points,"valid":valid,"K":K,"Kn":Kn}
        cv2.imwrite(str(args.out/f"{label.replace(' ','_')}_dynamic_mask.png"),dyn.astype(np.uint8)*255)

    ref=views[args.reference]
    bx=int(np.clip(round(bx_f),0,W-1)); by=int(np.clip(round(by_f),0,H-1))
    target=ref["points"][by,bx].astype(np.float64)
    if not np.all(np.isfinite(target)) or target[2]<=0.25:
        pts=ref["points"][max(0,by-4):min(H,by+5),max(0,bx-4):min(W,bx+5)].reshape(-1,3)
        pts=pts[np.all(np.isfinite(pts),axis=1)&(pts[:,2]>0.25)]
        if not len(pts): raise RuntimeError("No valid Right Slash MoGe depth around verified basketball")
        target=np.median(pts,axis=0)

    solves={}
    for label,v in views.items():
        if label==args.reference: continue
        s=solve_target_from_reference(ref,v); solves[label]=s
        print(label,{k:v for k,v in s.items() if k not in ("R","t","C")},flush=True)
    passed=[(lab,s) for lab,s in solves.items() if s.get("passed")]
    if not passed: raise RuntimeError("No secondary camera passed static PnP from Right Slash reference")

    options=[]
    for lab,s in passed:
        C=s["C"].astype(np.float64); ang=angle_between(-target,C-target)
        if ang>=3.0: options.append((ang,lab,s))
    if not options: raise RuntimeError("No solved camera supports >=3 degree physical baseline from Right Slash")
    options.sort(key=lambda x:x[0]); baseline,target_label,target_solve=options[0]
    render_max=min(float(args.max_degree),float(baseline))
    C_target=target_solve["C"].astype(np.float64)

    clouds={args.reference:reference_cloud(ref)}
    for lab,s in passed: clouds[lab]=scaled_world_cloud(views[lab],s)
    # Static-only secondary support. Their people/ball are never allowed into the frozen action.
    fill_order=[target_label]+[lab for lab,_ in passed if lab!=target_label]

    radius0=float(np.linalg.norm(target)); pivot0=project_point(ref["K"],np.eye(3),np.zeros(3),target)
    pose_rows=[]
    def pose(deg:float):
        R,C,_=true_orbit_pose(target,C_target,deg)
        rad=float(np.linalg.norm(C-target)); piv=project_point(ref["K"],R,C,target)
        pose_rows.append({"degree":float(deg),"radius":rad,"radius_drift":rad-radius0,
                          "pivot_pixel":piv.tolist(),"pivot_drift_px":float(np.linalg.norm(piv-pivot0))})
        return R,C

    def render(deg:float):
        if deg<=1e-9:
            return ref["image"].copy(),np.full((H,W),255,np.uint8),[]
        R,C=pose(deg)
        base,mask,_=raster_cloud_bounded(clouds[args.reference],ref["K"],R,C,radius=1)
        reports=[]
        for lab in fill_order:
            im,cm,cd=raster_cloud_bounded(clouds[lab],ref["K"],R,C,radius=1)
            take,stats=safe_static_fill(base,mask,im,cm,cd)
            base[take]=im[take]; mask[take]=255
            reports.append({"label":lab,"static_safe_fill":int(take.sum()),
                            "static_rejected":int(stats["static_rejected"]),"dynamic_fill":0,
                            "policy":"secondary dynamic pixels forbidden"})
        return base,mask,reports

    still=[]
    for deg in [0,1,2,3,5]:
        actual=min(float(deg),render_max)
        im,mask,reports=render(actual)
        cv2.imwrite(str(args.out/f"rightslash_true_orbit_{deg:02d}deg_native.png"),im)
        cv2.imwrite(str(args.out/f"rightslash_unresolved_{deg:02d}deg.png"),(mask==0).astype(np.uint8)*255)
        still.append({"degree":deg,"actual_degree":actual,"resolved_fraction":float((mask>0).mean()),
                      "unresolved_pixels":int((mask==0).sum()),"fills":reports})
    for i in range(args.frames):
        phase=i/max(args.frames-1,1); deg=render_max*math.sin(math.pi*phase)
        im,_,_=render(deg); cv2.imwrite(str(args.out/f"motion_{i:03d}.png"),im)

    max_radius=max(abs(float(r["radius_drift"])) for r in pose_rows) if pose_rows else 0.0
    max_pivot=max(float(r["pivot_drift_px"]) for r in pose_rows) if pose_rows else 0.0
    if max_radius>1e-6: raise RuntimeError(f"Orbit radius drift {max_radius}")
    if max_pivot>0.05: raise RuntimeError(f"Pivot drift {max_pivot}")

    serial={lab:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()} for lab,s in solves.items()}
    qa={"prototype":"rightslash_verified_apex_static_secondary_true_orbit_v21",
        "event":{"game_id":"0022500301","event_id":489,"description":"Steven Adams dunk vs Utah"},
        "source_resolution":[W,H],"render_resolution":[W,H],"resolution_policy":"native only",
        "reference":args.reference,"freeze_policy":"reference frame is the visually height-sensitive Right Slash ball-apex frame; no audio-mapped camera may replace the frozen player state",
        "verified_ball_observation":ball_obs,"verified_ball_pixel":[bx_f,by_f],"target_world_m":target.tolist(),
        "camera_method":"Right Slash MoGe reference 3D + person-masked static SIFT + solvePnPRansac",
        "orbit_method":"constant-radius rigid camera-centre rotation around verified 3D basketball, fixed Right Slash intrinsics/FOV",
        "secondary_fill_policy":"STATIC ONLY. Secondary people and basketball pixels are forbidden, eliminating cross-camera pose contamination from A/V latency.",
        "edge_splat_policy":"explicit bounded shifts; no np.roll edge wrap",
        "target_direction_camera":target_label,"real_baseline_angle_deg":float(baseline),"render_max_degree":float(render_max),
        "reference_focal_px":[float(ref["K"][0,0]),float(ref["K"][1,1])],"reference_radius":radius0,
        "max_radius_drift":max_radius,"max_pivot_drift_px":max_pivot,"camera_solves":serial,"pose_qa":pose_rows,"stills":still,
        "generation_policy":"no generated appearance, diffusion, optical-flow morph, focal zoom, or radial dolly; every output colour is a real NBA source pixel"}
    (args.out/"rightslash_true_orbit_qa_v21.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(json.dumps({"reference":args.reference,"target_direction_camera":target_label,"baseline_angle_deg":baseline,
                      "render_max_degree":render_max,"max_radius_drift":max_radius,"max_pivot_drift_px":max_pivot,"stills":still},indent=2),flush=True)

if __name__=="__main__": main()
