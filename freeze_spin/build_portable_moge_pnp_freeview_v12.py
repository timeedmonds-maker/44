from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

W, H = 960, 540
PERSON_CLASS = 1
BALL_CLASS = 37


def safe(label: str) -> str:
    return label.replace(" ", "_")


def label_from_frame(path: Path) -> str:
    return path.stem.replace("_predunk", "").replace("_", " ")


def normalized_to_pixel_K(Kn: np.ndarray, w: int, h: int) -> np.ndarray:
    K = np.asarray(Kn, dtype=np.float64).copy()
    K[0, 0] *= w
    K[0, 2] *= w
    K[1, 1] *= h
    K[1, 2] *= h
    return K


def detect_dynamic_and_ball(model, image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        p = model([to_tensor(rgb)])[0]
    dynamic = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    balls = []
    scores = p["scores"].cpu().numpy(); labels = p["labels"].cpu().numpy()
    boxes = p["boxes"].cpu().numpy(); masks = p["masks"].cpu().numpy()[:, 0]
    for sc, lab, box, m in zip(scores, labels, boxes, masks):
        sc = float(sc); lab = int(lab)
        if lab == PERSON_CLASS and sc >= 0.35:
            dynamic[m >= 0.38] = 255
        elif lab == BALL_CLASS and sc >= 0.10:
            x1, y1, x2, y2 = [float(v) for v in box]
            balls.append({"score": sc, "cx": (x1+x2)/2, "cy": (y1+y2)/2, "box": [x1,y1,x2,y2]})
    dynamic = cv2.dilate(dynamic, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
    balls.sort(key=lambda r: r["score"], reverse=True)
    return dynamic > 0, balls


def moge_infer(model, image: np.ndarray, tokens: int):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2,0,1)
    with torch.inference_mode():
        out = model.infer(t, num_tokens=tokens, use_fp16=False, apply_mask=False)
    depth = out["depth"].cpu().numpy().astype(np.float32)
    points = out["points"].cpu().numpy().astype(np.float32)
    valid = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.isfinite(depth)
    Kn = out["intrinsics"].cpu().numpy().astype(np.float64)
    K = normalized_to_pixel_K(Kn, image.shape[1], image.shape[0])
    return depth, points, valid, K, Kn


def sift_matches(im1, im2, mask1, mask2):
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.02, edgeThreshold=14, sigma=1.3)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, mask1.astype(np.uint8)*255)
    k2, d2 = sift.detectAndCompute(g2, mask2.astype(np.uint8)*255)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return [], [], []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(d1, d2, k=2)
    good=[]
    for pair in knn:
        if len(pair) < 2: continue
        a,b=pair
        if a.distance < 0.70*b.distance:
            good.append(a)
    # Enforce approximate one-to-one target assignment.
    best={}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx]=m
    good=list(best.values())
    p1=np.array([k1[m.queryIdx].pt for m in good],np.float64) if good else np.empty((0,2))
    p2=np.array([k2[m.trainIdx].pt for m in good],np.float64) if good else np.empty((0,2))
    return good,p1,p2


def sample_point_map(points: np.ndarray, valid: np.ndarray, uv: np.ndarray):
    x=np.rint(uv[:,0]).astype(int); y=np.rint(uv[:,1]).astype(int)
    ok=(x>=0)&(x<W)&(y>=0)&(y<H)
    idx=np.where(ok)[0]
    keep=[]; xyz=[]
    for i in idx:
        if valid[y[i],x[i]] and np.all(np.isfinite(points[y[i],x[i]])) and points[y[i],x[i],2] > 0.25:
            keep.append(i); xyz.append(points[y[i],x[i]])
    return np.asarray(keep,int), np.asarray(xyz,np.float64)


def solve_target_from_reference(ref, tgt):
    static_ref = (~ref["dynamic"]) & ref["valid"]
    static_tgt = (~tgt["dynamic"]) & tgt["valid"]
    _, uv_ref, uv_tgt = sift_matches(ref["image"], tgt["image"], static_ref, static_tgt)
    if len(uv_ref) < 35:
        return {"passed":False,"reason":f"only {len(uv_ref)} static SIFT matches"}
    keep, X = sample_point_map(ref["points"], ref["valid"], uv_ref)
    uv2 = uv_tgt[keep]
    if len(X) < 30:
        return {"passed":False,"reason":f"only {len(X)} valid reference-depth matches"}
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X, uv2, tgt["K"], None, flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=4.0, confidence=0.999, iterationsCount=2500)
    if not ok or inliers is None or len(inliers) < 25:
        return {"passed":False,"reason":"PnP RANSAC insufficient inliers","matches":int(len(X)),"inliers":0 if inliers is None else int(len(inliers))}
    ids=inliers.reshape(-1)
    try:
        rvec,tvec=cv2.solvePnPRefineLM(X[ids],uv2[ids],tgt["K"],None,rvec,tvec)
    except Exception:
        pass
    R,_=cv2.Rodrigues(rvec); t=tvec.reshape(3)
    proj,_=cv2.projectPoints(X[ids],rvec,tvec,tgt["K"],None); proj=proj.reshape(-1,2)
    err=np.linalg.norm(proj-uv2[ids],axis=1)
    # Estimate MoGe metric-scale correction in target camera from the same inlier correspondences.
    x2=np.rint(uv2[ids,0]).astype(int); y2=np.rint(uv2[ids,1]).astype(int)
    pred_z=(R@X[ids].T).T[:,2]+t[2]
    d2=tgt["depth"][y2,x2]
    good=np.isfinite(d2)&(d2>0.25)&np.isfinite(pred_z)&(pred_z>0.25)
    ratios=pred_z[good]/d2[good]
    scale=float(np.median(ratios)) if len(ratios)>=10 else float("nan")
    mad=float(np.median(np.abs(ratios-scale))) if len(ratios)>=10 else float("inf")
    C=-R.T@t
    passed=(len(ids)>=25 and float(np.median(err))<=2.5 and float(np.percentile(err,95))<=6.0 and np.isfinite(scale) and 0.35<scale<2.8 and mad/max(abs(scale),1e-6)<=0.25)
    return {
        "passed":bool(passed),"matches":int(len(X)),"inliers":int(len(ids)),"inlier_fraction":float(len(ids)/len(X)),
        "median_reprojection_px":float(np.median(err)),"p95_reprojection_px":float(np.percentile(err,95)),
        "depth_scale":scale,"depth_scale_mad":mad,"R":R,"t":t,"C":C,
    }


def scaled_world_cloud(view, solve):
    R=solve["R"]; t=solve["t"]; scale=float(solve["depth_scale"])
    valid=view["valid"] & np.all(np.isfinite(view["points"]),axis=2) & (view["depth"]>0.25)
    ys,xs=np.where(valid)
    Xc=view["points"][ys,xs].astype(np.float64)*scale
    Xw=(R.T@(Xc-t).T).T
    return Xw.astype(np.float32), view["image"][ys,xs].copy(), view["dynamic"][ys,xs].copy()


def reference_cloud(ref):
    valid=ref["valid"] & np.all(np.isfinite(ref["points"]),axis=2) & (ref["depth"]>0.25)
    ys,xs=np.where(valid)
    return ref["points"][ys,xs].astype(np.float32), ref["image"][ys,xs].copy(), ref["dynamic"][ys,xs].copy()


def project(P, X):
    h=np.column_stack([X,np.ones(len(X))]); q=(P@h.T).T
    ok=q[:,2]>1e-6; uv=np.zeros((len(X),2),np.float64); uv[ok]=q[ok,:2]/q[ok,2:3]
    return uv,ok


def raster_cloud(cloud, K, R, C, radius=1):
    X,col,dyn=cloud; t=-R@C; P=K@np.hstack([R,t.reshape(3,1)])
    uv,ok=project(P,X.astype(np.float64)); z=(R@(X.astype(np.float64)-C).T).T[:,2]
    u=np.rint(uv[:,0]).astype(int); v=np.rint(uv[:,1]).astype(int)
    ok &= (z>0.2)&(u>=0)&(u<W)&(v>=0)&(v<H)
    ids=np.where(ok)[0]
    image=np.zeros((H,W,3),np.uint8); mask=np.zeros((H,W),np.uint8); dynamic=np.zeros((H,W),np.uint8); zbuf=np.full(H*W,np.inf,np.float32)
    if len(ids):
        pix=v[ids]*W+u[ids]; np.minimum.at(zbuf,pix,z[ids].astype(np.float32)); win=ids[z[ids]<=zbuf[pix]+1e-4]
        image[v[win],u[win]]=col[win]; mask[v[win],u[win]]=255; dynamic[v[win],u[win]]=np.where(dyn[win],255,0).astype(np.uint8)
    if radius:
        kernel=np.ones((3,3),np.uint8)
        # Nearest-neighbour splat only into immediately adjacent holes; preserve original winners.
        for _ in range(radius):
            holes=mask==0
            dil=cv2.dilate(mask,kernel,iterations=1)
            # Use eight shifted copies, taking first available real source sample.
            base_img=image.copy(); base_mask=mask.copy(); base_dyn=dynamic.copy()
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                simg=np.roll(np.roll(base_img,dy,0),dx,1); sm=np.roll(np.roll(base_mask,dy,0),dx,1); sd=np.roll(np.roll(base_dyn,dy,0),dx,1)
                take=holes&(mask==0)&(sm>0)&(dil>0)
                image[take]=simg[take]; mask[take]=255; dynamic[take]=sd[take]
    return image,mask,dynamic


def angle_between(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); return math.degrees(math.acos(float(np.clip(np.dot(a,b),-1,1))))


def slerp_vector(a,b,alpha):
    ra=np.linalg.norm(a); rb=np.linalg.norm(b); ua=a/ra; ub=b/rb
    omega=math.acos(float(np.clip(np.dot(ua,ub),-1,1)))
    if omega<1e-7: u=(1-alpha)*ua+alpha*ub
    else: u=math.sin((1-alpha)*omega)/math.sin(omega)*ua+math.sin(alpha*omega)/math.sin(omega)*ub
    r=(1-alpha)*ra+alpha*rb
    return u/np.linalg.norm(u)*r


def safe_fill_gate(base_img, base_mask, cand_img, cand_mask, cand_dynamic):
    unresolved=base_mask==0
    candidate=cand_mask>0
    dyn=(cand_dynamic>0)&candidate&unresolved
    # Static camera fills are deliberately conservative: only narrow disocclusions near real reference pixels,
    # and only where local colour is compatible. This prevents the v11 green-crowd contamination.
    inv=(base_mask==0).astype(np.uint8)
    dist=cv2.distanceTransform(inv,cv2.DIST_L2,3)
    w=(base_mask>0).astype(np.float32)
    wf=cv2.GaussianBlur(w,(0,0),4.0)
    mean=np.zeros_like(base_img,np.float32)
    for c in range(3):
        mean[:,:,c]=cv2.GaussianBlur(base_img[:,:,c].astype(np.float32)*w,(0,0),4.0)/np.maximum(wf,1e-5)
    lab_c=cv2.cvtColor(cand_img,cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_m=cv2.cvtColor(np.clip(mean,0,255).astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
    delta=np.linalg.norm(lab_c-lab_m,axis=2)
    static=candidate&unresolved&(~dyn)&(dist<=14.0)&(wf>0.04)&(delta<=42.0)
    return dyn|static, {"dynamic_fill":int(dyn.sum()),"static_safe_fill":int(static.sum()),"static_rejected":int((candidate&unresolved&(~dyn)&(~static)).sum())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--locked-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--reference",default="In Arena")
    ap.add_argument("--tokens",type=int,default=1400)
    ap.add_argument("--max-degree",type=float,default=5.0)
    ap.add_argument("--frames",type=int,default=31)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    paths=sorted(args.locked_dir.glob("*_predunk.png")); images={label_from_frame(p):cv2.imread(str(p)) for p in paths}
    images={k:v for k,v in images.items() if v is not None}
    if args.reference not in images: raise RuntimeError(f"Reference {args.reference} unavailable: {list(images)}")

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    detector=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,progress=True).eval()
    moge=MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    views={}
    for label,image in images.items():
        dyn,balls=detect_dynamic_and_ball(detector,image)
        depth,points,valid,K,Kn=moge_infer(moge,image,args.tokens)
        views[label]={"label":label,"image":image,"dynamic":dyn,"balls":balls,"depth":depth,"points":points,"valid":valid,"K":K,"Kn":Kn}
        cv2.imwrite(str(args.out/f"{safe(label)}_dynamic_mask.png"),dyn.astype(np.uint8)*255)
        dv=np.zeros((H,W),np.uint8); ok=valid&np.isfinite(depth)&(depth>0)
        if ok.any():
            lo,hi=np.percentile(depth[ok],[2,98]); dv[ok]=np.clip((depth[ok]-lo)*255/max(float(hi-lo),1e-6),0,255).astype(np.uint8)
        cv2.imwrite(str(args.out/f"{safe(label)}_moge_depth.png"),dv)

    ref=views[args.reference]
    solves={}
    for label,v in views.items():
        if label==args.reference: continue
        s=solve_target_from_reference(ref,v); solves[label]=s
        print(label,{k:v for k,v in s.items() if k not in ("R","t","C")},flush=True)
    passed=[(lab,s) for lab,s in solves.items() if s.get("passed")]
    if not passed: raise RuntimeError("No secondary camera passed portable MoGe+SIFT+PnP calibration")

    # Reference ball target from detector + MoGe. Prefer highest confidence sports-ball detection near image centre.
    if not ref["balls"]: raise RuntimeError("No basketball detected in reference pre-dunk frame")
    b=ref["balls"][0]; bx=int(np.clip(round(b["cx"]),0,W-1)); by=int(np.clip(round(b["cy"]),0,H-1))
    target=ref["points"][by,bx].astype(np.float64)
    if not np.all(np.isfinite(target)) or target[2]<=0.25:
        ys=slice(max(0,by-3),min(H,by+4)); xs=slice(max(0,bx-3),min(W,bx+4)); pts=ref["points"][ys,xs].reshape(-1,3); pts=pts[np.all(np.isfinite(pts),axis=1)&(pts[:,2]>0.25)]
        if not len(pts): raise RuntimeError("No valid reference MoGe depth at basketball")
        target=np.median(pts,axis=0)

    # Pick the closest genuinely distinct solved camera to define a real camera-supported direction.
    options=[]
    for lab,s in passed:
        C=s["C"].astype(np.float64); ang=angle_between(-target,C-target)
        if ang>=4.0: options.append((ang,lab,s))
    if not options: options=[(angle_between(-target,s["C"].astype(np.float64)-target),lab,s) for lab,s in passed]
    options.sort(key=lambda x:x[0]); baseline_angle,target_label,target_solve=options[0]
    alpha_max=min(1.0,args.max_degree/max(float(baseline_angle),1e-6))

    clouds={args.reference:reference_cloud(ref)}
    for lab,s in passed: clouds[lab]=scaled_world_cloud(views[lab],s)
    # Rank secondary fill views by angular proximity to target path camera; chosen direction first.
    fill_order=[target_label]+[lab for lab,_ in passed if lab!=target_label]

    rots=Rotation.from_matrix(np.stack([np.eye(3),target_solve["R"]]))
    slerp=Slerp([0.0,1.0],rots)
    C_target=target_solve["C"].astype(np.float64)

    def render(alpha):
        if alpha<=1e-9: return ref["image"].copy(),np.full((H,W),255,np.uint8),{"secondary":[]}
        C=slerp_vector(-target,C_target-target,alpha)+target
        R=slerp([alpha]).as_matrix()[0]
        base,mask,dyn=raster_cloud(clouds[args.reference],ref["K"],R,C,radius=1)
        report=[]
        for lab in fill_order:
            im,cm,cd=raster_cloud(clouds[lab],ref["K"],R,C,radius=1)
            take,stats=safe_fill_gate(base,mask,im,cm,cd)
            base[take]=im[take]; mask[take]=255
            stats["label"]=lab; stats["accepted_total"]=int(take.sum()); report.append(stats)
        return base,mask,{"secondary":report}

    still=[]
    for deg in [0,1,2,3,5]:
        a=min(alpha_max,deg/max(float(baseline_angle),1e-6)) if deg>0 else 0.0
        frame,mask,rr=render(a); cv2.imwrite(str(args.out/f"portable_{deg:02d}deg_native.png"),frame); cv2.imwrite(str(args.out/f"portable_unresolved_{deg:02d}deg.png"),(mask==0).astype(np.uint8)*255)
        still.append({"degree":deg,"alpha":a,"resolved_fraction":float((mask>0).mean()),"unresolved_pixels":int((mask==0).sum()),"fills":rr["secondary"]})
    for i in range(args.frames):
        phase=i/max(args.frames-1,1); deg=args.max_degree*math.sin(math.pi*phase); a=min(alpha_max,deg/max(float(baseline_angle),1e-6))
        frame,_,_=render(a); cv2.imwrite(str(args.out/f"motion_{i:03d}.png"),frame)

    serial={}
    for lab,s in solves.items():
        serial[lab]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()}
    qa={
        "prototype":"portable_moge_sift_pnp_freeview_v12",
        "event":{"game_id":"0022500301","event_id":489,"description":"Steven Adams dunk vs Utah immediately after Adams block"},
        "source_resolution":[W,H],"render_resolution":[W,H],"resolution_policy":"native only",
        "reference":args.reference,"detected_ball_ref":b,"target_world_m":target.tolist(),
        "camera_method":"MoGe-2 reference 3D + person-masked static SIFT correspondences + solvePnPRansac; no manually annotated Jazz camera geometry",
        "camera_solves":serial,"passed_secondary_count":len(passed),"target_direction_camera":target_label,"real_baseline_angle_deg":baseline_angle,"render_max_degree":args.max_degree,
        "fill_policy":"reference real pixels first; secondary dynamic pixels may fill disocclusions; static background fill requires narrow-hole proximity plus local Lab agreement to prevent v11 green-crowd contamination",
        "generation_policy":"no generated appearance, no diffusion, no optical-flow morph; every output colour is an NBA source pixel",
        "stills":still,
        "success_gate":"at least one secondary camera passes strict PnP; 3-5 degree motion remains coherent; secondary background fill must not introduce broad source-colour contamination."
    }
    (args.out/"portable_moge_pnp_qa_v12.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(json.dumps({"passed_secondary_count":len(passed),"target_direction_camera":target_label,"baseline_angle_deg":baseline_angle,"stills":still},indent=2),flush=True)

if __name__=="__main__": main()
