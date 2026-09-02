from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2


def detect_static_mask(detector, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        pred = detector([x])[0]
    dynamic = np.zeros((h, w), np.uint8)
    for score, label, mask in zip(pred["scores"], pred["labels"], pred["masks"]):
        if float(score) < 0.55:
            continue
        if int(label) != 1:
            continue
        m = (mask[0].cpu().numpy() > 0.42).astype(np.uint8)
        dynamic = np.maximum(dynamic, m)
    dynamic = cv2.dilate(dynamic, np.ones((17, 17), np.uint8), iterations=1)
    return (dynamic == 0).astype(np.uint8) * 255


def sift_matches(im1: np.ndarray, im2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.018, edgeThreshold=12)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), mask1)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), mask2)
    if d1 is None or d2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    cand=[]
    for pair in knn:
        if len(pair) < 2:
            continue
        a,b=pair
        if a.distance < 0.70*b.distance:
            cand.append(a)
    # one-to-one target keypoint assignment, best distance wins
    best={}
    for m in cand:
        old=best.get(m.trainIdx)
        if old is None or m.distance < old.distance:
            best[m.trainIdx]=m
    rows=sorted(best.values(), key=lambda m:m.distance)
    uv1=np.array([k1[m.queryIdx].pt for m in rows], np.float64)
    uv2=np.array([k2[m.trainIdx].pt for m in rows], np.float64)
    return uv1,uv2


def sampson_px(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1=np.column_stack([p1,np.ones(len(p1))])
    x2=np.column_stack([p2,np.ones(len(p2))])
    Fx1=(F@x1.T).T
    Ftx2=(F.T@x2.T).T
    num=np.sum(x2*Fx1,axis=1)**2
    den=Fx1[:,0]**2+Fx1[:,1]**2+Ftx2[:,0]**2+Ftx2[:,1]**2
    return np.sqrt(num/np.maximum(den,1e-12))


def cell_count(pts: np.ndarray, w: int, h: int, n: int = 3) -> int:
    if len(pts)==0:
        return 0
    ix=np.clip((pts[:,0]/w*n).astype(int),0,n-1)
    iy=np.clip((pts[:,1]/h*n).astype(int),0,n-1)
    return len(set(zip(ix.tolist(),iy.tolist())))


def symmetric_transfer_error(H: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    q2=cv2.perspectiveTransform(p1.reshape(-1,1,2).astype(np.float64),H).reshape(-1,2)
    Hi=np.linalg.inv(H)
    q1=cv2.perspectiveTransform(p2.reshape(-1,1,2).astype(np.float64),Hi).reshape(-1,2)
    return 0.5*(np.linalg.norm(q2-p2,axis=1)+np.linalg.norm(q1-p1,axis=1))


def fit_f(p1: np.ndarray,p2: np.ndarray):
    F,m=cv2.findFundamentalMat(p1,p2,cv2.FM_RANSAC,1.5,0.999,10000)
    if F is None or m is None or F.shape!=(3,3):
        return None,None
    return F,m.reshape(-1).astype(bool)


def diagnose(im1,im2,mask1,mask2,seed=20260902):
    h,w=im1.shape[:2]
    p1,p2=sift_matches(im1,im2,mask1,mask2)
    out={"static_sift_matches":int(len(p1))}
    if len(p1)<35:
        out.update({"passed":False,"reason":"insufficient static SIFT matches"})
        return out
    F,fin=fit_f(p1,p2)
    if F is None or int(fin.sum())<25:
        out.update({"passed":False,"reason":"fundamental matrix insufficient consensus","fundamental_inliers":0 if fin is None else int(fin.sum())})
        return out
    e=sampson_px(F,p1[fin],p2[fin])
    H,hm=cv2.findHomography(p1,p2,cv2.RANSAC,2.0,maxIters=10000,confidence=0.999)
    hin=np.zeros(len(p1),bool) if hm is None else hm.reshape(-1).astype(bool)
    h_err=np.array([],dtype=float)
    if H is not None and int(hin.sum())>=4:
        h_err=symmetric_transfer_error(H,p1[hin],p2[hin])
    nonhom=int(np.sum(fin & ~hin))

    rng=np.random.default_rng(seed)
    fi=np.where(fin)[0]
    boot=[]
    B=60
    sample_n=max(20,int(round(0.75*len(fi))))
    for _ in range(B):
        idx=rng.choice(fi,size=sample_n,replace=False)
        Fb,mb=fit_f(p1[idx],p2[idx])
        if Fb is None:
            continue
        # Evaluate each resampled geometry on the original full-F consensus,
        # not on its own training residuals.
        eb=sampson_px(Fb,p1[fi],p2[fi])
        boot.append({
            "median_px":float(np.median(eb)),
            "p95_px":float(np.percentile(eb,95)),
            "fraction_under_2px":float(np.mean(eb<=2.0)),
        })
    medians=np.array([r["median_px"] for r in boot],float)
    p95s=np.array([r["p95_px"] for r in boot],float)
    under=np.array([r["fraction_under_2px"] for r in boot],float)
    ffrac=float(fin.mean())
    hfrac=float(hin.mean())
    h_to_f=float(hin.sum()/max(fin.sum(),1))
    spatial_ref=cell_count(p1[fin],w,h)
    spatial_tgt=cell_count(p2[fin],w,h)

    core_pass=(
        int(fin.sum())>=25
        and ffrac>=0.42
        and float(np.median(e))<=0.8
        and float(np.percentile(e,95))<=2.0
        and spatial_ref>=4
        and spatial_tgt>=4
        and len(boot)>=45
        and float(np.median(medians))<=1.0
        and float(np.percentile(medians,95))<=2.0
        and float(np.median(under))>=0.80
    )
    # Non-planar evidence is a separate flag. A camera can have a robust F
    # yet still be too close to a homography-dominated configuration for
    # metric translation/depth recovery.
    nonplanar_pass=bool(nonhom>=10 and h_to_f<=0.90)
    passed=bool(core_pass and nonplanar_pass)
    out.update({
        "passed":passed,
        "correspondence_geometry_passed":bool(core_pass),
        "nonplanar_evidence_passed":nonplanar_pass,
        "reason":"stable non-planar epipolar geometry" if passed else ("stable epipolar geometry but homography-dominated" if core_pass else "epipolar geometry not stable enough"),
        "fundamental_inliers":int(fin.sum()),
        "fundamental_inlier_fraction":ffrac,
        "median_sampson_px":float(np.median(e)),
        "p95_sampson_px":float(np.percentile(e,95)),
        "ref_3x3_cells":int(spatial_ref),
        "target_3x3_cells":int(spatial_tgt),
        "homography_inliers":int(hin.sum()),
        "homography_inlier_fraction":hfrac,
        "homography_to_fundamental_support_ratio":h_to_f,
        "nonhomography_fundamental_inliers":nonhom,
        "homography_median_symmetric_transfer_px":None if len(h_err)==0 else float(np.median(h_err)),
        "bootstrap_valid":int(len(boot)),
        "bootstrap_median_of_median_px":None if len(boot)==0 else float(np.median(medians)),
        "bootstrap_p95_of_median_px":None if len(boot)==0 else float(np.percentile(medians,95)),
        "bootstrap_median_of_p95_px":None if len(boot)==0 else float(np.median(p95s)),
        "bootstrap_median_fraction_under_2px":None if len(boot)==0 else float(np.median(under)),
        "F":F.tolist(),
        "policy":"No monocular depth, inferred focal length, PnP pose, player/ball geometry, or generated appearance contributes to this gate. It tests only whether static real-pixel correspondences support a stable non-planar two-view projective geometry.",
    })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--locked-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',default='In Arena')
    ap.add_argument('--targets',nargs='+',default=['Broadcast','Other Broadcast','Right Above Rim'])
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    labels=[args.reference]+args.targets
    images={}
    for label in labels:
        p=args.locked_dir/f"{label.replace(' ','_')}_apex.png"
        im=cv2.imread(str(p))
        if im is None: raise RuntimeError(f'Missing {p}')
        if im.shape[:2]!=(540,960): raise RuntimeError(f'Expected native 960x540: {p} {im.shape}')
        images[label]=im
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    detector=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,progress=True).eval()
    masks={label:detect_static_mask(detector,im) for label,im in images.items()}
    results={}
    for i,label in enumerate(args.targets):
        results[label]=diagnose(images[args.reference],images[label],masks[args.reference],masks[label],seed=20260902+i)
        print(label,json.dumps(results[label],indent=2),flush=True)
    payload={
        "prototype":"depth_free_fundamental_stability_v18",
        "event":{"game_id":"0022500301","event_id":489},
        "state":"accepted_ball_apex",
        "reference":args.reference,
        "native_resolution":[960,540],
        "results":results,
        "decision_rule":"A pass means the real pixels support stable non-planar projective correspondence geometry. It does not yet provide metric intrinsics, metric camera translation, or dense depth.",
    }
    (args.out/'fundamental_stability_v18.json').write_text(json.dumps(payload,indent=2))

if __name__=='__main__':
    main()
