from __future__ import annotations

"""v26: temporal multi-view court atlas + distant angular coverage.

v25 established two things: nearby official frames can be registered robustly for
some physical cameras, but image-space clean plates do not solve the growing
virtual-view holes; and identity-fallback on rejected temporal registrations is
unsafe.  v26 therefore never uses a rejected temporal registration.

The corrected v23 focal player and true three-view ball are unchanged.  Static
appearance is improved in two geometry-appropriate domains:

* FLOOR: each accepted temporal frame is related to the exact-state camera by a
  RANSAC homography.  The exact solved camera projects each metric z=0 court
  texel into exact-state image coordinates; the inverse temporal homography maps
  that coordinate into the real temporal frame.  Per-camera temporal samples are
  robustly reduced by choosing an ACTUAL source-frame RGB medoid.  Newly exposed
  atlas texels require support from at least two temporal frames.

* DISTANT ARENA: target virtual rays are projected into the solved exact camera
  and then through the inverse accepted temporal homographies.  This allows real
  source pixels revealed by pan/zoom in nearby frames to extend angular coverage.
  Metric floor/backboard regions are excluded from this infinity-depth layer.

Every output RGB pixel still descends from official native source footage.  The
0-degree anchor remains the exact source frame.  No generated texture, inpainting,
cross-fade, UHD or upscale path exists.  Output is native 960x540 only.
"""

import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v15 as v15
from freeze_spin import build_three_camera_mesh_v21 as v21
from freeze_spin import build_three_camera_mesh_v24 as v24
from freeze_spin import build_three_camera_mesh_v25 as v25


OFFSETS = (-120, -90, -60, -45, -30, -15, 15, 30, 45, 60, 90, 120)
_CLIPS_DIR: Path | None = None
_SYNC_QA_PATH: Path | None = None
_CONTEXT = None
_TEMPORAL = {}
_REG_QA = {}
_ATLAS_QA = {}
_BG_QA = []


def _pop_arg(name: str) -> str:
    i = sys.argv.index(name)
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


def _label_for_image(image):
    for label, src in v12.IMAGES.items():
        if image is src:
            return label
    return None


def _decode(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if idx < 0 or (n > 0 and idx >= n):
        cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, im = cap.read()
    actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
    cap.release()
    if not ok or actual != int(idx) or im is None or im.shape[:2] != (v12.H, v12.W):
        return None
    return im


def _poly_area(p):
    p = np.asarray(p, np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _register(candidate, anchor, anchor_dynamic):
    """Estimate candidate->anchor projective registration; reject instead of identity fallback."""
    g0 = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
    dyn = cv2.dilate(anchor_dynamic.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1) > 0
    # Avoid scorebug/footer where possible; retain enough static arena/court texture.
    mask = (~dyn).astype(np.uint8) * 255
    mask[int(0.88 * v12.H):, :] = 0
    sift = cv2.SIFT_create(nfeatures=6500, contrastThreshold=0.016, edgeThreshold=12, sigma=1.4)
    k0, d0 = sift.detectAndCompute(g0, mask)
    k1, d1 = sift.detectAndCompute(g1, mask)
    qa = {"candidate_keypoints": int(len(k0)), "anchor_keypoints": int(len(k1)), "accepted": False}
    if d0 is None or d1 is None or len(k0) < 20 or len(k1) < 20:
        qa["reason"] = "insufficient_features"; return None, qa
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for row in bf.knnMatch(d0, d1, k=2):
        if len(row) >= 2 and row[0].distance < 0.74 * row[1].distance:
            good.append(row[0])
    qa["ratio_matches"] = int(len(good))
    if len(good) < 24:
        qa["reason"] = "insufficient_ratio_matches"; return None, qa
    src = np.asarray([k0[m.queryIdx].pt for m in good], np.float32).reshape(-1, 1, 2)
    dst = np.asarray([k1[m.trainIdx].pt for m in good], np.float32).reshape(-1, 1, 2)
    H, inlier = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=9000, confidence=0.998)
    if H is None or not np.isfinite(H).all() or abs(float(H[2,2])) < 1e-9:
        qa["reason"] = "homography_failed"; return None, qa
    H = H / H[2,2]
    inlier = inlier.reshape(-1).astype(bool) if inlier is not None else np.zeros(len(good), bool)
    nin = int(inlier.sum()); frac = float(inlier.mean()) if len(inlier) else 0.0
    qa["ransac_inliers"] = nin; qa["ransac_inlier_fraction"] = frac
    if nin:
        pp = cv2.perspectiveTransform(src[inlier].reshape(-1,1,2), H).reshape(-1,2)
        rr = np.linalg.norm(pp - dst[inlier].reshape(-1,2), axis=1)
        qa["inlier_residual_median_px"] = float(np.median(rr))
        qa["inlier_residual_p90_px"] = float(np.percentile(rr,90))
    else:
        qa["inlier_residual_median_px"] = None; qa["inlier_residual_p90_px"] = None

    corners = np.asarray([[0,0],[v12.W-1,0],[v12.W-1,v12.H-1],[0,v12.H-1]],np.float32).reshape(-1,1,2)
    mapped = cv2.perspectiveTransform(corners,H).reshape(-1,2)
    area_ratio = _poly_area(mapped) / float((v12.W-1)*(v12.H-1))
    qa["mapped_corner_area_ratio"] = float(area_ratio)
    qa["mapped_corners"] = mapped.tolist()
    try:
        Hinv = np.linalg.inv(H)
        cond = float(np.linalg.cond(H / max(1.0, np.linalg.norm(H))))
    except np.linalg.LinAlgError:
        qa["reason"] = "singular_homography"; return None, qa
    qa["condition_number"] = cond
    med = qa["inlier_residual_median_px"] if qa["inlier_residual_median_px"] is not None else 999.0
    p90 = qa["inlier_residual_p90_px"] if qa["inlier_residual_p90_px"] is not None else 999.0
    accepted = (
        nin >= 28 and frac >= 0.34 and med <= 2.2 and p90 <= 4.8
        and 0.12 <= area_ratio <= 7.0 and np.isfinite(cond) and cond < 2.0e6
    )
    qa["accepted"] = bool(accepted)
    qa["reason"] = "accepted" if accepted else "robust_geometry_gate_rejected"
    if not accepted:
        return None, qa
    qa["H_candidate_to_anchor"] = H.tolist()
    return H.astype(np.float64), qa


def _ensure_temporal(label, cams, images, dynamic_masks):
    if label in _TEMPORAL:
        return _TEMPORAL[label]
    if _CLIPS_DIR is None or _SYNC_QA_PATH is None:
        raise RuntimeError("v26 temporal source configuration missing")
    clips = v25._clip_map(_CLIPS_DIR)
    sync = json.loads(_SYNC_QA_PATH.read_text())
    center = int(sync["selected"]["frames"][label])
    anchor = images[label]
    rows = []
    accepted = []
    for off in OFFSETS:
        idx = center + int(off)
        im = _decode(clips[label], idx)
        if im is None:
            rows.append({"offset":int(off),"frame":idx,"status":"decode_failed"}); continue
        H, rq = _register(im, anchor, dynamic_masks[label])
        row = {"offset":int(off),"frame":idx,"status":"accepted" if H is not None else "rejected","registration":rq}
        rows.append(row)
        if H is not None:
            accepted.append({"offset":int(off),"frame":idx,"image":im,"H":H,"Hinv":np.linalg.inv(H)})
    _TEMPORAL[label] = accepted
    _REG_QA[label] = {
        "source_clip": clips[label].name,
        "center_frame": center,
        "requested_offsets": list(OFFSETS),
        "accepted_offsets": [x["offset"] for x in accepted],
        "accepted_count": int(len(accepted)),
        "candidates": rows,
        "rejected_frames_are_never_sampled": True,
    }
    return accepted


def _perspective_points(H, uv):
    uv = np.asarray(uv, np.float64)
    h = np.column_stack([uv, np.ones(len(uv),np.float64)])
    q = (np.asarray(H,np.float64) @ h.T).T
    with np.errstate(divide="ignore",invalid="ignore"):
        out = q[:,:2] / q[:,2:3]
    return out


def _actual_medoid(stack, valid, min_support=1):
    """Return actual source RGB from candidate nearest robust median, never synthetic median RGB."""
    # stack N,H,W,3 uint8; valid N,H,W bool
    arr = stack.astype(np.float32)
    arr[~valid[...,None].repeat(3,axis=3)] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(arr, axis=0)
    dist = np.sum(np.square(arr - med[None,...]), axis=3)
    dist[~valid] = np.inf
    count = valid.sum(axis=0)
    winner = np.argmin(dist, axis=0)
    yy,xx = np.indices(count.shape)
    out = stack[winner,yy,xx]
    ok = count >= int(min_support)
    out[~ok] = 0
    return out, ok, count


def _source_atlas_temporal(src):
    label = _label_for_image(src["image"])
    if label is None or _CONTEXT is None:
        return v21._source_atlas_original(src) if hasattr(v21,"_source_atlas_original") else _ORIGINAL_SOURCE_ATLAS(src)
    cams, images, dynamic_masks = _CONTEXT
    temporals = _ensure_temporal(label,cams,images,dynamic_masks)
    P = v21._atlas_world_points()
    C,R,K = src["C"],src["R"],src["K"]
    uv,_,visible = v12.v8.project_metric((C,R,K),P)
    finite = np.isfinite(uv).all(axis=1)
    ua = np.rint(uv[:,0]).astype(np.int32); va = np.rint(uv[:,1]).astype(np.int32)
    inside_anchor = finite & visible & (ua>=0)&(ua<v12.W)&(va>=0)&(va<v12.H)
    exact_valid = np.zeros(len(P),bool)
    ids=np.where(inside_anchor)[0]
    if len(ids):
        exact_valid[ids] = src["floor_vis"][va[ids],ua[ids]].astype(bool)
    exact = np.zeros((len(P),3),np.uint8)
    if np.any(exact_valid):
        exact[exact_valid] = v12.v8.bilinear_sample(src["image"],uv[exact_valid])

    samples=[]; masks=[]
    for row in temporals:
        uc = _perspective_points(row["Hinv"],uv)
        ok = (
            finite & visible & np.isfinite(uc).all(axis=1)
            & (uc[:,0]>=0)&(uc[:,0]<v12.W-1)&(uc[:,1]>=0)&(uc[:,1]<v12.H-1)
        )
        # If a metric floor point is in the exact view and known to be occluded by
        # permanent/static geometry, do not override that visibility judgment.
        in_a = inside_anchor
        ok[in_a & (~exact_valid)] = False
        col=np.zeros((len(P),3),np.uint8)
        if np.any(ok): col[ok]=v12.v8.bilinear_sample(row["image"],uc[ok])
        samples.append(col.reshape(v21.AH,v21.AW,3)); masks.append(ok.reshape(v21.AH,v21.AW))

    out=exact.reshape(v21.AH,v21.AW,3).copy(); mask=exact_valid.reshape(v21.AH,v21.AW).copy()
    temporal_supported=0; expanded=0
    if samples:
        st=np.stack(samples,axis=0); vm=np.stack(masks,axis=0)
        medoid,tok,count=_actual_medoid(st,vm,min_support=2)
        add=tok & (~mask)
        # Temporal samples can also clean dynamic occlusion holes only if at least
        # two registered source frames agree in support.
        out[add]=medoid[add]; mask[add]=True
        temporal_supported=int(tok.sum()); expanded=int(add.sum())
    _ATLAS_QA[label]={
        "exact_floor_texels":int(exact_valid.sum()),
        "accepted_temporal_frames":int(len(temporals)),
        "temporal_two_frame_supported_texels":int(temporal_supported),
        "new_temporal_atlas_texels":int(expanded),
        "final_source_atlas_texels":int(mask.sum()),
        "pixel_policy":"actual registered source-frame RGB medoid; no median RGB output",
    }
    try:
        if v21._OUT is not None:
            cv2.imwrite(str(v21._OUT/f"v26_temporal_atlas_{label.replace(' ','_')}.png"),out)
            cv2.imwrite(str(v21._OUT/f"v26_temporal_atlas_mask_{label.replace(' ','_')}.png"),mask.astype(np.uint8)*255)
    except Exception:
        pass
    return out,mask


def _sample_candidate_by_anchor_uv(row, uv_anchor):
    uc=_perspective_points(row["Hinv"],uv_anchor)
    ok=(np.isfinite(uc).all(axis=1)&(uc[:,0]>=0)&(uc[:,0]<v12.W-1)&(uc[:,1]>=0)&(uc[:,1]<v12.H-1))
    col=np.zeros((len(uv_anchor),3),np.uint8)
    if np.any(ok): col[ok]=v12.v8.bilinear_sample(row["image"],uc[ok])
    return col,ok


def far_background_temporal_angular(Kt,Rt,cams,images,dynamic_masks):
    global _CONTEXT
    _CONTEXT=(cams,images,dynamic_masks)
    # Build candidate sets even at 0 degrees so the floor atlas can immediately use them.
    for label in v12.CAMERAS:
        _ensure_temporal(label,cams,images,dynamic_masks)
    if v13._is_anchor_pose(cams,Rt,v12.CURRENT_CT):
        return images[v12.A].copy(),np.ones((v12.H,v12.W),bool)

    yy,xx=np.indices((v12.H,v12.W),np.float64)
    hp=np.stack([xx.ravel(),yy.ravel(),np.ones(v12.H*v12.W)],axis=0)
    dcam=np.linalg.inv(Kt)@hp; dw=Rt.T@dcam
    # Keep rigid metric planes out of the distant/infinite layer.
    _,sf=v12.v4.virtual_plane_points(Kt,Rt,v12.CURRENT_CT,'floor')
    _,sb=v12.v4.virtual_plane_points(Kt,Rt,v12.CURRENT_CT,'board')
    blocked=(sf|sb).reshape(-1)

    target_axis=v12.RIM-v12.CURRENT_CT; target_axis/=max(1e-9,float(np.linalg.norm(target_axis)))
    pref=[]
    for label in v12.CAMERAS:
        a=v12.RIM-cams[label][0]; a/=max(1e-9,float(np.linalg.norm(a)))
        pref.append((float(np.dot(a,target_axis)),label))
    pref.sort(reverse=True)

    out=np.zeros((v12.H*v12.W,3),np.uint8); owned=np.zeros(v12.H*v12.W,bool)
    source_counts={}
    for _,label in pref:
        Cc,Rc,Kc=cams[label]
        ds=Rc@dw; q=Kc@ds
        with np.errstate(divide="ignore",invalid="ignore"):
            ua=(q[:2]/q[2:3]).T
        sign=float(v12.v3.forward_sign(Rc,Cc))
        base_geom=np.isfinite(ua).all(axis=1)&(sign*ds[2]>1e-6)&(~blocked)
        ui=np.rint(ua[:,0]).astype(np.int32); vi=np.rint(ua[:,1]).astype(np.int32)
        inside=base_geom&(ui>=0)&(ui<v12.W)&(vi>=0)&(vi<v12.H)
        exact_ok=inside.copy()
        ids=np.where(inside)[0]
        if len(ids):
            excluded=v13._static_exclusion(label,cams)
            exact_ok[ids]&=~dynamic_masks[label][vi[ids],ui[ids]]
            exact_ok[ids]&=~excluded[vi[ids],ui[ids]]
        exact_col=np.zeros((len(ua),3),np.uint8)
        exact_valid=exact_ok&(ua[:,0]>=0)&(ua[:,0]<v12.W-1)&(ua[:,1]>=0)&(ua[:,1]<v12.H-1)
        if np.any(exact_valid): exact_col[exact_valid]=v12.v8.bilinear_sample(images[label],ua[exact_valid])

        rows=_TEMPORAL[label]
        samples=[exact_col.reshape(v12.H,v12.W,3)]; masks=[exact_valid.reshape(v12.H,v12.W)]
        for row in rows:
            col,ok=_sample_candidate_by_anchor_uv(row,ua)
            ok &= base_geom
            # Where the equivalent anchor pixel lies inside the exact frame, keep
            # its static/dynamic exclusion semantics for every temporal candidate.
            ok[inside & (~exact_ok)] = False
            samples.append(col.reshape(v12.H,v12.W,3)); masks.append(ok.reshape(v12.H,v12.W))
        st=np.stack(samples,axis=0); vm=np.stack(masks,axis=0)
        medoid,valid,count=_actual_medoid(st,vm,min_support=1)
        take=valid.reshape(-1)&(~owned)
        flat=medoid.reshape(-1,3); out[take]=flat[take]; owned[take]=True
        source_counts[label]={
            "owned_pixels":int(take.sum()),
            "candidate_count":int(len(rows)),
            "pixels_with_temporal_only_support":int(np.sum((count>=1)&(~masks[0]))),
        }
    _BG_QA.append({"owned_fraction":float(owned.mean()),"source_counts":source_counts})
    return out.reshape(v12.H,v12.W,3),owned.reshape(v12.H,v12.W)


def main():
    global _CLIPS_DIR,_SYNC_QA_PATH,_CONTEXT,_ORIGINAL_SOURCE_ATLAS
    _CLIPS_DIR=Path(_pop_arg('--clips-dir'))
    si=sys.argv.index('--sync-qa'); _SYNC_QA_PATH=Path(sys.argv[si+1])
    _CONTEXT=None; _TEMPORAL.clear(); _REG_QA.clear(); _ATLAS_QA.clear(); _BG_QA.clear()

    _ORIGINAL_SOURCE_ATLAS=v21._source_atlas
    # Preserve an explicit handle because our wrapper may be called before context.
    v21._source_atlas_original=_ORIGINAL_SOURCE_ATLAS
    v21._source_atlas=_source_atlas_temporal
    v15.far_background_multisource_clean=far_background_temporal_angular

    assert (int(v12.W),int(v12.H))==(960,540)
    v24.main()

    oi=sys.argv.index('--out'); out=Path(sys.argv[oi+1])
    qp=out/'three_camera_mesh_v12_qa.json'; q=json.loads(qp.read_text())
    q['v26_temporal_multiview_static']={
        'resolution':[960,540],
        'temporal_registration':_REG_QA,
        'per_camera_temporal_atlas':_ATLAS_QA,
        'background_render_calls':_BG_QA,
        'court_method':'metric world-space temporal atlas; accepted candidate->exact homographies only; >=2 temporal support for newly exposed texels',
        'arena_method':'distant angular source reprojection extended only by accepted temporal homographies',
        'rejected_temporal_frames_sampled':False,
        'foreground_geometry_change':False,
        'ball_geometry_change':False,
        'generated_texture':False,
        'inpainting':False,
        'crossfade':False,
        'upscale':False,
        'uhd':False,
    }
    qp.write_text(json.dumps(q,indent=2))
    print(json.dumps({
        'v26_registration':{k:v['accepted_offsets'] for k,v in _REG_QA.items()},
        'v26_atlas':_ATLAS_QA,
        'v26_background':_BG_QA,
    },indent=2),flush=True)


if __name__=='__main__':
    main()
