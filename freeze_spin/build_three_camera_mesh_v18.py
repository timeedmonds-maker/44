from __future__ import annotations

"""Jazz event 489 v18: native combined proof with stable single-view fallback.

Keeps v17's passed v16 temporal real-pixel clean plate and v15's multi-view
foreground. The one LAR player absent from Broadcast/RAR is reconstructed with
a deliberately labelled single-view convex 2.5-D surface instead of noisy raw
monocular depth. Every texture sample is still an observed LAR pixel and every
surface point remains on that source pixel's calibrated ray, so the proxy
reprojects exactly to the source view. The only prior is smooth human-scale
front-surface depth anchored to the detected floor contact.

QA is corrected to distinguish exact segmentation recall from the deliberately
dilated support masks used only for cross-view matching.
All outputs remain native 960x540.
"""

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v15 as v15
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_reference_surface_v10 as v10
from freeze_spin import build_three_camera_diagnostic_v3 as v3

W,H=v15.W,v15.H
REF=v15.REF
CAMERA_ORDER=v15.CAMERA_ORDER
_CLEAN_IMAGE=None
_CLEAN_NEW=None
_orig_background=v15.background_sample
_orig_annotate=v15.annotate


def background_sample_v18(P,support,sources,plane):
    if plane!='floor' or _CLEAN_IMAGE is None or _CLEAN_NEW is None:
        return _orig_background(P,support,sources,plane)
    s2={k:dict(v) for k,v in sources.items()}
    lar=dict(s2[REF]); lar['image']=_CLEAN_IMAGE; lar['floor_vis']=lar['floor_vis']|_CLEAN_NEW; s2[REF]=lar
    return _orig_background(P,support,s2,plane)


def convex_single_view_surface(inst,cam,image,zprior):
    mask=inst['mask'].astype(bool)
    ys,xs=np.where(mask)
    if len(xs)<80: return None
    C,R,K=cam; s=float(v3.forward_sign(R,C))
    foot=np.asarray(inst['foot_world_cm'],np.float64); fx,fy=[int(x) for x in inst['foot_px']]
    # Vertical plane through detected floor contact, facing the source camera in XY.
    n=foot[:2]-C[:2]; nn=float(np.linalg.norm(n))
    if not np.isfinite(nn) or nn<1e-6: return None
    n3=np.asarray([n[0]/nn,n[1]/nn,0.0],np.float64)
    xn=(xs.astype(np.float64)-K[0,2])/K[0,0]; yn=(ys.astype(np.float64)-K[1,2])/K[1,1]
    dc=np.column_stack([xn,yn,np.ones_like(xn)])
    dw=(s*dc)@R
    num=float(np.dot(n3,foot-C)); den=dw@n3
    with np.errstate(divide='ignore',invalid='ignore'): zbase=num/den
    fallback=float(inst.get('floor_camera_depth_cm',np.nanmedian(zbase)))
    bad=~np.isfinite(zbase)|(zbase<60)|(zbase>12000)
    zbase[bad]=fallback

    # Smooth convex front profile per source scanline.  The depth adjustment is
    # along each calibrated source ray, preserving the exact source projection.
    bulge=np.zeros(len(xs),np.float64)
    h=max(1,int(ys.max())-int(ys.min())+1)
    for y in np.unique(ys):
        ids=np.where(ys==y)[0]
        xx=xs[ids].astype(np.float64); lo=float(xx.min()); hi=float(xx.max()); half=max(1.0,(hi-lo)/2.0); cen=(lo+hi)/2.0
        xnorm=np.clip((xx-cen)/half,-1.0,1.0)
        local_z=float(np.median(zbase[ids])); half_world=half*local_z/max(float(K[0,0]),1e-6)
        radius=min(18.0,max(2.5,0.72*half_world))
        # Taper to zero at the actual floor contact and modestly near the head.
        above=max(0.0,float(fy-y)); foot_taper=min(1.0,above/22.0)
        topfrac=(float(y)-float(ys.min()))/h
        head_taper=0.65+0.35*min(1.0,topfrac/0.18) if topfrac<0.18 else 1.0
        bulge[ids]=radius*np.sqrt(np.maximum(0.0,1.0-xnorm*xnorm))*foot_taper*head_taper
    z=zbase-bulge
    # Keep a tight, physically plausible slab around the grounded plane.
    z=np.clip(z,fallback-85.0,fallback+55.0)
    pts=v10.backproject_pixels(cam,xs,ys,z).astype(np.float32)
    physical=(np.isfinite(pts).all(1)&(pts[:,0]>-450)&(pts[:,0]<1100)&(np.abs(pts[:,1])<850)&(pts[:,2]>-80)&(pts[:,2]<500))
    pts=pts[physical]; cols=image[ys[physical],xs[physical]].astype(np.uint8); zd=z[physical].astype(np.float32)
    if len(pts)<80: return None
    return {'points':pts,'colors':cols,'source_label':REF,'source_depth':zd,'support':np.zeros(len(pts),np.int8),'camera':cam,'qa':{
        'single_view_fallback':True,'single_view_geometry':'upright_vertical_plane_plus_scanline_convex_front_on_source_rays',
        'source_mask_pixels':int(mask.sum()),'surface_points':int(len(pts)),'floor_anchor_world_cm':[float(x) for x in foot],
        'floor_camera_depth_cm':fallback,'median_convex_bulge_cm':float(np.median(bulge[physical])),'p95_convex_bulge_cm':float(np.percentile(bulge[physical],95)),
        'native_source_pixels_only':True}}


def self_projection_exact(meshes,used_indices,instances,cams):
    out={}
    for lab in CAMERA_ORDER:
        ms=[m for m in meshes if m['source_label']==lab]
        if not ms: continue
        _,rm,_,rqa=v13.raster_one_source(ms,cams[lab][2],cams[lab][1],cams[lab][0])
        exact=np.zeros((H,W),bool); support=np.zeros((H,W),bool)
        for idx in sorted(used_indices.get(lab,set())):
            exact|=instances[lab][idx]['mask'].astype(bool)
            support|=instances[lab][idx]['support_mask'].astype(bool)
        ie=int(np.sum(rm&exact)); ue=int(np.sum(rm|exact)); isp=int(np.sum(rm&support)); usp=int(np.sum(rm|support))
        out[lab]={
            'iou':float(ie/max(1,ue)),'recall':float(ie/max(1,int(exact.sum()))),
            'exact_mask_iou':float(ie/max(1,ue)),'exact_mask_recall':float(ie/max(1,int(exact.sum()))),'exact_mask_pixels':int(exact.sum()),
            'support_mask_iou':float(isp/max(1,usp)),'support_mask_recall':float(isp/max(1,int(support.sum()))),'support_mask_pixels':int(support.sum()),
            'rendered_pixels':int(rm.sum()),**rqa}
    return out


def annotate_v18(img,angle,qa,fqa):
    out=_orig_annotate(img,angle,qa,fqa)
    cv2.rectangle(out,(0,0),(960,20),(0,0,0),-1)
    cv2.putText(out,'JAZZ EVENT 489 | v18 NATIVE | CLEAN PLATE + MULTI-VIEW MESH + CONVEX SINGLE-VIEW FALLBACK',(7,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    global _CLEAN_IMAGE,_CLEAN_NEW
    pre=argparse.ArgumentParser(add_help=False); pre.add_argument('--clean-plate-dir',type=Path,required=True); known,rest=pre.parse_known_args(); cp=known.clean_plate_dir
    cq=json.loads((cp/'temporal_clean_plate_v16_qa.json').read_text()); _CLEAN_IMAGE=cv2.imread(str(cp/'lar_temporal_clean_plate.png')); nm=cv2.imread(str(cp/'lar_temporal_new_floor_pixels.png'),cv2.IMREAD_GRAYSCALE)
    if _CLEAN_IMAGE is None or nm is None or _CLEAN_IMAGE.shape[:2]!=(540,960): raise RuntimeError('missing/non-native v16 clean plate')
    if not cq.get('gates',{}).get('numeric_pass'): raise RuntimeError('v16 clean plate not passed')
    _CLEAN_NEW=nm>0
    v15.background_sample=background_sample_v18; v15.monocular_lar_surface=convex_single_view_surface; v15.self_projection_all=self_projection_exact; v15.annotate=annotate_v18
    sys.argv=[sys.argv[0]]+rest; v15.main()
    out=None
    for i,a in enumerate(rest):
        if a=='--out' and i+1<len(rest): out=Path(rest[i+1]); break
    if out is None: raise RuntimeError('--out missing')
    q=json.loads((out/'three_camera_mesh_v15_qa.json').read_text()); q['schema_version']=18; q['native_dimensions']=[960,540]; q['native_only']=True; q['v16_clean_plate']={'accepted_temporal_count':cq['accepted_temporal_count'],'new_temporal_floor_pixels':cq['new_temporal_floor_pixels'],'exact_dynamic_floor_hole_fill_fraction':cq['exact_dynamic_floor_hole_fill_fraction']}
    q['qa_definition_change']='self_projection recall/iou now use exact instance segmentation; support-mask metrics are reported separately because 3x3 dilation is matching tolerance, not target body area'
    q['single_view_policy']='only the LAR person absent from both other solved views uses convex source-ray proxy; no claim of multi-view recovery for that player'
    q['gates']['native_dimensions_pass']=True; q['gates']['v16_clean_plate_pass']=True; q['status']='STATIC_NATIVE_CONVEX_FALLBACK_RENDERED'; q['method']='v11 exact sync + v16 real-pixel temporal clean floor + v15/v14 multi-view player mesh + exact-source-reprojecting convex 2.5D proxy only for unmatched LAR player; native 960x540'
    q['gates']['numeric_pass']=all(v for k,v in q['gates'].items() if k!='numeric_pass')
    (out/'three_camera_mesh_v18_qa.json').write_text(json.dumps(q,indent=2),encoding='utf-8')
    print(json.dumps({'status':q['status'],'native':q['native_dimensions'],'self_projection':q['self_projection'],'single_view_players':[p for p in q['players'] if p.get('mode')=='LAR_ONLY_UNMATCHED'],'gates':q['gates']},indent=2),flush=True)

if __name__=='__main__': main()
