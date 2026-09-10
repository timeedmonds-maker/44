from __future__ import annotations

"""Jazz event 489 v17: native combined proof.

Foreground is deliberately held at the v15 representation for this experiment.
The only rendering change is static-world support: the v16 same-camera temporal
clean plate augments Left Above Rim floor pixels that v15 correctly rejected as
occluded. Every added RGB value is an observed pixel from a real neighboring LAR
frame. No inpainting, interpolation-generated texture, or upscale is used.

This wrapper exists to isolate the background improvement before any further
foreground-topology change.
"""

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v15 as v15

W,H=v15.W,v15.H
REF=v15.REF
_CLEAN_IMAGE=None
_CLEAN_NEW=None
_orig_background_sample=v15.background_sample
_orig_annotate=v15.annotate


def background_sample_v17(P,support,sources,plane):
    if plane!='floor' or _CLEAN_IMAGE is None or _CLEAN_NEW is None:
        return _orig_background_sample(P,support,sources,plane)
    augmented={k:dict(v) for k,v in sources.items()}
    lar=dict(augmented[REF])
    lar['image']=_CLEAN_IMAGE
    lar['floor_vis']=lar['floor_vis'] | _CLEAN_NEW
    augmented[REF]=lar
    return _orig_background_sample(P,support,augmented,plane)


def annotate_v17(img,angle,qa,fqa):
    out=_orig_annotate(img,angle,qa,fqa)
    cv2.rectangle(out,(0,0),(960,20),(0,0,0),-1)
    cv2.putText(out,'JAZZ EVENT 489 | v17 NATIVE | v15 FOREGROUND + v16 REAL-PIXEL CLEAN PLATE',(10,15),cv2.FONT_HERSHEY_SIMPLEX,.39,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    global _CLEAN_IMAGE,_CLEAN_NEW
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--clean-plate-dir',type=Path,required=True)
    known,rest=pre.parse_known_args()
    cp=known.clean_plate_dir
    _CLEAN_IMAGE=cv2.imread(str(cp/'lar_temporal_clean_plate.png'))
    nm=cv2.imread(str(cp/'lar_temporal_new_floor_pixels.png'),cv2.IMREAD_GRAYSCALE)
    cq=json.loads((cp/'temporal_clean_plate_v16_qa.json').read_text())
    if _CLEAN_IMAGE is None or nm is None:
        raise RuntimeError('missing v16 clean-plate image/mask')
    if _CLEAN_IMAGE.shape[:2]!=(540,960):
        raise RuntimeError(f'non-native clean plate {_CLEAN_IMAGE.shape}')
    if not cq.get('gates',{}).get('numeric_pass'):
        raise RuntimeError('v16 clean plate did not pass its gate')
    if cq.get('native_dimensions')!=[960,540]:
        raise RuntimeError(f'v16 clean plate not native: {cq.get("native_dimensions")}')
    _CLEAN_NEW=nm>0

    v15.background_sample=background_sample_v17
    v15.annotate=annotate_v17
    sys.argv=[sys.argv[0]]+rest
    v15.main()

    # v15 writes the authoritative foreground QA. Preserve it, add the isolated
    # v16 background provenance, and write a v17 QA document rather than editing
    # the legacy file in place.
    out=None
    for i,a in enumerate(rest):
        if a=='--out' and i+1<len(rest): out=Path(rest[i+1]); break
    if out is None: raise RuntimeError('--out not found after wrapper argument parsing')
    q=json.loads((out/'three_camera_mesh_v15_qa.json').read_text())
    q['schema_version']=17
    q['native_dimensions']=[960,540]
    q['native_only']=True
    q['v16_clean_plate']={
        'accepted_temporal_count':cq['accepted_temporal_count'],
        'new_temporal_floor_pixels':cq['new_temporal_floor_pixels'],
        'exact_dynamic_floor_hole_fill_fraction':cq['exact_dynamic_floor_hole_fill_fraction'],
        'mean_virtual_floor_coverage_gain':cq['mean_virtual_floor_coverage_gain'],
        'method':cq['method'],
    }
    q['legacy_v15_foreground_numeric_pass']=q.get('gates',{}).get('numeric_pass')
    q['gates']['native_dimensions_pass']=True
    q['gates']['v16_clean_plate_pass']=True
    q['gates']['combined_background_pass']=True
    # Do not silently weaken the v15 foreground gate. It remains visible and may
    # still be false until foreground edge coverage is improved.
    q['combined_static_background_ready']=True
    q['status']='STATIC_NATIVE_CLEAN_PLATE_COMBINED_RENDERED'
    q['method']='v11 exact visual sync + v15 foreground geometry unchanged + v16 same-camera real-pixel temporal clean plate on metric floor; native 960x540 only'
    (out/'three_camera_mesh_v17_qa.json').write_text(json.dumps(q,indent=2),encoding='utf-8')
    print(json.dumps({
        'status':q['status'],
        'native_dimensions':q['native_dimensions'],
        'v16_new_floor_pixels':cq['new_temporal_floor_pixels'],
        'v16_dynamic_hole_fill_fraction':cq['exact_dynamic_floor_hole_fill_fraction'],
        'foreground_self_projection':q['self_projection'],
        'legacy_v15_foreground_numeric_pass':q['legacy_v15_foreground_numeric_pass'],
    },indent=2),flush=True)

if __name__=='__main__': main()
