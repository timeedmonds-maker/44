from __future__ import annotations

"""Add a smooth 61-camera 0..25 degree orbit to a passed v32f dataset.

Every virtual camera is a rigid metric pose.  No RGB interpolation is performed.
The copied anchor PNGs exist only because the PanopticSports loader requires a
file path for each render camera; they are byte-identical placeholders and are
not ground truth.
"""

import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np


def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def zrot(deg):
    a=np.deg2rad(float(deg)); c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]],np.float64)


def orbit(M0,pivot,deg):
    M0=np.asarray(M0,np.float64); R0=M0[:3,:3]
    C0=np.linalg.inv(M0)[:3,3]; Q=zrot(deg)
    C=pivot+Q@(C0-pivot); R=R0@Q.T
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=-R@C
    return M


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',type=Path,required=True); ap.add_argument('--frames',type=int,default=61)
    a=ap.parse_args(); r=a.dataset
    tr=json.load(open(r/'train_meta.json')); b=json.load(open(r/'v32f_backend.json'))
    if len(tr['fn'])!=1 or tr['cam_id']!=[[0,1,2]]: raise RuntimeError('not a freeze-only v32f dataset')
    n=int(a.frames)
    if n<2: raise RuntimeError('frames must be >=2')
    angles=np.linspace(0.0,25.0,n)
    K=np.asarray(tr['k'][0][0],np.float64); M0=np.asarray(tr['w2c'][0][0],np.float64)
    pivot=np.asarray(b['focal_player_seed']['center_world_cm'],np.float64)/100.0
    Ms=[orbit(M0,pivot,x) for x in angles]
    if np.max(np.abs(Ms[0]-M0))>1e-10: raise RuntimeError('dense orbit 0deg mismatch')
    radii=[np.linalg.norm(np.linalg.inv(M)[:3,3]-pivot) for M in Ms]
    if max(abs(x-radii[0]) for x in radii)>1e-9: raise RuntimeError('dense orbit radius drift')
    anchor=r/'ims'/tr['fn'][0][0]; ah=sha256(anchor)
    fns=[]
    for i,deg in enumerate(angles):
        fn=f'dense_orbit_placeholder_{i:03d}_{deg:06.3f}deg.png'; dst=r/'ims'/fn
        shutil.copy2(anchor,dst)
        if sha256(dst)!=ah: raise RuntimeError('placeholder bytes changed')
        fns.append(fn)
    meta={'w':tr['w'],'h':tr['h'],'fn':[fns],'k':[[K.tolist() for _ in angles]],'w2c':[[M.tolist() for M in Ms]],'cam_id':[[1000+i for i in range(n)]]}
    (r/'test_meta_orbit_dense_0_25.json').write_text(json.dumps(meta,indent=2))
    q={'version':'v32f_dense_orbit','frames':n,'fps_target':30,'angles_deg':[float(x) for x in angles],
       'start_deg':0.0,'end_deg':25.0,'orbit_pivot_world_m':pivot.tolist(),'orbit_radius_m':float(radii[0]),
       'zero_degree_w2c_max_abs_error':float(np.max(np.abs(Ms[0]-M0))),
       'max_orbit_radius_error_m':float(max(abs(x-radii[0]) for x in radii)),
       'camera_motion':'rigid metric orbit about world +Z; no 2D interpolation',
       'rgb_placeholders_are_ground_truth':False,'generated_rgb':False,'optical_flow':False,'crossfade':False,
       'status':'PASS_V32F_DENSE_61_VIEW_ORBIT'}
    (r/'v32f_dense_orbit_qa.json').write_text(json.dumps(q,indent=2)); print(json.dumps(q,indent=2))

if __name__=='__main__': main()
