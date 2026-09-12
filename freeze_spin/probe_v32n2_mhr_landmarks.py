from __future__ import annotations

"""v32n2: inspect neutral MHR joint/locator landmarks before NBA fitting.

This is a mapping audit only.  It records exact world-space neutral joint origins,
locator names/parents/offsets, and skinned locators when available so COCO-style
shoulder/elbow/wrist/hip/knee/ankle observations are not attached to the wrong
anatomical origin.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def _jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, 'tolist'):
        try: return x.tolist()
        except Exception: pass
    return str(x)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--work',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    subprocess.run(['mhr-download-assets','--output',str(a.work)], check=False)
    # The CLI's default layout is the most stable contract; use package default if it exists.
    from mhr.mhr import MHR
    from mhr.io import get_default_asset_folder
    assets=get_default_asset_folder()
    if not (assets/'lod1.fbx').exists():
        subprocess.run(['mhr-download-assets'], check=True)
    model=MHR.from_files(device=torch.device('cpu'),lod=1,wants_pose_correctives=False)
    character=model.character
    z_id=torch.zeros((1,45),dtype=torch.float32)
    z_pose=torch.zeros((1,204),dtype=torch.float32)
    with torch.no_grad():
        _,state=model(z_id,z_pose,None,apply_correctives=False)
    st=state[0].cpu().numpy(); names=list(character.skeleton.joint_names); parents=np.asarray(character.skeleton.joint_parents,int)
    joints=[{'index':i,'name':n,'parent':int(parents[i]),'xyz_cm':st[i,:3].tolist()} for i,n in enumerate(names)]
    focus_tokens=('upleg','lowleg','foot','talocrural','subtalar','ball','clavicle','uparm','lowarm','wrist','spine','neck','head','eye')
    focus=[r for r in joints if any(t in r['name'].lower() for t in focus_tokens)]
    locs=[]
    for i,l in enumerate(getattr(character,'locators',[]) or []):
        d={'index':i,'repr':str(l)}
        for k in ('name','parent','parent_joint','joint','offset','position'):
            if hasattr(l,k): d[k]=_jsonable(getattr(l,k))
        locs.append(d)
    slocs=[]
    for i,l in enumerate(getattr(character,'skinned_locators',[]) or []):
        d={'index':i,'repr':str(l)}
        for k in ('name','parent','parent_joint','joint','offset','position','weights','joint_indices'):
            if hasattr(l,k): d[k]=_jsonable(getattr(l,k))
        slocs.append(d)
    q={'version':'v32n2_mhr_landmark_probe','status':'PASS_V32N2_MHR_LANDMARK_PROBE','joint_count':len(names),'focus_joints':focus,'locators':locs,'skinned_locators':slocs,'generated_rgb':False}
    (a.out/'v32n2_mhr_landmark_probe.json').write_text(json.dumps(q,indent=2))
    print(json.dumps({'status':q['status'],'focus_joint_count':len(focus),'locator_count':len(locs),'skinned_locator_count':len(slocs)},indent=2))

if __name__=='__main__': main()
