from __future__ import annotations

"""v2 gate: recover the exact v33e metric scene before any free-view render.

This deliberately does NOT render a novel view.  It rebuilds the accepted
three physical camera states for event 489, reproduces the exact v33e body
reprojection gate, triangulates Adams' body in world centimetres, and audits the
old v1 ball-pixel pivot.  Only validated source geometry is exported.
"""

import argparse, copy, importlib, json, math, sys
from pathlib import Path
import cv2, numpy as np

W,H=960,540
CAMS=("Left Above Rim","Right Above Rim","Broadcast")
RIM=np.array([38.1,0.0,304.8],float)
OLD_BALL={"Right Above Rim":np.array([601.,191.]),"Broadcast":np.array([531.,105.])}

def safe(s): return s.replace(" ","_")
def readj(p): return json.loads(Path(p).read_text())
def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.floating,np.integer)): return x.item()
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [serial(v) for v in x]
    return x

def loadmods(root):
    root=Path(root).resolve(); sys.path.insert(0,str(root))
    v33e=importlib.import_module("freeze_spin.run_v33e_locked_three_camera_wide_flow_state_search")
    v33d=importlib.import_module("freeze_spin.run_v33d_locked_three_camera_joint_state_search")
    v33b=importlib.import_module("freeze_spin.run_v33b_rar_exact_state_sweep")
    v32j=importlib.import_module("freeze_spin.solve_v32j_rfdetr_freeze_pose")
    return v33e,v33d,v33b,v32j

def ray(cam,uv):
    K=np.asarray(cam['K_px'],float); R=np.asarray(cam['R_world_to_camera'],float)
    d=R.T@(np.linalg.inv(K)@np.array([uv[0],uv[1],1.]))
    return d/np.linalg.norm(d)

def closest_rays(c1,d1,c2,d2):
    w=c1-c2; a=d1@d1; b=d1@d2; c=d2@d2; d=d1@w; e=d2@w
    den=a*c-b*b
    s=(b*e-c*d)/den; t=(a*e-b*d)/den
    p1=c1+s*d1; p2=c2+t*d2
    return p1,p2,(p1+p2)/2,float(np.linalg.norm(p1-p2))
def project(cam,X):
    K=np.asarray(cam['K_px'],float); R=np.asarray(cam['R_world_to_camera'],float); C=np.asarray(cam['C_world_cm'],float)
    q=K@(R@(np.asarray(X,float)-C)); return q[:2]/q[2]
def overlay(img,row,joints,cam,v32j,title):
    out=img.copy(); xy=np.asarray(row['xy'],float); cf=np.asarray(row['conf'],float)
    for a,b in v32j.DRAW:
        if cf[a]>=.20 and cf[b]>=.20: cv2.line(out,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),(0,255,255),2,cv2.LINE_AA)
    for j,X in joints.items():
        uv=v32j.project(cam,X)
        if uv is not None: cv2.circle(out,tuple(np.rint(uv).astype(int)),4,(255,0,255),1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,32),(0,0,0),-1); cv2.putText(out,title,(7,21),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clips',type=Path,required=True); ap.add_argument('--b32-root',type=Path,required=True)
    ap.add_argument('--v73-frame0257',type=Path,required=True); ap.add_argument('--v33e-json',type=Path,required=True)
    ap.add_argument('--validated-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    v33e,v33d,v33b,v32j=loadmods(a.validated_root)
    scene=readj(a.b32_root/'stage_a'/'v32_scene_manifest.json'); q=readj(a.v33e_json)
    if q.get('status')!='PASS_V33E_EXACT_THREE_VIEW_STATE' or scene.get('resolution')!=[W,H]: raise RuntimeError('v33e/B32 source lock failed')
    rel=q['winner']['rels']; expected={'Left Above Rim':-1,'Right Above Rim':-6,'Broadcast':18}
    if rel!=expected or not q['winner']['exact_state']: raise RuntimeError(f'v33e winner changed {rel}')
    centers={k:int(v) for k,v in scene['freeze']['chosen_frame_indices'].items()}; frames={c:centers[c]+rel[c] for c in CAMS}
    if frames!={'Left Above Rim':259,'Right Above Rim':250,'Broadcast':294}: raise RuntimeError(f'frame lock changed {frames}')

    rels=tuple(range(-20,21)); v33d.RELS=rels
    stage=a.out/'wide'; stage.mkdir(exist_ok=True); v33e.export_wide_burst(a.clips,centers,stage,rels)
    fs={c:v33d.frames(stage,c) for c in CAMS}
    cert=cv2.imread(str(a.v73_frame0257),0)
    if cert is None or cert.shape!=(H,W): raise RuntimeError('v73 cert missing')
    exact={}; transfers={}
    for c in ('Left Above Rim','Broadcast'):
        cc,au=v33d.transfer(fs[c][0][2],fs[c][rel[c]][2],scene['cameras'][c]);
        if cc is None: raise RuntimeError(f'{c} transfer failed {au}')
        exact[c]=cc; transfers[c]=au
    c='Right Above Rim'; cc,au=v33b.transfer_rar_camera(cert,fs[c][rel[c]][2],scene['cameras'][c])
    if cc is None: raise RuntimeError(f'RAR transfer failed {au}')
    exact[c]=cc; transfers[c]=au

    camqa={}
    for c in CAMS:
        exp=q['camera_transfers'][c][str(rel[c])]; expK=np.asarray(exp.get('K',exp.get('K_px')),float); gotK=np.asarray(exact[c]['K_px'],float)
        kd=float(np.max(np.abs(expK-gotK))); cd=float(np.linalg.norm(np.asarray(exact[c]['C_world_cm'])-np.asarray(scene['cameras'][c]['C_world_cm'])))
        gate=kd<=2.0 and cd<=1e-5; camqa[c]={'gate':gate,'K_max_abs_delta_vs_v33e_px':kd,'center_delta_cm':cd,'transfer':transfers[c]}
        if not gate: raise RuntimeError(f'{c} camera fingerprint failed kd={kd} cd={cd}')
    escene=copy.deepcopy(scene)
    for c in CAMS: escene['cameras'][c]=exact[c]

    from rfdetr import RFDETRKeypointPreview
    model=RFDETRKeypointPreview(); ids={}; obs={}; oba={}
    for c in CAMS:
        ids[c],_=v33d.track(model,c,fs[c]); obs[c],oba[c]=v33e.flow_observations(fs[c],ids[c],c,rels)
    rows={c:obs[c][rel[c]] for c in CAMS}
    repro=v33d.repro(escene,rows['Left Above Rim'],rows['Right Above Rim'],rows['Broadcast']); exp=q['winner']['reprojection']
    bodygate=bool(repro['gate'] and abs(repro['median_px']-exp['median_px'])<=2 and abs(repro['p90_px']-exp['p90_px'])<=4)
    if not bodygate: raise RuntimeError(f'body repro fingerprint failed {repro} expected {exp}')

    cams={c:v32j.cam(escene,c) for c in CAMS}; joints={}; jq={}
    for j in v33e.BODY:
        o={c:np.asarray(rows[c]['xy'],float)[j] for c in CAMS if np.asarray(rows[c]['conf'],float)[j]>=v33e.MIN_CONF}
        if len(o)<2: continue
        X=v32j.triangulate_rays(cams,o)
        if X is None or not(-350<=X[0]<=1250 and -750<=X[1]<=750 and -60<=X[2]<=450): continue
        joints[int(j)]=X; jq[str(j)]={'name':v32j.NAMES[j],'world_xyz_cm':X.tolist(),'reproj':{c:float(np.linalg.norm(v32j.project(cams[c],X)-uv)) for c,uv in o.items()}}
    if len(joints)<7: raise RuntimeError(f'only {len(joints)} body joints triangulated')
    for c in CAMS:
        cv2.imwrite(str(a.out/f'v2_body_{safe(c)}.png'),overlay(fs[c][rel[c]][1],rows[c],joints,cams[c],v32j,f'v2 {c}: yellow observation / magenta 3D reprojection'))

    # Audit and retire the stale v1 ball pivot rather than silently reusing it.
    c1,c2='Right Above Rim','Broadcast'; C1=np.asarray(exact[c1]['C_world_cm'],float); C2=np.asarray(exact[c2]['C_world_cm'],float)
    p1,p2,Xball,sep=closest_rays(C1,ray(exact[c1],OLD_BALL[c1]),C2,ray(exact[c2],OLD_BALL[c2]))
    ballerrs={c:float(np.linalg.norm(project(exact[c],Xball)-OLD_BALL[c])) for c in (c1,c2)}
    stale=bool(sep>10.0 or max(ballerrs.values())>10.0)

    out={
      'version':'reshoot_v2_exact_geometry_gate','status':'PASS_V2_EXACT_GEOMETRY_GATE','event':{'game_id':'0022500301','event_id':489},
      'source_lock':{'cameras':frames,'rels':rel,'native_resolution':[W,H]},'camera_fingerprint':camqa,
      'exact_cameras':{c:{k:exact[c][k] for k in ('K_px','R_world_to_camera','C_world_cm','extrinsic_world_to_camera_3x4')} for c in CAMS},
      'body_reprojection':{'gate':bodygate,'recomputed':repro,'validated_v33e':exp},'triangulated_body_joints':jq,
      'observation_sources':{c:oba[c][str(rel[c])] for c in CAMS},
      'old_v1_ball_pivot_audit':{'pixels':{k:v.tolist() for k,v in OLD_BALL.items()},'ray_separation_cm':sep,'world_midpoint_cm':Xball.tolist(),'reprojection_errors_px':ballerrs,'stale_invalid_for_v33e_exact_state':stale},
      'render_pivot':{'source':'fixed accepted metric-world rim centre','world_xyz_cm':RIM.tolist()},
      'freeview_render_unlocked':False,'next_gate':'metric depth alignment from exact cameras + body joints, then leave-one-camera-out visual QA before novel render',
      'pixel_policy':'no generated RGB; no fourth camera; native 960x540 only'
    }
    (a.out/'v2_exact_geometry_lock.json').write_text(json.dumps(serial(out),indent=2))
    print(json.dumps({'status':out['status'],'frames':frames,'reprojection':repro,'joint_count':len(joints),'stale_ball':out['old_v1_ball_pivot_audit'],'next_gate':out['next_gate']},indent=2),flush=True)

if __name__=='__main__': main()
