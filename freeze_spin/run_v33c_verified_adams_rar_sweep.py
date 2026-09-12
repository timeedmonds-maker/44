from __future__ import annotations

"""v33c: repeat the RAR exact-state sweep with Adams identity hard-locked
from previously verified LAR and Broadcast observations.

v33b exposed a false cross-view match between two other Houston players.  This
run does not let geometry choose the reference identities.  It locks:
- LAR to the visually verified v32w Adams t+03 detection under the rim;
- Broadcast to the visually verified v32z Adams t+03 detection (#12 at basket).
Each is dense-flow tracked to t+00 exactly as before.  RAR is then swept across
all real rel-06..rel+06 frames with a static-only exact camera transfer for each
frame.  A RAR state passes only if the same whole-body candidate satisfies the
existing raw-state gate against BOTH verified Adams reference views.

No player-based camera fitting, no per-joint temporal mixing, no generated RGB,
no interpolation, no threshold weakening, no upscale, no free-view render.
"""

import argparse, copy, json
from pathlib import Path
import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview

from freeze_spin import run_v33b_rar_exact_state_sweep as v33b
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import run_v32w_lar_all_candidate_epipolar_mhr as v32w
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

LAR,BCAST,RAR=v32v.LAR,v32v.BCAST,v32v.RAR
BODY=v32v.BODY; MIN_CONF=v32v.MIN_CONF; LEFT_WRIST=9
W,H=v32j.W,v32j.H
RAR_RELS=tuple(range(-6,7))
VERIFIED={
 LAR:{"anchor_rel":3,"box":[467.1226806640625,179.22335815429688,504.54278564453125,286.54254150390625],"source":"v32w visually verified Adams #12"},
 BCAST:{"anchor_rel":3,"box":[499.87261962890625,141.63543701171875,539.3041381835938,278.45684814453125],"source":"v32z visually verified Adams #12"},
}

def box_center(b):
 b=np.asarray(b,float); return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2],float)

def iou(a,b):
 a=np.asarray(a,float); b=np.asarray(b,float)
 x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
 inter=max(0.,x2-x1)*max(0.,y2-y1)
 aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]); bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
 return inter/max(aa+bb-inter,1e-9)

def locked_reference(model,stage,label,drop_left_wrist=False):
 spec=VERIFIED[label]; rel=int(spec["anchor_rel"]); target=np.asarray(spec["box"],float)
 frames=v32v.load_burst(stage,label,range(0,rel+1))
 dets=v32j.infer(model,frames[rel]); rows=[]
 for i,d in enumerate(dets):
  dark,torso,confident=v32w.appearance(d,frames[rel])
  b=np.asarray(d["box"],float); ov=iou(b,target); dist=float(np.linalg.norm(box_center(b)-box_center(target)))
  if confident<7: continue
  rows.append((-(ov),dist,i,d,dark,torso,confident,b))
 if not rows: raise RuntimeError(f"v33c no reference candidates for {label}")
 rows.sort(key=lambda r:(r[0],r[1])); _,dist,i,d,dark,torso,confident,b=rows[0]; ov=iou(b,target)
 if ov<0.55 or dist>15.0:
  raise RuntimeError(f"v33c verified identity lock failed {label}: IoU={ov:.3f} center={dist:.2f}px box={b.tolist()}")
 xy0,valid0,trace=v32l.dense_track_back(frames,rel,d["xy"],d["conf"])
 valid0=np.asarray(valid0,bool)&(np.asarray(d["conf"],float)>=MIN_CONF)
 if drop_left_wrist: valid0[LEFT_WRIST]=False
 return {
  "label":label,"source":f"verified_temporal_t+{rel}","anchor_rel":rel,"selected_index":int(i),
  "anchor_box_xyxy":b.tolist(),"verified_target_box_xyxy":target.tolist(),"anchor_iou":float(ov),
  "anchor_center_delta_px":dist,"xy0":np.asarray(xy0,float),"valid0":valid0,
  "anchor_conf":np.asarray(d["conf"],float),"trace":trace,"dark_fraction":float(dark),
  "torso_dark_score":float(torso),"confident_body_joints":int(confident),"frames":frames,
  "identity_provenance":spec["source"],
 }

def serial_ref(r):
 return {k:v for k,v in r.items() if k not in {"xy0","valid0","anchor_conf","trace","frames"}}

def pseudo_t0(r):
 c=np.asarray(r["anchor_conf"],float).copy(); c[~np.asarray(r["valid0"],bool)]=0.0
 return {"xy":np.asarray(r["xy0"],float),"conf":c}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--b32-root",type=Path,required=True); ap.add_argument("--v73-frame0257",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 stage=a.b32_root/"stage_a"; scene=json.loads((stage/"v32_scene_manifest.json").read_text()); qual=json.loads((stage/"v32_quality_cluster.json").read_text())
 cert=cv2.imread(str(a.v73_frame0257),cv2.IMREAD_GRAYSCALE)
 if cert is None or cert.shape!=(H,W): raise RuntimeError("v33c missing certified RAR frame0257")
 model=RFDETRKeypointPreview()
 lref=locked_reference(model,stage,LAR,drop_left_wrist=True); bref=locked_reference(model,stage,BCAST,drop_left_wrist=False)
 lb=v32v.epipolar_stats(scene,LAR,BCAST,lref["xy0"],lref["valid0"],bref["xy0"],bref["valid0"]); lb_gate=v33b.raw_pair_gate(lb)
 cv2.imwrite(str(a.out/"v33c_verified_lar_t00.png"),v33b.draw_detection(lref["frames"][0],pseudo_t0(lref),f"v33c VERIFIED Adams LAR | L-B {lb['median_px']:.1f}px"))
 cv2.imwrite(str(a.out/"v33c_verified_broadcast_t00.png"),v33b.draw_detection(bref["frames"][0],pseudo_t0(bref),f"v33c VERIFIED Adams Broadcast | L-B {lb['median_px']:.1f}px"))
 rb=np.asarray(qual.get("focal_player",{}).get("observations",{}).get(RAR,{}).get("bbox_xyxy",[430,180,590,340]),float); rt=box_center(rb); source_cam=scene["cameras"][RAR]
 sweep=[]; visuals=[]
 for rel in RAR_RELS:
  p=v32k.burst_path(stage,RAR,rel); img=cv2.imread(str(p)); gray=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
  if img is None or gray is None: sweep.append({"rel":rel,"frame_file":p.name,"status":"FAIL_IMAGE"}); continue
  cam,audit=v33b.transfer_rar_camera(cert,gray,source_cam)
  if cam is None: sweep.append({"rel":rel,"frame_file":p.name,"status":"FAIL_STATIC_TRANSFER","static":audit}); continue
  sr=copy.deepcopy(scene); sr["cameras"][RAR]=cam; candidates=[]
  for i,d in enumerate(v32j.infer(model,img)):
   dark,torso,confident=v32w.appearance(d,img)
   if confident<7 or not (torso>=0.35 or dark>=0.45): continue
   valid=np.asarray(d["conf"],float)>=MIN_CONF
   el=v32v.epipolar_stats(sr,RAR,LAR,np.asarray(d["xy"],float),valid,lref["xy0"],lref["valid0"])
   eb=v32v.epipolar_stats(sr,RAR,BCAST,np.asarray(d["xy"],float),valid,bref["xy0"],bref["valid0"])
   if min(int(el["joint_count"]),int(eb["joint_count"]))<5: continue
   b=np.asarray(d["box"],float); dist=float(np.linalg.norm(box_center(b)-rt)); shortage=max(0,7-int(el["joint_count"]))+max(0,7-int(eb["joint_count"]))
   score=float(el["median_px"]+eb["median_px"]+.25*(el["p90_px"]+eb["p90_px"])+10*shortage+.01*dist-.20*torso-.10*dark)
   gates={"lar_rar":v33b.raw_pair_gate(el),"broadcast_rar":v33b.raw_pair_gate(eb)}
   candidates.append({"index":int(i),"score":score,"box_xyxy":b.tolist(),"dark_fraction":float(dark),"torso_dark_score":float(torso),"confident_body_joints":int(confident),"target_center_distance_px":dist,"lar_rar":v33b.serialize_epi(el),"broadcast_rar":v33b.serialize_epi(eb),"gates":gates,"exact_state_gate":bool(gates["lar_rar"] and gates["broadcast_rar"]),"_det":d})
  candidates.sort(key=lambda r:(not r["exact_state_gate"],r["score"])); best=candidates[0] if candidates else None
  sweep.append({"rel":int(rel),"frame_file":p.name,"status":"CANDIDATE" if best else "NO_CANDIDATE","static":audit,"candidate_count":len(candidates),"best":None if best is None else {k:v for k,v in best.items() if k!="_det"},"top_candidates":[{k:v for k,v in x.items() if k!="_det"} for x in candidates[:12]]})
  if best:
   ov=v33b.draw_detection(img,best["_det"],f"v33c RAR rel{rel:+d} | L {best['lar_rar']['median_px']:.1f}px B {best['broadcast_rar']['median_px']:.1f}px gate={best['exact_state_gate']}"); cv2.imwrite(str(a.out/f"v33c_rar_rel{rel:+03d}.png"),ov); visuals.append((rel,ov))
 valid=[r for r in sweep if r.get("best")]; valid.sort(key=lambda r:(not r["best"]["exact_state_gate"],r["best"]["score"])); winner=valid[0] if valid else None; exact=bool(lb_gate and winner and winner["best"]["exact_state_gate"])
 if visuals:
  canvas=np.zeros((H,W,3),np.uint8)
  for s,(rel,ov) in enumerate(sorted(visuals)[:16]):
   rr,cc=divmod(s,4); th=cv2.resize(ov,(240,135),interpolation=cv2.INTER_AREA); canvas[rr*135:(rr+1)*135,cc*240:(cc+1)*240]=th
  cv2.imwrite(str(a.out/"v33c_rar_sweep_contact_native_960x540.png"),canvas)
 qa={"version":"v33c_verified_adams_rar_exact_state_sweep","status":"PASS_V33C_RAR_EXACT_STATE_FOUND" if exact else "FAIL_CLOSED_V33C_NO_RAR_EXACT_STATE","native_resolution":[W,H],"generated_rgb":False,"novel_view_rendered":False,"upscaled":False,"verified_references":{"lar":serial_ref(lref),"broadcast":serial_ref(bref),"lar_broadcast_epipolar":v33b.serialize_epi(lb),"lar_broadcast_gate":lb_gate},"rar_sweep":sweep,"winner":winner,"gate":{"verified_lar_broadcast_consistent":lb_gate,"rar_candidate_found":winner is not None,"rar_agrees_with_lar":bool(winner and winner['best']['gates']['lar_rar']),"rar_agrees_with_broadcast":bool(winner and winner['best']['gates']['broadcast_rar']),"exact_three_view_state_found":exact,"freeview_render_unlocked":False},"next_if_pass":"fit one MHR body to all three identity-locked exact-state observations, then leave-one-view-out QA","next_if_fail":"RAR dynamic foreground is unsupported for this freeze; seek another independent camera rather than forcing RAR"}
 (a.out/"v33c_verified_adams_rar_sweep.json").write_text(json.dumps(qa,indent=2)); print(json.dumps({"status":qa["status"],"verified_references":qa["verified_references"],"winner":winner,"gate":qa["gate"]},indent=2),flush=True)
 if not exact: raise SystemExit(6)

if __name__=="__main__": main()
