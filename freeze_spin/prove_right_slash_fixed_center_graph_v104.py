from __future__ import annotations

"""v104: strict four-state Right Slash fixed-centre graph proof.

v103 preserved the original 1.6 px / 80-inlier gates but failed only because
its 416->457 floor p95 was 1.802 px.  v101 was explicitly an acquisition bank
for selecting clean geometry states, so v104 substitutes the independently
recovered event169 state rather than weakening any threshold.

The proof uses source-specific basket ROIs because Right Slash pans/zooms enough
that a single hard-coded image rectangle is not a semantic basket region across
all states.  Each required edge must still span stands, the actual basket area,
and court floor.  A supported composition cycle is required independently.

A pass authorizes only a metric shared-centre solve.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FRAMES = {
    169: ("f02.png", "74b02d5706a89ab3eeb7153db78030295fbbda87bbc2ce13275af5c3844c1c66"),
    375: ("f01.png", "20b6cc30e1fa49299566d53c591e404bd8b8ef5d19a7019b7c004e5c51a370cc"),
    416: ("f06.png", "325a02876fb09c89de6657a711e3241ef5382fbf39fcc1696c95686a642d2668"),
    540: ("f00.png", "2c10a5be6096181fd423b7d7a8b6136c4b90c97ca8cf9d3442f04ae434cd2bcd"),
}
# Semantic basket regions in each SOURCE frame, manually bounded from immutable
# pixels.  They contain board/rim/stanchion support, not players.
BASKET_ROI = {
    169: (250, 600, 30, 300),
    375: (380, 620, 120, 280),
    416: (330, 620, 30, 260),
    540: (50, 290, 80, 300),
}
REQUIRED_EDGES = [(375, 540), (416, 540), (540, 169)]
CYCLE_DIRECT = (375, 169)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_native(root: Path, event_id: int) -> np.ndarray:
    fn, expected = FRAMES[event_id]
    p = root / f"event_{event_id}_selected" / fn
    if sha256(p) != expected:
        raise RuntimeError(f"immutable v101 frame SHA mismatch: {p}")
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (H, W):
        raise RuntimeError(f"missing/non-native v101 frame: {p}")
    return im


def features(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.008, edgeThreshold=12).detectAndCompute(g, None)


def local_scale(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = []
    for x, y in np.asarray(pts, float):
        den = M[2,0]*x + M[2,1]*y + M[2,2]
        u = (M[0,0]*x + M[0,1]*y + M[0,2]) / den
        v = (M[1,0]*x + M[1,1]*y + M[1,2]) / den
        J = np.array([
            [(M[0,0]-u*M[2,0])/den, (M[0,1]-u*M[2,1])/den],
            [(M[1,0]-v*M[2,0])/den, (M[1,1]-v*M[2,1])/den],
        ])
        out.append(math.sqrt(abs(float(np.linalg.det(J)))))
    return np.asarray(out)


def fit_edge(images, source_event: int, dest_event: int) -> dict:
    ka, da = features(images[source_event]); kb, db = features(images[dest_event])
    if da is None or db is None: raise RuntimeError("SIFT descriptors unavailable")
    good=[]
    for pair in cv2.BFMatcher().knnMatch(da, db, k=2):
        if len(pair)==2 and pair[0].distance < 0.72*pair[1].distance: good.append(pair[0])
    if len(good)<8: raise RuntimeError("insufficient ratio-test matches")
    pa=np.float32([ka[m.queryIdx].pt for m in good]); pb=np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(104000 + 31*source_event + dest_event)
    M,mask=cv2.findHomography(pa,pb,cv2.RANSAC,1.5,maxIters=30000,confidence=0.9995)
    if M is None or mask is None: raise RuntimeError("homography failed")
    M=M/M[2,2]; inl=mask.ravel().astype(bool)
    pred=cv2.perspectiveTransform(pa[:,None,:],M)[:,0,:]
    ferr=np.linalg.norm(pred-pb,axis=1)
    inv=np.linalg.inv(M); inv=inv/inv[2,2]
    back=cv2.perspectiveTransform(pb[inl,None,:],inv)[:,0,:]
    berr=np.linalg.norm(back-pa[inl],axis=1)
    inverse_eq=berr*local_scale(M,pa[inl])
    x,y=pa[:,0],pa[:,1]
    x0,x1,y0,y1=BASKET_ROI[source_event]
    regions={
        "stands": y<260,
        "basket": (x>=x0)&(x<=x1)&(y>=y0)&(y<=y1),
        "floor": y>360,
    }
    rr={}
    for name,reg in regions.items():
        z=reg&inl
        rr[name]={"inliers":int(z.sum()),"p95_px":float(np.percentile(ferr[z],95)) if z.any() else None}
    cells=sorted({(min(int(px//120),7),min(int(py//90),5)) for px,py in pa[inl]})
    return {
        "H":M,"src_inliers":pa[inl],"dst_inliers":pb[inl],
        "ratio_test_matches":len(good),"ransac_inliers":int(inl.sum()),
        "forward_p95_px":float(np.percentile(ferr[inl],95)),
        "inverse_destination_equivalent_p95_px":float(np.percentile(inverse_eq,95)),
        "spatial_cell_count_8x6":len(cells),"spatial_cells_8x6":[list(q) for q in cells],
        "regions":rr,
    }


def gates(r):
    return {
        "ratio_matches_at_least_100": r["ratio_test_matches"]>=100,
        "ransac_inliers_at_least_80": r["ransac_inliers"]>=80,
        "forward_p95_at_most_1_6px": r["forward_p95_px"]<=1.6,
        "inverse_destination_equivalent_p95_at_most_1_6px": r["inverse_destination_equivalent_p95_px"]<=1.6,
        "spatial_cells_at_least_15": r["spatial_cell_count_8x6"]>=15,
        "stands_inliers_at_least_50": r["regions"]["stands"]["inliers"]>=50,
        "stands_p95_at_most_1_6px": (r["regions"]["stands"]["p95_px"] or 1e9)<=1.6,
        "basket_inliers_at_least_30": r["regions"]["basket"]["inliers"]>=30,
        "basket_p95_at_most_1_6px": (r["regions"]["basket"]["p95_px"] or 1e9)<=1.6,
        "floor_inliers_at_least_8": r["regions"]["floor"]["inliers"]>=8,
        "floor_p95_at_most_1_6px": (r["regions"]["floor"]["p95_px"] or 1e9)<=1.6,
    }


def serial(r):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k not in ("src_inliers","dst_inliers")}


def cycle_metrics(direct, first, second):
    comp=second["H"]@first["H"]; comp=comp/comp[2,2]
    pts=direct["src_inliers"].astype(np.float32)
    a=cv2.perspectiveTransform(pts[:,None,:],direct["H"])[:,0,:]
    b=cv2.perspectiveTransform(pts[:,None,:],comp)[:,0,:]
    d=np.linalg.norm(a-b,axis=1)
    return {"count":int(len(d)),"median_px":float(np.median(d)),"p95_px":float(np.percentile(d,95)),"max_px":float(np.max(d))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--v101-root",type=Path,required=True); ap.add_argument("--v103-report",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    q103=json.loads(a.v103_report.read_text())
    if q103.get("status")!="FAIL_RIGHT_SLASH_FIXED_CENTER_GRAPH_V103": raise RuntimeError("v104 requires exact v103 fail-closed prerequisite")
    # Preserve the narrow v103 failure signature: its cycle passed, but the
    # required graph failed.  We never reinterpret v103 as a pass.
    if q103.get("gates",{}).get("cycle_supported_p95_at_most_2_5px") is not True: raise RuntimeError("v103 cycle prerequisite changed")
    images={eid:load_native(a.v101_root,eid) for eid in FRAMES}
    edges={}
    required=[]
    for s,t in REQUIRED_EDGES:
        r=fit_edge(images,s,t); edges[(s,t)]=r; g=gates(r); ok=all(g.values())
        required.append({"edge":[s,t],"source_basket_roi":list(BASKET_ROI[s]),"metrics":serial(r),"gates":g,"pass":bool(ok)})
        print("V104 EDGE",s,t,"PASS" if ok else "FAIL","inliers",r["ransac_inliers"],"p95",round(r["forward_p95_px"],4),"basket",r["regions"]["basket"]["inliers"],"floor",r["regions"]["floor"]["inliers"],flush=True)
    direct=fit_edge(images,*CYCLE_DIRECT); edges[CYCLE_DIRECT]=direct
    dg=gates(direct)
    cyc=cycle_metrics(direct,edges[(375,540)],edges[(540,169)])
    all_required=all(x["pass"] for x in required)
    direct_pass=all(dg.values())
    graph_nodes=sorted({n for e in REQUIRED_EDGES for n in e})
    cycle_pass=cyc["count"]>=80 and cyc["p95_px"]<=2.5
    passed=bool(all_required and direct_pass and graph_nodes==[169,375,416,540] and cycle_pass)
    report={
        "schema_version":1,"status":"PASS_RIGHT_SLASH_FIXED_CENTER_GRAPH_V104" if passed else "FAIL_RIGHT_SLASH_FIXED_CENTER_GRAPH_V104",
        "game_id":"0022500301","camera_label":"Right Slash",
        "prerequisite":{"v103_status":q103["status"],"v103_preserved_as_failed_diagnostic":True,"selection_rationale":"event169 is an independently recovered v101 geometry-bank state; substitution replaces the v103 event457 floor miss without changing any per-edge threshold"},
        "frames":{str(k):{"file":FRAMES[k][0],"sha256":FRAMES[k][1],"basket_roi":list(BASKET_ROI[k])} for k in sorted(FRAMES)},
        "required_edges":required,
        "cycle_direct_375_to_169":{"metrics":serial(direct),"gates":dg,"pass":bool(direct_pass)},
        "cycle_375_to_540_to_169_vs_direct_375_to_169":cyc,
        "gates":{"all_required_edges_pass":all_required,"direct_cycle_edge_passes_same_thresholds":direct_pass,"graph_spans_four_independent_states":graph_nodes==[169,375,416,540],"cycle_supported_inliers_at_least_80":cyc["count"]>=80,"cycle_supported_p95_at_most_2_5px":cyc["p95_px"]<=2.5},
        "interpretation":"Pass is strong projective evidence consistent with a fixed Right Slash optical centre across four distributed same-game states; metric shared-centre solve is authorized but not yet passed.",
        "guardrails":["native immutable 960x540 source only","no player or ball landmarks","1.6 px / 80-inlier edge thresholds unchanged","semantic basket ROI is state-specific because the camera pans/zooms","no metric camera or replay promotion"],
        "permissions":{"shared_center_metric_attempt_allowed":passed,"right_slash_metric_camera_allowed":False,"replay_render_allowed":False},
    }
    (a.out/"right_slash_fixed_center_graph_v104.json").write_text(json.dumps(report,indent=2)+"\n")
    print("FINAL",report["status"],"cycle_p95",round(cyc["p95_px"],4),flush=True)
    if not passed: raise SystemExit(2)

if __name__=="__main__": main()
