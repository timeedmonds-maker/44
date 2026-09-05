from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree

W,H=960,540
FT=30.48
RIM_X=15*2.54
PAINT_HALF=8*FT
FT_X=15*FT
FT_R=6*FT
RESTRICTED_R=4*FT
EXPECTED_SHA='7092757ccbc61ebe97f38620afa9535564c7636f70647fd3f8cbfb58aa91178b'
SEED_H=np.array([[0.,1.02,480.],[-.52,0.,345.],[0.,0.,1.]],float)
SCALE=np.array([.3,.3,80.,.3,.3,80.,.0005,.0005],float)

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def pvec(M):
    M=np.asarray(M,float)/M[2,2]
    return np.r_[M[0],M[1],M[2,:2]]
Q0=pvec(SEED_H)
def Hfrom(z):
    q=Q0+np.asarray(z,float)*SCALE
    return np.array([[q[0],q[1],q[2]],[q[3],q[4],q[5]],[q[6],q[7],1.]],float)

def project_h(M,xy):
    a=np.asarray(xy,float);q=(M@np.c_[a,np.ones(len(a))].T).T
    return q[:,:2]/q[:,2,None]

def invworld(M,uv): return project_h(np.linalg.inv(M),uv)

def ridge(mask:np.ndarray,min_dt=1.5,win=5)->np.ndarray:
    dt=distance_transform_edt(mask)
    mx=maximum_filter(dt,size=win,mode='constant')
    r=mask & (dt>=min_dt) & (dt>=mx-1e-6)
    yy,xx=np.where(r)
    return np.c_[xx,yy].astype(float)

def extract(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);hh,ss,vv=cv2.split(hsv)
    cyan=(hh>=105)&(hh<=122)&(ss>=50)&(vv>=120)
    Y,X=np.indices(cyan.shape)
    out={}
    for name,roi in [('left_lane',(X<285)&(Y<390)),('right_lane',(X>675)&(Y<390))]:
        yy,xx=np.where(cyan&roi);pts=np.c_[xx,yy].astype(np.float32)
        if len(pts)<500: raise RuntimeError(f'{name} sparse')
        vx,vy,x0,y0=cv2.fitLine(pts,cv2.DIST_L1,0,.01,.01).ravel()
        d=np.abs((pts[:,0]-x0)*vy-(pts[:,1]-y0)*vx);pts=pts[d<3.0]
        rows=[]
        for yb in range(0,390,8):
            q=pts[(pts[:,1]>=yb)&(pts[:,1]<yb+8)]
            if len(q)>=3: rows.append([np.median(q[:,0]),np.median(q[:,1])])
        rows=np.asarray(rows,float)
        vx,vy,x0,y0=cv2.fitLine(rows.astype(np.float32),cv2.DIST_L1,0,.01,.01).ravel()
        d=np.abs((rows[:,0]-x0)*vy-(rows[:,1]-y0)*vx)
        out[name]=rows[d<1.5]
    roi=(X>280)&(X<680)&(Y<170)
    n,lab,stats,_=cv2.connectedComponentsWithStats((cyan&roi).astype(np.uint8),8)
    fm=np.zeros_like(cyan,bool);rejected=[]
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA]);x,y,w,h=map(int,stats[i,:4])
        if area<40: continue
        if y>=130 and w<=22 and h<=10:
            rejected.append({'bbox':[x,y,w,h],'area':area});continue
        fm |= lab==i
    out['ft_circle']=ridge(fm)
    roi=(X>300)&(X<670)&(Y>205)&(Y<385)
    n,lab,stats,_=cv2.connectedComponentsWithStats((cyan&roi).astype(np.uint8),8)
    if n<=1: raise RuntimeError('restricted arc absent')
    i=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));rm=(lab==i)
    rp=ridge(rm);rp=rp[rp[:,0]<575]
    out['restricted']=rp
    if min(map(len,out.values()))<20: raise RuntimeError({k:len(v) for k,v in out.items()})
    return cyan,out,rejected

def split(samples):
    tr={};te={}
    for off,(k,p) in enumerate(samples.items()):
        use=np.asarray([((int(x//12)+3*int(y//12)+off)%4)!=0 for x,y in p],bool)
        tr[k]=p[use];te[k]=p[~use]
        if len(te[k])<8: raise RuntimeError(f'heldout sparse {k}')
    return tr,te

def residual(z,data):
    try:
        M=Hfrom(z);out=[]
        for k,p in data.items():
            w=invworld(M,p)
            if k=='left_lane': r=w[:,1]+PAINT_HALF
            elif k=='right_lane': r=w[:,1]-PAINT_HALF
            elif k=='ft_circle': r=np.sqrt((w[:,0]-FT_X)**2+w[:,1]**2)-FT_R
            elif k=='restricted': r=np.sqrt((w[:,0]-RIM_X)**2+w[:,1]**2)-RESTRICTED_R
            out.append(r/2.5)
        out.append(np.asarray(z)*.001)
        return np.concatenate(out)
    except Exception:
        return np.full(sum(len(v) for v in data.values())+8,1e6,float)

def solve(data,warm=None,seed=69,nstarts=12,max_nfev=18000):
    rng=np.random.default_rng(seed);starts=[np.zeros(8)]
    if warm is not None:starts.insert(0,np.asarray(warm,float))
    while len(starts)<nstarts:starts.append(rng.uniform(-.8,.8,8))
    rows=[]
    for s in starts:
        f=least_squares(lambda z:residual(z,data),s,loss='soft_l1',f_scale=2,x_scale='jac',max_nfev=max_nfev)
        rows.append((float(f.cost),np.asarray(f.x,float),Hfrom(f.x)))
    rows.sort(key=lambda r:r[0]);return rows

def dense(k,n=5000):
    if k=='left_lane':
        x=np.linspace(-300,1100,n);return np.c_[x,np.full(n,-PAINT_HALF)]
    if k=='right_lane':
        x=np.linspace(-300,1100,n);return np.c_[x,np.full(n,PAINT_HALF)]
    t=np.linspace(0,2*math.pi,n)
    if k=='ft_circle':return np.c_[FT_X+FT_R*np.cos(t),FT_R*np.sin(t)]
    return np.c_[RIM_X+RESTRICTED_R*np.cos(t),RESTRICTED_R*np.sin(t)]

def pixmetric(M,obs,k):
    d=cKDTree(project_h(M,dense(k))).query(obs)[0]
    return {'count':int(len(d)),'median_px':float(np.median(d)),'p95_px':float(np.percentile(d,95)),'max_px':float(np.max(d))}

def support(M,samples):
    out={}
    for k,p in samples.items():
        w=invworld(M,p)
        if 'lane' in k:
            xs=[float(np.percentile(w[:,0],1)),float(np.percentile(w[:,0],99))];x=np.linspace(*xs,1600)
            out[k]=np.c_[x,np.full_like(x,-PAINT_HALF if k=='left_lane' else PAINT_HALF)]
        else:
            cx,rad=(FT_X,FT_R) if k=='ft_circle' else (RIM_X,RESTRICTED_R)
            aa=np.mod(np.arctan2(w[:,1],w[:,0]-cx),2*math.pi);ss=np.sort(aa);g=np.diff(np.r_[ss,ss[0]+2*math.pi]);j=int(np.argmax(g));st=ss[(j+1)%len(ss)];en=ss[j]+(2*math.pi if ss[j]<st else 0);t=np.linspace(st,en,1600)
            out[k]=np.c_[cx+rad*np.cos(t),rad*np.sin(t)]
    return out

def shift(A,B):
    d=np.linalg.norm(A-B,axis=1);return {'median_px':float(np.median(d)),'p95_px':float(np.percentile(d,95)),'max_px':float(np.max(d))}

def draw(im,cyan,M,samples,test,path):
    ov=im.copy();ov[cyan]=(0,255,255)
    colors={'left_lane':(255,0,255),'right_lane':(255,0,255),'ft_circle':(0,165,255),'restricted':(0,0,255)}
    for k in samples:
        q=np.round(project_h(M,dense(k))).astype(int);ok=(q[:,0]>=0)&(q[:,0]<W)&(q[:,1]>=0)&(q[:,1]<H)
        for x,y in q[ok][::8]:cv2.circle(ov,(int(x),int(y)),1,colors[k],-1,cv2.LINE_AA)
        for x,y in np.round(test[k]).astype(int):cv2.circle(ov,(int(x),int(y)),3,(0,255,0),1,cv2.LINE_AA)
    cv2.imwrite(str(path),ov)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frame',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    im=cv2.imread(str(a.frame));actual=sha256(a.frame)
    if im is None or im.shape[:2]!=(H,W):raise RuntimeError('native 960x540 required')
    if actual!=EXPECTED_SHA:raise RuntimeError(f'event225 immutable SHA mismatch {actual}')
    cyan,samples,rejected=extract(im);tr,te=split(samples)
    roots=solve(tr,nstarts=20);cost,z,M=roots[0];sup=support(M,samples);nom={k:project_h(M,v) for k,v in sup.items()}
    held={k:pixmetric(M,te[k],k) for k in samples}
    low=[r for r in roots if r[0]<=cost+1.0];cluster=[]
    for c,zz,MM in low:
        cluster.append({'cost':c,**{k:shift(nom[k],project_h(MM,sup[k]))['p95_px'] for k in samples}})
    maxcluster=max(max(row[k] for k in samples) for row in cluster)
    red={k:v[:-1] for k,v in tr.items()};rred=solve(red,warm=z,nstarts=12,seed=6901)[0];Mr=rred[2]
    reduction={k:shift(nom[k],project_h(Mr,sup[k])) for k in samples}
    rng=np.random.default_rng(6902);pert=[]
    for t in range(64):
        pg={k:v+rng.uniform(-.5,.5,v.shape) for k,v in tr.items()}
        f=least_squares(lambda zz:residual(zz,pg),z,loss='soft_l1',f_scale=2,x_scale='jac',max_nfev=7000);MM=Hfrom(f.x)
        pert.append({'trial':t,**{k:shift(nom[k],project_h(MM,sup[k])) for k in samples}})
    maxpert={k:max(r[k]['p95_px'] for r in pert) for k in samples}
    whole={}
    for drop in samples:
        dat={k:v for k,v in tr.items() if k!=drop};f=least_squares(lambda zz:residual(zz,dat),z,loss='soft_l1',f_scale=2,x_scale='jac',max_nfev=10000);MM=Hfrom(f.x)
        whole[drop]={k:shift(nom[k],project_h(MM,sup[k])) for k in samples}
    gates={'immutable_native_event225_frame':True,'source_only_static_regulation_paint':True,'heldout_left_lane_p95_at_most_2px':held['left_lane']['p95_px']<=2,'heldout_right_lane_p95_at_most_2px':held['right_lane']['p95_px']<=2,'heldout_ft_circle_p95_at_most_2px':held['ft_circle']['p95_px']<=2,'heldout_restricted_arc_p95_at_most_2px':held['restricted']['p95_px']<=2,'nominal_multistart_cluster_at_most_0_05px':maxcluster<=.05,'support_reduction_all_p95_at_most_2_5px':max(v['p95_px'] for v in reduction.values())<=2.5,'half_pixel_all_p95_at_most_2_5px':max(maxpert.values())<=2.5}
    status='PASS_RIGHT_ABOVE_RIM_EVENT225_FLOOR_V69' if all(gates.values()) else 'FAIL_RIGHT_ABOVE_RIM_EVENT225_FLOOR_V69'
    permissions={'floor_homography_allowed':status.startswith('PASS_'),'physical_camera_center_allowed':False,'metric_event_camera_allowed':False,'static_novel_view_allowed':False,'replay_render_allowed':False}
    report={'schema_version':1,'status':status,'game_id':'0022500301','event_id':225,'camera_label':'Right Above Rim','frame':a.frame.name,'sha256':actual,'homography_world_to_source_px':M.tolist(),'source_extraction':{'cyan_hsv':'H 105..122, S>=50, V>=120','rejected_ft_hash_components':rejected,'restricted_arc_fixed_clean_x_max':575,'counts':{k:int(len(v)) for k,v in samples.items()},'train_counts':{k:int(len(v)) for k,v in tr.items()},'heldout_counts':{k:int(len(v)) for k,v in te.items()}},'heldout_pixel':held,'nominal_multistart_low_cost_root_count':len(low),'max_low_cost_root_projection_p95_px':maxcluster,'support_reduction':reduction,'half_pixel_max_projection_p95_px':maxpert,'whole_family_removal_diagnostic_only':whole,'gates':gates,'permissions':permissions,'next_gate':'Use the accepted event225 metric floor plus independent regulation rim/board evidence and exact Frame C transfer to prove a stable physical Right Above Rim camera centre; this floor pass alone cannot promote a metric camera.'}
    (a.out/'right_above_rim_event225_floor_v69.json').write_text(json.dumps(report,indent=2));draw(im,cyan,M,samples,te,a.out/'right_above_rim_event225_floor_overlay_v69.png')
    print(json.dumps({'status':status,'heldout':held,'maxcluster':maxcluster,'support_reduction_max':max(v['p95_px'] for v in reduction.values()),'maxpert':maxpert,'gates':gates,'permissions':permissions},indent=2))
    if not all(gates.values()):raise SystemExit(2)
if __name__=='__main__':main()
