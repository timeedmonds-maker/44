from __future__ import annotations
import argparse, cv2, numpy as np, math, json, hashlib
from pathlib import Path
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree

FT=30.48; RIM_X=15*2.54; PAINT=8*FT; FTX=15*FT; FTR=6*FT; RESTRICT=4*FT; RIMR=9*2.54
EV_SHA='7092757ccbc61ebe97f38620afa9535564c7636f70647fd3f8cbfb58aa91178b'
TG_SHA='0740ccde03fd7839ae92a0dd617312472559ceedafe94b5825fe8ad512d96a4a'
LAR_CENTER=np.array([1954.0944213029006,-20.657870280048282,370.3129555117168],float)

def sha256(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def hsv_masks(im):
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
 cyan=((h>=105)&(h<=122)&(s>=50)&(v>=120)).astype(np.uint8)
 red=(((h<=18)|(h>=170))&(s>=80)&(v>=60)).astype(np.uint8)
 return cyan,red

def ridge(mask,min_dt=1.5,win=5):
 dt=distance_transform_edt(mask);mx=maximum_filter(dt,size=win,mode='constant');r=mask&(dt>=min_dt)&(dt>=mx-1e-6);y,x=np.where(r);return np.c_[x,y].astype(float)

def extract_event(im):
 cyan,red=hsv_masks(im);Y,X=np.indices(cyan.shape);out={}
 for name,roi in [('left',(X<285)&(Y<390)),('right',(X>675)&(Y<390))]:
  y,x=np.where(cyan.astype(bool)&roi);pts=np.c_[x,y].astype(np.float32)
  vx,vy,x0,y0=cv2.fitLine(pts,cv2.DIST_L1,0,.01,.01).ravel();d=np.abs((pts[:,0]-x0)*vy-(pts[:,1]-y0)*vx);pts=pts[d<3]
  rows=[]
  for yb in range(0,390,8):
   q=pts[(pts[:,1]>=yb)&(pts[:,1]<yb+8)]
   if len(q)>=3: rows.append([np.median(q[:,0]),np.median(q[:,1])])
  rows=np.asarray(rows,float);vx,vy,x0,y0=cv2.fitLine(rows.astype(np.float32),cv2.DIST_L1,0,.01,.01).ravel();d=np.abs((rows[:,0]-x0)*vy-(rows[:,1]-y0)*vx);out[name]=rows[d<1.5]
 roi=(X>280)&(X<680)&(Y<170);n,lab,stats,_=cv2.connectedComponentsWithStats((cyan.astype(bool)&roi).astype(np.uint8),8);fm=np.zeros_like(cyan,bool)
 for i in range(1,n):
  a=int(stats[i,cv2.CC_STAT_AREA]);x,y,w,h=map(int,stats[i,:4])
  if a<40: continue
  if y>=130 and w<=22 and h<=10: continue
  fm|=lab==i
 out['ft']=ridge(fm)
 roi=(X>300)&(X<670)&(Y>205)&(Y<385);n,lab,stats,_=cv2.connectedComponentsWithStats((cyan.astype(bool)&roi).astype(np.uint8),8);i=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));rp=ridge(lab==i);out['restricted']=rp[rp[:,0]<575]
 cx,cy=480.,313.;rr=np.sqrt((X-cx)**2+(Y-cy)**2);ang=(np.degrees(np.arctan2(Y-cy,X-cx))+360)%360;rim=[]
 for a0 in np.arange(0,360,4):
  sel=red.astype(bool)&(rr>42)&(rr<65)&(X>410)&(X<555)&(Y>240)&(Y<380)&(ang>=a0)&(ang<a0+4);ys,xs=np.where(sel)
  if len(xs)>=2:
   rv=np.sqrt((xs-cx)**2+(ys-cy)**2);j=np.argmin(rv);rim.append([xs[j],ys[j],a0])
 return out,np.asarray(rim,float)

def extract_target(im):
 cyan,red=hsv_masks(im);Y,X=np.indices(cyan.shape);out={}
 def line(xlo,xhi,ylo,yhi):
  y,x=np.where(cyan.astype(bool)&(X>=xlo)&(X<=xhi)&(Y>=ylo)&(Y<=yhi));pts=np.c_[x,y].astype(np.float32)
  vx,vy,x0,y0=cv2.fitLine(pts,cv2.DIST_L1,0,.01,.01).ravel();d=np.abs((pts[:,0]-x0)*vy-(pts[:,1]-y0)*vx);pts=pts[d<3];rows=[]
  for yb in range(ylo,yhi+1,8):
   q=pts[(pts[:,1]>=yb)&(pts[:,1]<yb+8)]
   if len(q)>=3: rows.append([np.median(q[:,0]),np.median(q[:,1])])
  rows=np.asarray(rows,float);vx,vy,x0,y0=cv2.fitLine(rows.astype(np.float32),cv2.DIST_L1,0,.01,.01).ravel();d=np.abs((rows[:,0]-x0)*vy-(rows[:,1]-y0)*vx);return rows[d<1.8]
 out['left']=line(220,290,120,390);out['right']=line(695,760,0,385)
 roi=np.zeros_like(cyan);roi[20:175,330:665]=1;n,lab,stats,_=cv2.connectedComponentsWithStats(cyan*roi,8);fm=np.zeros_like(cyan,bool)
 for i in range(1,n):
  x,y,w,h,a=map(int,stats[i,:5])
  if 120<=a<=400 and w>=14 and h>=10: fm|=lab==i
 out['ft']=ridge(fm,1.4);out['ft']=out['ft'][np.lexsort((out['ft'][:,0],out['ft'][:,1]))]
 ctr=np.array([484.8,299.4]);rr=np.sqrt((X-ctr[0])**2+(Y-ctr[1])**2);ang=(np.degrees(np.arctan2(Y-ctr[1],X-ctr[0]))+360)%360;rim=[]
 for a0 in np.arange(95,251,4):
  sel=red.astype(bool)&(rr>40)&(rr<60)&(X>415)&(X<525)&(Y>232)&(Y<368)&(ang>=a0)&(ang<a0+4);ys,xs=np.where(sel)
  if len(xs)>=2:
   rv=np.sqrt((xs-ctr[0])**2+(ys-ctr[1])**2);j=np.argmin(rv);rim.append([xs[j],ys[j],a0])
 return out,np.asarray(rim,float)

def split_dict(obs,offs):
 tr={};te={}
 for k,v in obs.items():
  ii=np.arange(len(v));m=((ii+offs.get(k,0))%4==0);tr[k]=v[~m,:2];te[k]=v[m,:2]
 return tr,te

def split_rim(rim,off):
 ii=np.arange(len(rim));m=((ii+off)%4==0);return rim[~m,:2],rim[m,:2]

def curve(k,n=720):
 if k=='left':
  x=np.linspace(-400,1200,n);return np.c_[x,np.full(n,-PAINT),np.zeros(n)]
 if k=='right':
  x=np.linspace(-400,1200,n);return np.c_[x,np.full(n,PAINT),np.zeros(n)]
 t=np.linspace(0,2*np.pi,n)
 if k=='ft': return np.c_[FTX+FTR*np.cos(t),FTR*np.sin(t),np.zeros(n)]
 if k=='restricted': return np.c_[RIM_X+RESTRICT*np.cos(t),RESTRICT*np.sin(t),np.zeros(n)]
 return np.c_[RIM_X+RIMR*np.cos(t),RIMR*np.sin(t),np.full(n,10*FT)]
CUR={k:curve(k,900 if k in ['left','right'] else 720) for k in ['left','right','ft','restricted','rim']}
ACTION=np.array([[x,y,z] for x in np.linspace(-30,250,8) for y in np.linspace(-180,180,9) for z in np.linspace(20,350,8)],float)

def project_full(p,P):
 C=p[:3];f=np.exp(p[3]);cx,cy=p[4:6];R=cv2.Rodrigues(p[6:9].reshape(3,1))[0];q=(R@(P-C).T).T;return np.c_[f*q[:,0]/q[:,2]+cx,f*q[:,1]/q[:,2]+cy],q[:,2]
def project_fixed(p,C,P):
 f=np.exp(p[0]);cx,cy=p[1:3];R=cv2.Rodrigues(p[3:6].reshape(3,1))[0];q=(R@(P-C).T).T;return np.c_[f*q[:,0]/q[:,2]+cx,f*q[:,1]/q[:,2]+cy],q[:,2]
def nv(obs,pred):
 d,j=cKDTree(pred).query(obs);return (pred[j]-obs).ravel()
def pstats(obs,k,p,full=True,C=None):
 pr,z=(project_full(p,CUR[k]) if full else project_fixed(p,C,CUR[k]));d=cKDTree(pr).query(obs)[0];return {'count':int(len(d)),'median_px':float(np.median(d)),'p95_px':float(np.percentile(d,95)),'max_px':float(np.max(d))}
def dstat(a,b):
 d=np.linalg.norm(a-b,axis=1);return {'median_px':float(np.median(d)),'p95_px':float(np.percentile(d,95)),'max_px':float(np.max(d))}

def init_rvec(H69,f=615,cx=471.6,cy=261.7):
 K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]]);Ki=np.linalg.inv(K);a=Ki@H69[:,0];b=Ki@H69[:,1];c=Ki@H69[:,2];lam=2/(np.linalg.norm(a)+np.linalg.norm(b));sg=-1.;r1=sg*lam*a;r2=sg*lam*b;r3=np.cross(r1,r2);R0=np.c_[r1,r2,r3];U,_,V=np.linalg.svd(R0);R=U@V
 if np.linalg.det(R)<0: U[:,-1]*=-1;R=U@V
 return cv2.Rodrigues(R)[0].ravel()

def draw(im,path,p,C=None,full=True,held=None):
 ov=im.copy();cols={'left':(255,255,0),'right':(255,0,255),'ft':(0,255,255),'restricted':(255,128,0),'rim':(0,255,0)}
 for k in cols:
  pr,z=(project_full(p,CUR[k]) if full else project_fixed(p,C,CUR[k]));q=np.round(pr).astype(int);ok=(q[:,0]>=0)&(q[:,0]<im.shape[1])&(q[:,1]>=0)&(q[:,1]<im.shape[0])
  for x,y in q[ok][::6]: cv2.circle(ov,(int(x),int(y)),1,cols[k],-1,cv2.LINE_AA)
 if held:
  for pts in held.values():
   for x,y in np.round(pts).astype(int): cv2.circle(ov,(int(x),int(y)),4,(255,255,255),1,cv2.LINE_AA)
 cv2.imwrite(str(path),ov)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--event225-frame',type=Path,required=True);ap.add_argument('--target-frame',type=Path,required=True);ap.add_argument('--floor-proof',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if sha256(a.event225_frame)!=EV_SHA: raise RuntimeError('immutable event225 SHA mismatch')
 if sha256(a.target_frame)!=TG_SHA: raise RuntimeError('immutable target SHA mismatch')
 floor=json.loads(a.floor_proof.read_text())
 if floor.get('status')!='PASS_RIGHT_ABOVE_RIM_EVENT225_FLOOR_V69' or not floor.get('permissions',{}).get('floor_homography_allowed'): raise RuntimeError('v69 floor not accepted')
 H69=np.array(floor['homography_world_to_source_px'],float);rv0=init_rvec(H69)
 ie=cv2.imread(str(a.event225_frame));oe,re=extract_event(ie);etr,ete=split_dict(oe,{'left':0,'right':1,'ft':2,'restricted':3});rtr,rte=split_rim(re,1)
 def eres(p,data=etr,rimdata=rtr):
  out=[]
  for k in ['left','right','ft','restricted']:
   pr,z=project_full(p,CUR[k]);out.append(nv(data[k],pr)*(1.3 if k=='restricted' else 1.0))
  pr,z=project_full(p,CUR['rim']);out.append(nv(rimdata,pr)*1.15)
  return np.concatenate(out)
 lo=np.r_[[-1000,-1000,100],math.log(150),100,50,[-10]*3];hi=np.r_[[1000,1000,1200],math.log(2000),850,520,[10]*3]
 starts=[]
 for C0,f in [(np.array([-9.,-1.,558.]),615),(np.array([56.,0.,210.]),231),(np.array([0.,0.,700.]),750),(np.array([-100.,100.,500.]),600)]:
  for cx,cy in [(470,260),(480,300),(520,260)]: starts.append(np.r_[C0,math.log(f),cx,cy,rv0])
 erows=[]
 for s in starts:
  o=least_squares(eres,s,bounds=(lo,hi),loss='soft_l1',f_scale=1.2,x_scale='jac',max_nfev=2200);erows.append((float(o.cost),o.x))
 erows.sort(key=lambda x:x[0]);phys=[]
 for c,p in erows:
  uv,z=project_full(p,ACTION)
  if np.all(z>0) or np.all(z<0): phys.append((c,p))
 if not phys: raise RuntimeError('no physically valid event225 root')
 eb=phys[0][1];C=eb[:3];enom=project_full(eb,ACTION)[0]
 eheld={k:pstats(ete[k],k,eb) for k in ete};eheld['rim']=pstats(rte,'rim',eb)
 ecomp=[]
 for c,p in phys:
  if c>phys[0][0]+1.0: break
  ecomp.append({'cost':c,**dstat(enom,project_full(p,ACTION)[0]),'center_shift_cm':float(np.linalg.norm(p[:3]-C))})
 rng=np.random.default_rng(7010);ep=[]
 for t in range(64):
  dat={k:v+rng.uniform(-.5,.5,v.shape) for k,v in etr.items()};rr=rtr+rng.uniform(-.5,.5,rtr.shape);o=least_squares(lambda p:eres(p,dat,rr),eb,bounds=(lo,hi),loss='soft_l1',f_scale=1.2,x_scale='jac',max_nfev=1400);p=o.x
  ep.append({'center_shift_cm':float(np.linalg.norm(p[:3]-C)),**dstat(enom,project_full(p,ACTION)[0]),'max_floor_heldout_p95_px':float(max(pstats(ete[k],k,p)['p95_px'] for k in ete)),'rim_heldout_p95_px':pstats(rte,'rim',p)['p95_px']})
 tg=cv2.imread(str(a.target_frame));ot,rt=extract_target(tg);ttr,tte=split_dict(ot,{'left':0,'right':1,'ft':2});rrtr,rrte=split_rim(rt,3);ttr['rim']=rrtr;tte['rim']=rrte
 def tres(p,data=ttr):
  out=[]
  for k in ['left','right','ft','rim']:
   pr,z=project_fixed(p,C,CUR[k]);out.append(nv(data[k],pr)*(1.25 if k=='rim' else 1.0))
  return np.concatenate(out)
 lt=np.r_[math.log(200),100,50,[-10]*3];ht=np.r_[math.log(2500),850,520,[10]*3];starts=[]
 for f in [500,560,620,750]:
  for cx,cy in [(450,240),(480,270),(500,290),(520,320)]:
   for dr in [np.zeros(3),[.04,0,0],[-.04,0,0]]: starts.append(np.r_[math.log(f),cx,cy,rv0+np.array(dr)])
 trows=[]
 for s in starts:
  o=least_squares(tres,s,bounds=(lt,ht),loss='soft_l1',f_scale=1.2,x_scale='jac',max_nfev=1600);trows.append((float(o.cost),o.x))
 trows.sort(key=lambda x:x[0]);tb=trows[0][1];tnom=project_fixed(tb,C,ACTION)[0];theld={k:pstats(tte[k],k,tb,False,C) for k in tte}
 tcomp=[]
 for c,p in trows:
  if c>trows[0][0]+1.0: break
  tcomp.append({'cost':c,**dstat(tnom,project_fixed(p,C,ACTION)[0])})
 rng=np.random.default_rng(7011);tp=[]
 for t in range(64):
  dat={k:v+rng.uniform(-.5,.5,v.shape) for k,v in ttr.items()};o=least_squares(lambda p:tres(p,dat),tb,bounds=(lt,ht),loss='soft_l1',f_scale=1.2,x_scale='jac',max_nfev=1000);p=o.x
  tp.append({**dstat(tnom,project_fixed(p,C,ACTION)[0]),'max_heldout_p95_px':float(max(pstats(tte[k],k,p,False,C)['p95_px'] for k in tte))})
 baseline=float(np.linalg.norm(C-LAR_CENTER))
 gates={'immutable_event225_and_target_frames':True,'accepted_v69_floor_input':True,'event225_all_heldout_floor_p95_at_most_2px':max(eheld[k]['p95_px'] for k in ['left','right','ft','restricted'])<=2.0,'event225_heldout_inner_rim_p95_at_most_2px':eheld['rim']['p95_px']<=2.0,'event225_competitive_physical_roots_action_p95_at_most_1px':max(x['p95_px'] for x in ecomp)<=1.0,'event225_half_pixel_center_shift_at_most_25cm':max(x['center_shift_cm'] for x in ep)<=25.0,'event225_half_pixel_action_p95_at_most_2px':max(x['p95_px'] for x in ep)<=2.0,'event225_half_pixel_heldout_floor_p95_at_most_2_5px':max(x['max_floor_heldout_p95_px'] for x in ep)<=2.5,'event225_half_pixel_heldout_rim_p95_at_most_2_5px':max(x['rim_heldout_p95_px'] for x in ep)<=2.5,'distinct_from_left_above_rim_by_at_least_50cm':baseline>=50.0,'target_all_heldout_geometry_p95_at_most_2px':max(x['p95_px'] for x in theld.values())<=2.0,'target_competitive_roots_action_p95_at_most_0_5px':max(x['p95_px'] for x in tcomp)<=0.5,'target_half_pixel_action_p95_at_most_2px':max(x['p95_px'] for x in tp)<=2.0,'target_half_pixel_heldout_p95_at_most_2_5px':max(x['max_heldout_p95_px'] for x in tp)<=2.5}
 passed=all(gates.values());status='PASS_RIGHT_ABOVE_RIM_PHYSICAL_EVENT_V70' if passed else 'FAIL_RIGHT_ABOVE_RIM_PHYSICAL_EVENT_V70';permissions={'physical_camera_center_allowed':passed,'metric_event_camera_allowed':passed,'static_novel_view_allowed':False,'replay_render_allowed':False}
 report={'schema_version':1,'status':status,'camera_label':'Right Above Rim','game_id':'0022500301','source_event':225,'target_event':'0022500527/18','physical_center_cm':C.tolist(),'baseline_to_left_above_rim_cm':baseline,'event225':{'focal_px':float(np.exp(eb[3])),'principal_point_px':eb[4:6].tolist(),'rvec':eb[6:9].tolist(),'heldout':eheld,'competitive_roots':ecomp,'perturbation_64':{'max_center_shift_cm':max(x['center_shift_cm'] for x in ep),'max_action_p95_px':max(x['p95_px'] for x in ep),'max_action_max_px':max(x['max_px'] for x in ep),'max_floor_heldout_p95_px':max(x['max_floor_heldout_p95_px'] for x in ep),'max_rim_heldout_p95_px':max(x['rim_heldout_p95_px'] for x in ep)}},'target_frame_c':{'focal_px':float(np.exp(tb[0])),'principal_point_px':tb[1:3].tolist(),'rvec':tb[3:6].tolist(),'heldout':theld,'competitive_roots':tcomp,'perturbation_64':{'max_action_p95_px':max(x['p95_px'] for x in tp),'max_action_max_px':max(x['max_px'] for x in tp),'max_heldout_p95_px':max(x['max_heldout_p95_px'] for x in tp)}},'gates':gates,'permissions':permissions,'next_gate':'Right Above Rim is camera #2 only if this report passes remotely. Static novel view/replay remain locked pending the broader multi-camera milestone.'}
 (a.out/'right_above_rim_physical_event_v70.json').write_text(json.dumps(report,indent=2));draw(ie,a.out/'right_above_rim_event225_physical_overlay_v70.png',eb,full=True,held={**ete,'rim':rte});draw(tg,a.out/'right_above_rim_target_physical_overlay_v70.png',tb,C=C,full=False,held=tte)
 print(json.dumps({'status':status,'physical_center_cm':C.tolist(),'baseline_cm':baseline,'event225_heldout':eheld,'event225_perturbation':report['event225']['perturbation_64'],'target_heldout':theld,'target_perturbation':report['target_frame_c']['perturbation_64'],'gates':gates,'permissions':permissions},indent=2))
 if not passed: raise SystemExit(2)
if __name__=='__main__': main()
