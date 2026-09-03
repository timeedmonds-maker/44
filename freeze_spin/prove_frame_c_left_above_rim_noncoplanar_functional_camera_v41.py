from __future__ import annotations
import argparse,json,math,os,re
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
W,H=960,540; FT=30.48; IN=2.54
EVENTS=[40,220,440,620]
RIM_ROI={40:[450,505,120,145],220:[465,510,155,175],440:[455,505,180,200],489:[455,505,160,190],620:[455,505,165,185]}
TARGET_RIM_CENTER=np.array([479.0,174.0],float)

def K(f,pp):return np.array([[f,0,pp[0]],[0,f,pp[1]],[0,0,1.]],float)
def normh(h):h=np.asarray(h,float);return h/h[2,2]
def terr(h,p,q):
 z=cv2.perspectiveTransform(p[:,None,:].astype(np.float32),h.astype(float))[:,0];return np.linalg.norm(z-q,axis=1)
def sift(a,b):
 s=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015);ka,da=s.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);kb,db=s.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
 if da is None or db is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
 g=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.72*n.distance];d={}
 for m in g:
  if m.trainIdx not in d or m.distance<d[m.trainIdx].distance:d[m.trainIdx]=m
 g=list(d.values());return np.float32([ka[m.queryIdx].pt for m in g]),np.float32([kb[m.trainIdx].pt for m in g])
def core(x):return (x[:,0]>.2*W)&(x[:,0]<.8*W)&(x[:,1]>.48*H)&(x[:,1]<.98*H)
def audit(src,tgt):
 a=cv2.imread(str(src));b=cv2.imread(str(tgt));p,q=sift(a,b)
 if len(p)<30:return None
 tr=(((p[:,1]<.46*H)|(p[:,0]<.14*W)|(p[:,0]>.86*W))&((q[:,1]<.46*H)|(q[:,0]<.14*W)|(q[:,0]>.86*W))&~core(p)&~core(q));wh=~tr&~core(p)&~core(q)
 if tr.sum()<24 or wh.sum()<10:return None
 hm,m=cv2.findHomography(p[tr],q[tr],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
 if hm is None:return None
 ii=m.ravel().astype(bool);pi,qi=p[tr][ii],q[tr][ii];te=terr(hm,pi,qi);we=terr(hm,p[wh],q[wh])
 if len(pi)<24 or np.percentile(te,95)>1.5 or np.median(we)>2.5 or np.percentile(we,90)>4:return None
 keep=we<=4
 return {'H':hm,'p':pi.astype(float),'q':qi.astype(float),'pw':p[wh][keep].astype(float),'qw':q[wh][keep].astype(float),'train_p95':float(np.percentile(te,95))}
def best(paths,tgt):
 rr=[]
 for p in paths:
  r=audit(p,tgt)
  if r is not None:r['source']=str(p);rr.append(r)
 if not rr:raise RuntimeError('no same-camera sample')
 rr.sort(key=lambda z:(len(z['p']),-z['train_p95']),reverse=True);return rr[0]
def decomp(h,cx,cy):
 h=np.asarray(h,float);h1,h2,h3=h[:,0],h[:,1],h[:,2];a1=np.array([h1[0]-cx*h1[2],h1[1]-cy*h1[2]]);a2=np.array([h2[0]-cx*h2[2],h2[1]-cy*h2[2]]);c=[]
 if abs(h1[2]*h2[2])>1e-12:
  x=-(a1@a2)/(h1[2]*h2[2]);c += [x] if x>0 else []
 d=h1[2]**2-h2[2]**2
 if abs(d)>1e-12:
  x=-(a1@a1-a2@a2)/d;c += [x] if x>0 else []
 if not c:raise RuntimeError('decomp')
 f=math.sqrt(float(np.median(c)));ki=np.linalg.inv(K(f,[cx,cy]));q1,q2,q3=ki@h1,ki@h2,ki@h3;l=2/(np.linalg.norm(q1)+np.linalg.norm(q2));r0=np.c_[l*q1,l*q2,np.cross(l*q1,l*q2)];u,_,v=np.linalg.svd(r0);R=u@v
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
 rv,_=cv2.Rodrigues(R);C=-R.T@(l*q3);return f,C,rv.ravel()
def grid():return np.array([[x,y,0.] for x in np.linspace(-4*FT,28*FT,9) for y in np.linspace(-25*FT,25*FT,11)])
GRID=grid()
def hobs(h):
 z=(h@np.c_[GRID[:,:2],np.ones(len(GRID))].T).T;uv=z[:,:2]/z[:,2,None];m=(z[:,2]>0)&(uv[:,0]>20)&(uv[:,0]<940)&(uv[:,1]>20)&(uv[:,1]<520);return GRID[m],uv[m]
def project(C,pp,par,P):
 f=np.exp(par[0]);R,_=cv2.Rodrigues(par[1:].reshape(3,1));ca=(R@(P-C).T).T;return np.c_[f*ca[:,0]/ca[:,2]+pp[0],f*ca[:,1]/ca[:,2]+pp[1]]
th=np.linspace(0,2*math.pi,721);RIM=np.c_[15*IN+9*IN*np.cos(th),9*IN*np.sin(th),np.full_like(th,10*FT)];RIM_CENTER=np.array([[15*IN,0,10*FT]],float)
def nearvec(pred,obs):
 d=((pred[None,:,:]-obs[:,None,:])**2).sum(2);return (pred[np.argmin(d,axis=1)]-obs).ravel()
def rim_pixels(im,e):
 x0,x1,y0,y1=RIM_ROI[e];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[y0:y1,x0:x1];hh,ss,vv=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2];m=(((hh<=20)|(hh>=170))&(ss>=70)&(vv>=60)).astype(np.uint8);ys,xs=np.where(m)
 if len(xs)<25:raise RuntimeError(f'rim pixels sparse event {e}')
 pts=np.c_[xs+x0,ys+y0].astype(float);bins={}
 for p in pts:bins.setdefault((int(p[0]//2),int(p[1]//2)),[]).append(p)
 return np.array([np.mean(v,axis=0) for v in bins.values()])
def solve_center(Hs,rims,evs,seed_pp=(458.,450.),warm=None,max_nfev=12000):
 ob={e:hobs(Hs[e]) for e in evs};pars=[]
 for e in evs:
  d=decomp(Hs[e],*seed_pp);pars.append([math.log(d[0]),*d[2]])
 x0=np.r_[[1970.,-20.,377.],seed_pp,np.array(pars).ravel()] if warm is None else warm
 def fun(x):
  C=x[:3];pp=x[3:5];ps=x[5:].reshape(len(evs),4);out=[]
  for i,e in enumerate(evs):
   P,O=ob[e];out += [(project(C,pp,ps[i],P)-O).ravel(),nearvec(project(C,pp,ps[i],RIM),rims[e])]
  return np.concatenate(out)
 lo=np.r_[[-5000,-3000,50],[100,50],np.tile(np.r_[[math.log(250)],[-10,-10,-10]],len(evs))];hi=np.r_[[5000,3000,1500],[850,520],np.tile(np.r_[[math.log(2500)],[10,10,10]],len(evs))]
 o=least_squares(fun,x0,bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=max_nfev);return o.x,float(np.sqrt(np.mean(fun(o.x)**2)))
def curves():
 bx=15*IN;r3=23.75*FT;yc=22*FT;xt=bx+math.sqrt(r3*r3-yc*yc);fx=15*FT;rf=6*FT;n=900;y=np.linspace(-yc,yc,n);out={'three_point_arc':np.c_[bx+np.sqrt(r3*r3-y*y),y,np.zeros(n)]};t=np.linspace(-math.pi/2,math.pi/2,n);out['free_throw_front_semicircle']=np.c_[fx+rf*np.cos(t),rf*np.sin(t),np.zeros(n)];x=np.linspace(-15*FT,xt,n);out['left_corner_three_straight']=np.c_[x,np.full(n,-yc),np.zeros(n)];out['right_corner_three_straight']=np.c_[x,np.full(n,yc),np.zeros(n)];y=np.linspace(-8*FT,8*FT,n);out['free_throw_line']=np.c_[np.full(n,fx),y,np.zeros(n)];return out
CURVES=curves()
def nearest_res(pred,o):return pred[np.argmin(np.linalg.norm(pred-o,axis=1))]-o
def clip_pair(src,target,meta):
 r=audit(src,target)
 if r is None:return None
 r['name']=Path(src).name;r['relative_seconds']=float(meta[r['name']]['relative_to_freeze_seconds']);return r
def nearest_rot(Hm,pp,ft,fs):
 M=np.linalg.inv(K(ft,pp))@Hm@K(fs,pp);d=np.linalg.det(M);M=M/np.cbrt(abs(d));M=-M if d<0 else M;u,_,v=np.linalg.svd(M);R=u@v
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
 return cv2.Rodrigues(R)[0].ravel()
def pair_project(p,core,sp):
 ft=np.exp(core[3]);pp=core[4:6];fs=np.exp(sp[0]);R,_=cv2.Rodrigues(sp[1:].reshape(3,1));hm=K(ft,pp)@R@np.linalg.inv(K(fs,pp));ph=np.c_[p,np.ones(len(p))];qh=(hm@ph.T).T;return qh[:,:2]/qh[:,2,None]
def init_source(pr,core):
 pp=core[4:6];ft=np.exp(core[3]);x0=np.r_[math.log(ft),nearest_rot(pr['H'],pp,ft,ft)];lo=np.r_[math.log(150),[-10]*3];hi=np.r_[math.log(4000),[10]*3]
 o=least_squares(lambda x:(pair_project(pr['p'],core,x)-pr['q']).ravel(),x0,bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=2500);return o.x
def action_grid():return np.array([[x,y,z] for x in np.linspace(-30,250,8) for y in np.linspace(-180,180,9) for z in np.linspace(20,350,8)],float)
ACTION=action_grid()

def json_safe(value):
 if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.bool_):return bool(value)
 if isinstance(value,np.integer):return int(value)
 if isinstance(value,np.floating):return float(value)
 return value

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target-frame',type=Path,required=True);ap.add_argument('--same-game-samples',type=Path,required=True);ap.add_argument('--target-clip-samples',type=Path,required=True);ap.add_argument('--target-manifest',type=Path,required=True);ap.add_argument('--floor-proof',type=Path,required=True);ap.add_argument('--wide-court',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--perturbation-trials',type=int,default=12);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 floor=json.loads(a.floor_proof.read_text());wide=json.loads(a.wide_court.read_text());Ht=normh(np.array(floor['floor_homography_world_to_image'],float));sel={};Hs={};rims={}
 for e in EVENTS:
  r=best(sorted(a.same_game_samples.glob(f'Left_Above_Rim__event{e:04d}__s*.png')),a.target_frame);sel[e]=r;Hs[e]=normh(np.linalg.inv(r['H'])@Ht);rims[e]=rim_pixels(cv2.imread(r['source']),e)
 roots=[]
 for pp in [(458,450),(456,249),(480,370),(430,450),(500,420)]:roots.append(solve_center(Hs,rims,EVENTS,seed_pp=pp)[0])
 base=roots[0];C=base[:3];gamepp=base[3:5];center_spread=max(np.linalg.norm(x[:3]-y[:3]) for i,x in enumerate(roots) for y in roots[i+1:]);gamepp_spread=max(np.linalg.norm(x[3:5]-y[3:5]) for i,x in enumerate(roots) for y in roots[i+1:])
 loo={}
 for omit in EVENTS:
  ev=[e for e in EVENTS if e!=omit];x,_=solve_center(Hs,rims,ev);loo[str(omit)]={'center_shift_cm':float(np.linalg.norm(x[:3]-C)),'pp_shift_px':float(np.linalg.norm(x[3:5]-gamepp))}
 obs={k:np.array(v,float) for k,v in wide['observations_px'].items()};held={k:set(v) for k,v in wide['held_out_indices'].items()};target_image=cv2.imread(str(a.target_frame));target_rim=rim_pixels(target_image,489)
 def metric_res(core,obsx=obs,rimc=TARGET_RIM_CENTER,rimobs=target_rim):
  pp=core[4:6];par=np.r_[core[3],core[:3]];z=[]
  for n,oo in obsx.items():
   pred=project(C,pp,par,CURVES[n])
   for i,p in enumerate(oo):
    if i not in held[n]:z.extend(nearest_res(pred,p))
  z.extend(((project(C,pp,par,RIM_CENTER)-np.asarray(rimc).reshape(1,2))*3).ravel());z.extend(nearvec(project(C,pp,par,RIM),rimobs));return np.asarray(z)
 d=decomp(Ht,459,430);metric0=np.r_[d[2],math.log(d[0]),459.,430.]
 man=json.loads(a.target_manifest.read_text());meta={x['file']:x for x in man['samples']};pairs=[]
 for p in sorted(a.target_clip_samples.glob('Left_Above_Rim_target_event__*.png')):
  q=clip_pair(p,a.target_frame,meta)
  if q is not None:pairs.append(q)
 est=[]
 for pr in pairs:
  sp=init_source(pr,metric0);est.append((pr,sp,float(np.exp(sp[0]))))
 med=float(np.median([x[2] for x in est]));sett=[x[0] for x in est if abs(x[2]-med)/med<=.01]
 if len(sett)<6:raise RuntimeError('settled static support')
 lo=np.r_[[-10]*3,math.log(250),100,50];hi=np.r_[[10]*3,math.log(2500),850,520]
 for _ in sett:lo=np.r_[lo,math.log(150),[-10]*3];hi=np.r_[hi,math.log(4000),[10]*3]
 def jseed(core):return np.concatenate([core]+[init_source(pr,core) for pr in sett])
 def jres(x):
  co=x[:6];out=[metric_res(co)];off=6
  for pr in sett:out.append((pair_project(pr['p'],co,x[off:off+4])-pr['q']).ravel());off+=4
  return np.concatenate(out)
 eroot=[]
 for dx,dy in [(0,0),(-20,0),(20,0),(0,-20),(0,20)]:
  c=metric0.copy();c[4]+=dx;c[5]+=dy;o=least_squares(jres,jseed(c),bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=8000);eroot.append(o.x)
 eroot.sort(key=lambda x:float(np.mean(jres(x)**2)));bestx=eroot[0];core0=bestx[:6]
 pp=core0[4:6];par=np.r_[core0[3],core0[:3]];tr=[];ho=[]
 for n,oo in obs.items():
  pred=project(C,pp,par,CURVES[n])
  for i,p in enumerate(oo):
   e=float(np.min(np.linalg.norm(pred-p,axis=1)));(ho if i in held[n] else tr).append(e)
 rim_pred=project(C,pp,par,RIM);rimerr=float(np.linalg.norm(project(C,pp,par,RIM_CENTER)-TARGET_RIM_CENTER.reshape(1,2)));rim_contour=np.linalg.norm(nearvec(rim_pred,target_rim).reshape(-1,2),axis=1)
 rim_overlay=target_image.copy()
 for x,y in target_rim:cv2.circle(rim_overlay,(int(round(x)),int(round(y))),2,(255,255,0),-1,cv2.LINE_AA)
 q=np.round(rim_pred).astype(int);ok=(q[:,0]>=0)&(q[:,0]<W)&(q[:,1]>=0)&(q[:,1]<H)
 for x,y in q[ok]:cv2.circle(rim_overlay,(int(x),int(y)),1,(255,0,255),-1,cv2.LINE_AA)
 cv2.imwrite(str(a.out/'left_above_rim_target_rim_contour_v41.png'),rim_overlay)
 static=[];off=6
 for pr in sett:
  sp=bestx[off:off+4];off+=4;ew=np.linalg.norm(pair_project(pr['pw'],core0,sp)-pr['qw'],axis=1);static.append({'frame':pr['name'],'p95_px':float(np.percentile(ew,95)),'median_px':float(np.median(ew))})
 fe=[]
 for i,x in enumerate(eroot):
  for j,y in enumerate(eroot[i+1:],i+1):
   ua=project(C,x[4:6],np.r_[x[3],x[:3]],ACTION);ub=project(C,y[4:6],np.r_[y[3],y[:3]],ACTION);m=(ua[:,0]>0)&(ua[:,0]<W)&(ua[:,1]>0)&(ua[:,1]<H)&(ub[:,0]>0)&(ub[:,0]<W)&(ub[:,1]>0)&(ub[:,1]<H);dd=np.linalg.norm(ua[m]-ub[m],axis=1);fe.append({'i':i,'j':j,'p95_px':float(np.percentile(dd,95)),'max_px':float(dd.max())})
 pp_root=max(np.linalg.norm(x[4:6]-y[4:6]) for i,x in enumerate(eroot) for y in eroot[i+1:]);fe95=max(x['p95_px'] for x in fe);femax=max(x['max_px'] for x in fe)
 gates={'center_root_spread':center_spread<=1.0,'center_loo':max(x['center_shift_cm'] for x in loo.values())<=15.0,'target_floor_heldout':float(np.percentile(ho,95))<=1.5,'target_rim_center':rimerr<=1.0,'target_rim_contour_support':len(target_rim)>=12,'target_rim_contour_p95':float(np.percentile(rim_contour,95))<=2.5,'target_static_heldout':max(x['p95_px'] for x in static)<=3.0,'functional_root_p95':fe95<=0.5,'functional_root_max':femax<=0.75}
 passed=all(gates.values())
 rep=json_safe({'schema_version':1,'status':'PASS_NONCOPLANAR_FUNCTIONAL_CAMERA_V41' if passed else 'FAIL_NONCOPLANAR_FUNCTIONAL_CAMERA_V41','game_id':'0022500301','event_id':489,'camera_label':'Left Above Rim','method':'independent same-game v35 floor transfers + regulation rim-circle pixels solve physical centre; held-out Frame C raw wide-court + rim-centre + full source-pixel rim contour + settled static background; multistart accepted only if functionally sub-pixel equivalent across 3D action volume','independent_events':EVENTS,'selected_samples':{str(e):Path(sel[e]['source']).name for e in EVENTS},'camera_center_cm':C.tolist(),'game_level_pp_diagnostic_px':gamepp.tolist(),'center_multistart_max_cm':center_spread,'game_pp_multistart_max_px':gamepp_spread,'leave_one_event_out':loo,'target_camera':{'rvec':core0[:3].tolist(),'focal_px':float(np.exp(core0[3])),'principal_point_px':core0[4:6].tolist(),'parameter_pp_root_spread_px':float(pp_root)},'target_metric':{'floor_train_p95_px':float(np.percentile(tr,95)),'floor_heldout_p95_px':float(np.percentile(ho,95)),'rim_center_error_px':rimerr,'rim_contour_source_pixel_count':len(target_rim),'rim_contour_median_px':float(np.median(rim_contour)),'rim_contour_p95_px':float(np.percentile(rim_contour,95)),'rim_contour_max_px':float(np.max(rim_contour))},'settled_static':{'count':len(sett),'median_initial_source_focal_px':med,'heldout':static},'functional_root_equivalence':{'action_volume_cm':{'x':[-30,250],'y':[-180,180],'z':[20,350]},'max_p95_px':fe95,'max_px':femax,'pairs':fe},'gates':gates,'permissions':{'physical_camera_center_allowed':passed,'metric_event_camera_allowed':passed,'replay_render_allowed':False},'retired_constraint':'legacy backboard inner-target pixels are diagnostic only and excluded from v41 fit'})
 (a.out/'left_above_rim_noncoplanar_functional_camera_v41.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2));raise SystemExit(0 if passed else 2)
if __name__=='__main__':main()
