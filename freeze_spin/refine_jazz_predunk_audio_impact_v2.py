from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def extract_audio(path: Path, sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix='.s16', delete=False) as f:
        raw_path = Path(f.name)
    try:
        subprocess.run([
            'ffmpeg','-nostdin','-y','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(sr),'-f','s16le',str(raw_path)
        ], check=True)
        x=np.fromfile(raw_path,dtype='<i2').astype(np.float32)/32768.0
    finally:
        raw_path.unlink(missing_ok=True)
    return x


def spectral_flux(x: np.ndarray, sr: int, nfft: int=512, hop: int=128):
    if len(x)<nfft*2:
        return np.empty(0),np.empty(0)
    win=np.hanning(nfft).astype(np.float32)
    frames=[]
    for i in range(0,len(x)-nfft+1,hop):
        spec=np.abs(np.fft.rfft(x[i:i+nfft]*win))
        frames.append(np.log1p(spec))
    S=np.asarray(frames,np.float32)
    freqs=np.fft.rfftfreq(nfft,1.0/sr)
    band=(freqs>=900)&(freqs<=7200)
    d=np.maximum(S[1:,band]-S[:-1,band],0.0)
    flux=d.mean(axis=1)
    # Add a small broadband impulsiveness term so rim/ball contact is favoured over sustained crowd energy.
    rms=np.sqrt(np.mean(np.exp(S[:,band]*2)-1,axis=1)+1e-9)
    dr=np.maximum(rms[1:]-rms[:-1],0.0)
    dr=dr/(np.median(dr)+1e-6)
    flux=flux/(np.median(flux)+1e-6)+0.18*dr
    med=float(np.median(flux)); mad=float(np.median(np.abs(flux-med)))+1e-6
    z=(flux-med)/(1.4826*mad)
    times=(np.arange(len(z))+1.0)*hop/sr+nfft/(2*sr)
    return times,z


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clips',type=Path,required=True)
    ap.add_argument('--sync',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--sample-rate',type=int,default=16000)
    ap.add_argument('--pre-impact-seconds',type=float,default=0.12)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)

    sync=json.loads(args.sync.read_text())
    cfg=json.loads(args.config.read_text())
    files={r['label']:r['file'] for r in cfg['angles']}
    rows={r['label']:r for r in sync['angles']}
    ref=sync['reference_angle']; provisional=float(sync['reference_freeze_time'])
    # The provisional timestamp was center-minus-0.35s. The actual rim impact should be shortly after it.
    search_lo=provisional-0.10; search_hi=provisional+0.85
    grid=np.arange(search_lo,search_hi+0.0001,0.004)
    aligned=[]; per_view=[]
    for label,row in rows.items():
        if row['confidence']=='low': continue
        p=args.clips/files[label]
        x=extract_audio(p,args.sample_rate)
        t,z=spectral_flux(x,args.sample_rate)
        if not len(t): continue
        offset=float(row['offset_seconds_vs_reference'])
        tref=t-offset
        zg=np.interp(grid,tref,z,left=np.nan,right=np.nan)
        quality=float(row['mean_incident_quality'])
        weight=float(np.clip(quality,0.15,0.95))
        aligned.append((weight,zg,label))
        # strongest local transient in reference timeline for agreement QA
        inside=(tref>=search_lo)&(tref<=search_hi)
        if inside.any():
            ii=np.where(inside)[0]; k=ii[int(np.argmax(z[ii]))]
            per_view.append({'label':label,'peak_ref_time':float(tref[k]),'peak_z':float(z[k]),'weight':weight})
    if len(aligned)<6: raise RuntimeError(f'Only {len(aligned)} usable audio views')
    A=np.stack([a[1] for a in aligned])
    weights=np.array([a[0] for a in aligned],np.float64)
    # Weighted mean of clipped robust-z novelty; clipping prevents one commentary feed dominating.
    Ac=np.clip(A,-1.0,12.0)
    valid=np.isfinite(Ac)
    num=np.nansum(Ac*weights[:,None],axis=0)
    den=np.nansum(valid*weights[:,None],axis=0)
    consensus=num/np.maximum(den,1e-6)
    # Mild temporal smoothing across ~20 ms.
    kernel=np.ones(5,np.float64)/5.0
    consensus=np.convolve(consensus,kernel,mode='same')
    k=int(np.nanargmax(consensus)); impact=float(grid[k]); peak=float(consensus[k])
    freeze=impact-float(args.pre_impact_seconds)
    supporting=[r for r in per_view if abs(r['peak_ref_time']-impact)<=0.055]
    confidence='high' if peak>=3.0 and len(supporting)>=6 else ('moderate' if peak>=2.0 and len(supporting)>=4 else 'low')
    payload={
        'method':'multi-camera synchronized high-frequency spectral-flux consensus',
        'reference_angle':ref,
        'provisional_reference_time':provisional,
        'search_window_reference_seconds':[search_lo,search_hi],
        'estimated_dunk_impact_reference_time':impact,
        'recommended_predunk_reference_time':freeze,
        'pre_impact_lead_seconds':args.pre_impact_seconds,
        'consensus_peak_score':peak,
        'usable_views':len(aligned),
        'supporting_peak_views_within_55ms':len(supporting),
        'confidence':confidence,
        'per_view_peaks':sorted(per_view,key=lambda r:r['peak_ref_time']),
        'supporting_views':[r['label'] for r in supporting],
        'policy':'Audio identifies a common physical impact transient; freeze is deliberately placed ~0.12 s earlier. This is a temporal cue, not a visual ball/rim proof.'
    }
    args.out.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__': main()
