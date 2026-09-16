#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tracks',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--context',type=Path,required=True);ap.add_argument('--roster',type=Path,required=True);ap.add_argument('--nbacv-src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.tracks)
 if 'raw_track_id' not in d: raise RuntimeError('raw_track_id missing')
 d=d.copy(); d['stitched_track_id']=d['track_id']; d['track_id']=d['raw_track_id'].astype(int)
 raw=a.out/'tracks_raw_identity_input.csv';d.to_csv(raw,index=False)
 tool=Path(__file__).resolve().parents[1]/'tools'/'adams_event_track_identity_ocr.py'
 subprocess.run(['python',str(tool),'--tracks',str(raw),'--source',str(a.source),'--context',str(a.context),'--roster',str(a.roster),'--nbacv-src',str(a.nbacv_src),'--out',str(a.out/'ocr')],check=True)
 print(a.out/'ocr'/'track_identity.csv')
if __name__=='__main__':main()
