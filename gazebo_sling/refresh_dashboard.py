#!/usr/bin/python3
"""Refresh only telemetry regions from the same run's causal, timestamped logs.

Retains recorded Gazebo viewport pixels, labels, resolution and frame rate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as S
import cv2
import numpy as np
from presentation import Presentation

SKIPPED = {}
def rows(path):
    with path.open(errors='replace') as f:lines=f.readlines()
    SKIPPED[path.name]=sum('\0' in line or '\ufffd' in line for line in lines)
    return list(csv.DictReader(line for line in lines if '\0' not in line and '\ufffd' not in line))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ffmpeg',required=True);args=p.parse_args();root=args.root
    cfg=json.loads((root/'runtime.json').read_text());metrics=json.loads((root/'gazebo_metrics.json').read_text());execution=metrics['execution']
    output=root/'gazebo_transport_data.mp4'
    if output.exists():raise SystemExit('Output exists; preserve previous video')
    load=[r for r in rows(root/'gazebo_states.csv') if r['model']=='payload'];lt=np.array([float(r['time_s']) for r in load])
    cables=rows(root/'cables.csv');ct=np.array([float(r['time_s']) for r in cables])
    estimates=rows(root/'estimates.csv')
    valid=[]
    for row in estimates:
        try:
            assert None not in row and row.get('uav_id') in ['1','2','3'] and row.get('armed') in ['True','False']
            float(row['time_s'])
            valid.append(row)
        except (AssertionError,ValueError,TypeError):SKIPPED['estimates.csv']+=1
    er=[sorted([r for r in valid if r['uav_id']==str(i)],key=lambda r:float(r['time_s'])) for i in range(1,4)];et=[np.array([float(r['time_s']) for r in rlist]) for rlist in er]
    commands=rows(root/'commands.csv');clock=np.array([float(r['time_s']) for r in commands])
    def previous(times,t):return int(np.searchsorted(times,t,side='right')-1)
    source=root/'gazebo_transport.mp4';cap=cv2.VideoCapture(str(source));view=Presentation(cfg['payload_reference'])
    encoder=subprocess.Popen([args.ffmpeg,'-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate','10','-i','-','-an','-c:v','libx264','-preset','ultrafast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    count=0
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            stamp=execution['video_start_sim_s']+count/10
            li=previous(lt,stamp);ci=previous(ct,stamp);ki=previous(clock,stamp)
            stage=commands[ki]['stage'] if ki>=0 else 'initializing'
            truth=None;truth_stamp=None;cable=None
            if li>=0:
                row=load[li];truth_stamp=lt[li]
                pose=S(position=S(**{k:float(row[k+'_m']) for k in ['x','y','z']}),orientation=S(**{k:float(row['q'+k]) for k in ['x','y','z','w']}))
                truth=S(name=['payload'],pose=[pose])
            if ci>=0:
                row=cables[ci];cable=[ct[ci]]+[float(row[k+'_'+str(i)]) for i in range(1,4) for k in ['tension_N','length_m','radial_speed_m_s']]
            states=[]
            for rlist,times in zip(er,et):
                index=previous(times,stamp);row=rlist[index] if index>=0 else None
                states.append(S(mode=row['mode'],armed=row['armed'].lower()=='true') if row else None)
            start=metrics['transport_clock_start_s'] if stamp>=metrics['transport_clock_start_s'] else None
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            canvas=view.render(np.zeros_like(rgb),stamp,stage,start,truth,cable,states,truth_stamp)
            rgb[:,960:]=canvas[:,960:];rgb[604:]=canvas[604:]
            encoder.stdin.write(rgb.tobytes());count+=1
    finally:
        cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
    if code or count!=execution['video_frames']:raise SystemExit('Encoding or frame count check failed')
    (root/'dashboard_refresh.json').write_text(json.dumps({'source_video_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_run':'same Gazebo run, causal log samples at each original frame timestamp','replaced_regions':'right telemetry panel and bottom numeric/status panel only','skipped_diagnostic_lines':SKIPPED,'frames':count,'fps':10,'output':output.name,'presentation_sha256':hashlib.sha256((Path(__file__).parent/'presentation.py').read_bytes()).hexdigest()},indent=2))
    print(output)
if __name__=='__main__':main()
