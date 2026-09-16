#!/usr/bin/env python3
"""Transform synchronized map references into surveyed per-PX4 local ENU files.

This exports files only; it never connects to or arms aircraft. MAVROS accepts
local ENU fields and converts to PX4 NED. Derivatives rotate without translation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

def transform(p,v,a,yaw,origin,heading):
    c,s=math.cos(heading),math.sin(heading)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    # Row-vector equivalent of R^T (p_map-origin).
    return (p-np.asarray(origin))@rotation,v@rotation,a@rotation,(yaw-heading+np.pi)%(2*np.pi)-np.pi

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-example',action='store_true')
    args=p.parse_args();profile=json.loads(args.frames.read_text());manifest=json.loads((args.candidate/'manifest.json').read_text())
    if profile.get('hardware_calibrated') is not True and not args.allow_example:
        p.error('Measured frame calibration required; --allow-example exports explicitly uncalibrated review files only')
    if len(profile['uavs'])!=3 or {r['id'] for r in profile['uavs']}!={1,2,3}:
        p.error('Exactly UAV ids 1, 2, 3 are required')
    clocks=[];args.output.mkdir(parents=True,exist_ok=True);hashes={}
    for uav in profile['uavs']:
        source=args.candidate/('uav%d_enu_50hz.csv'%uav['id'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['csv_sha256'][source.name]:raise ValueError('Candidate CSV hash mismatch')
        with source.open() as f:
            reader=csv.reader(f);header=next(reader);data=np.array(list(reader),float)
        if not np.isfinite(data).all() or np.any(np.diff(data[:,0])<=0):raise ValueError('Invalid trajectory data')
        origin=np.array(uav['origin_in_map_m'],float);heading=float(uav['yaw_map_from_local_rad'])
        if origin.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(heading):raise ValueError('Invalid frame transform')
        position,velocity,acceleration,yaw=transform(data[:,1:4],data[:,4:7],data[:,7:10],data[:,10],origin,heading)
        converted=np.column_stack((data[:,0],position,velocity,acceleration,yaw));clocks.append(data[:,0])
        target=args.output/('uav%d_local_enu_50hz.csv'%uav['id'])
        with target.open('w',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(converted)
        hashes[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    if any(not np.array_equal(clocks[0],t) for t in clocks[1:]):raise ValueError('UAV clocks differ')
    (args.output/'manifest.json').write_text(json.dumps({'source_manifest':manifest,'frames':profile,'csv_sha256':hashes,
        'coordinate_frame':'per-UAV local ENU for ROS/MAVROS; not preconverted NED','hardware_calibrated':profile['hardware_calibrated'],
        'flight_ready':False,'note':'Frame calibration alone does not validate carrying capacity, localization, tracking, or coordinated failsafes'},indent=2))
    print(args.output)

if __name__=='__main__':main()
