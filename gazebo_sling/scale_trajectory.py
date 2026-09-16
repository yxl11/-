#!/usr/bin/env python3
"""Time-scale frozen p/v/a consistently and resample on a 50 Hz clock."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sling_sim.replay import ReplayTrajectory

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--factor',type=float,required=True)
    args=p.parse_args()
    if not np.isfinite(args.factor) or args.factor<=0:p.error('factor must be finite and positive')
    if args.output.exists():p.error('output exists; preserve previous trajectories')
    ref=ReplayTrajectory.load(args.source);duration=ref.duration/args.factor
    samples=round(duration*50)
    if not np.isclose(samples/50,duration):p.error('duration must lie on 50 Hz grid')
    times=np.arange(samples+1)/50
    values=[ref.at(t*args.factor) for t in times]
    position=np.array([x[0] for x in values]);velocity=np.array([x[1] for x in values])*args.factor;acceleration=np.array([x[2] for x in values])*args.factor**2
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,time=times,position=position,velocity=velocity,acceleration=acceleration,yaw=np.zeros((len(times),3)))
    print(f'{duration:g} s, {len(times)} samples, speed x{args.factor:g}, acceleration x{args.factor**2:g}')
if __name__=='__main__':main()
