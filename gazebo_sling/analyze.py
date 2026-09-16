#!/usr/bin/env python3
"""Audit Gazebo physical transport and export synchronized hardware candidates.

Run in the project's Python 3.12 venv. Exported references are commands, never
the noisy measured response. Hardware calibration remains a separate gate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sling_sim.replay import ReplayTrajectory
from sling_sim.trajectory import TransportTrajectory

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    cfg=json.loads((root/'runtime.json').read_text());execution=json.loads((root/'execution.json').read_text())
    source=Path(cfg['trajectory_source']);ref=ReplayTrajectory.load(source)
    config=json.loads((ROOT/'results/experiment_v1/planning/config_snapshot.json').read_text());config.update(cfg.get('payload_reference',{}));path=TransportTrajectory(config)
    commands=read(root/'commands.csv');transport=[r for r in commands if r['stage']=='transport']
    if not transport:raise SystemExit('No transport stage to analyze')
    t0=float(transport[0]['time_s']);t1=max(float(r['time_s']) for r in transport)
    rows=read(root/'gazebo_states.csv'); metrics={}; positions={}; times={};tilts={};quaternions={};velocities={}
    for name in ['payload','uav1','uav2','uav3']:
        body=[r for r in rows if r['model']==name and r['stage']=='transport' and t0<=float(r['time_s'])<=t1]
        t=np.array([float(r['time_s'])-t0 for r in body]);pos=np.array([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in body])
        q=np.array([[float(r[k]) for k in ['qx','qy','qz','qw']] for r in body]);q/=np.linalg.norm(q,axis=1)[:,None]
        tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,0]**2+q[:,1]**2),-1,1)))
        desired=np.array([path.at(s).position if name=='payload' else ref.at(s)[0][int(name[-1])-1] for s in t])
        error=np.linalg.norm(pos-desired,axis=1);final=t>=2*config['leg_duration']+2*config['hold_duration']
        metrics[name]={'position_RMSE_m':float(np.sqrt(np.mean(error**2))),'position_peak_m':float(error.max()),
            'tilt_peak_deg':float(tilt.max()),'final_hold_tilt_peak_deg':float(tilt[final].max()) if final.any() else None,
            'final_hold_RMSE_m':float(np.sqrt(np.mean(error[final]**2))) if final.any() else None,
            'position_min_m':pos.min(axis=0).tolist(),'position_max_m':pos.max(axis=0).tolist()}
        positions[name]=pos;times[name]=t;tilts[name]=tilt
        quaternions[name]=q;velocities[name]=np.array([[float(r[k]) for k in ['vx','vy','vz']] for r in body])
    cables=[r for r in read(root/'cables.csv') if r['stage']=='transport' and t0<=float(r['time_s'])<=t1]
    tension=np.array([[float(r['tension_N_'+str(i)]) for i in range(1,4)] for r in cables]);length=np.array([[float(r['length_m_'+str(i)]) for i in range(1,4)] for r in cables])
    clock=times['payload'];u=np.stack([np.column_stack([np.interp(clock,times['uav'+str(i)],positions['uav'+str(i)][:,j]) for j in range(3)]) for i in range(1,4)],axis=1)
    distances=np.stack([np.linalg.norm(u[:,i]-u[:,j],axis=1) for i,j in [(0,1),(0,2),(1,2)]],axis=1)
    minimum_distance=float(distances.min())
    # Independent Newton force-balance audit: measured Gazebo load acceleration
    # must agree with the three cable forces. This uses only evaluation logs.
    from sling_sim.rotations import quaternion_matrix
    clock_abs=clock+t0;ct_abs=np.array([float(r['time_s']) for r in cables])
    interpolated_tension=np.column_stack([np.interp(clock_abs,ct_abs,tension[:,i]) for i in range(3)])
    phi=np.arange(3)*2*np.pi/3
    anchors=np.column_stack((.25*np.cos(phi),.25*np.sin(phi),np.full(3,.125)))
    rot=quaternion_matrix(quaternions['payload'][:,[3,0,1,2]])
    attachment=positions['payload'][:,None,:]+np.einsum('nij,kj->nki',rot,anchors)
    directions=u-attachment;directions/=np.linalg.norm(directions,axis=2)[:,:,None]
    resultant=np.sum(interpolated_tension[:,:,None]*directions,axis=1)
    acceleration=np.gradient(velocities['payload'],clock,axis=0)
    balance=cfg['payload_mass']*(acceleration+np.array([0,0,9.81]))-resultant
    balance_rms=float(np.sqrt(np.mean(np.sum(balance[2:-2]**2,axis=1))))
    summary={'execution':execution,'engine':'Gazebo Classic 11 / ODE + three PX4 SITL 1.13.2 + ROS Noetic / MAVROS',
        'measurement':'ideal UAV pose at 50 Hz through MAVROS vision_pose / EKF2; no load or tension controller feedback',
        'task_speed_factor':cfg.get('task_speed_factor',1.),'reference_duration_s':ref.duration,'transport_clock_start_s':t0,'transport_logged_duration_s':t1-t0,'bodies':metrics,
        'tension_min_N':float(tension.min()),'tension_max_N':float(tension.max()),
        'rope_extension_peak_m':float((length-cfg['rope_length']).max()),
        'minimum_uav_center_distance_m':minimum_distance,'minimum_assumed_surface_clearance_m':minimum_distance-.96,
        'sampling_hz':50,'cable_model':'unilateral Kelvin-Voigt, massless; slack transmits zero force; no segmented rope/contact',
        'physics_dt_s':cfg['physics_dt'],'flight_ready':False}
    summary['independent_load_Newton_balance_RMS_N']=balance_rms
    summary['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'gazebo_sling').rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
    summary['px4_binary_sha256']=hashlib.sha256((Path(cfg['px4'])/'build/px4_sitl_default/bin/px4').read_bytes()).hexdigest()
    checks={'execution_completed':execution['status']=='completed','full_reference_duration':t1-t0>=ref.duration-.1,
        'payload_peak_below_15cm':metrics['payload']['position_peak_m']<=.15,
        'uav_peaks_below_10cm':all(metrics['uav'+str(i)]['position_peak_m']<=.1 for i in range(1,4)),
        'payload_tilt_below_5deg':metrics['payload']['tilt_peak_deg']<=5,
        'final_hold_tilt_below_1deg':metrics['payload']['final_hold_tilt_peak_deg'] is not None and metrics['payload']['final_hold_tilt_peak_deg']<=1,
        'tension_4_to_18N':float(tension.min())>=4 and float(tension.max())<=18,
        'assumed_uav_clearance_25cm':minimum_distance-.96>=.25,
        'no_transport_slack':bool(np.all(tension>0)),
        'finite_positions':all(np.isfinite(x).all() for x in positions.values()),
        'independent_Newton_balance_RMS_below_05N':balance_rms<.5,
        'camera_video_present':(root/'gazebo_transport.mp4').exists() and (root/'gazebo_transport.mp4').stat().st_size>1000}
    summary['simulation_checks']=checks;summary['simulation_screen_passed']=all(checks.values())
    (root/'gazebo_metrics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    output=root/'hardware_candidate';output.mkdir(exist_ok=True)
    hashes={}
    for i in range(3):
        file=output/('uav%d_enu_50hz.csv'%(i+1))
        with file.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_s','x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','ax_m_s2','ay_m_s2','az_m_s2','yaw_rad'])
            for n,t in enumerate(ref.time):writer.writerow([t,*ref.p[n,i],*ref.v[n,i],*ref.a[n,i],0.])
        hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    manifest={'duration_s':ref.duration,'sampling_hz':50,'samples_per_uav':len(ref.time),
        'source_trajectory_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'csv_sha256':hashes,
        'frame':'common map ENU (x east, y north, z up); yaw 0 relative to map x',
        'units':'m, s, rad','interpolation':'quintic Hermite with p/v/a constraints',
        'synchronization':'one shared clock and start trigger; all UAVs use the same progress',
        'interface':'ROS 1 MAVROS setpoint_raw/local; ROS fields ENU, MAVROS converts to MAVLink LOCAL_NED once',
        'physical_assumptions':{k:cfg[k] for k in ['payload_mass','uav_mass','rope_length','cable_stiffness_N_m','cable_damping_N_s_m']},
        'simulation_screen_passed':all(checks.values()),'hardware_calibrated':False,'flight_ready':False,
        'phase_scope':'already airborne suspended transport only; actual takeoff/landing are separate procedures',
        'hardware_requirements':['measured masses, rigging and thrust margin','actual attachment positions and rope properties',
            'surveyed map-to-each-PX4-origin transforms and yaw offsets','validated localization noise/delay and clock synchronization',
            'hardware-specific tracking and coordinated failsafe validation']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    from matplotlib import pyplot as plt
    from sling_sim.visualization import configure_style
    configure_style();fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axs[0,0].plot(positions['payload'][:,0],positions['payload'][:,1],label='Gazebo负载响应')
    desired=np.array([path.at(t).position for t in clock]);axs[0,0].plot(desired[:,0],desired[:,1],'--',label='参考')
    axs[0,0].set(xlabel='x / m',ylabel='y / m');axs[0,0].legend()
    axs[0,1].plot(clock,np.linalg.norm(positions['payload']-desired,axis=1)*100);axs[0,1].set(xlabel='运输时间 / s',ylabel='负载误差 / cm')
    ct=np.array([float(r['time_s'])-t0 for r in cables]);axs[1,0].plot(ct,tension);axs[1,0].set(xlabel='运输时间 / s',ylabel='物理绳张力 / N')
    axs[1,1].plot(clock,tilts['payload']);axs[1,1].set(xlabel='运输时间 / s',ylabel='负载倾角 / °')
    fig.suptitle('Gazebo ODE + ROS 1 + 三套PX4：协同吊运往返物理验证')
    fig.savefig(root/'gazebo_summary.png',dpi=150);fig.savefig(root/'gazebo_summary.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if not all(checks.values()):raise SystemExit(2)

if __name__=='__main__':main()
