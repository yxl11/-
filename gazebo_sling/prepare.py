#!/usr/bin/python3
"""Generate isolated Gazebo/PX4 runtime from the installed PX4 model template."""
import argparse
import json
import math
import numpy as np
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--px4', type=Path, default=Path('/home/yhr/PX4_Firmware'))
    p.add_argument('--output', type=Path, default=ROOT/'results/gazebo_transport')
    p.add_argument('--speed-factor',type=float,default=1.)
    p.add_argument('--trajectory',type=Path)
    args = p.parse_args()
    if not math.isfinite(args.speed_factor) or args.speed_factor <= 0:p.error('positive finite speed-factor required')
    if args.speed_factor != 1 and args.trajectory is None:p.error('scaled --trajectory required with speed-factor')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    px4 = args.px4.resolve(); sitl = px4/'Tools/sitl_gazebo'
    etc = out/'etc'
    if etc.exists():
        raise SystemExit('Runtime exists; choose a new --output to preserve PX4 parameters/logs.')
    shutil.copytree(px4/'build/px4_sitl_default/etc', etc)
    # Explicitly isolate local XTDrone port changes in the pre-existing checkout.
    mavlink = etc/'init.d-posix/px4-rc.mavlink'
    text = mavlink.read_text().replace('34580+px4_instance', '14580+px4_instance').replace('24540+px4_instance', '14540+px4_instance')
    mavlink.write_text(text)
    params = etc/'init.d-posix/px4-rc.params'
    params.write_text('''#!/bin/sh
param set MAV_SYS_ID $((px4_instance+1))
param set COM_RCL_EXCEPT 4
param set NAV_RCL_ACT 0
param set COM_OBL_RC_ACT 4
param set COM_OF_LOSS_T 0.5
param set MPC_THR_HOVER 0.73
param set MPC_THR_MAX 0.95
param set MPC_XY_VEL_MAX 1.0
param set MPC_Z_VEL_MAX_UP 0.5
param set MPC_Z_VEL_MAX_DN 0.4
param set MPC_ACC_HOR_MAX 1.0
param set MPC_TILTMAX_AIR 25
param set MPC_XY_P 1.0
param set MPC_XY_VEL_P_ACC 3.0
param set MPC_XY_VEL_I_ACC 1.0
param set MPC_XY_VEL_D_ACC 0.3
param set EKF2_AID_MASK 24
param set EKF2_HGT_MODE 3
param set EKF2_EV_DELAY 0
param set EKF2_EV_POS_X 0
param set EKF2_EV_POS_Y 0
param set EKF2_EV_POS_Z 0
param set EKF2_EVP_NOISE 0.01
param set EKF2_EVA_NOISE 0.01
''')
    config = json.loads((ROOT/'results/experiment_v1/planning/config_snapshot.json').read_text())
    source=(args.trajectory or ROOT/'results/experiment_v1/trajectory/trajectory.npz').resolve()
    with np.load(source,allow_pickle=False) as data:
        expected=(2*config['leg_duration']+3*config['hold_duration'])/args.speed_factor
        if not np.isclose(data['time'][-1],expected):raise SystemExit('Trajectory duration disagrees with speed factor')
    if args.speed_factor != 1:
        params.write_text(params.read_text().replace('MPC_XY_VEL_MAX 1.0',f'MPC_XY_VEL_MAX {args.speed_factor:g}'))
    config['leg_duration']/=args.speed_factor;config['hold_duration']/=args.speed_factor
    cfg = {'px4':str(px4), 'physics_dt':.001, 'cable_stiffness_N_m':5000., 'cable_damping_N_s_m':35.,
           'payload_mass':config['payload_mass'], 'uav_mass':config['uav_mass'], 'rope_length':config['rope_length'],
           'trajectory_source':str(source), 'task_speed_factor':args.speed_factor,
           'payload_reference':{k:config[k] for k in ['leg_duration','hold_duration','payload_com_height']}, 'workspace':[20,10,5], 'spawn_world':[], 'localization_frame':'absolute_map_enu', 'flight_ready':False}
    world = ET.Element('sdf', version='1.6'); w = ET.SubElement(world,'world',name='default')
    def add(parent, tag, value): ET.SubElement(parent,tag).text=str(value)
    add(w,'gravity','0 0 -9.81')
    phys=ET.SubElement(w,'physics',name='ode',type='ode')
    add(phys,'max_step_size',.001); add(phys,'real_time_update_rate',1000)
    ode=ET.SubElement(phys,'ode'); solver=ET.SubElement(ode,'solver'); add(solver,'type','quick'); add(solver,'iters',80)
    for uri in ['model://ground_plane']:
        include=ET.SubElement(w,'include'); add(include,'uri',uri)
    scene=ET.SubElement(w,'scene'); add(scene,'ambient','0.42 0.46 0.52 1'); add(scene,'background','0.72 0.83 0.94 1'); add(scene,'shadows','true')
    light=ET.SubElement(w,'light',name='soft_daylight',type='directional')
    add(light,'pose','0 0 10 0 0 0');attenuation=ET.SubElement(light,'attenuation');add(attenuation,'range',1000);add(attenuation,'constant',.9);add(attenuation,'linear',.01);add(attenuation,'quadratic',.001)
    add(light,'cast_shadows','true');add(light,'diffuse','0.88 0.85 0.78 1');add(light,'specular','0.25 0.25 0.25 1');add(light,'direction','-0.35 0.45 -0.82')
    # Static survey grid and S-path markers give the native camera scale cues.
    floor=ET.SubElement(w,'model',name='survey_grid');add(floor,'static','true')
    fl=ET.SubElement(floor,'link',name='grid')
    def marker(name,pose,size,color):
        v=ET.SubElement(fl,'visual',name=name);add(v,'pose',pose)
        g=ET.SubElement(v,'geometry');b=ET.SubElement(g,'box');add(b,'size',size)
        mat=ET.SubElement(v,'material');add(mat,'ambient',color);add(mat,'diffuse',color)
    marker('floor','0 0 0.001 0 0 0','200 200 0.001','0.72 0.76 0.78 1')
    for j in range(-10,11):marker('gx'+str(j),f'{j} 0 0.004 0 0 0','0.008 10 0.003','0.60 0.65 0.68 1')
    for j in range(-5,6):marker('gy'+str(j),f'0 {j} 0.004 0 0 0','20 0.008 0.003','0.60 0.65 0.68 1')
    for j in range(121):
        s=j/120;marker('path'+str(j),f'{-3+6*s} {math.sin(2*math.pi*s)} 0.008 0 0 0','0.055 0.055 0.006','0.05 0.35 0.8 1')
    for j,x in enumerate([-3,3]):marker('endpoint'+str(j),f'{x} 0 0.01 0 0 0','0.6 0.6 0.008','0.95 0.7 0.1 1')
    # A simple open test courtyard, outside the 20 x 10 m flight area.
    # Background is visual-only and does not change the flight dynamics.
    marker('back_fence','0 5.5 0.5 0 0 0','22 0.08 1.0','0.72 0.78 0.78 1')
    marker('left_fence','-11 0 0.5 0 0 0','0.08 11 1.0','0.72 0.78 0.78 1')
    marker('right_fence','11 0 0.5 0 0 0','0.08 11 1.0','0.72 0.78 0.78 1')
    for j in range(-10,11,5):marker('fence_post'+str(j),f'{j} 5.5 0.6 0 0 0','0.13 0.13 1.2','0.92 0.94 0.92 1')
    for i in range(3):
        result=subprocess.run(['/usr/bin/python3',str(sitl/'scripts/jinja_gen.py'),'--stdout',
            '--mavlink_id',str(i+1),'--mavlink_tcp_port',str(4560+i),'--mavlink_udp_port',str(14560+i),
            str(sitl/'models/iris/iris.sdf.jinja'),str(sitl)],check=True,capture_output=True,text=True)
        model=ET.fromstring(result.stdout).find('model'); model.set('name','uav'+str(i+1))
        angle=2*math.pi*i/3
        radial=config['anchor_radius']+config['rope_length']*math.sin(math.radians(20))
        pos=[-3+radial*math.cos(angle),radial*math.sin(angle),.15]
        cfg['spawn_world'].append(pos); add(model,'pose',' '.join(map(str,pos))+' 0 0 0')
        base=model.find("link[@name='base_link']/inertial")
        base.find('mass').text=str(config['uav_mass']-.035)
        for tag,val in zip(['ixx','iyy','izz'],config['uav_inertia']): base.find('inertia/'+tag).text=str(val)
        for plugin in model.findall('plugin'):
            if plugin.find('motorConstant') is not None:
                plugin.find('motorConstant').text=str(config['max_rotor_thrust']/1000**2)
                plugin.find('maxRotVelocity').text='1000'
            if plugin.get('name')=='mavlink_interface':
                plugin.find('enable_lockstep').text='0'
        # Distinct body color keeps UAV numbering readable in camera output.
        visual=model.find("link[@name='base_link']/visual/material")
        for child in list(visual): visual.remove(child)
        color=['0.0 0.5 0.85 1','0.9 0.25 0.08 1','0.55 0.15 0.75 1'][i]
        add(visual,'ambient',color); add(visual,'diffuse',color);add(visual,'specular','0.35 0.35 0.35 1')
        # Additional geometry is visual-only: retain all validated inertial/collision data.
        body=model.find("link[@name='base_link']")
        def detail(parent,name,pose,size,tint):
            v=ET.SubElement(parent,'visual',name=name);add(v,'pose',pose)
            g=ET.SubElement(v,'geometry');b=ET.SubElement(g,'box');add(b,'size',size)
            m=ET.SubElement(v,'material');add(m,'ambient',tint);add(m,'diffuse',tint);add(m,'specular','0.3 0.3 0.3 1')
        detail(body,'avionics_cover','0 0 0.055 0 0 0','0.20 0.13 0.055',color)
        detail(body,'battery_pack','-0.02 0 0.092 0 0 0','0.11 0.07 0.018','0.10 0.13 0.17 1')
        for side in [-1,1]:
            detail(body,'landing_skid_'+str(side),f'0 {side*.14} -0.04 0 0 0','0.28 0.014 0.018','0.16 0.19 0.23 1')
        w.append(model)
    load=ET.SubElement(w,'model',name='payload'); add(load,'pose','-3 0 0.13 0 0 0')
    link=ET.SubElement(load,'link',name='body'); inertia=ET.SubElement(link,'inertial'); add(inertia,'mass',config['payload_mass'])
    tensor=ET.SubElement(inertia,'inertia'); x,y,z=config['payload_dimensions']; mass=config['payload_mass']
    for tag,val in zip(['ixx','iyy','izz'],[mass*(y*y+z*z)/12,mass*(x*x+z*z)/12,mass*(x*x+y*y)/12]): add(tensor,tag,val)
    for kind in ['collision','visual']:
        shape=ET.SubElement(link,kind,name='load_'+kind); geom=ET.SubElement(shape,'geometry'); box=ET.SubElement(geom,'box'); add(box,'size','0.6 0.5 0.25')
        if kind=='visual':
            material=ET.SubElement(shape,'material'); add(material,'ambient','0.15 0.45 0.25 1'); add(material,'diffuse','0.15 0.45 0.25 1')
    for offset in [-.18,.18]:
        v=ET.SubElement(link,'visual',name='cargo_band_'+str(offset));add(v,'pose',f'{offset} 0 0 0 0 0')
        g=ET.SubElement(v,'geometry');b=ET.SubElement(g,'box');add(b,'size','0.026 0.504 0.254')
        m=ET.SubElement(v,'material');add(m,'ambient','0.12 0.16 0.18 1');add(m,'diffuse','0.12 0.16 0.18 1')
    # These are visual elements only. Cable forces are calculated independently.
    # Pre-existing SDF visuals can be updated in both GUI and sensor scenes.
    for i in range(3):
        v=ET.SubElement(link,'visual',name='rope_'+str(i+1));add(v,'pose','0 0 1 0 0 0')
        g=ET.SubElement(v,'geometry');cyl=ET.SubElement(g,'cylinder');add(cyl,'radius',.009);add(cyl,'length',1.8)
        mat=ET.SubElement(v,'material');add(mat,'ambient','0.85 0.55 0.12 1');add(mat,'diffuse','0.85 0.55 0.12 1')
    plugin=ET.SubElement(w,'plugin',name='sling_cables',filename='libsling_cables.so')
    add(plugin,'length',config['rope_length']); add(plugin,'stiffness',cfg['cable_stiffness_N_m']); add(plugin,'damping',cfg['cable_damping_N_s_m'])
    camera=ET.SubElement(w,'model',name='overview_camera'); add(camera,'static','true'); add(camera,'pose','0 -9 6 0 0.45 1.57079632679')
    cl=ET.SubElement(camera,'link',name='camera_link'); sensor=ET.SubElement(cl,'sensor',name='camera',type='camera')
    add(sensor,'always_on','true'); add(sensor,'update_rate',10)
    cam=ET.SubElement(sensor,'camera'); add(cam,'horizontal_fov',1.20)
    image=ET.SubElement(cam,'image'); add(image,'width',1280); add(image,'height',720); add(image,'format','R8G8B8')
    clip=ET.SubElement(cam,'clip'); add(clip,'near',.1); add(clip,'far',100)
    cp=ET.SubElement(sensor,'plugin',name='camera_ros',filename='libgazebo_ros_camera.so')
    add(cp,'robotNamespace','/')
    add(cp,'cameraName','sling_camera'); add(cp,'imageTopicName','image_raw'); add(cp,'frameName','camera_link')
    gui=ET.SubElement(w,'gui'); gc=ET.SubElement(gui,'camera',name='user_camera'); add(gc,'pose','0 -9 6 0 0.45 1.57079632679')
    ET.ElementTree(world).write(out/'transport.world',encoding='utf-8',xml_declaration=True)
    (out/'runtime.json').write_text(json.dumps(cfg,indent=2))
    launch=ET.Element('launch')
    for i in range(3):
        group=ET.SubElement(launch,'group',ns='uav'+str(i+1))
        include=ET.SubElement(group,'include',file='/opt/ros/noetic/share/mavros/launch/px4.launch')
        for name,value in [('fcu_url','udp://:1454%d@127.0.0.1:1458%d'%(i,i)),('gcs_url',''),('tgt_system',str(i+1))]: ET.SubElement(include,'arg',name=name,value=value)
    ET.ElementTree(launch).write(out/'mavros.launch',encoding='utf-8',xml_declaration=True)
    print(out)

if __name__=='__main__': main()
