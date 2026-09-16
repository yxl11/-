#!/usr/bin/python3
"""Synchronized ROS 1 Offboard execution; control uses only each UAV estimate.

Gazebo body/cable truth is logged and used for video presentation, never for flight tracking.
All references are ROS ENU; MAVROS performs the ENU/NED conversion once.
"""
import argparse
import csv
from collections import deque
import json
import math
from pathlib import Path
import time
import threading
import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
import subprocess
from presentation import Presentation

def smooth(u):
    u=np.clip(u,0.,1.)
    return (35*u**4-84*u**5+70*u**6-20*u**7,
            140*u**3-420*u**4+420*u**5-140*u**6,
            420*u**2-1680*u**3+2100*u**4-840*u**5)

class RecordedReference:
    # Self-contained interpolation compatible with ROS's Python 3.8.
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as d:
            if set(d.files)!={'time','position','velocity','acceleration','yaw'} or np.any(d['yaw']!=0):
                raise ValueError('Trajectory must contain only time, UAV p/v/a and zero yaw')
            self.t=d['time'].copy(); self.p=d['position'].copy(); self.v=d['velocity'].copy(); self.a=d['acceleration'].copy()
        if self.t[0]!=0 or np.any(np.diff(self.t)<=0) or any(x.shape!=(len(self.t),3,3) or not np.isfinite(x).all() for x in [self.p,self.v,self.a]):
            raise ValueError('Invalid reference table')
        h=np.diff(self.t)[:,None,None]
        c0=self.p[:-1]; c1=h*self.v[:-1]; c2=.5*h*h*self.a[:-1]
        dp=self.p[1:]-c0-c1-c2; dv=h*self.v[1:]-c1-2*c2; da=h*h*self.a[1:]-2*c2
        self.c=np.stack([c0,c1,c2,10*dp-4*dv+.5*da,-15*dp+7*dv-da,6*dp-3*dv+.5*da],axis=1)
    def at(self,t):
        t=np.clip(t,0,self.t[-1]); i=min(np.searchsorted(self.t,t,side='right')-1,len(self.t)-2)
        h=self.t[i+1]-self.t[i]; u=(t-self.t[i])/h; c=self.c[i]
        return (sum(c[k]*u**k for k in range(6)),sum(k*c[k]*u**(k-1) for k in range(1,6))/h,
                sum(k*(k-1)*c[k]*u**(k-2) for k in range(2,6))/h**2)

class Executor:
    def __init__(self,args):
        self.args=args; self.root=args.root.resolve(); self.cfg=json.loads((self.root/'runtime.json').read_text())
        self.ref=RecordedReference(self.cfg['trajectory_source'])
        self.states=[None]*3; self.positions=[None]*3; self.received=[0.]*3
        self.truth=None; self.cables=None; self.stage='initializing'; self.recording=False; self.video=None; self.frames=0
        self.video_start=None; self.video_end=None
        self.presentation=Presentation(self.cfg['payload_reference']); self.transport_start=None
        self.truth_history=deque(maxlen=100); self.cable_history=deque(maxlen=100)
        self.closed=False
        self.states_log=(self.root/'gazebo_states.csv').open('w',newline=''); self.sw=csv.writer(self.states_log)
        self.sw.writerow(['time_s','stage','model','x_m','y_m','z_m','qx','qy','qz','qw','vx','vy','vz'])
        self.cables_log=(self.root/'cables.csv').open('w',newline=''); self.cw=csv.writer(self.cables_log)
        self.cw.writerow(['time_s','stage']+[f'{k}_{i}' for i in range(1,4) for k in ['tension_N','length_m','radial_speed_m_s']])
        self.command_log=(self.root/'commands.csv').open('w',newline=''); self.rw=csv.writer(self.command_log)
        self.rw.writerow(['time_s','stage','uav_id','x_world','y_world','z_world','vx','vy','vz','ax','ay','az','x_local','y_local','z_local'])
        self.estimate_lock=threading.Lock()
        self.estimate_log=(self.root/'estimates.csv').open('w',newline=''); self.ew=csv.writer(self.estimate_log)
        self.ew.writerow(['time_s','uav_id','x_local','y_local','z_local','mode','armed'])
        self.pubs=[]; self.arm=[]; self.mode=[]
        for i in range(3):
            prefix='/uav%d/mavros'%(i+1)
            rospy.Subscriber(prefix+'/state',State,lambda m,i=i:self.state_cb(m,i))
            rospy.Subscriber(prefix+'/local_position/pose',PoseStamped,lambda m,i=i:self.pose_cb(m,i))
            self.pubs.append(rospy.Publisher(prefix+'/setpoint_raw/local',PositionTarget,queue_size=1))
            self.arm.append(rospy.ServiceProxy(prefix+'/cmd/arming',CommandBool))
            self.mode.append(rospy.ServiceProxy(prefix+'/set_mode',SetMode))
        rospy.Subscriber('/gazebo/model_states',ModelStates,self.truth_cb,queue_size=1)
        rospy.Subscriber('/sling/cables',Float64MultiArray,self.cable_cb,queue_size=2)
        rospy.Subscriber('/sling_camera/image_raw',Image,self.image_cb,queue_size=2)
        self.offsets=None; self.last_truth=-1

    def state_cb(self,m,i): self.states[i]=m
    def pose_cb(self,m,i):
        if self.closed:return
        self.positions[i]=np.array([m.pose.position.x,m.pose.position.y,m.pose.position.z]); self.received[i]=time.monotonic()
        s=self.states[i]
        with self.estimate_lock:
            self.ew.writerow([rospy.Time.now().to_sec(),i+1,*self.positions[i],s.mode if s else '',s.armed if s else False])
    def truth_cb(self,m):
        if self.closed:return
        self.truth=m
        t=rospy.Time.now().to_sec()
        if t-self.last_truth<.0199:return
        self.last_truth=t
        self.truth_history.append((t,m))
        for name,p,v in zip(m.name,m.pose,m.twist):
            if name not in ['payload','uav1','uav2','uav3']:continue
            self.sw.writerow([t,self.stage,name,p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w,v.linear.x,v.linear.y,v.linear.z])
    def cable_cb(self,m):
        if self.closed:return
        self.cables=list(m.data); self.cable_history.append(self.cables); self.cw.writerow([m.data[0],self.stage,*m.data[1:]])
    def image_cb(self,m):
        if not self.recording:return
        if self.video is None:
            self.video_start=m.header.stamp.to_sec()
            if m.encoding not in ('rgb8','bgr8') or m.step!=m.width*3:
                raise RuntimeError('Expected tightly packed RGB/BGR Gazebo camera image')
            self.video=subprocess.Popen([self.args.ffmpeg,'-y','-v','error','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','%dx%d'%(m.width,m.height),'-framerate','10','-i','-','-c:v','libx264','-preset','ultrafast',
                '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(self.root/'gazebo_transport.mp4')],stdin=subprocess.PIPE)
        try:
            # Gazebo publishes tightly packed rgb8/bgr8 images at 10 simulation Hz.
            rgb=np.frombuffer(m.data,dtype=np.uint8).reshape(m.height,m.width,3)
            if m.encoding=='bgr8':rgb=rgb[:,:,::-1]
            stamp=m.header.stamp.to_sec()
            truth_sample=next((row for row in reversed(list(self.truth_history)) if row[0]<=stamp),None)
            cable_sample=next((row for row in reversed(list(self.cable_history)) if row[0]<=stamp),None)
            frame=self.presentation.render(rgb,stamp,self.stage,self.transport_start,truth_sample[1] if truth_sample else None,cable_sample,self.states,truth_sample[0] if truth_sample else None)
            self.video.stdin.write(frame.tobytes()); self.frames+=1; self.video_end=m.header.stamp.to_sec()
        except (BrokenPipeError,OSError): self.recording=False; rospy.logerr('Video encoder failed')
    def send(self,p,v,a):
        for i in range(3):
            msg=PositionTarget(); msg.header.stamp=rospy.Time.now(); msg.coordinate_frame=PositionTarget.FRAME_LOCAL_NED
            msg.type_mask=PositionTarget.IGNORE_YAW_RATE
            local=p[i]-self.offsets[i]
            msg.position.x,msg.position.y,msg.position.z=map(float,local)
            msg.velocity.x,msg.velocity.y,msg.velocity.z=map(float,v[i])
            msg.acceleration_or_force.x,msg.acceleration_or_force.y,msg.acceleration_or_force.z=map(float,a[i])
            msg.yaw=0.; self.pubs[i].publish(msg)
            self.rw.writerow([rospy.Time.now().to_sec(),self.stage,i+1,*p[i],*v[i],*a[i],*local])
    def wait(self,p,v,a,seconds):
        start=rospy.Time.now().to_sec(); rate=rospy.Rate(50)
        while not rospy.is_shutdown() and rospy.Time.now().to_sec()-start<seconds:self.send(p,v,a);rate.sleep()
    def run(self):
        start=time.monotonic()
        while not rospy.is_shutdown():
            if all(s and s.connected for s in self.states) and all(p is not None for p in self.positions) and rospy.Time.now().to_sec()>0:break
            if time.monotonic()-start>120:raise RuntimeError('PX4/MAVROS connection/local-position timeout')
            time.sleep(.1)
        # The external-vision sensor supplies absolute map coordinates to EKF2.
        # A spawned model settles onto the floor before arming: spawn z is NOT a
        # valid survey of its settled position. Never turn that settling into an
        # artificial map-origin offset. Real deployments need surveyed transforms.
        spawn=np.array(self.cfg['spawn_world']); self.offsets=np.zeros((3,3))
        if self.cfg.get('localization_frame')!='absolute_map_enu':
            raise RuntimeError('Explicit map/PX4 frame calibration is required')
        (self.root/'frame_calibration.json').write_text(json.dumps({'world_minus_local':self.offsets.tolist(),'method':'absolute_map_enu_external_vision','real_hardware_calibrated':False},indent=2))
        zeros=np.zeros((3,3)); self.stage='prestream'; self.wait(spawn,zeros,zeros,2.)
        self.stage='arming'; deadline=time.monotonic()+60; last=0
        while not all(s.armed and s.mode=='OFFBOARD' for s in self.states):
            self.send(spawn,zeros,zeros)
            if time.monotonic()-last>2:
                for i,s in enumerate(self.states):
                    if s.mode!='OFFBOARD':self.mode[i](0,'OFFBOARD')
                    if not s.armed:self.arm[i](True)
                last=time.monotonic()
            if time.monotonic()>deadline:raise RuntimeError('Failed to arm all three PX4 instances in OFFBOARD')
            rospy.sleep(.02)
        target=self.ref.at(0)[0]; duration=16.; self.stage='takeoff'; self.recording=True
        start=rospy.Time.now().to_sec(); rate=rospy.Rate(50)
        while not rospy.is_shutdown():
            t=rospy.Time.now().to_sec()-start
            if t>=duration:break
            s,sd,sdd=smooth(t/duration); delta=target-spawn
            self.send(spawn+s*delta,sd*delta/duration,sdd*delta/duration**2);rate.sleep()
        self.stage='settling'; self.wait(target,zeros,zeros,12.)
        error=np.linalg.norm(np.array(self.positions)+self.offsets-target,axis=1)
        if error.max()>.2:raise RuntimeError('Initial UAV tracking error exceeds 20 cm: '+str(error))
        self.stage='transport'; start=rospy.Time.now().to_sec()
        self.transport_start=start
        while not rospy.is_shutdown():
            t=rospy.Time.now().to_sec()-start
            if t>self.ref.t[-1]:break
            if any(not s.armed or s.mode!='OFFBOARD' for s in self.states):raise RuntimeError('PX4 left OFFBOARD or disarmed')
            if max(time.monotonic()-x for x in self.received)>1:raise RuntimeError('Stale UAV position estimate')
            self.send(*self.ref.at(t));rate.sleep()
        self.stage='final_hold'; self.wait(self.ref.at(self.ref.t[-1])[0],zeros,zeros,4.)
        for mode in self.mode:mode(0,'AUTO.LAND')
        self.stage='landing'; start=rospy.Time.now().to_sec()
        while not rospy.is_shutdown() and rospy.Time.now().to_sec()-start<25 and any(s.armed for s in self.states):rospy.sleep(.1)
        return {'status':'completed','transport_duration_s':float(self.ref.t[-1]),'video_frames':self.frames,
                'video_start_sim_s':self.video_start,'video_end_sim_s':self.video_end,
                'all_disarmed':all(not s.armed for s in self.states)}
    def close(self):
        self.recording=False; self.closed=True
        time.sleep(.2)
        if self.video:
            self.video.stdin.close(); self.video.wait(timeout=30)
        for f in [self.states_log,self.cables_log,self.command_log,self.estimate_log]:f.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--ffmpeg',required=True);args=p.parse_args()
    rospy.init_node('sling_executor'); executor=Executor(args)
    try:result=executor.run()
    except Exception as exc:
        rospy.logerr(str(exc)); result={'status':'failed','reason':str(exc),'stage':executor.stage}
        for mode in executor.mode:
            try:mode(0,'AUTO.LAND')
            except rospy.ServiceException:pass
    finally:executor.close()
    (args.root/'execution.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if result['status']!='completed':raise SystemExit(2)

if __name__=='__main__':main()
