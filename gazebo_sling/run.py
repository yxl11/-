#!/usr/bin/python3
"""Run an isolated ROS master, Gazebo ODE, three PX4 instances and MAVROS."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/gazebo_transport');p.add_argument('--gui',action='store_true')
    p.add_argument('--snapshot',action='store_true',help='Save a camera frame from the running isolated simulation')
    p.add_argument('--ffmpeg',default=str(ROOT/'.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'))
    args=p.parse_args(); root=args.root.resolve();cfg=json.loads((root/'runtime.json').read_text());px4=Path(cfg['px4'])
    if (root/'execution.json').exists() and not args.snapshot:raise SystemExit('Run already exists; prepare a fresh --root.')
    env=os.environ.copy();env.update({'ROS_MASTER_URI':'http://127.0.0.1:11321','ROS_HOME':str(root/'ros_home'),
        'ROS_LOG_DIR':str(root/'ros_logs'),'GAZEBO_MASTER_URI':'http://127.0.0.1:11351','GAZEBO_MODEL_DATABASE_URI':'',
        'PX4_SIM_MODEL':'iris','PX4_ESTIMATOR':'ekf2','QT_X11_NO_MITSHM':'1'})
    env['GAZEBO_PLUGIN_PATH']=str(ROOT/'build/gazebo_sling')+':'+str(px4/'build/px4_sitl_default/build_gazebo')+':/opt/ros/noetic/lib'
    env['GAZEBO_MODEL_PATH']=str(px4/'Tools/sitl_gazebo/models')+':/usr/share/gazebo-11/models'
    env['GAZEBO_RESOURCE_PATH']='/usr/share/gazebo-11:'+str(px4/'Tools/sitl_gazebo')
    env['PATH']=str(px4/'build/px4_sitl_default/bin')+':/opt/ros/noetic/bin:'+env['PATH']
    env['LD_LIBRARY_PATH']='/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:/opt/ros/noetic/lib:'+env.get('LD_LIBRARY_PATH','')
    # Keep ROS Python on the system interpreter, independent of the research venv.
    env['PYTHONPATH']='/opt/ros/noetic/lib/python3/dist-packages'
    if args.snapshot:
        code="import rospy,cv2,numpy as np; from sensor_msgs.msg import Image; rospy.init_node('sling_snapshot',anonymous=True); m=rospy.wait_for_message('/sling_camera/image_raw',Image,timeout=20); a=np.frombuffer(m.data,dtype=np.uint8).reshape(m.height,m.width,3); cv2.imwrite('camera_snapshot.png',a[:,:,::-1] if m.encoding=='rgb8' else a)"
        subprocess.run(['/usr/bin/python3','-c',code],cwd=root,env=env,check=True)
        return
    processes=[]; files=[]
    def start(name,cmd,cwd=root):
        f=(root/(name+'.log')).open('w');files.append(f)
        proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc);return proc
    try:
        start('roscore',['/opt/ros/noetic/bin/roscore','-p','11321']);time.sleep(3)
        start('gazebo',['gzserver','--verbose','-s','libgazebo_ros_api_plugin.so',str(root/'transport.world')])
        for i in range(3):
            work=root/('px4_%d'%i);work.mkdir(exist_ok=True)
            start('px4_%d'%i,[str(px4/'build/px4_sitl_default/bin/px4'),str(root/'etc'),'-s','etc/init.d-posix/rcS','-i',str(i),'-w',str(work),'-d'],work)
        start('mavros',['/opt/ros/noetic/bin/roslaunch',str(root/'mavros.launch')])
        start('localization',['/usr/bin/python3',str(ROOT/'gazebo_sling/localization.py')])
        if args.gui:start('gazebo_gui',['gzclient'])
        time.sleep(5)
        executor=start('executor',['/usr/bin/python3',str(ROOT/'gazebo_sling/execute.py'),'--root',str(root),'--ffmpeg',args.ffmpeg])
        deadline=time.monotonic()+900
        while executor.poll() is None:
            if time.monotonic()>deadline:raise RuntimeError('Simulation exceeded 15 minute wall-time deadline')
            if any(proc.poll() is not None for proc in processes[:5]):raise RuntimeError('ROS/Gazebo/PX4 process exited; inspect logs')
            time.sleep(1)
        if executor.returncode:raise SystemExit(executor.returncode)
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGINT)
                except ProcessLookupError:pass
        for proc in processes:
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        for f in files:f.close()

if __name__=='__main__':main()
