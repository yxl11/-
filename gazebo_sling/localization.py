#!/usr/bin/python3
"""Ideal simulated UAV-only localization sensor, replacing indoor radar/LIO.

It sends UAV pose measurements to PX4 EKF2 through MAVROS vision_pose.
Payload and cable state never leave the evaluation topics.
"""
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped

def main():
    rospy.init_node('simulated_uav_localization')
    pubs=[rospy.Publisher('/uav%d/mavros/vision_pose/pose'%i,PoseStamped,queue_size=1) for i in range(1,4)]
    last=[0.]
    def callback(m):
        now=rospy.Time.now()
        if now.to_sec()-last[0]<.0199:return
        last[0]=now.to_sec()
        for i,pub in enumerate(pubs):
            name='uav%d'%(i+1)
            if name not in m.name:continue
            pose=PoseStamped();pose.header.stamp=now;pose.header.frame_id='map'
            pose.pose=m.pose[m.name.index(name)];pub.publish(pose)
    rospy.Subscriber('/gazebo/model_states',ModelStates,callback,queue_size=1)
    rospy.spin()

if __name__=='__main__':main()
