#!/usr/bin/env python3

import rospy
import control_arm_toolbox as catb
from actionlib_msgs.msg import GoalStatusArray
from moveit_commander import MoveGroupCommander
from franka_gripper.msg import GraspEpsilon
import numpy as np
import math


#iEpsilon = {'inner': 0.04, 'outer': 0.04 }
iEpsilon = GraspEpsilon()
iEpsilon.inner=0.04
iEpsilon.outer = 0.04

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    roll_x = np.round(np.rad2deg(roll_x), 2)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    pitch_y = np.round(np.rad2deg(pitch_y), 2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    yaw_z = np.round(np.rad2deg(yaw_z), 2)
    return roll_x, pitch_y, yaw_z


def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  roll = np.round(np.deg2rad(roll), 2)
  pitch = np.round(np.deg2rad(pitch), 2)
  yaw = np.round(np.deg2rad(yaw), 2)
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qx, qy, qz, qw]


def main():
    rospy.init_node('simple_goto')
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    commander = MoveGroupCommander('panda_arm')
    catb.go_to_ready(commander)
    catb.control_gripper(0.08,0.02)
    ee_pose=commander.get_current_pose().pose
    print(euler_from_quaternion(ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w))
    
    #pre-pose 
    qx, qy, qz, qw = get_quaternion_from_euler(-180,-0,40)   
    ee_pose_first=ee_pose
    ee_pose_first.position.x= 0.1
    ee_pose_first.position.y= -0.5
    ee_pose_first.position.z= 0.6
    ee_pose_first.orientation.x = qx
    ee_pose_first.orientation.y = qy
    ee_pose_first.orientation.z = qz
    ee_pose_first.orientation.w = qw
    catb.cartesian_go_to(commander,ee_pose_first)
    #pick
    ee_pose_desired=ee_pose
    ee_pose_desired.position.x= 0.1
    ee_pose_desired.position.y= -0.5
    ee_pose_desired.position.z= 0.3
    catb.cartesian_go_to(commander,ee_pose_desired)
    ee_pose_pick=ee_pose
    ee_pose_pick.position.z=0.171
    catb.cartesian_go_to(commander,ee_pose_pick)
    #gripper
    catb.grasp(0.04, iEpsilon, 0.1, 0.1)
    #ready
    catb.go_to_ready(commander)
    #place
    #ee_pose_place=ee_pose
    #ee_pose_place.position.x= 0.1
    #ee_pose_place.position.y= -0.5
    #ee_pose_place.position.z=0.2
    #catb.cartesian_go_to(commander,ee_pose_place)
    #drop
    catb.control_gripper(0.08,0.02)
    #ready
    #catb.go_to_ready(commander)
        
    

if __name__ == '__main__':
    main()
