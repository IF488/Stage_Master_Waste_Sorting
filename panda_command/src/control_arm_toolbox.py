#!/usr/bin/env python3

import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Pose
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspActionGoal, GraspGoal
import actionlib


def control_gripper(iWidth, iSpeed):
    client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    client.wait_for_server()
    goal = MoveGoal(width = iWidth, speed = iSpeed)
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))

def close_gripper():
    control_gripper(0, 0.1)

def open_gripper():
    control_gripper(0.08,0.1)

def test_closing_gripper():
    open_gripper()
    for i in range(9):
        control_gripper((0.08-i*0.01),0.04)
        time.sleep(2)

def test_opening_gripper():
    close_gripper()
    for i in range(9):
        control_gripper(i*0.01, 0.04)
        time.sleep(2)

def cartesian_go_to(iCommander,iPose):
    iCommander.set_pose_target(iPose)
    iCommander.go(wait=True)
    iCommander.stop()

def joint_go_to(iCommander, iJointArray):
    iCommander.go(iJointArray, wait=True)
    iCommander.stop()

def go_to_ready(iCommander):
    iCommander.set_named_target('ready')
    iCommander.go()
    
def grasp(iWidth, iEpsilon, iSpeed, iForce):
    client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
    client.wait_for_server()
    goal = GraspGoal(width = iWidth, epsilon = iEpsilon, speed = iSpeed, force = iForce)
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))
