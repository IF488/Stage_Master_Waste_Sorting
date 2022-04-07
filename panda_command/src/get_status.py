#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Pose
from franka_gripper.msg import MoveAction, MoveGoal
import actionlib
import time
import yaml
import rospkg




if __name__ == '__main__':
    rospy.init_node('get_joint_status')
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    commander = MoveGroupCommander('panda_arm')
    joint=commander.get_current_joint_values()
    ee_pose=commander.get_current_pose().pose