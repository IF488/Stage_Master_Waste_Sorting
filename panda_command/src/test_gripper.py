#!/usr/bin/env python3

import rospy
import control_arm_toolbox as catb



def main():
	rospy.init_node("test_gripper")
	catb.open_gripper()
	for i in range(1,10):
		# more than 0.1ms the gripper sound weird.
		print(i*0.01)
		catb.control_gripper(0, i*0.01)
		catb.control_gripper(0.08,i*0.01)


if __name__ == '__main__':
	main()