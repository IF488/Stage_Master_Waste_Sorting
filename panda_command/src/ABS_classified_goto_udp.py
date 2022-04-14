#!/usr/bin/env python3

"""
* Filename    : ABS_classified_goto_udp.py
* Description : Pick and Place Program developed for the Waste Sorting Robot Arm System
* Author      : Ishan FOOLELL
* University  : Universite des Mascareignes
* E-mail      : ifoolell@student.udm.ac.mu
* Course      : Master Artificial Intelligence and Robotics
* Version     : v2.0.0
"""


import rospy
import control_arm_toolbox as catb
from actionlib_msgs.msg import GoalStatusArray
from moveit_commander import MoveGroupCommander
from franka_gripper.msg import GraspEpsilon
import numpy as np
import math
import socket
import json
import time
import threading
import os


# Parameters Initialisation
detected = False
x_client = 0.3
y_client = 0.0
port = 5006
iEpsilon = GraspEpsilon()
iEpsilon.inner=0.02
iEpsilon.outer = 0.04
predict_list = []


def most_frequent(List):
    """ Return Most frequent item in list """
    return max(set(List), key = List.count)

def task1():
    """ UDP server """
    global x_client
    global y_client
    global angle
    global prediction
    global detected
    print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 1: {}".format(os.getpid()))
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind(("", port))
    print("server started")
    print("Socket is listening")
    while True:
        recieved = sock.recvfrom(1024)
        #print(type(recieved))
        #print(recieved)
        #print(time.asctime()) #bytes object
        result = json.loads(recieved[0].decode('utf-8'))
        x_client = result['x']
        print("x: ", x_client)
        y_client = result['y']
        print("y: ", y_client)
        iAngle = result['angle']
        angle = iAngle - 50 + 90
        print("angle: ", angle)
        prediction = result['prediction']
        print("predicted class: ", prediction)
        detected = result['detection']
        print("detection: ", detected)


def get_quaternion_from_euler(roll, pitch, yaw):
    """ Converts Euler angle into quaternion """
    roll = np.round(np.deg2rad(roll), 2)
    pitch = np.round(np.deg2rad(pitch), 2)
    yaw = np.round(np.deg2rad(yaw), 2)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def pick_and_place():
    """ Performs pick and place with Panda robot """
    while True:
        #print("x: ", x_client)
        #print("y: ", y_client)
        #print("detection: ", detected)
        if detected == False:
            catb.go_to_ready(commander)          
            time.sleep(0.5)
        elif detected == True:
            time.sleep(1.5)  
            #pick and place
            catb.go_to_ready(commander)
            catb.control_gripper(0.08,0.02)
            ee_pose=commander.get_current_pose().pose
            predict_list = []
            #pre-pose    
            ee_pose_first=ee_pose
            ee_pose_first.position.x= 0.25
            ee_pose_first.position.y=0.25
            ee_pose_first.position.z=0.6
            catb.cartesian_go_to(commander,ee_pose_first)
            #pick
            qx, qy, qz, qw = get_quaternion_from_euler(-180,-0,angle) 
            ee_pose_desired=ee_pose
            ee_pose_desired.position.x= x_client
            ee_pose_desired.position.y= y_client
            ee_pose_desired.position.z= 0.6
            ee_pose_desired.orientation.x = qx
            ee_pose_desired.orientation.y = qy
            ee_pose_desired.orientation.z = qz
            ee_pose_desired.orientation.w = qw
            catb.cartesian_go_to(commander,ee_pose_desired)
            ee_pose_first=ee_pose
            ee_pose_first.position.z=0.4
            catb.cartesian_go_to(commander,ee_pose_first)
            ee_pose_pick=ee_pose
            ee_pose_pick.position.z=0.18
            catb.cartesian_go_to(commander,ee_pose_pick)
            #gripper
            catb.grasp(0.04, iEpsilon, 0.1, 0.1)
            #ready
            catb.go_to_ready(commander)
            
            predict_list.append(prediction)
            predicted_class = str(most_frequent(predict_list))
            print(predicted_class)
            #place
            ee_pose_place=ee_pose
            if predicted_class == "paper" or predicted_class == "cardboard":
                ee_pose_place.position.x= 0.1
                ee_pose_place.position.y= -0.5
                ee_pose_place.position.z=0.4
            elif predicted_class == "plastic":
                ee_pose_place.position.x= 0.3
                ee_pose_place.position.y= -0.3
                ee_pose_place.position.z=0.4
            elif predicted_class == "metal":
                ee_pose_place.position.x= 0.4
                ee_pose_place.position.y= 0.1
                ee_pose_place.position.z=0.4
            else:
                ee_pose_place.position.x= 0.6
                ee_pose_place.position.y= 0.0
                ee_pose_place.position.z=0.4
            catb.cartesian_go_to(commander,ee_pose_place)
            #drop
            catb.control_gripper(0.08,0.02)
            #ready
            catb.go_to_ready(commander)
            predict_list[:] = []
        time.sleep(1)


if __name__ == '__main__':
    rospy.init_node('simple_goto')
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    commander = MoveGroupCommander('panda_arm')
    # print ID of current process
    print("ID of process running main program: {}".format(os.getpid()))
  
    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))
  
    # creating threads
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=pick_and_place, name='t2')  
  
    # starting threads
    t1.start()
    time.sleep(2)
    t2.start()
  
    # wait until all threads finish
    #t1.join()
    #t2.join()

