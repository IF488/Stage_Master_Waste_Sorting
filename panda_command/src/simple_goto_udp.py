#!/usr/bin/env python3

import rospy
import control_arm_toolbox as catb
from actionlib_msgs.msg import GoalStatusArray
from moveit_commander import MoveGroupCommander
import socket
import json
import time
import threading
import os

detected = False
x_client = 0.3
y_client = 0.0

port = 5012

def task1():
    global x_client
    global y_client
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
        detected = result['detection']
        print("detection: ", detected)

def pick_and_place():
    while True:
        #print("x: ", x_client)
        #print("y: ", y_client)
        #print("detection: ", detected)
        if detected == False:
            catb.go_to_ready(commander)          
            time.sleep(0.5)
        elif detected == True:
            #pick and place
            catb.go_to_ready(commander)
            catb.control_gripper(0.08,0.02)
            ee_pose=commander.get_current_pose().pose
            #pre-pose    
            ee_pose_first=ee_pose
            ee_pose_first.position.x= 0.25
            ee_pose_first.position.y=0.25
            ee_pose_first.position.z=0.6
            catb.cartesian_go_to(commander,ee_pose_first)
            #pick
            ee_pose_desired=ee_pose
            ee_pose_desired.position.x= x_client
            ee_pose_desired.position.y= y_client
            ee_pose_desired.position.z= 0.3
            catb.cartesian_go_to(commander,ee_pose_desired)
            ee_pose_pick=ee_pose
            ee_pose_pick.position.z=0.12
            catb.cartesian_go_to(commander,ee_pose_pick)
            #gripper
            catb.control_gripper(0.023,0.2)
            #ready
            catb.go_to_ready(commander)
            #place
            ee_pose_place=ee_pose
            ee_pose_place.position.x= 0.1
            ee_pose_place.position.y= -0.5
            ee_pose_place.position.z=0.2
            catb.cartesian_go_to(commander,ee_pose_place)
            #drop
            catb.control_gripper(0.08,0.02)
            #ready
            catb.go_to_ready(commander)
        time.sleep(1)

"""        
def task2(): 

    if detected == False:
        catb.go_to_ready(commander)          
        time.sleep(0.5)
    elif detected == True:
        time.sleep(2)
        print(x_client, y_client)
        pick_and_place(x_client, y_client)
        time.sleep(0.5)
"""

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

