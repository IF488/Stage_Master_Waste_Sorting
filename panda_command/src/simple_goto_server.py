#!/usr/bin/env python3

import rospy
import control_arm_toolbox as catb
from actionlib_msgs.msg import GoalStatusArray
from moveit_commander import MoveGroupCommander
import socket
import json
import time

#detected = False
#x_client = 0.3
#y_client = 0.0

port = 9990

def pick_and_place(xR,yR):
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
    ee_pose_desired.position.x= xR
    ee_pose_desired.position.y= yR
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
        
def main(): 
    try:
        sock = socket.socket()
        print("Socket created ...")

        sock.bind(('', port))
        sock.listen(5)
        print('socket is listening')
        c, addr = sock.accept()
        print('got connection from ', addr)
        while True:
            jsonReceived = c.recv(1024)
            #print("Json received -->", jsonReceived)
            #c.sendall(b'OK!')
            result = json.loads(jsonReceived.decode('utf-8'))
            x_client = result['x']
            print("x: ", x_client)
            y_client = result['y']
            print("y: ", y_client)
            detected = result['detection']
            print("detection: ", detected)
            if detected == False:
                catb.go_to_ready(commander)          
                time.sleep(0.5)

            elif detected == True:
                time.sleep(2)
                print(x_client, y_client)
                pick_and_place(x_client, y_client)
                time.sleep(0.5)
                
        #time.sleep(1)
    except:
        pass

if __name__ == '__main__':
    rospy.init_node('simple_goto')
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    commander = MoveGroupCommander('panda_arm')
    main()

