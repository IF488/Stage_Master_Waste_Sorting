#!/usr/bin/env python3
import rospy
import read_yaml
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
import control_arm_toolbox as catb
import rospkg

def load_pose():  
    rospack=rospkg.RosPack()
    file_path=rospack.get_path("panda_command")+"/config/poseList.yaml"
    lData=read_yaml.load_yaml(file_path)
    
    oPrePickingPose=read_yaml.dict_to_pose(lData["prepicking_pose"])
    oPickingPose=read_yaml.dict_to_pose(lData["picking_pose"])
    oPrePlacePose=read_yaml.dict_to_pose(lData["preplace_pose"])
    oPlacePose=read_yaml.dict_to_pose(lData["place_pose"])
        
    return oPrePickingPose, oPickingPose, oPrePlacePose,oPlacePose 

def place_and_pick(iCommander):
    mPrepickingPose, mPickingPose, mPrePlacePose, mPlacePose=load_pose()

    #step 0 open gripper
    catb.open_gripper()

    #step one go to start position
    catb.go_to_ready(iCommander)

    #step two go to prepicking place
    catb.cartesian_go_to(iCommander, mPrePlacePose)
    
    #step three go to picking place
    catb.cartesian_go_to(iCommander, mPlacePose)
    
    #step four pick the cube
    catb.control_gripper(0.020,0.1)

    #step five go to prepicking place
    catb.cartesian_go_to(iCommander, mPrePlacePose)
    
    #step six go to start position
    catb.go_to_ready(iCommander)
    
    #step seven go to preplace place
    catb.cartesian_go_to(iCommander,mPrepickingPose)

    #step eight go to place place
    catb.cartesian_go_to(iCommander, mPickingPose)
    
    #step nine drop the cube
    catb.open_gripper()

    #step ten go to preplace place
    catb.cartesian_go_to(iCommander,mPrepickingPose)

    #step eleven go to start position
    catb.go_to_ready(iCommander)

def pick_and_place(iCommander):
    mPrepickingPose, mPickingPose, mPrePlacePose, mPlacePose=load_pose()

    #step 0 open gripper
    catb.open_gripper()

    #step one go to start position
    catb.go_to_ready(iCommander)

    #step two go to prepicking place
    catb.cartesian_go_to(iCommander, mPrepickingPose)
    
    #step three go to picking place
    catb.cartesian_go_to(iCommander, mPickingPose)
    
    #step four pick the cube
    catb.control_gripper(0.020,0.1)

    #step five go to prepicking place
    catb.cartesian_go_to(iCommander, mPrepickingPose)
    
    #step six go to start position
    catb.go_to_ready(iCommander)
    
    #step seven go to preplace place
    catb.cartesian_go_to(iCommander,mPrePlacePose)


    #step eight go to place place
    catb.cartesian_go_to(iCommander, mPlacePose)
 
    
    #step nine drop the cube
    catb.open_gripper()


    #step ten go to preplace place
    catb.cartesian_go_to(iCommander,mPrePlacePose)


    #step eleven go to start position
    catb.go_to_ready(iCommander)

def demo_pick_and_place_loop(iCommander):
    for i in range(5):
        pick_and_place(iCommander)
        place_and_pick(iCommander)


def main():
    rospy.init_node('demo_pick_and_place')
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    commander = MoveGroupCommander('panda_arm')
    commander.set_max_velocity_scaling_factor(0.1)
    commander.set_max_acceleration_scaling_factor(0.1)
    for i in range(1):
        pick_and_place(commander)
        place_and_pick(commander)

if __name__ == '__main__':
    main()
