import yaml
import rospkg
import rospy
from geometry_msgs.msg import Pose

def dict_to_pose(iPoseDict):
	oPose=Pose()
	oPose.position.x=iPoseDict['position']['x']
	oPose.position.y=iPoseDict['position']['y']
	oPose.position.z=iPoseDict['position']['z']
	oPose.orientation.x=iPoseDict['orientation']['x']
	oPose.orientation.y=iPoseDict['orientation']['y']
	oPose.orientation.z=iPoseDict['orientation']['z']
	oPose.orientation.w=iPoseDict['orientation']['w']
	return oPose

def load_yaml(iPath):
    with open(iPath,"r") as stream:
        oData=yaml.safe_load(stream)
    return oData


if __name__ == '__main__':
	rospack=rospkg.RosPack()
	file_path=rospack.get_path("panda_command")+"/config/poseList.yaml"
	data_loaded=load_yaml(iPath)
	print("data loaded from yaml file: %s"%(file_path))
	print(data_loaded)

	print("Conversion from yaml pose to geometry_msgs Pose")
	mPose=dictToPose(data_loaded["prepicking_pose"])
	print(mPose)