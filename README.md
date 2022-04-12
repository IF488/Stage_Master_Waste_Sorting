# Stage Master IAR (Internship Master AIR)
The internship was done at Université des Mascareignes in the AIR Lab. The task assigned to the author was the development of Autonomous Waste Sorting Systems using Artificial Intelligence and Robotics. This repository contains the source code developed during the internship, along with the dataset used, and the MobileNetV2 model.

## Waste Sorting Robot Arm System
The Waste Sorting Robot Arm System requires a PC with ROS Noetic connected to the Panda robot arm via Franka ROS, and a Raspberry Pi with OpenCV and Tensorflow lite.
Three items in this repository are needed:
- The *panda_command* ROS package
- The *Pi_trash_classifier.py* Raspberry Pi algorithm
- The *Mob_garbage_v2.tflite* MobileNetV2 model

### Installation
In order to run the Waste Sorting Robot Arm System, the *panda_command* ROS package shall first be cloned in the *src* folder of your catkin workspace. Once cloned, you shall *build* and *source* your catkin workspace.

Then, the *Pi_trash_classifier.py* and the *Mob_garbage_v2.tflite* shall be placed in your Raspberry Pi in one and the same folder.

### Execution
1. Open a first terminal on your PC and launch the *robot.launch* file available in *panda_command* using your Panda robot's IP as argument.

2. In a second terminal on your PC, run the *ABS_classified_goto_udp.py* program located in *panda_command*. 

3. On your Raspberry Pi, run the *Pi_trash_classifier.py* program using Python 3.

NB: An UDP client/server is used for communication between PC and Raspberry Pi, therefore, the PC IP address shall be set in *Pi_trash_classifier.py* and the Port number shall be set on both *ABS_classified_goto_udp.py* and *Pi_trash_classifier.py*.

## Intelligent Waste Sorting Bin
The Intelligent Waste Sorting Bin requires a Raspberry Pi with OpenCV and Tensorflow lite, and a PCA9685 card. Only the *test_PCA9685* folder is needed, it contains the Python algorithm, the MobileNetV2 model and the PCA9685 libraries. 

### Installation
Clone the *test_PCA9685* folder in your Raspberry Pi.

### Execution
Run the *trashbin_v3.py* program using Python3.

## Deep Learning Classifier comparison
The algorithm used to train our models are available in the following notebooks:
- *ResNet34_trashnet.ipynb* : The notebook used to train our ResNet34 model
- *trashnet_CNN.ipynb* : The notebook used to train our CNN model
- *trashnet_mobileNet.ipynb* : The notebook used to train our MobileNetV2 model

The dataset used in this study is availble in *dataset-resized* folder. The dataset is a modified version of [Trashnet](https://github.com/garythung/trashnet)  by G. Thung.