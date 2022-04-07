# -*- coding: utf-8 -*-
"""
* Filename    : Pi_trash_classifier.py
* Description : A program to detect, classify, retrieve angle,X,Y position of trash and send data using UDP
* Author      : Ishan FOOLELL
* University  : Universite des Mascareignes
* E-mail      : ifoolell@student.udm.ac.mu
* Course      : Master Artificial Intelligence and Robotics
* Version     : v2.0.0
"""


import threading
import os
import time
from collections import deque
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
from imutils.video import VideoStream
import numpy as np   
import argparse
import cv2
import imutils
import sys
import socket
import json
import tflite_runtime.interpreter as tflite



def find_if_close(cnt1,cnt2):
    """ Find if two contours are close to each other based on distance"""
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False


def get_center_angle(thresh):
    """ Detect contour and retrieve X, y, width, height, angle of contour """
    contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    area = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(area)
    cnt = contours[max_index]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    center = (int(rect[0][0]),int(rect[0][1]))
    width = int(rect[1][0])
    height = int(rect[1][1])
    myangle = int(rect[2])
    if width < height:
        myangle = 90 - myangle
    else:
        myangle = -myangle
    x = center[0] + 320
    y = center[1] + 60
    return x,y,width,height,myangle


def gammaCorrection(src, gamma):
    """ Function for Gamma Correction """
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)


def most_frequent(List):
    """ Function to find most common item in list """
    return max(set(List), key = List.count)


def task1():
    """ Main Function to detect trash and retrieve data """
    global xR
    global yR
    global detected
    global angle
    global third_image
    global absolute_difference
    print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 1: {}".format(os.getpid()))
    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(camera, size=(1920, 1080))
    time.sleep(0.1)
    first_frame = None
    kernel = np.ones((20,20),np.uint8)
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Window', 1600, 900)
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        third_image = image[60:1080, 320:1600]  # For classification with Deep Learning
        sub_image = image[60:1080, 320:1600]    # For trash detection
        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        gray = cv2.medianBlur(gray,9)
        if first_frame is None:
            first_frame = gray
            raw_capture.truncate(0)
            continue
        absolute_difference = cv2.absdiff(first_frame, gray)
        absolute_difference = cv2.adaptiveThreshold(absolute_difference, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        area = [cv2.contourArea(c) for c in contours]
        if len(area) < 30:
            detected = False
            cv2.imshow('Resized Window', image)
            key = cv2.waitKey(1) & 0xFF
            raw_capture.truncate(0)
            if key == ord("q"):
               break
            continue
        else:
            detected = True
            LENGTH = len(contours)
            status = np.zeros((LENGTH,1))
            for i,cnt1 in enumerate(contours):
                x = i    
                if i != LENGTH-1:
                    for j,cnt2 in enumerate(contours[i+1:]):
                        x = x+1
                        #dist = find_if_close(cnt1,cnt2)
                        dist = True
                        if dist == True:
                            val = min(status[i],status[x])
                            status[x] = status[i] = val
                        else:
                            if status[x]==status[i]:
                                status[x] = i+1
            unified = []
            maximum = int(status.max())+1
            for i in range(maximum):
                pos = np.where(status==i)[0]
                if pos.size != 0:
                    cont = np.vstack(contours[i] for i in pos)
                    hull = cv2.convexHull(cont)
                    unified.append(hull)
            new_thresh = cv2.drawContours(absolute_difference, unified,-1,255,-1)
            x,y,width,height,angle = get_center_angle(new_thresh)
            # Convert to centimeter
            x2 = x * CM_TO_PIXEL
            y2 = y * CM_TO_PIXEL
            # Transpose to robot frame            
            xR = y2 - dist_x_robot  # Distance from robot base to top of image = 40.6cm
            yR = x2 - dist_y_robot  # Distance from robot base to left side of image = 5.8cm
            # Convert to meters
            xR = xR/100
            yR = yR/100
            xR = np.round(xR,3)
            yR = np.round(yR,3)
            # Create text and box for image
            label = "  Rotation Angle: " + str(angle) + " degrees"
            textbox = cv2.rectangle(image, (0, 0), (350, 30), (255,255,255), -1)
            cv2.putText(image, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            cv2.circle(image,(x,y),4,(0,255,0),-1)
            text = "x: " + str(xR) + ", y: " + str(yR)
            cv2.putText(image, text, (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            classified = "type: " + str(predicted_class)
            cv2.putText(image, classified, (x + 40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.drawContours(sub_image, unified,-1,(0,255,0),2)
        # Show resulting image
        cv2.imshow('Resized Window', image)
        key = cv2.waitKey(1) & 0xFF
        raw_capture.truncate(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
        

def task2():
    """ UDP client which send data to PC running Franka ROS """
    my_ip = socket.gethostbyname(socket.getfqdn())
    print("Socket created")
    print("Task 2 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 2: {}".format(os.getpid()))
    while True:
        print(time.asctime()) #printing to the terminal values
        json_string = {}
        json_string  = {"x": xR, "y": yR, "angle": angle, "prediction": predicted_class, "detection": detected}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)#initalising socket and UDP connection
        #Message = str(json_string)
        Message = json.dumps(json_string)
        print('Message sent:= ', Message)
        sock.sendto(Message.encode('utf-8'),(UDP_IP, PORT))
        print("SENT to:-", UDP_IP, PORT, "From", my_ip)
        time.sleep(1)#delay
        

def task3():
    """ Deep Learning trash classification """
    global predicted_class
    print("Task 3 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 3: {}".format(os.getpid()))
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5
    while True:
        third_image_rgb = cv2.cvtColor(third_image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = third_image.shape 
        image_resized = cv2.resize(third_image_rgb, (input_width, input_height))
        if detected == False:
            predicted_class = "None"
            predict_list[:] = []
        else:
            input_data = np.array(image_resized, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)
            MaxProb = np.max(results)
            if MaxProb >= 0.85:
                predict_list.append(np.argmax(results))
            if len(predict_list) > 0 :
                predicted_class = str(labels[most_frequent(predict_list)])
            else:
                predicted_class = "trash"
                
 
  
if __name__ == "__main__":
    
    # Initialize the camera
    camera = PiCamera()
    # Set the camera resolution
    camera.resolution = (1920, 1080)
    # Set the number of frames per second
    camera.framerate = 30
    #Centimeter to pixel ratio
    # 100cm measured on the workspace plane from top to bottom of the image
    CM_TO_PIXEL = 100.0 / 1920
    # Distance from robot base to image origin in centimiter
    dist_x_robot = 40.6
    dist_y_robot = 5.8
    #Initialize values for task 1
    xR = 0.3
    yR = 0.0
    detected = False
    angle = 0
    predicted_class = "None"
    # IP Address and Port for task 2
    UDP_IP = "10.42.0.64"
    PORT = 5008
    # Labels for classifier task 3
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    # Path of classifier 
    interpreter = tflite.Interpreter(model_path="Mob_garbage_v2.tflite")
    # Initialize Predict List 
    predict_list = []

    # print ID of current process
    print("ID of process running main program: {}".format(os.getpid()))
  
    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))
  
    # creating threads
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2') 
    t3 = threading.Thread(target=task3, name='t3') 
  
    # starting threads
    t1.start()
    time.sleep(2)
    t3.start()
    time.sleep(1)
    t2.start()
  

