# -*- coding: utf-8 -*-
"""
Stage Master: Waste Sorting using AI and Robotics

@author: Ishan FOOLELL
Master IAR
Universite des Mascareignes

Description: Python program for Smart Waste Sorting Bin
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
import tflite_runtime.interpreter as tflite
from SunFounder_PCA9685 import Servo
from tkinter import *
import tkinter as tk
import RPi.GPIO as GPIO



def task1():
    """ Object detection """
    global detected
    global third_image
    global absolute_difference
    print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 1: {}".format(os.getpid()))
    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(camera, size=(1600, 900))
    time.sleep(0.1)
    first_frame = None
    kernel = np.ones((20,20),np.uint8)
    #kernel2 = np.ones((1,1), np.uint8)
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Window', 640, 360)
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        third_image = image[60:900, 320:1600]
        sub_image = image[60:900, 320:1600]
        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        gray = cv2.medianBlur(gray,9)
        if first_frame is None:
            first_frame = gray
            raw_capture.truncate(0)
            continue
        absolute_difference = cv2.absdiff(first_frame, gray)
        absolute_difference = cv2.adaptiveThreshold(absolute_difference, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        #absolute_difference = cv2.erode(absolute_difference, kernel2, iterations=1)
        #absolute_difference = cv2.dilate(absolute_difference, kernel2, iterations=1)
        contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        area = [cv2.contourArea(c) for c in contours]
        if len(area) < 30:
            detected = False
            cv2.imshow('Resized Window', image)
            #cv2.imshow('Abs diff Threshold',absolute_difference)
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
            classified = "type: " + str(predicted_class)
            #cv2.putText(image, classified, (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #cv2.drawContours(sub_image, unified,-1,(0,255,0),2)
        textconfirm = "confirm: " + str(confirm)
        #cv2.putText(image, textconfirm, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Resized Window', image)
        key = cv2.waitKey(1) & 0xFF
        raw_capture.truncate(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
           


def task2():
    """ Servo control for pan and tilt and container door """
    global confirm
    print("Task 2 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 2: {}".format(os.getpid()))
    while True:
        if detected == False:
            time.sleep(1)
            myservo[0].write(90)
            myservo[1].write(90)
            myservo[2].write(90)
        elif detected == True:
            if confirm == True:
                time.sleep(5)
                if predicted_class == "plastic":
                    myservo[0].write(90)
                    myservo[1].write(135)
                    time.sleep(0.5)
                    myservo[2].write(0)
                elif predicted_class == "metal":
                    myservo[0].write(0)
                    myservo[1].write(135)
                    time.sleep(0.5)
                    myservo[2].write(0)
                elif predicted_class == "paper" or predicted_class == "cardboard":
                    myservo[0].write(0)
                    myservo[1].write(45)
                    time.sleep(0.5)
                    myservo[2].write(0)
                else:
                    myservo[0].write(90)
                    myservo[1].write(45)
                    time.sleep(0.5)
                    myservo[2].write(0)
            else:
                time.sleep(5)
                if predicted_class == "plastic":
                    myservo[0].write(90)
                    myservo[1].write(135)
                    time.sleep(0.5)
                    myservo[2].write(0)
                elif predicted_class == "metal":
                    myservo[0].write(0)
                    myservo[1].write(135)
                    time.sleep(0.5)
                    myservo[2].write(0)
                elif predicted_class == "paper" or predicted_class == "cardboard":
                    myservo[0].write(0)
                    myservo[1].write(45)
                    time.sleep(0.5)
                    myservo[2].write(0)
                else:
                    myservo[0].write(90)
                    myservo[1].write(45)
                    time.sleep(0.5)
                    myservo[2].write(0)


# Program to find most frequent
# element in a list
def most_frequent(List):
    return max(set(List), key = List.count)


def task3():
    """ Trash classification """
    global predicted_class
    global predict_list
    global confirm
    print("Task 2 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 2: {}".format(os.getpid()))
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
        #third_image_rgb = gammaCorrection(third_image_rgb, 2)
        imH, imW, _ = third_image.shape 
        image_resized = cv2.resize(third_image_rgb, (input_width, input_height))
        if detected == False:
            predicted_class = "None"
            predict_list[:] = []
            confirm = False
        else:
            input_data = np.array(image_resized, dtype=np.float32)
            #input_data = (input_data)/255.0
            input_data = np.expand_dims(input_data, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)
            MaxProb = np.max(results)
            #print(output_data)
            #print("MAX: ", MaxProb)
            if MaxProb >= 0.85:
                predict_list.append(np.argmax(results))
            #print(predict_list)
            if len(predict_list) > 0 :
                #print(most_frequent(predict_list))
                predicted_class = str(labels[most_frequent(predict_list)])
            else:
                predicted_class = "trash"
                
def update():
    """ update predicted class """
    lbl.config(text = predicted_class)
    lbl.after(1000, update)
    
def confirmation():
    """ OK button """
    global confirm
    confirm = True
    btn3.place_forget()
    btn4.place_forget()
    btn5.place_forget()
    btn6.place_forget()
    
def other():
    """ Other button """
    btn3.place(x = 500, y= 500)
    btn4.place(x = 700, y= 500)
    btn5.place(x = 500, y= 600)
    btn6.place(x = 700, y= 600)
    
def plastic():
    """ Plastic button """
    global predicted_class
    global predict_list
    global p
    if detected == True:
        filename = "dataset/plastic/plastic{0}.jpg".format(p)
        cv2.imwrite(filename, third_image)
        p+=1
    for i in range(20):
        predict_list.append(4)

def metal():
    """ Metal button """
    global predicted_class
    global predict_list
    global m
    global third_image
    if detected == True:
        filename = "dataset/metal/metal{0}.jpg".format(m)
        cv2.imwrite(filename, third_image)
        m+=1
    for i in range(20):
        predict_list.append(2)
        
def paper():
    """ Paper button """
    global predicted_class
    global predict_list
    global r
    if detected == True:
        filename = "dataset/paper/paper{0}.jpg".format(r)
        cv2.imwrite(filename, third_image)
        r+=1
    for i in range(20):
        predict_list.append(3)
        
def trash():
    """ Trash button """
    global predicted_class
    global predict_list
    global t
    if detected == True:
        filename = "dataset/trash/trash{0}.jpg".format(t)
        cv2.imwrite(filename, third_image)
        t+=1
    for i in range(20):
        predict_list.append(5)


def task4():
    """ Graphical User Interface """
    global lbl, btn3, btn4, btn5, btn6
    global predict_list
    print("Task 4 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 4: {}".format(os.getpid()))
    
    #while True:
    window = Tk()
    window.title("Trashbin app")
    window.geometry('1280x720')
    window.configure(bg='white')
    bgimg = PhotoImage(file = "logo.png")
    logo = Label(window, image = bgimg)
    logo.place(x=400, y=20)
    lbl = Label(window, font=("Arial Bold", 30), bg="#ffffff")
    lbl.place(x=610, y=260)
    update()
    btnOK = Button(window, text="OK", fg='black', font=(None, 20), bg='green', command=confirmation, height=3, width=7)
    btnOK.place(x = 700, y= 350)    
    btnOther = Button(window, text="Other", fg='black', font=(None, 20), bg='yellow', command=other, height=3, width=7)
    btnOther.place(x = 500, y= 350)
    btn3 = Button(window, text="Plastic", fg='white', font=(None, 20), bg='blue', command=plastic, height=2, width=7)
    btn4 = Button(window, text="Metal", fg='white', font=(None, 20), bg='blue', command=metal, height=2, width=7)
    btn5 = Button(window, text="Paper", fg='white', font=(None, 20), bg='blue', command=paper, height=2, width=7)
    btn6 = Button(window, text="Trash", fg='white', font=(None, 20), bg='blue', command=trash, height=2, width=7)
    window.mainloop()
  

def task5():
    """ Bin opening """
    global waiting
    print("Task 5 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 5: {}".format(os.getpid()))
    sensorPin = 11
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(sensorPin, GPIO.IN)
    time.sleep(5)
    while True:
        if GPIO.input(sensorPin)==GPIO.HIGH and detected==False:
            time.sleep(1)
            myservo[3].write(0)
            time.sleep(5)
            myservo[3].write(90)
        elif GPIO.input(sensorPin)==GPIO.HIGH and detected==True:
            time.sleep(1)
            if GPIO.input(sensorPin)==GPIO.HIGH:
                waiting = True
        elif waiting == True and detected == False:
            time.sleep(1)
            waiting = False
            myservo[3].write(0)
            time.sleep(5)
            myservo[3].write(90)
        else:
            myservo[3].write(90)
        time.sleep(1)                


if __name__ == "__main__":
    
    # Initialize the camera
    camera = PiCamera()
    # Set the camera resolution
    camera.resolution = (1600, 900)
    # Set the number of frames per second
    camera.framerate = 30

    # Initialize values
    detected = False
    predicted_class = "None"
    confirm = False
    waiting = False
    global p
    listp = os.listdir("dataset/plastic")
    p = len(listp)
    global m
    listm = os.listdir("dataset/metal")
    m = len(listm)
    global r
    listr = os.listdir("dataset/paper")
    r = len(listr)
    global t
    listt = os.listdir("dataset/trash")
    t = len(listt)
    
    # Servo setup
    myservo = []
    myservo.append(Servo.Servo(0, bus_number=1))  # channel 0
    myservo.append(Servo.Servo(2, bus_number=1))  # channel 2
    myservo.append(Servo.Servo(4, bus_number=1))  # channel 4
    myservo.append(Servo.Servo(6, bus_number=1))  # channel 6
    Servo.Servo(0, bus_number=1).setup()
    Servo.Servo(2, bus_number=1).setup()
    Servo.Servo(4, bus_number=1).setup()
    Servo.Servo(6, bus_number=1).setup()
    print(myservo)
    myservo[0].write(90)
    myservo[1].write(90)
    myservo[2].write(90)
    myservo[3].write(90)
    
    # Deep Learning Classifier Setup
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    interpreter = tflite.Interpreter(model_path="Mob_garbage_v2.tflite")
    predict_list = []
    
    # print ID of current process
    print("ID of process running main program: {}".format(os.getpid()))
  
    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))
  
    # creating threads
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2') 
    t3 = threading.Thread(target=task3, name='t3')
    t4 = threading.Thread(target=task4, name='t4')
    t5 = threading.Thread(target=task5, name='t5')
  
    # starting threads
    t1.start()
    time.sleep(2)
    t3.start()
    time.sleep(1)
    t4.start()
    time.sleep(1)
    t2.start()
    time.sleep(1)
    t5.start()
  

