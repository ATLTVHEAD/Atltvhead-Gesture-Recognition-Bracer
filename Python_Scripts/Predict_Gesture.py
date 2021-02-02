# Predict_Gesture.py
# Description: Recieved Data from ESP32 Micro via the AGRB-Training-Data-Capture.ino file, make gesture prediction  
# Written by: Nate Damen
# Created on July 13th 2020
# Updated on JAN 29th 2021

import numpy as np 
import pandas as pd 
import datetime
import re
import os, os.path
import time
import random
import tensorflow as tf
import serial

PORT = "/dev/ttyUSB0"
#PORT = "/dev/ttyUSB1"
#PORT = "COM8"

serialport = None
serialport = serial.Serial(PORT, 115200, timeout=0.05)

#load Model
model = tf.keras.models.load_model('../Model/cnn_model.h5')

#Get Data from imu. Waits for incomming data and data stop
def get_imu_data():
    global serialport
    if not serialport:
        # open serial port
        serialport = serial.Serial(PORT, 115200, timeout=0.05)
        # check which port was really used
        print("Opened", serialport.name)
        # Flush input
        time.sleep(3)
        serialport.readline()

    # Poll the serial port
    line = str(serialport.readline(),'utf-8')
    if not line:
        return None
 
    vals = line.replace("Uni:", "").strip().split(',')
 
    if len(vals) != 7:
        return None
    try:
        vals = [float(i) for i in vals]
    except ValueError:
        return ValueError
 
    return vals

# Create Reshape function for each row of the dataset
def reshape_function(data):
    reshaped_data = tf.reshape(data, [-1, 3, 1])
    return reshaped_data


#Create a pipeline to process incomming data for the model to read and handle
def data_pipeline(data_a):
    tensor_set = tf.data.Dataset.from_tensor_slices((data_a[:,:,1:4]))
    tensor_set_cnn = tensor_set.map(reshape_function)
    tensor_set_cnn = tensor_set_cnn.batch(192)
    return tensor_set_cnn

def gesture_Handler(sock, rw, data, dataholder, dataCollecting, gesture, old_gesture):
    dataholder = np.array(get_imu_data())
    if dataholder.all() != None:
        #print(dataholder)
        dataCollecting = True
        data[0, rw, :] = dataholder
        rw += 1
        if rw > 380:
            rw = 380
    if dataholder.all() == None and dataCollecting == True:
        if rw == 380:
            prediction = np.argmax(model.predict(data_pipeline(data)), axis=1)
            gesture=gest_id[prediction[0]]
        rw = 0
        dataCollecting = False
    return rw, gesture, old_gesture, dataCollecting


if __name__ == "__main__":
    #define Gestures, current data, temp data holder, a first cylce boolean,
    gest_id = {0:'wave_mode', 1:'fist_pump_mode', 2:'random_motion_mode', 3:'speed_mode', 4:'pumped_up_mode'}
    data = np.zeros(shape=(1,380,7))
    dataholder = np.zeros(shape=(1,7))
    row = 0
    dataCollecting = False
    gesture = ''
    old_gesture = ''

    #flush the serial port
    time.sleep(3)
    serialport.readline()

    while(1):
        t=time.time()
        row, gesture, old_gesture, dataCollecting = gesture_Handler(sock,row,data,dataholder,dataCollecting,gesture,old_gesture)
        if gesture != old_gesture:
            print(gesture)    
            old_gesture=gesture
        
