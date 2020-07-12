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

serialport = None
serialport = serial.Serial(PORT, 115200, timeout=0.05)

model = tf.keras.models.load_model('../Model/cnn_model.h5')

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
    line = str(serialport.readline(), 'utf-8')
    if not line:
        return None
    #print(line)
    #if not "Uni:" in line:
        #return None
    vals = line.replace("Uni:", "").strip().split(',')
    #print(vals)
    if len(vals) != 7:
        return None
    try:
        vals = [float(i) for i in vals]
    except ValueError:
        return ValueError
    #print(vals)
    return vals

def reshape_function(data):
    reshaped_data = tf.reshape(data, [-1, 3, 1])
    return reshaped_data

header = ["deltaTime","Acc_X","Acc_Y","Acc_Z","Gyro_X","Gyro_Y","Gyro_Z"]

def date_pipeline(data):
    df = pd.DataFrame(data, columns = header)
    tensor_set = tf.data.Dataset.from_tensor_slices(
        (np.array(data["Acc_X","Acc_Y","Acc_Z"].tolist(),dtype=np.float64)))
    tensor_set_cnn = tensor_set.map(reshape_function)
    tensor_set_cnn = tensor_set_cnn.batch(192)
    return tensor_set_cnn


data = []
dataholder=[]
dataCollecting = False

while(1):
    #serialport.flush()
    dataholder = get_imu_data()
    if dataholder != None:
        dataCollecting=True
        #print(dataholder[0])
        data.append(dataholder)
    if dataholder == None and dataCollecting == True:
        prediction = np.argmax(model.predict(date_pipeline(data)), axis=1)
        print(prediction)
        data = []
        dataCollecting = False