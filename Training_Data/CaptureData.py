# CaptureData.py
# Description: Recieved Data from ESP32 Micro via the AGRB-Training-Data-Capture.ino file. Save data stream as csv files
# Written by: Nate Damen
# Created on June 17th 2020
# Updated on June 18th 2020

import time
import datetime
import os
import numpy as np
import pandas as pd
import serial
import re

# PORT = "/dev/ttyUSB0"
PORT = "COM8"

# How many sensor samples we want to store
HISTORY_SIZE = 2500

# Pause re-sampling the sensor and drawing for INTERVAL seconds
INTERVAL = 0.01

serialport = None
serialport = serial.Serial(PORT, 115200, timeout=0.05)

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


j=0

filename = input("\Type the Folder and Filename: ")
if not os.path.exists(filename):
  os.mkdir(filename + '/')

header = ["deltaTime","Acc_X","Acc_Y","Acc_Z","Gyro_X","Gyro_Y","Gyro_Z"]

def saveData():
    global j
    file_name = filename + "/" + filename + '{0:03d}'.format(j) + ".csv"
    df = pd.DataFrame(data, columns = header)
    #df['deltaTime']=df.apply(lambda row: row.currentTime - row.startTime ,axis=1)
    df.to_csv(file_name, index=False)
    j += 1
    

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
        saveData()
        data = []
        dataCollecting = False
