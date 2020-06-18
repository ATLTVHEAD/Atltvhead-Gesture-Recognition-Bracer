# Capture Data.py
# Description: Recieved Data from ESP32 Micro via the AGRB-Training-Data-Capture.ino file. Save data stream as csv files
# Written by: Nate Damen
# Created on June 17th 2020
# Updated on 


import time
import datetime
import matplotlib.dates as mdates
from collections import deque
import numpy as np
import os
import pandas as pd
import serial
import re

PORT = "COM43"

# How many sensor samples we want to store
HISTORY_SIZE = 2500

# Pause re-sampling the sensor and drawing for INTERVAL seconds
INTERVAL = 0.01

serialport = None

def get_imu_data():
    global serialport
    if not serialport:
        # open serial port
        serialport = serial.Serial(PORT, 115200, timeout=0.1)
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
    if not "Uni:" in line:
        return None
    vals = line.replace("Uni:", "").strip().split(',')
    #print(vals)
    if len(vals) != 9:
        return None
    try:
        vals = [float(i) for i in vals]
    except ValueError:
        return None
    #print(vals)
    return vals

for _ in range(20):
    print(get_imu_data())

filename = input("Name the folder where data will be stored: ")
if not os.path.exists(filename):
  os.mkdir(filename + '/')
starting_index = int(input("What number should we start on? "))

file_name = filename + "/" + filename + '{0:03d}'.format(i) + ".csv"
  df = pd.DataFrame(data, columns = header)
  df.to_csv(file_name, header=True)
  i += 1