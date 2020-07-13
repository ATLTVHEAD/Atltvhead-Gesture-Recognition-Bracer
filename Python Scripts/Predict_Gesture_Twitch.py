import numpy as np 
import pandas as pd 
import datetime
import re
import os, os.path
import time
import random
import tensorflow as tf
import serial
import socket
import cfg

PORT = "/dev/ttyUSB0"
#PORT = "/dev/ttyUSB1"
#PORT = "COM8"

serialport = None
serialport = serial.Serial(PORT, 115200, timeout=0.05)

#load Model
model = tf.keras.models.load_model('../Model/cnn_model.h5')

#Creating our socket and passing on info for twitch
sock = socket.socket()
sock.connect((cfg.HOST,cfg.PORT))
sock.send("PASS {}\r\n".format(cfg.PASS).encode("utf-8"))
sock.send("NICK {}\r\n".format(cfg.NICK).encode("utf-8"))
sock.send("JOIN {}\r\n".format(cfg.CHAN).encode("utf-8"))
sock.setblocking(0)

#handling of some of the string characters in the twitch message
chat_message = re.compile(r"^:\w+!\w+@\w+.tmi.twitch.tv PRIVMSG #\w+ :")

#Lets create a new function that allows us to chat a little easier. Create two variables for the socket and messages to be passed in and then the socket send function with proper configuration for twitch messages. 
def chat(s,msg):
    s.send("PRIVMSG {} :{}\r\n".format(cfg.CHAN,msg).encode("utf-8"))

#The next two functions allow for twitch messages from socket receive to be passed in and searched to parse out the message and the user who typed it. 
def getMSG(r):
    mgs = chat_message.sub("", r)
    return mgs


def getUSER(r):
    try:
        user=re.search(r"\w+",r).group(0)
    except AttributeError:
        user ="tvheadbot"
        print(AttributeError)
    return user

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

# header for the incomming data
header = ["deltaTime","Acc_X","Acc_Y","Acc_Z","Gyro_X","Gyro_Y","Gyro_Z"]

#Create a way to see the length of the data incomming, needs to be 760 points. Used for testing incomming data
def dataFrameLenTest(data):
    df=pd.DataFrame(data,columns=header)
    x=len(df[['Acc_X','Acc_Y','Acc_Z']].to_numpy())
    print(x)
    return x

#Create a pipeline to process incomming data for the model to read and handle
def data_pipeline(data_a):
    df = pd.DataFrame(data_a, columns = header)
    temp=df[['Acc_X','Acc_Y','Acc_Z']].to_numpy()
    tensor_set = tf.data.Dataset.from_tensor_slices(
        (np.array([temp.tolist()],dtype=np.float64)))
    tensor_set_cnn = tensor_set.map(reshape_function)
    tensor_set_cnn = tensor_set_cnn.batch(192)
    return tensor_set_cnn

#define Gestures, current data, temp data holder
gest_id = {0:'wave_mode', 1:'fist_pump_mode', 2:'random_motion_mode', 3:'speed_mode'}
data = []
dataholder=[]
dataCollecting = False
gesture=''
old_gesture=''

#flush the serial port
serialport.flush()

while(1):
    try:
        response = sock.recv(1024).decode("utf-8")
    except:
        dataholder = get_imu_data()
        if dataholder != None:
            dataCollecting=True
            data.append(dataholder)
        if dataholder == None and dataCollecting == True:
            if len(data) == 760:
                prediction = np.argmax(model.predict(data_pipeline(data)), axis=1)
                gesture=gest_id[prediction[0]]
            if gesture != old_gesture:
                chat(sock,'!' + gesture)
                print(gesture)
            data = []
            dataCollecting = False
            old_gesture=gesture
    else:
        if len(response)==0:
            print('orderly shutdown on the server end')
            sock = socket.socket()
            sock.connect((cfg.HOST,cfg.PORT))
            sock.send("PASS {}\r\n".format(cfg.PASS).encode("utf-8"))
            sock.send("NICK {}\r\n".format(cfg.NICK).encode("utf-8"))
            sock.send("JOIN {}\r\n".format(cfg.CHAN).encode("utf-8"))
            sock.setblocking(0)
        else:
            print(response)
            #pong the pings to stay connected 
            if response == "PING :tmi.twitch.tv\r\n":
                sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
