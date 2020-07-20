import numpy as np
import serial
import pandas as pd
import time 

#import tflite_runtime.interpreter as tflite
import tensorflow as tf


#PORT = "/dev/ttyUSB0"
PORT = "COM8"


serialport = None
serialport = serial.Serial(PORT, 115200, timeout=0.05)
header = ["deltaTime","Acc_X","Acc_Y","Acc_Z","Gyro_X","Gyro_Y","Gyro_Z"]
gest_id = {0:'wave_mode', 1:'fist_pump_mode', 2:'random_motion_mode', 3:'speed_mode'}
data = []
dataholder=[]
dataCollecting = False
gesture=''
old_gesture=''

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

if __name__ =="__main__":
    # Load the TFLite model and allocate tensors.
    
    #interpreter = tflite.Interpreter(model_path="../Model/cnn_model_quantized.tflite")    
    interpreter = tf.lite.Interpreter(model_path="../Model/cnn_model_quantized.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    print(input_details[0])
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    #print(input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)


    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    gesture=gest_id[np.argmax(output_data, axis=1)[0]]

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    print(gesture)
    while(1):
        dataholder = get_imu_data()
        if dataholder != None:
            dataCollecting=True
            data.append(dataholder)
        if dataholder == None and dataCollecting == True:
            if len(data) == 760:
                df = pd.DataFrame(data, columns = header)
                temp=df[['Acc_X','Acc_Y','Acc_Z']].to_numpy()
                #print(np.reshape(np.array([temp.tolist()],dtype=np.float32), input_shape))
                input_data = np.reshape(np.array([temp.tolist()],dtype=np.float32), input_shape)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                gesture=gest_id[np.argmax(output_data, axis=1)[0]]
                print(gesture)
            dataCollecting=False
            data=[]
