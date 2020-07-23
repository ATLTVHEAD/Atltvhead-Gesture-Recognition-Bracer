# Atltvhead Gesture Recognition Bracer
 Atltvhead Gesture Recognition Bracer - A TensorflowLite gesture detector for the atltvhead project and for exploration into Data Science

This repository is my own spin on Jennifer Wang's and Google Tensorflow's magic wand project. 

# TLDR:
1) Get started by uploading the AGRB-Traning-Data-Capture.ino in the Arduino_Sketch folder onto an arduino of your choice with a button and adafruit LSM6DSOX 9dof IMU. 
2) Use the CaptureData.py in the Training_Data folder. Initiate this file, type in the gesture name, start recording the motion data by pressing the button on the arduino
3) After several gestures were recorded change to a different gesture and do it again. I tried to get 50 motion recordings of each gesture, you can try less if you like. 
4) Once all the data is collected, navigate to the Python Scripts folder and run DataPipeline.py and ModelPipeline.py in that order. Models are trained here and can time some time. 
5) Run Predict_Gesture.py and press the button on the arduino to take a motion recording and see the results printed out. 

# The problem:
I run an interactive livestream. I wear an old tv (with working led display) as a helmet and backpack with a display. Twitch chat controls what's displayed on the television screen and on the backpack screen through chat commands. Together Twitch chat and I go through the city of Atlanta, Ga spreading cheer. 

As time has gone on, I have over 20 channel commands for the tv display. Remembering and even copypasting all has become complicated and tedious. So it's time to simplicy my interface to the tvhead.
 What are my resources? 
During the livestream, I am on rollerblades, my right hand is holding the camera, my left hand has a high five detecting glove I've built from a lidar sensor and esp32, my backpack has a raspberry pi 4, and a website with buttons that post commands in twitch chat. 
 What to simplify?
I'd like to simplify the channel commands and gamify it a bit more.
 What resources to use?
I am going to change my high five glove, removing the lidar sensor, and feed the raspberry pi with acceleration and gyroscope data. So that the Pi can inference a gesture performed from the arm data.

# The Goals



# Arduino

# Data Collection

# Model Building

# Raspberry Pi Deployment

# Future Work
    -shrink snapshot window (down from 3sec to 1.5 ~ 2sec)
    -Test if gyro data improves continuous snapshot mode
    -Deploy on ESP32 with TinyML/TensorflowLite