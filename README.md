# Atltvhead Gesture Recognition Bracer
 Atltvhead Gesture Recognition Bracer - A TensorflowLite gesture detector for the atltvhead project and exploration into Data Science

This repository is my spin on Jennifer Wang's and Google Tensorflow's magic wand project, but for arm gestures

# TLDR:
1) Get started by uploading the AGRB-Traning-Data-Capture.ino in the Arduino_Sketch folder onto an Arduino of your choice with a button and Adafruit LSM6DSOX 9dof IMU. 
2) Use the CaptureData.py in the Training_Data folder. Initiate this file, type in the gesture name, start recording the motion data by pressing the button on the Arduino
3) After several gestures were recorded change to a different gesture and do it again. I tried to get 50 motion recordings of each gesture, you can try less if you like. 
4) Once all the data is collected, navigate to the Python Scripts folder and run DataPipeline.py and ModelPipeline.py in that order. Models are trained here and can time some time. 
5) Run Predict_Gesture.py and press the button on the Arduino to take a motion recording and see the results printed out. 

# The problem:
I run an interactive live stream. I wear an old tv (with working led display) like a helmet and backpack with a display. Twitch chat controls what's displayed on the television screen and the backpack screen through chat commands. Together Twitch chat and I go through the city of Atlanta, Ga spreading cheer. 

As time has gone on, I have over 20 channel commands for the tv display. Remembering and even copypasting all has become complicated and tedious. So it's time to simplify my interface to the tvhead.
 What are my resources? 
During the live stream, I am on rollerblades, my right hand is holding the camera, my left hand has a high five detecting glove I've built from a lidar sensor and esp32, my backpack has a raspberry pi 4, and a website with buttons that post commands in twitch chat. 
 What to simplify?
I'd like to simplify the channel commands and gamify it a bit more.
 What resources to use?
I am going to change my high five gloves, removing the lidar sensor, and feed the raspberry pi with acceleration and gyroscope data. So that the Pi can inference a gesture performed from the arm data.

# Goals

A working gesture detection model using TensorFlow and sensor data.

# Arduino
The AGRB-Traning-Data-Capture.ino in the Arduino_Sketch folder is my Arduino script to pipe acceleration and gyroscope data from an Adafruit LSM6DSOX 9dof IMU out of the USB serial port. An esp32 Thingplus by SparkFun is the board I've chosen due to the Qwiic connector support between this board and the Adafruit IMU. A push-button is connected between ground and pin 33 with the internal pullup resistor on. Eventually, I plan to deploy a tflite model on the esp32, so I've included a battery.

The data stream is started after the button on the Arduino is pressed and stops after 3 seconds. It is similar to a photograph, but instead of an x/y of camera pixels, its sensor data/time. 

# Data Collection
In the Training_Data folder, locate the CaptureData.py script. With the Arduino loaded with the AGRB-Traning-Data-Capture.ino script and connected to the capture computer with a USB cable, run the CaptureData.py script. It'll ask for the name of the gesture you are capturing. Then when you are ready to perform the gesture press the button on the Arduino and perform the gesture within 3 seconds. When you have captured enough of one gesture, stop the python script. Rinse and Repeat. 

I choose 3 seconds of data collection or roughly 760 data points mainly because I wasn't positive how long each gesture would take to be performed. Anywho, more data is better right?

# Docker File
To ensure you and I will get the "same" model I've included a Docker make file for you! I used this docker container while processing my data and training my model. It's based on the Jupiter Notebooks TensorFlow docker container. 

# Data Exploration and Augmentation
As said in the TLDR, the DataPipeline.py script, found in the Python_Scripts folder, will take all of your data from the data collection, split them between training/test/validation sets, augment the training data, and finalized CSVs ready for the model training.

## The following conclusions and findings are found in Jypter_Scripts/Data_Exploration.ipynb file:

- The first exploration task I conducted was to use seaborn's pairplot to plot all variables against one another for each different type of gesture. I was looking to see if there was any noticable outright linear or logistic relationships between variables. None popped out to me. 
![GitHub Logo](/Jypter_Scripts/images/Accels.png)
Format: ![Gesture Acclerations](url)

- Looking at the descriptions, I noticed that each gesture sampling had a different number of points, and are not consistant between samples of the same gesture.

- Each gestures acceleration data and gyroscope data is pretty unqiue when looking at time series plots. With fist pump mode and speed mode looking the most similar and will probably be the trickiest to differentiate from one another.

- Conducting a PCA of the different gestures yielded that the most "important" type of raw data is acceleration. However, when conducting a PCA with min/max normalized acceleration and gyroscope data, the most important feature became the normalized gyroscope data. Specifically Gyro_Z seems to contribute the most to principal component, across all gestures. 

- So now the decision. The PCA of Raw Data says that accelerations work. The PCA of Normalized Data seems to conclude that gyroscope data works. Since I'd like to eventually move this project over to the esp32, less data processing will reduce processing overhead on the micro. So lets try just using the **raw acceleration data** first. If that doesn't work, I'll add in the raw gyroscope data. If none of those work well, I'll normalize the data. 

## The following information is can be found in more detail in the Jypter_Scripts/Data Cleaning and Augmentation.ipynb file:




# Model Building
As said in the TLDR, the ModelPipeline.py script, found in the Python_Scripts folder, will import all finalized data from the finalized CSVs, create 2 different models an LSTM and CNN, compare the models' performances, and save all models. Note the LSTM will not have a size optimized tflite model. 



# Raspberry Pi Deployment
I used a raspberry pi 4 for my deployment. It was already in a previous tvhead build, has the compute power for model inference, and can be powered by a battery. I'll eventually phase it out for the ESP32 and TinyML once that model is solid. 



# Future Work
    -shrink data capture window from 3secs to 1.5 ~ 2secs
    -Test if gyro data improves continuous snapshot mode
    -Deploy on ESP32 with TinyML/TensorflowLite