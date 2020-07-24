# Atltvhead Gesture Recognition Bracer
 Atltvhead Gesture Recognition Bracer - A TensorflowLite gesture detector for the atltvhead project and exploration into Data Science

This repository is my spin on Jennifer Wang's and Google Tensorflow's magic wand project, but for arm gestures

# TLDR:
1) Get started by uploading the AGRB-Traning-Data-Capture.ino in the Arduino_Sketch folder onto an Arduino of your choice with a button and Adafruit LSM6DSOX 9dof IMU. 
2) Use the CaptureData.py in the Training_Data folder. Initiate this file, type in the gesture name, start recording the motion data by pressing the button on the Arduino
3) After several gestures were recorded change to a different gesture and do it again. I tried to get 50 motion recordings of each gesture, you can try less if you like. 
4) Once all the data is collected, navigate to the Python Scripts folder and run DataPipeline.py and ModelPipeline.py in that order. Models are trained here and can time some time. 
5) Run Predict_Gesture.py and press the button on the Arduino to take a motion recording and see the results printed out. 

# Problem:
I run an interactive live stream. I wear an old tv (with working led display) like a helmet and backpack with a display. Twitch chat controls what's displayed on the television screen and the backpack screen through chat commands. Together Twitch chat and I go through the city of Atlanta, Ga spreading cheer. 

As time has gone on, I have over 20 channel commands for the tv display. Remembering and even copypasting all has become complicated and tedious. So it's time to simplify my interface to the tvhead.
 What are my resources? 
During the live stream, I am on rollerblades, my right hand is holding the camera, my left hand has a high five detecting glove I've built from a lidar sensor and esp32, my backpack has a raspberry pi 4, and a website with buttons that post commands in twitch chat. 
 What to simplify?
I'd like to simplify the channel commands and gamify it a bit more.
 What resources to use?
I am going to change my high five gloves, removing the lidar sensor, and feed the raspberry pi with acceleration and gyroscope data. So that the Pi can inference a gesture performed from the arm data.

# Goal:

A working gesture detection model using TensorFlow and sensor data.

# Machine Learning Flow:

![My Flow](/Jypter_Scripts/images/Machine_Learning_Flow_Chart-02.png)

# Arduino Setup:
The AGRB-Traning-Data-Capture.ino in the Arduino_Sketch folder is my Arduino script to pipe acceleration and gyroscope data from an Adafruit LSM6DSOX 9dof IMU out of the USB serial port. An esp32 Thingplus by SparkFun is the board I've chosen due to the Qwiic connector support between this board and the Adafruit IMU. A push-button is connected between ground and pin 33 with the internal pullup resistor on. Eventually, I plan to deploy a tflite model on the esp32, so I've included a battery.

![ESP32 Layout](/Arduino_Sketch/images/Esp32_layout.png)

The data stream is started after the button on the Arduino is pressed and stops after 3 seconds. It is similar to a photograph, but instead of an x/y of camera pixels, its sensor data/time. 

# Data Collection:
In the Training_Data folder, locate the CaptureData.py script. With the Arduino loaded with the AGRB-Traning-Data-Capture.ino script and connected to the capture computer with a USB cable, run the CaptureData.py script. It'll ask for the name of the gesture you are capturing. Then when you are ready to perform the gesture press the button on the Arduino and perform the gesture within 3 seconds. When you have captured enough of one gesture, stop the python script. Rinse and Repeat. 

![Bracer Gif](/Arduino_Sketch/images/Biting.gif)

I choose 3 seconds of data collection or roughly 760 data points mainly because I wasn't positive how long each gesture would take to be performed. Anywho, more data is better right?

# Docker File:
To ensure you and I will get the "same" model I've included a Docker make file for you! I used this docker container while processing my data and training my model. It's based on the Jupiter Notebooks TensorFlow docker container. 

# Data Exploration and Augmentation:
As said in the TLDR, the DataPipeline.py script, found in the Python_Scripts folder, will take all of your data from the data collection, split them between training/test/validation sets, augment the training data, and finalized CSVs ready for the model training.

## The following conclusions and findings are found in Jypter_Scripts/Data_Exploration.ipynb file:

- The first exploration task I conducted was to use seaborn's pair plot to plot all variables against one another for each different type of gesture. I was looking to see if there was any noticeable outright linear or logistic relationships between variables. None popped out to me. 
![Fist Pump Pairplot](/Jypter_Scripts/images/fist_pump_pairplot.png)

- Looking at the descriptions, I noticed that each gesture sampling had a different number of points, and are not consistent between samples of the same gesture.

- Each gesture's acceleration data and gyroscope data is pretty unique when looking at time series plots. With fist-pump mode and speed mode looking the most similar and will probably be the trickiest to differentiate from one another.
![Gesture Acclerations](/Jypter_Scripts/images/Accels.png)
![Gesture Gyroscopes](/Jypter_Scripts/images/Gyros.png)

- Conducting a PCA of the different gestures yielded that the most "important" type of raw data is acceleration. However, when conducting a PCA with min/max normalized acceleration and gyroscope data, the most important feature became the normalized gyroscope data. Specifically, Gyro_Z seems to contribute the most to the first principal component, across all gestures.
![PCA's](/Jypter_Scripts/images/PCAs.png)

- So now the decision. The PCA of Raw Data says that accelerations work. The PCA of Normalized Data seems to conclude that gyroscope data works. Since I'd like to eventually move this project over to the esp32, less pre-processing will reduce processing overhead on the micro. So let's try just using the **raw acceleration data** first. If that doesn't work, I'll add in the raw gyroscope data. If none of those work well, I'll normalize the data. 

## The following information is can be found in more detail in the Jypter_Scripts/Data Cleaning and Augmentation.ipynb file:

Since I was collecting the data myself, I have a super small data set. Meaning I will most likely overfit my model. To overcome this I implemented augmentation techniques to my data set.

The augmentation techniques used are as follows:
1) Increase and decrease the peaks of the XYZ data
2) Shift the data to complete faster or slower. Time stretch and shrink.
3) Add noise to the data points
4) Increase and decrease the magnitude the XYZ data uniformly
5) Shift the snapshot window around the time series data, making the gesture start sooner or later

To address the number of data points inconsistency, I found that 760 data points per sample was the average. I then implemented a script that cut off the front and end of my data by a certain number of samples depending on the length. Saving the first half and the second half as two different samples, to keep as much data as possible. This cut data had a final length of 760 for each sample. 

Before Augmenting I had 168 samples in my training set, 23 in my test set, and 34 in my validation set. After I augmenting I ended up with 8400 samples in my training set, 46 in my test set, and 68 in my validation set. Still small, but way better than before. 

# Model Building and Selection:
As said in the TLDR, the ModelPipeline.py script, found in the Python_Scripts folder, will import all finalized data from the finalized CSVs, create 2 different models an LSTM and CNN, compare the models' performances, and save all models. Note the LSTM will not have a size optimized tflite model. 

For which model to use, I looked towards my predecessors works. They used Scikit-learn's SDG Classifier, a CNN, and an LSTM. Since I want to eventually deploy on the esp32 with TinyML, I scikit learn is out. 2D CNN's and LSTM's are both valid options for deployment, so Lets define the two models. 

**CNN** 
I made a 10 layer CNN. The first layer being a 2D convolution layer, going into a maxpool, dropout, another 2D convolution, another maxpool, another dropout, a flattening, a dense, a final dropout, and a dense output layer for the 4 gestures. 

After tuning hyperparameters, I ended up with a batch size of 192, 300 steps per epoch, and 20 epochs. I optimized with an adam optimizer and used sparse categorical corssentropy for my loss, having accuracy as the metric to measure. 

**LSTM**
Using Tensorflow I made a senquencial LTSM model with 22 bidirectional layers and a dense output layer classifying to my 4 gestures.  

After tuning hyperparameters, I ended up with a batch size of 64, 200 steps per epoch, and 20 epochs. I optimized with an adam optimizer and used sparse categorical corssentropy for my loss, having accuracy as the metric to measure. 

**Model Selection** 
Both the CNN and LSTM perfectly predicted the gestures of the training set. The LSTM with a loss of 0.04 and the CNN with a loss of 0.007 during the test. 

Next I looked at the Training Validation loss per epoch of training. From the look of it, the CNN with batch size of 192 is pretty close to being fit correctly. The CNN batch size of 64 and the LSTM both seem a little overfit.
![Training Validation Loss](/Jypter_Scripts/images/Model_Losses.png)

So I chose to proceed with the CNN model, trained with a batch size of 192. I saved the model, as well as saved a tflite version of the model optimized for size.

# Testing 
I wrote a gesture prediction test script for both the regular model and the tflight model. Both models work!

# Raspberry Pi Deployment:
I used a raspberry pi 4 for my current deployment, since it was already in a previous tvhead build, has the compute power for model inference, and can be powered by a battery. 

The pi is in a backpack with a display. On the display is a positive message that changes based on what is said in my Twitch chat, during live streams. I used this same script, but added the tensorflow model gesture prediction components from the Predict_Gesture_Twitch.py script to create the PositivityPack.py script. 

To infer gestures and send them to Twitch, use the PositivityPack.py or the Predict_Gesture_Twitch.py. They run the heavy .h5 model file. To run the tflite model on the raspberry pi run the Test_TFLite_Inference_Serial_Input.py script. You'll need to connect the raspberry pi with the ESP32 in the arm attachment using a USB cable. Press the button the arm attachment to send data and predict gesture. Long press the button to continually detect gestures, continuous snapshot mode.

**Note:** When running the scripts that communicate with Twitch you'll need to follow [Twitch's chatbot development](https://dev.twitch.tv/docs/irc) documentation for creating your own chatbot and authenticating it. 

# Conclusions:
It works! The gesture prediction works perfectly, when triggering a gesture prediction from the arm attachment. Continous snapshot mode works well, but feels sluggish in use due to the 3 seconds data sampling between gesture predictions.

# Future Work:
    -Shrink data capture window from 3secs to 1.5 ~ 2secs, improving Continuous snapshot mode
    -Test if gyro data improves continuous snapshot mode
    -Deploy on ESP32 with TinyML/TensorflowLite