/*
  IMU Classifier
  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.
  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.
  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.
  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry
  This example code is in the public domain.
*/

#include <Adafruit_LSM6DSOX.h>

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h>
#include <tensorflow/lite/experimental/micro/micro_error_reporter.h>
#include <tensorflow/lite/experimental/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"


const int numSamples = 760;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 100 * 1024;
byte tensorArena[tensorArenaSize];

// array to map gesture index to a name
const char* GESTURES[] = {
  "Single_Wave",
  "Fist_Pump",
  "Random_Motion",
  "Speed_Mode"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// For SPI mode, we need a CS pin
#define LSM_CS 10
// For software-SPI mode we need SCK/MOSI/MISO pins
#define LSM_SCK 13
#define LSM_MISO 12
#define LSM_MOSI 11

Adafruit_LSM6DSOX sox;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!sox.begin_I2C()) {
    while (1) {
      delay(10);
    }
  }

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;
  
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  sox.getEvent(&accel, &gyro, &temp);

  aX=accel.acceleration.x;
  aY=accel.acceleration.y;
  aZ=accel.acceleration.z;

  // normalize the IMU data between 0 to 1 and store in the model's
  // input tensor
  tflInputTensor->data.f[samplesRead * 3 + 0] = aX;
  tflInputTensor->data.f[samplesRead * 3 + 1] = aY;
  tflInputTensor->data.f[samplesRead * 3 + 2] = aZ;
    
  samplesRead++;
    
  if (samplesRead == numSamples) {
    // Run inferencing
    samplesRead=0;
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }

    // Loop through the output tensor values from the model
    for (int i = 0; i < NUM_GESTURES; i++) {
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(tflOutputTensor->data.f[i], 3);
    }
    Serial.println();
  }
}
