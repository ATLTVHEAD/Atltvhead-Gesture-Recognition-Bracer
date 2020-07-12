/* AGRB-Training-Data-Capture.ino

Description: Upon button press, send 2 seconds of data reads over serial. It will be recieved by Capture Data.py

Written by Nate Damen  
Created: June 17,2020
Updated: June 17,2020

*/

//#include <WiFi.h>
//#include <IRCClient.h>
#include <Math.h>
#include <Adafruit_LSM6DSOX.h>

#include<AceButton.h>
using namespace ace_button;


const int BUTTON_PIN = 33;
#ifdef ESP32
  const int LED_PIN = 13;
#else
  const int LED_PIN = LED_BUILTIN;
#endif

const float gyrocal[3]={0.00904999,-0.01169999,-0.00905}; // rads/s
const float altgyrocal[3]={0.00904999,-0.01169999,-0.0035};; // rads/s
const float magcal[3]={1.35999999,-22.63,69.08};  // utesla

const int LED_ON = HIGH;
const int LED_OFF = LOW;
AceButton button(BUTTON_PIN);
void handleEvent(AceButton*, uint8_t, uint8_t);

bool bpress = false;
bool lpress = false;
bool dpress = false;

int startTime = 0;
int currTime = 0;

int id = 0;

// For SPI mode, we need a CS pin
#define LSM_CS 10
// For software-SPI mode we need SCK/MOSI/MISO pins
#define LSM_SCK 13
#define LSM_MISO 12
#define LSM_MOSI 11

Adafruit_LSM6DSOX sox;



void setup() {
  delay(1000); // some microcontrollers reboot twice
  Serial.begin(115200);
  while (! Serial); // Wait until Serial is ready - Leonardo/Micro
  Serial.println(F("setup(): begin"));

  // initialize built-in LED as an output
  pinMode(LED_PIN, OUTPUT);

  // Button uses the built-in pull up register.
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Configure the ButtonConfig with the event handler, and enable all higher
  // level events.
  ButtonConfig* buttonConfig = button.getButtonConfig();
  buttonConfig->setEventHandler(handleEvent);
  buttonConfig->setFeature(ButtonConfig::kFeatureClick);
  buttonConfig->setFeature(ButtonConfig::kFeatureDoubleClick);
  buttonConfig->setFeature(ButtonConfig::kFeatureLongPress);
  buttonConfig->setFeature(ButtonConfig::kFeatureRepeatPress);
  buttonConfig->setFeature(ButtonConfig::kFeatureSuppressClickBeforeDoubleClick);


  if (!sox.begin_I2C()) {
    while (1) {
      delay(10);
    }
  }
}

void loop() {

  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  sox.getEvent(&accel, &gyro, &temp);
  id++;
  currTime = millis();
  if(bpress){
    startTime = currTime;
    bpress = false;
    id = 0;
    //Serial.print("start");
  }
  if(currTime-startTime <= 3200 && id < 760){
    //Serial.print(id);
    //Serial.print(',');
    Serial.print(currTime - startTime);
    Serial.print(',');
    //Serial.print(currTime);
    //Serial.print(',');
    
    /* Display the results (acceleration is measured in m/s^2) */
    Serial.print(accel.acceleration.x,3);
    Serial.print(',');
    Serial.print(accel.acceleration.y,3);
    Serial.print(',');
    Serial.print(accel.acceleration.z,3);
    Serial.print(',');

    /* Display the results (rotation is measured in rad/s) */
    Serial.print(gyro.gyro.x - gyrocal[0],3);
    Serial.print(',');
    Serial.print(gyro.gyro.y - gyrocal[1],3);
    Serial.print(',');
    Serial.print(gyro.gyro.z - gyrocal[2],3);
    Serial.println(); 
  } 
  button.check();
}



void handleEvent(AceButton* /* button */, uint8_t eventType,
    uint8_t buttonState) {

  switch (eventType) {
    case AceButton::kEventPressed:
      digitalWrite(LED_PIN, LED_ON);
      bpress = true;
      //bpress = !bpress;
      //Serial.println("Press");
      break;
    case AceButton::kEventReleased:
      digitalWrite(LED_PIN, LED_OFF);
      //Serial.println("Release");
      break;
    case AceButton::kEventLongPressed:
      //Serial.println("long press");
      lpress = true;
      break;
    case AceButton::kEventDoubleClicked:
      //Serial.println("Double Click");
      dpress = true;
      break;
    //case AceButton::kEventClicked:
      //Serial.println("single Click");
      //break;        
  }
}
