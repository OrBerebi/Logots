#include <Wire.h>
#include <AFMotor.h>
#include <Servo.h>  // <--- RE-ADDED: This fixes the compile error.

// Motors
AF_DCMotor motorLeft(3);
AF_DCMotor motorRight(4);

// Servos
Servo armServo;
Servo armServo2;
// --- PIN CHANGE: Moved off 9 & 10 to avoid timer conflict ---
const int SERVO_PIN = 10;  // Was A0
const int SERVO2_PIN = 9; // Was A1

// --- New Global Variables for Smoothing ---
volatile int target_left_pwm = 0;
volatile int target_right_pwm = 0;
volatile int target_arm_angle = 90;
int current_arm_angle = 90;

unsigned long lastServoMoveTime = 0;
const int SERVO_SPEED_DELAY = 1; // ms delay per 1-degree step

// I2C buffer
#define I2C_BUFFER_SIZE 32
char i2cBuffer[I2C_BUFFER_SIZE + 1];
byte bufferIndex = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("Ready to receive motor commands over I2C.");

  motorLeft.setSpeed(0); motorRight.setSpeed(0);
  motorLeft.run(RELEASE); motorRight.run(RELEASE);

  armServo.attach(SERVO_PIN);   // Now attached to A0
  armServo2.attach(SERVO2_PIN); // Now attached to A1
  armServo.write(current_arm_angle);
  armServo2.write(current_arm_angle);

  Wire.begin(0x08);
  Wire.onReceive(receiveEvent);
}

void loop() {
  // Main loop handles all movement
  controlMotor(motorLeft, target_left_pwm);
  controlMotor(motorRight, target_right_pwm);
  smoothServoMove();
}

void smoothServoMove() {
  if (millis() - lastServoMoveTime > SERVO_SPEED_DELAY) {
    lastServoMoveTime = millis(); 

    if (current_arm_angle < target_arm_angle) {
      current_arm_angle++;
    } 
    else if (current_arm_angle > target_arm_angle) {
      current_arm_angle--;
    }
    
    armServo.write(current_arm_angle);
    armServo2.write(current_arm_angle);
  }
}

// ISR: Just capture the data, don't do any work
void receiveEvent(int bytesReceived) {
  while (Wire.available()) {
    char c = Wire.read();

    if (c == '\n') {
      i2cBuffer[bufferIndex] = '\0';
      parseMessage(i2cBuffer); 
      bufferIndex = 0;
    } else {
      if (bufferIndex < I2C_BUFFER_SIZE) {
        i2cBuffer[bufferIndex++] = c;
      } else {
        bufferIndex = 0;
        Serial.println("I2C buffer overflow! Discarding message.");
      }
    }
  }
}

// Parse message and set global targets
void parseMessage(const char* msg) {
  int left_pwm = 0, right_pwm = 0, arm_angle = 90;
  
  int n = sscanf(msg, "%d,%d,%d", &left_pwm, &right_pwm, &arm_angle);
  
  if (n == 3) {
    // Update the 'volatile' target variables.
    target_left_pwm = left_pwm;
    target_right_pwm = right_pwm;
    target_arm_angle = constrain(arm_angle, 0, 180);
  } else {
    Serial.println("Parsing error! Message ignored.");
  }
}

void controlMotor(AF_DCMotor& motor, int pwm) {
  pwm = constrain(pwm, -255, 255);
  motor.setSpeed(abs(pwm));
  if (pwm > 0) motor.run(FORWARD);
  else if (pwm < 0) motor.run(BACKWARD);
  else motor.run(RELEASE);
}