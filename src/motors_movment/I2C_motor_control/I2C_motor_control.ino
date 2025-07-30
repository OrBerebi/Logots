#include <Wire.h>
#include <AFMotor.h>
#include <Servo.h>

// Motor definitions
AF_DCMotor motorLeft(2);   // Motor connected to M2
AF_DCMotor motorRight(3);  // Motor connected to M3

Servo armServo;            // Servo for arm control
Servo armServo2;           // Duplicate servo
const int SERVO_PIN = 10;  // Original servo pin
const int SERVO2_PIN = 9;  // Duplicate servo pin

// Buffer to hold incoming I2C data
char i2cBuffer[33];  // 32-byte I2C limit + null terminator
byte bufferIndex = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("Ready to receive motor commands over I2C.");

  // Motor setup
  motorLeft.setSpeed(0);
  motorRight.setSpeed(0);
  motorLeft.run(RELEASE);
  motorRight.run(RELEASE);

  // Servo setup
  armServo.attach(SERVO_PIN);
  armServo2.attach(SERVO2_PIN);
  armServo.write(90);  // Neutral position
  armServo2.write(90); // Neutral position for second servo

  // I2C setup
  Wire.begin(0x08); // I2C address must match ESP32 sender
  Wire.onReceive(receiveEvent);
}

void loop() {
  // Nothing needed here – logic handled in receiveEvent
}

void receiveEvent(int bytesReceived) {
  while (Wire.available()) {
    char c = Wire.read();

    if (c == '\n') {
      i2cBuffer[bufferIndex] = '\0'; // End current message
      parseMessage(i2cBuffer);
      bufferIndex = 0; // Reset for next message
    } else if (bufferIndex < sizeof(i2cBuffer) - 1) {
      i2cBuffer[bufferIndex++] = c;
    } else {
      // Overflow: discard and reset buffer
      bufferIndex = 0;
      Serial.println("I2C buffer overflow! Discarding message.");
    }
  }
}

void parseMessage(const char* msg) {
  char buf[33];
  strncpy(buf, msg, 32);
  buf[32] = '\0';

  char* token;
  int index = 0;

  int frame_id = 0;
  float timestamp = 0;
  int left_pwm = 0, right_pwm = 0, arm_angle = 0;

  token = strtok(buf, ",");
  while (token != NULL) {
    switch (index) {
      case 0: frame_id = atoi(token); break;
      case 1: timestamp = atof(token); break;
      case 2: left_pwm = atoi(token); break;
      case 3: right_pwm = atoi(token); break;
      case 4: arm_angle = atoi(token); break;
    }
    index++;
    token = strtok(NULL, ",");
  }

  if (index == 5) {
    Serial.print("Parsed: ");
    Serial.print("L=");
    Serial.print(left_pwm);
    Serial.print(" R=");
    Serial.print(right_pwm);
    Serial.print(" Arm=");
    Serial.println(arm_angle);

    controlMotor(motorLeft, left_pwm);
    controlMotor(motorRight, right_pwm);

    // Move both servos
    arm_angle = constrain(arm_angle, 0, 180);
    armServo.write((int)arm_angle);
    armServo2.write((int)arm_angle);
  } else {
    Serial.println("Parsing error!");
  }
}

void controlMotor(AF_DCMotor& motor, int pwm) {
  pwm = constrain(pwm, -255, 255);
  motor.setSpeed(abs(pwm));
  if (pwm > 0) {
    motor.run(FORWARD);
  } else if (pwm < 0) {
    motor.run(BACKWARD);
  } else {
    motor.run(RELEASE);
  }
}
