#include <AFMotor.h>
#include <Servo.h>

// Motor definitions
AF_DCMotor motorLeft(2);   // Motor connected to M2
AF_DCMotor motorRight(3);  // Motor connected to M3

Servo armServo;
Servo armServo2;

const int SERVO_PIN = 10;
const int SERVO2_PIN = 9;

// Joystick pins
const int VRX_PIN = A0;
const int VRY_PIN = A1;
const int SW_PIN  = 2;  // Digital pin for joystick button

// Potentiometer pin
const int POT_PIN = A2;

// Constants
const int DEADZONE = 100;  // Adjust based on your joystick
const int MOTOR_SPEED = 75;  // Constant speed (0–255)

void setup() {
  Serial.begin(9600);

  // Motor setup
  motorLeft.setSpeed(0);
  motorRight.setSpeed(0);
  motorLeft.run(RELEASE);
  motorRight.run(RELEASE);

  // Servo setup
  armServo.attach(SERVO_PIN);
  armServo2.attach(SERVO2_PIN);
  armServo.write(90);
  armServo2.write(90);

  // Joystick setup
  pinMode(SW_PIN, INPUT_PULLUP);  // Usually active LOW
}

void loop() {
  int x = analogRead(VRX_PIN) - 523;
  int y = analogRead(VRY_PIN) - 517 ;
  bool pressed = digitalRead(SW_PIN) == LOW;
  int potVal = analogRead(POT_PIN);  // 0–1023
  int servoAngle = map(potVal, 0, 1023, 0, 180);
  armServo.write((int)servoAngle);

  // Debug print
  Serial.print("X: "); Serial.print(x);
  Serial.print(" Y: "); Serial.print(y);
  Serial.print(" SW: "); Serial.print(pressed);
  Serial.print(" Pot: "); Serial.println(potVal);

  // Decide movement
  if (abs(y) > DEADZONE && abs(y) > abs(x)) {
    // Forward or backward
    if (y > 0) {
      moveForward();
    } else {
      moveBackward();
    }
  } else if (abs(x) > DEADZONE) {
    // Rotate left or right
    if (x > 0) {
      rotateRight();
    } else {
      rotateLeft();
    }
  } else {
    stopMotors();
  }

  delay(100);  // Small delay for stability
}

void moveForward() {
  motorLeft.setSpeed(MOTOR_SPEED);
  motorRight.setSpeed(MOTOR_SPEED);
  motorLeft.run(FORWARD);
  motorRight.run(FORWARD);
}

void moveBackward() {
  motorLeft.setSpeed(MOTOR_SPEED);
  motorRight.setSpeed(MOTOR_SPEED);
  motorLeft.run(BACKWARD);
  motorRight.run(BACKWARD);
}

void rotateLeft() {
  motorLeft.setSpeed(MOTOR_SPEED);
  motorRight.setSpeed(MOTOR_SPEED);
  motorLeft.run(BACKWARD);
  motorRight.run(FORWARD);
}

void rotateRight() {
  motorLeft.setSpeed(MOTOR_SPEED);
  motorRight.setSpeed(MOTOR_SPEED);
  motorLeft.run(FORWARD);
  motorRight.run(BACKWARD);
}

void stopMotors() {
  motorLeft.run(RELEASE);
  motorRight.run(RELEASE);
}
