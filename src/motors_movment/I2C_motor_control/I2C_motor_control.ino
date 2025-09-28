#include <Wire.h>
#include <AFMotor.h>
#include <Servo.h>

// Motors
AF_DCMotor motorLeft(3);
AF_DCMotor motorRight(4);

// Servos
Servo armServo;
Servo armServo2;
const int SERVO_PIN = 10;
const int SERVO2_PIN = 9;

// I2C buffer
#define I2C_BUFFER_SIZE 32
char i2cBuffer[I2C_BUFFER_SIZE + 1]; // +1 for null terminator
byte bufferIndex = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("Ready to receive motor commands over I2C.");

  motorLeft.setSpeed(0); motorRight.setSpeed(0);
  motorLeft.run(RELEASE); motorRight.run(RELEASE);

  armServo.attach(SERVO_PIN);
  armServo2.attach(SERVO2_PIN);
  armServo.write(90); armServo2.write(90);

  Wire.begin(0x08); // Arduino I2C address
  Wire.onReceive(receiveEvent);
}

void loop() {
  // Nothing needed here
}

void receiveEvent(int bytesReceived) {
  while (Wire.available()) {
    char c = Wire.read();

    if (c == '\n') {
      i2cBuffer[bufferIndex] = '\0'; // terminate string
      parseMessage(i2cBuffer);
      bufferIndex = 0; // reset for next message
    } else {
      if (bufferIndex < I2C_BUFFER_SIZE) {
        i2cBuffer[bufferIndex++] = c;
      } else {
        // Overflow, discard current message
        bufferIndex = 0;
        Serial.println("I2C buffer overflow! Discarding message.");
      }
    }
  }
}

void parseMessage(const char* msg) {
  int left_pwm = 0, right_pwm = 0, arm_angle = 90;
  
  int n = sscanf(msg, "%d,%d,%d", &left_pwm, &right_pwm, &arm_angle);
  if (n == 3) {
    //Serial.print("Parsed: L="); Serial.print(left_pwm);
    //Serial.print(" R="); Serial.print(right_pwm);
    //Serial.print(" Arm="); Serial.println(arm_angle);

    controlMotor(motorLeft, left_pwm);
    controlMotor(motorRight, right_pwm);

    arm_angle = constrain(arm_angle, 0, 180);
    armServo.write(arm_angle);
    armServo2.write(arm_angle);
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
