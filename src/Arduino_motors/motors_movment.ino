#include <AFMotor.h>

// Motor definitions
AF_DCMotor motorLeft(3);  // Motor connected to M3
AF_DCMotor motorRight(4); // Motor connected to M4

void setup() {
  Serial.begin(9600);
  Serial.println("Robot iterative movement routine");

  // Initial setup: stop both motors
  //motorLeft.setSpeed(90);  // Set speed (0-255)
  //motorRight.setSpeed(75);
  motorLeft.setSpeed(190);  // Set speed (0-255)
  motorRight.setSpeed(175);
  motorLeft.run(RELEASE);
  motorRight.run(RELEASE);
}

void loop() {

  Serial.println("Pausing");
  motorLeft.run(RELEASE);
  motorRight.run(RELEASE);
  delay(5000); // 0.5 second pause before repeating

  // Step 1: Move forward for 3 seconds
  Serial.println("Moving forward");
  motorLeft.run(BACKWARD); // Left motor moves backward
  motorRight.run(FORWARD); // Right motor moves forward
  delay(4000); // 3 seconds

  Serial.println("Pausing");
  motorLeft.run(RELEASE);
  motorRight.run(RELEASE);
  delay(2000); // 0.5 second pause before repeating

  // Step 3: Move backward for 2 seconds
  Serial.println("Moving backward");
  motorLeft.run(FORWARD); // Left motor moves forward
  motorRight.run(BACKWARD); // Right motor moves backward
  delay(4000); // 2 seconds

  Serial.println("Pausing");
  motorLeft.run(RELEASE);
  motorRight.run(RELEASE);
  delay(2000); // 0.5 second pause before repeating


  // Step 2: Rotate left for 0.3 seconds
  //Serial.println("Rotating left");
  //motorLeft.run(FORWARD);
  //motorRight.run(FORWARD);
  //delay(3000); // 0.3 seconds

  // Step 2: Rotate left for 0.3 seconds
  //Serial.println("Rotating left");
  //motorLeft.run(BACKWARD);
  //motorRight.run(BACKWARD);
  //delay(2000); // 0.3 seconds



  // Step 4: Rotate right for 0.7 seconds
 // Serial.println("Rotating right");
 // motorLeft.run(BACKWARD);
 // motorRight.run(BACKWARD);
 // delay(1000); // 0.7 seconds

}
