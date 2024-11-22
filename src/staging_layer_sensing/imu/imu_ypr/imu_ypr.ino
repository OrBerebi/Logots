#include <Wire.h>
#include <MPU9250_asukiaaa.h>

MPU9250_asukiaaa imu;

// Variables to store initial offsets
float yawOffset = 0;
float pitchOffset = 0;
float rollOffset = 0;

// Complementary filter variables
float yawGyro = 0;  // Gyroscope-integrated yaw
float yawMag = 0;   // Magnetometer-derived yaw
float yawFiltered = 0; // Filtered yaw value
float alpha = 0.95; // Complementary filter coefficient (adjust as needed)
unsigned long previousTime = 0;

// Gyroscope calibration offsets
float gyroXOffset = 0, gyroYOffset = 0, gyroZOffset = 0;

void setup() {
  Wire.begin(16, 17); // SDA = GPIO 16, SCL = GPIO 17
  Serial.begin(1000000);

  // Initialize the IMU
  imu.setWire(&Wire);
  imu.beginAccel();
  imu.beginGyro();
  imu.beginMag();

  Serial.println("MPU9250 initialization complete");

  // Calibrate the gyroscope
  calibrateGyro();

  // Capture initial values to set offsets
  imu.accelUpdate();
  imu.magUpdate();

  // Calculate initial pitch and roll from the accelerometer
  float normAccel = sqrt(imu.accelX() * imu.accelX() +
                         imu.accelY() * imu.accelY() +
                         imu.accelZ() * imu.accelZ());
  float ax = imu.accelX() / normAccel;
  float ay = imu.accelY() / normAccel;
  float az = imu.accelZ() / normAccel;

  pitchOffset = atan2(-ax, sqrt(ay * ay + az * az));
  rollOffset = atan2(ay, az);

  // Calculate initial yaw from the magnetometer
  float mx = imu.magX();
  float my = imu.magY();
  float mz = imu.magZ();
  
  float mx_comp = mx * cos(pitchOffset) + mz * sin(pitchOffset);
  float my_comp = mx * sin(rollOffset) * sin(pitchOffset) + my * cos(rollOffset) - mz * sin(rollOffset) * cos(pitchOffset);

  yawOffset = atan2(my_comp, mx_comp);

  // Convert offsets from radians to degrees
  pitchOffset *= 180.0 / M_PI;
  rollOffset *= 180.0 / M_PI;
  yawOffset *= 180.0 / M_PI;
  if (yawOffset < 0) yawOffset += 360; // Ensure yaw is in the range [0, 360)

  previousTime = millis();
}

void loop() {
  // Update sensor data
  imu.accelUpdate();
  imu.gyroUpdate();
  imu.magUpdate();

  // Subtract offsets from gyroscope readings
  float gyroX = imu.gyroX() - gyroXOffset;
  float gyroY = imu.gyroY() - gyroYOffset;
  float gyroZ = imu.gyroZ() - gyroZOffset;

  // Calculate the time step (delta time)
  unsigned long currentTime = millis();
  float dt = (currentTime - previousTime) / 1000.0; // Convert to seconds
  previousTime = currentTime;

  // Normalize the accelerometer data
  float normAccel = sqrt(imu.accelX() * imu.accelX() +
                         imu.accelY() * imu.accelY() +
                         imu.accelZ() * imu.accelZ());
  float ax = imu.accelX() / normAccel;
  float ay = imu.accelY() / normAccel;
  float az = imu.accelZ() / normAccel;

  // Calculate pitch and roll
  float pitch = atan2(-ax, sqrt(ay * ay + az * az));
  float roll = atan2(ay, az);

  // Compensate magnetometer data for pitch and roll
  float mx = imu.magX();
  float my = imu.magY();
  float mz = imu.magZ();
  
  float mx_comp = mx * cos(pitch) + mz * sin(pitch);
  float my_comp = mx * sin(roll) * sin(pitch) + my * cos(roll) - mz * sin(roll) * cos(pitch);

  // Calculate yaw from the magnetometer
  yawMag = atan2(my_comp, mx_comp) * 180.0 / M_PI;
  if (yawMag < 0) yawMag += 360;

  // Integrate gyroscope data to get yawGyro
  yawGyro += gyroZ * dt; // Assuming gyroZ() returns degrees per second
  if (yawGyro < 0) yawGyro += 360;
  if (yawGyro >= 360) yawGyro -= 360;

  // Apply the complementary filter
  yawFiltered = alpha * yawGyro + (1 - alpha) * yawMag;

  // Convert pitch and roll from radians to degrees
  pitch *= 180.0 / M_PI;
  roll *= 180.0 / M_PI;

  // Subtract initial offsets to reset YPR to [0, 0, 0]
  yawFiltered -= yawOffset;
  pitch -= pitchOffset;
  roll -= rollOffset;

  // Adjust yaw to keep it in the range [0, 360)
  if (yawFiltered < 0) yawFiltered += 360;
  if (yawFiltered >= 360) yawFiltered -= 360;

  // Print yaw, pitch, and roll
  //Serial.print("Yaw: ");
  //Serial.print(yawFiltered, 2);
  //Serial.print(", Pitch: ");
  //Serial.print(pitch, 2);
  //Serial.print(", Roll: ");
  //Serial.println(roll, 2);


  Serial.print(yawFiltered, 2);
  Serial.print(", ");
  Serial.print(pitch, 2);
  Serial.print(", ");
  Serial.println(roll, 2);

  // Delay for stability
  delay(10);
}

void calibrateGyro() {
  Serial.println("Calibrating gyroscope...");

  int numSamples = 500; // Number of samples for calibration
  for (int i = 0; i < numSamples; i++) {
    imu.gyroUpdate();
    gyroXOffset += imu.gyroX();
    gyroYOffset += imu.gyroY();
    gyroZOffset += imu.gyroZ();
    delay(10); // Small delay between samples
  }
  gyroXOffset /= numSamples;
  gyroYOffset /= numSamples;
  gyroZOffset /= numSamples;

  Serial.println("Gyroscope calibration complete");
}
