#include <Wire.h>
#include <WiFi.h>
#include <MPU9250_asukiaaa.h>

MPU9250_asukiaaa imu;

// Wi-Fi credentials
const char *ssid = "Mushkins";
const char *password = "besserbros";

WiFiServer wifiServer(12345); // Create a TCP server on port 12345
WiFiClient client;

// Variables to store initial offsets
float yawOffset = 0;
float pitchOffset = 0;
float rollOffset = 0;

// Complementary filter variables
float yawGyro = 0;      // Gyroscope-integrated yaw
float yawMag = 0;       // Magnetometer-derived yaw
float yawFiltered = 0;  // Filtered yaw value
float alpha = 0.95;     // Complementary filter coefficient
unsigned long previousTime = 0;

// Gyroscope calibration offsets
float gyroXOffset = 0, gyroYOffset = 0, gyroZOffset = 0;

void setup() {
  Wire.begin(16, 17); // SDA = GPIO 16, SCL = GPIO 17
  Serial.begin(115200);

  // Initialize Wi-Fi
  setupWiFi();

  // Initialize the IMU
  imu.setWire(&Wire);
  imu.beginAccel();
  imu.beginGyro();
  imu.beginMag();

  Serial.println("MPU9250 initialization complete");

  // Calibrate the gyroscope
  calibrateGyro();

  // Capture initial offsets
  imu.accelUpdate();
  imu.magUpdate();

  float normAccel = sqrt(imu.accelX() * imu.accelX() +
                         imu.accelY() * imu.accelY() +
                         imu.accelZ() * imu.accelZ());
  float ax = imu.accelX() / normAccel;
  float ay = imu.accelY() / normAccel;
  float az = imu.accelZ() / normAccel;

  pitchOffset = atan2(-ax, sqrt(ay * ay + az * az));
  rollOffset = atan2(ay, -az);

  float mx = imu.magX();
  float my = imu.magY();
  float mz = imu.magZ();

  float mx_comp = mx * cos(pitchOffset) + mz * sin(pitchOffset);
  float my_comp = mx * sin(rollOffset) * sin(pitchOffset) + my * cos(rollOffset) - mz * sin(rollOffset) * cos(pitchOffset);

  yawOffset = atan2(my_comp, mx_comp);

  pitchOffset *= 180.0 / M_PI;
  rollOffset *= 180.0 / M_PI;
  yawOffset *= 180.0 / M_PI;
  if (yawOffset < 0) yawOffset += 360;
  previousTime = millis();
}

void loop() {
  // Check for new client connections
  if (!client || !client.connected()) {
    client = wifiServer.available();
    if (client) {
      Serial.println("Client connected!");
    }
  }

  if (client && client.connected()) {
    // Update sensor data
    imu.accelUpdate();
    imu.gyroUpdate();
    imu.magUpdate();

    float gyroX = imu.gyroX() - gyroXOffset;
    float gyroY = imu.gyroY() - gyroYOffset;
    float gyroZ = imu.gyroZ() - gyroZOffset;

    unsigned long currentTime = millis();
    float dt = (currentTime - previousTime) / 1000.0; // Convert to seconds
    previousTime = currentTime;

    float normAccel = sqrt(imu.accelX() * imu.accelX() +
                           imu.accelY() * imu.accelY() +
                           imu.accelZ() * imu.accelZ());
    float ax = imu.accelX() / normAccel;
    float ay = imu.accelY() / normAccel;
    float az = imu.accelZ() / normAccel;

    float pitch = atan2(-ax, sqrt(ay * ay + az * az));
    float roll = atan2(ay, -az);


    

    float mx = imu.magX();
    float my = imu.magY();
    float mz = imu.magZ();

    float mx_comp = mx * cos(pitch) + mz * sin(pitch);
    float my_comp = mx * sin(roll) * sin(pitch) + my * cos(roll) - mz * sin(roll) * cos(pitch);

    yawMag = atan2(my_comp, mx_comp) * 180.0 / M_PI;
    if (yawMag < 0) yawMag += 360;

    yawGyro += gyroZ * dt;
    if (yawGyro < 0) yawGyro += 360;
    if (yawGyro >= 360) yawGyro -= 360;

    yawFiltered = alpha * yawGyro + (1 - alpha) * yawMag;

    pitch *= 180.0 / M_PI;
    roll *= 180.0 / M_PI;


    yawFiltered -= yawOffset;
    pitch -= pitchOffset;
    roll -= rollOffset;

    

    if (yawFiltered < 0) yawFiltered += 360;
    if (yawFiltered >= 360) yawFiltered -= 360;

    // Send yaw, pitch, roll over Wi-Fi
    String yprData = String(yawFiltered, 2) + "," + String(pitch, 2) + "," + String(roll, 2) + "\n";
    client.print(yprData);

    
    //Serial.println(yprData);
    
    //Serial.println(yprData);

    delay(10); // Delay for stability
  }
}

void setupWiFi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  wifiServer.begin();
}

void calibrateGyro() {
  Serial.println("Calibrating gyroscope...");
  int numSamples = 500;
  for (int i = 0; i < numSamples; i++) {
    imu.gyroUpdate();
    gyroXOffset += imu.gyroX();
    gyroYOffset += imu.gyroY();
    gyroZOffset += imu.gyroZ();
    delay(10);
  }
  gyroXOffset /= numSamples;
  gyroYOffset /= numSamples;
  gyroZOffset /= numSamples;
  Serial.println("Gyroscope calibration complete");
}
