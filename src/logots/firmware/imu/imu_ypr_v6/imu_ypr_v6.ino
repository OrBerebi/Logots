#include <Wire.h>
#include <WiFi.h>
#include <MPU9250_asukiaaa.h>
#include <MadgwickAHRS.h>

MPU9250_asukiaaa imu;
Madgwick filter;  // Madgwick filter instance

// IMU sampling rate (Hz)
const float imuRate = 50.0f;  // Adjust to your sensor's actual sampling rate
unsigned long lastUpdate = 0;

float calGyroX = 0, calGyroY = 0, calGyroZ = 0;
bool calibrated = false;

float yawOffset = 0, pitchOffset = 0, rollOffset = 0;
bool offsetsSet = false;

// Calibration values
float calAccelX, calAccelY, calAccelZ;


const char *ssid = "Mushkins";
const char *password = "besserbros";

//const char* ssid = "self.object";
//const char* password = "FRTZ35%%grmnySF";

//const char *ssid = "A-6-168";
//const char *password = "62488453";

WiFiServer wifiServer(12345); // Create a TCP server on port 12345
WiFiClient client;


void calibrateGyro() {
    const int samples = 2000;
    float sumX = 0, sumY = 0, sumZ = 0;
    
    for (int i = 0; i < samples; i++) {
        imu.gyroUpdate();
        sumX += imu.gyroX();
        sumY += imu.gyroY();
        sumZ += imu.gyroZ();
        delay(5);
    }
    
    calGyroX = sumX / samples;
    calGyroY = sumY / samples;
    calGyroZ = sumZ / samples;
    calibrated = true;
}

void calibrateOffsets() {
    const int samples = 2000;
    float sumPitch = 0, sumRoll = 0, sumYaw = 0;
    
    for (int i = 0; i < samples; i++) {
        imu.accelUpdate();
        imu.gyroUpdate();

        // GYRO
        float gz = imu.gyroZ() - calGyroZ;   
        float gx = imu.gyroX() - calGyroX;    
        float gy = imu.gyroY() - calGyroY;   

        // ACCEL
        float az = imu.accelZ();   
        float ax = imu.accelX(); //pitch  
        float ay = imu.accelY();

        filter.updateIMU(gx, gy, gz, ax, ay, az);
                
        sumPitch += filter.getPitch();
        sumRoll += filter.getRoll();
        sumYaw += filter.getYaw();
        delay(5);
    }
    
    pitchOffset = sumPitch / samples;
    rollOffset = sumRoll / samples;
    yawOffset = sumYaw / samples;
    offsetsSet = true;
}

void setup() {
  Wire.begin(16, 17); // SDA = GPIO 16, SCL = GPIO 17
  Serial.begin(115200);

  // Initialize the IMU first
  imu.setWire(&Wire);
  imu.beginAccel();
  imu.beginGyro();
  imu.beginMag();

  // Initialize the Madgwick filter
  filter.begin(imuRate);
  filter.setBeta(0.2f);   // <-- Added to match Code 1

  // Calibrate gyro and offsets BEFORE Wi-Fi
  calibrateGyro();
  calibrateOffsets();

  Serial.println("MPU9250 initialization and calibration complete");

  // Now initialize Wi-Fi
  setupWiFi();
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
    // Update the IMU sensor data
    if (millis() - lastUpdate > 1000.0f / imuRate) {
      lastUpdate = millis();

      // Get raw data from the IMU
      imu.accelUpdate();
      imu.gyroUpdate();
      //imu.magUpdate();
      
       // GYRO
      float gz = imu.gyroZ() - calGyroZ;   
      float gx = imu.gyroX() - calGyroX;    
      float gy = imu.gyroY() - calGyroY;   

      // ACCEL
      float az = imu.accelZ();   
      float ax = imu.accelX(); //pitch  
      float ay = imu.accelY();

      // Feed the data into the Madgwick filter with no sign changes
      if (calibrated) {
        filter.updateIMU(gx, gy, gz, ax, ay, az);
      }
    
      

      float yaw = filter.getYaw()- yawOffset;
      float pitch = filter.getPitch()- pitchOffset;
      float roll = filter.getRoll()- rollOffset;

      // Normalize to [0, 360]
      yaw = fmod(yaw + 360.0, 360.0);
      // Normalize to [-180, 180]
      if (roll > 180.0) roll -= 360.0;
      if (roll < -180.0) roll += 360.0;

      // Normalize to [-180, 180]
      if (pitch > 180.0) pitch -= 360.0;
      if (pitch < -180.0) pitch += 360.0;
      
      

      // Send yaw, pitch, roll over Wi-Fi
      String yprData = String(yaw, 1) + "," + String(pitch, 1) + "," + String(roll, 1) + "\n";
      client.print(yprData);
      //Serial.println(yprData);
    }
  }
  delay(5); // Delay for stability

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
