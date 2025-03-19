#include <Wire.h>
#include <WiFi.h>
#include <MPU9250_asukiaaa.h>
#include <MadgwickAHRS.h>

MPU9250_asukiaaa imu;
Madgwick filter;  // Madgwick filter instance

// IMU sampling rate (Hz)
const float imuRate = 25.0f;  // Adjust to your sensor's actual sampling rate
unsigned long lastUpdate = 0;

float calGyroX = 0, calGyroY = 0, calGyroZ = 0;
bool calibrated = false;

float yawOffset = 0, pitchOffset = 0, rollOffset = 0;
bool offsetsSet = false;

// Calibration values
float calAccelX, calAccelY, calAccelZ;


// Wi-Fi credentials
const char *ssid = "Mushkins";
const char *password = "besserbros";

WiFiServer wifiServer(12345); // Create a TCP server on port 12345
WiFiClient client;


void calibrateGyro() {
    const int samples = 500;
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
    const int samples = 500;
    float sumPitch = 0, sumRoll = 0, sumYaw = 0;
    
    for (int i = 0; i < samples; i++) {
        imu.accelUpdate();
        imu.gyroUpdate();

        float gx = imu.gyroX() - calGyroX;
        float gy = imu.gyroY() - calGyroY;
        float gz = imu.gyroZ() - calGyroZ;
        
        float ax = imu.accelX();
        float ay = imu.accelY();
        float az = imu.accelZ();

        filter.updateIMU(gx, gy, gz, ax, ay, -az);
                
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

  // Initialize Wi-Fi
  setupWiFi();

  // Initialize the IMU
  imu.setWire(&Wire);
  imu.beginAccel();
  imu.beginGyro();
  imu.beginMag();
  
  // Initialize the Madgwick filter with the IMU rate
  filter.begin(imuRate);
  calibrateGyro();
  calibrateOffsets();

  Serial.println("MPU9250 initialization complete");
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
      
      float gx = imu.gyroX() - calGyroX;
      float gy = imu.gyroY() - calGyroY;
      float gz = imu.gyroZ() - calGyroZ;
      
      float ax = imu.accelX();
      float ay = imu.accelY();
      float az = imu.accelZ();
      
      //float mx = imu.magX();
      //float my = imu.magY();
      //float mz = imu.magZ();


      // Feed the data into the Madgwick filter
      if (calibrated) {
        //filter.update(gx, gy, -gz, ax, ay, -az, mx, my, -mz);
        filter.updateIMU(gx, gy, gz, ax, ay, -az);
      }
    
      

      float yaw = filter.getYaw()- yawOffset;
      float pitch = filter.getPitch()- pitchOffset;
      float roll = filter.getRoll()- rollOffset;
      
      

      // Send yaw, pitch, roll over Wi-Fi
      String yprData = String(yaw, 2) + "," + String(pitch, 2) + "," + String(roll, 2) + "\n";
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
