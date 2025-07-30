#include <WiFi.h>
#include <Wire.h>

const char* ssid = "self.object";
const char* password = "FRTZ35%%grmnySF";

WiFiServer wifiServer(12345);
WiFiClient client;

const uint8_t ARDUINO_UNO_I2C_ADDRESS = 0x08; // Match this on the Uno side

void setup() {
  Serial.begin(115200);

  // I2C Master init (ESP32)
  Wire.begin();  // Uses default SDA (GPIO21), SCL (GPIO22). You can customize if needed.

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWi-Fi connected!");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  wifiServer.begin();
}

void loop() {
  if (!client || !client.connected()) {
    client = wifiServer.available();
    if (client) {
      Serial.println("Client connected.");
    }
  }

  if (client && client.connected() && client.available()) {
    String message = client.readStringUntil('\n');
    message.trim();
    if (message.length() > 0) {
      Serial.println("Forwarding to Arduino via I2C: " + message);
      //sendI2C(message);
      sendI2C(message + "\n");
    }
  }

  delay(5);
}

void sendI2C(const String& data) {
  const size_t maxI2CBytes = 32;
  size_t len = data.length();
  size_t sent = 0;

  while (sent < len) {
    Wire.beginTransmission(ARDUINO_UNO_I2C_ADDRESS);
    size_t chunkSize = min(maxI2CBytes, len - sent);
    Wire.write((const uint8_t*)data.substring(sent, sent + chunkSize).c_str(), chunkSize);
    Wire.endTransmission();
    sent += chunkSize;
    delay(5); // Let the slave catch up
  }
}
