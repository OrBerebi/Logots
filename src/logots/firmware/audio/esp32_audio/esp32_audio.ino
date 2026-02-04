#include "AudioTools.h"
#include <WiFi.h>

const char *ssid = "Mushkins";
const char *password = "besserbros";

//const char* ssid = "self.object";
//const char* password = "FRTZ35%%grmnySF";

//const char *ssid = "A-6-168";
//const char *password = "62488453";
unsigned long lastDebugPrint = 0; // for throttling debug prints


WiFiServer wifiServer(12345); // Create a TCP server on port 12345

// Audio Configuration
//AudioInfo info(48000, 1, 16); // Audio format: 8kHz, mono, 16-bit
AudioInfo info(16000, 1, 16); // Audio format: 8kHz, mono, 16-bit
I2SStream i2sStream;         // Access I2S as a stream

StreamCopy copier;           // StreamCopy instance

WiFiClient client;           // Current connected client

// Custom Wi-Fi Output Stream
class WiFiOutputStream : public Print {
  WiFiClient* client; // Pointer to current client

public:
  WiFiOutputStream(WiFiClient* cli) : client(cli) {}

  virtual size_t write(uint8_t data) override {
    if (client && client->connected()) {
      return client->write(data);
    }
    return 0;
  }

  virtual size_t write(const uint8_t* buffer, size_t size) override {
    if (client && client->connected()) {
      return client->write(buffer, size);
    }
    return 0;
  }
};

WiFiOutputStream wifiStream(&client); // Create the Wi-Fi output stream

// Function to set up Wi-Fi
void setupWiFi() {
  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  wifiServer.begin(); // Start the Wi-Fi server
}

// Arduino Setup
void setup(void) {
  Serial.begin(115200); // Use for logging
  //AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Info);

  setupWiFi(); // Initialize Wi-Fi

  auto cfg = i2sStream.defaultConfig(RX_MODE);
  cfg.copyFrom(info);
  cfg.i2s_format = I2S_STD_FORMAT; // Use I2S standard format
  cfg.is_master = true;
  cfg.use_apll = false; // Enable if needed
  cfg.pin_mck = 3;
  cfg.pin_bck = 18;    // SCK -> GPIO18
  cfg.pin_ws = 17;     // WS -> GPIO17
  cfg.pin_data = 16;   // SD -> GPIO16
  cfg.port_no = 0;     // Use I2S port 0
  cfg.buffer_size = 512;
  cfg.buffer_count = 16;
  i2sStream.begin(cfg);
  Serial.println("I2S stream started!");
}

// Arduino Loop
void loop() {
  // Check for new client connections
  if (!client || !client.connected()) {
    client = wifiServer.available(); // Accept a new client
    if (client) {
      Serial.println("Client connected!");
      copier.begin(wifiStream, i2sStream); // Set the Wi-Fi output and I2S input
    }
  }

  // Copy I2S audio stream to Wi-Fi output
  if (client && client.connected()) {
    copier.copy();

    //static unsigned long lastDebugPrint = 0;
    //if (millis() - lastDebugPrint > 1000) { // every 1 second
    //    int sample = i2sStream.read();       // <-- NO arguments
    //    Serial.print("Sample: ");
    //    Serial.println(sample);
    //    lastDebugPrint = millis();
    //}
  }
}
