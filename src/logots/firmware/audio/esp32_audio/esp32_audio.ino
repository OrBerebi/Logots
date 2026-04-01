#include "AudioTools.h"
#include <WiFi.h>

const char *ssid = "Mushkins";
const char *password = "besserbros";

WiFiServer wifiServer(12345);

// The INMP441 requires 64 clock cycles per frame. 32-bit Stereo provides this!
AudioInfo info_mic(16000, 2, 32); 

I2SStream i2sStream;
WiFiClient client;

// --- THE FIX: BULK PROCESSING ARRAYS ---
// We will read 256 frames at a time. 
// 256 frames * 2 channels (Stereo) = 512 samples.
const int READ_FRAMES = 256;
int32_t i2s_rx_buffer[READ_FRAMES * 2]; // Buffer to hold the raw 32-bit stereo data
int16_t tx_buffer[READ_FRAMES];         // Buffer to hold the clean 16-bit mono data

void setupWiFi() {
  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected!");
  wifiServer.begin(); 
}

void setup(void) {
  Serial.begin(115200); 

  setupWiFi(); 

  auto cfg = i2sStream.defaultConfig(RX_MODE);
  cfg.copyFrom(info_mic); 
  cfg.i2s_format = I2S_STD_FORMAT; 
  cfg.is_master = true;
  cfg.use_apll = false; 
  cfg.pin_bck = 18;    
  cfg.pin_ws = 17;     
  cfg.pin_data = 16;   
  cfg.port_no = 0;     
  
  // Keep the buffers large to give the DMA plenty of room
  cfg.buffer_size = 1024;
  cfg.buffer_count = 16;
  
  i2sStream.begin(cfg);
  Serial.println("I2S stream started!");
}

void loop() {
  if (!client || !client.connected()) {
    client = wifiServer.available(); 
    if (client) {
      Serial.println("Client connected!");
      // Turn off TCP packet batching for real-time streaming
      client.setNoDelay(true); 
    }
  }

  if (client && client.connected()) {
    
    // 1. Grab a massive chunk of audio all at once (2048 bytes)
    size_t bytes_read = i2sStream.readBytes((uint8_t*)i2s_rx_buffer, sizeof(i2s_rx_buffer));
    
    // Calculate how many full frames we actually received (8 bytes per 32-bit stereo frame)
    int frames_read = bytes_read / 8; 
    
    if (frames_read > 0) {
      // 2. Process the whole chunk in a lightning-fast loop
      for (int i = 0; i < frames_read; i++) {
        // i2s_rx_buffer holds [Left, Right, Left, Right...]
        // We want the Left channel, which is at index (i * 2)
        // Shift it by 16 bits to safely convert to 16-bit audio
        tx_buffer[i] = (int16_t)(i2s_rx_buffer[i * 2] >> 16);
      }
      
      // 3. Blast the processed 16-bit chunk over Wi-Fi
      // 2 bytes per 16-bit sample
      client.write((uint8_t*)tx_buffer, frames_read * 2);
    }
  }
}