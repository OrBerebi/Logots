#include <WiFi.h>
#include <Wire.h>
#include <driver/i2s.h>

//const char *ssid = "Mushkins";
//const char *password = "besserbros";

const char* ssid = "self.object";
const char* password = "FRTZ35%%grmnySF";

// --- Motor Server (Original) ---
WiFiServer motorServer(12345);
WiFiClient motorClient;
const uint8_t ARDUINO_UNO_I2C_ADDRESS = 0x08;

// --- Audio Server (New) ---
WiFiServer audioServer(12346);

// --- I2S Pins for MAX98357A ---
#define I2S_LRC_PIN  25 // Word Select / Left-Right Clock
#define I2S_BCLK_PIN 26 // Bit Clock
#define I2S_DOUT_PIN 27 // Data Out (DIN on the amp)

// --- Configure the I2S Audio Hardware ---
void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = 44100, // Standard audio sample rate
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Stereo (Amp mixes to mono)
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = true // Prevents loud buzzing when stream pauses
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_LRC_PIN,
    .data_out_num = I2S_DOUT_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

// --- Background Task: Runs on Core 0 ---
void audioTask(void *pvParameters) {
  audioServer.begin();
  Serial.println("Audio Server started on port 12346.");

  uint8_t audioBuffer[1024];
  size_t bytesWritten;

  while (true) {
    WiFiClient audioClient = audioServer.available();
    if (audioClient) {
      Serial.println("Audio Client connected to stream.");
      
      while (audioClient.connected()) {
        if (audioClient.available()) {
          // Read chunk of raw audio data from Wi-Fi
          int bytesRead = audioClient.read(audioBuffer, sizeof(audioBuffer));
          if (bytesRead > 0) {
            // Push directly to the I2S hardware DMA buffer
            i2s_write(I2S_NUM_0, audioBuffer, bytesRead, &bytesWritten, portMAX_DELAY);
          }
        } else {
          // Prevent the task watchdog from crashing if connection goes idle
          vTaskDelay(1 / portTICK_PERIOD_MS); 
        }
      }
      Serial.println("Audio stream disconnected.");
      i2s_zero_dma_buffer(I2S_NUM_0); // Mute speaker gracefully
    }
    // Give Core 0 network tasks time to breathe while waiting for client
    vTaskDelay(10 / portTICK_PERIOD_MS); 
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(); // ESP32 I2C master

  setupI2S(); // Initialize the amplifier

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected! IP: " + WiFi.localIP().toString());

  motorServer.begin();

  // Launch the continuous Audio Task on Core 0
  xTaskCreatePinnedToCore(
    audioTask,       // Task function
    "AudioStream",   // Name of task
    10000,           // Stack size (Memory)
    NULL,            // Parameters
    1,               // Priority
    NULL,            // Task handle
    0                // Core ID (0 = Background, 1 = Arduino Loop)
  );
}

// --- Main Loop: Runs on Core 1 (Untouched motor logic) ---
void loop() {
  if (!motorClient || !motorClient.connected()) {
    motorClient = motorServer.available();
    if (motorClient) Serial.println("Motor Client connected.");
  }

  if (motorClient && motorClient.connected() && motorClient.available()) {
    String message = motorClient.readStringUntil('\n');
    message.trim();
    if (message.length() > 0) {
      // Forward to Arduino
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
    delay(5);
  }
}