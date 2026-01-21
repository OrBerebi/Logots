#include "esp_camera.h"
#include <WiFi.h>

// ===================
// Select camera model
// ===================
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
#include "camera_pins.h"

// ===========================
// WiFi Credentials
// ===========================
const char *ssid = "Mushkins";
const char *password = "besserbros";

//const char* ssid = "self.object";
//const char* password = "FRTZ35%%grmnySF";

void startCameraServer();
void setupLedFlash(int pin);

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // === OPTIMIZATION 1: Frame Size ===
  // VGA (640x480) is significantly faster to transfer than SVGA 
  // and close enough for your 640x640 target.
  config.frame_size = FRAMESIZE_VGA;

  // === OPTIMIZATION 2: PSRAM & Buffering ===
  if (psramFound()) {
    // Keep config quality high (low number) here to ensure a Large Buffer is allocated
    config.jpeg_quality = 10; 
    config.fb_count = 2; // Double buffering
    config.grab_mode = CAMERA_GRAB_LATEST; // Grab the newest frame, discard old ones
  } else {
    // Fallback for boards without PSRAM
    config.frame_size = FRAMESIZE_VGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Camera Init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  
  // === OPTIMIZATION 3: Sensor Tuning ===
  // Set runtime quality to 50 (Higher number = smaller file = faster stream)
  // You can adjust this between 30 (better looking) and 60 (super fast)
  s->set_quality(s, 20); 
  
  // Explicitly enforce VGA size on the sensor
  s->set_framesize(s, FRAMESIZE_VGA);

  // User Preferences (Visuals)
  //s->set_vflip(s, 1);        // Flip image
  s->set_brightness(s, 1);   // Boost brightness
  s->set_saturation(s, -2);  // Reduce saturation

  // WiFi Connection
  WiFi.begin(ssid, password);
  // === OPTIMIZATION 4: Power ===
  WiFi.setSleep(false); // Disable WiFi power saving for max throughput

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

void loop() {
  // Main task is handled by the WebServer in the background
  delay(10000);
}