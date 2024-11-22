/**
 * @file streams-i2s-serial.ino
 * @author Phil Schatzmann
 * @brief see https://github.com/pschatzmann/arduino-audio-tools/blob/main/examples/examples-stream/streams-i2s-serial/README.md
 * 
 * @author Phil Schatzmann
 * @copyright GPLv3
 */


#include "AudioTools.h"

//AudioInfo info(22050, 1, 16);
AudioInfo info(8000, 1, 16);
I2SStream i2sStream; // Access I2S as stream
CsvOutput<int16_t> csvStream(Serial);

StreamCopy copier(csvStream, i2sStream); // copy i2sStream to csvStream

// Arduino Setup
void setup(void) {
    Serial.begin(1000000);
    AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Info);
    
    auto cfg = i2sStream.defaultConfig(RX_MODE);
    

    cfg.copyFrom(info);
    cfg.bits_per_sample = 16;
    cfg.i2s_format = I2S_STD_FORMAT; // or try with I2S_LSB_FORMAT
    cfg.is_master = true;
     // this module nees a master clock if the ESP32 is master
    cfg.use_apll = false;  // try with yes
    cfg.pin_mck = 3; 
    cfg.pin_bck = 18;    // SCK -> GPI18
    cfg.pin_ws = 17;    // WS -> GPIO17
    cfg.pin_data = 16;   // SD -> GPI16
    cfg.port_no = 0;    // Use I2S port 0
    i2sStream.begin(cfg);

    // make sure that we have the correct channels set up
    csvStream.begin(info);

}

// Arduino loop - copy data
void loop() {
    copier.copy();
}