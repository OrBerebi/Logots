import socket
import time  # <-- Add this right under import socket
from pydub import AudioSegment

# --- Configuration ---
ESP32_IP = "192.168.68.101"  # <-- Change this to the IP printed in your ESP32 Serial Monitor
PORT = 12346              # The dedicated audio port we set up
AUDIO_FILE = "dog-barking.wav" # Path to your local audio file

def stream_audio():
    print(f"Loading and preprocessing '{AUDIO_FILE}'...")
    
    try:
        # Load the audio file (pydub can handle wav, mp3, ogg, etc.)
        audio = AudioSegment.from_file(AUDIO_FILE)
        
        # Force the audio to match the ESP32's I2S hardware configuration
        # This prevents it from sounding like Alvin and the Chipmunks or slow-motion demons
        audio = audio.set_frame_rate(44100) # 44.1 kHz
        audio = audio.set_sample_width(2)   # 2 bytes = 16-bit audio
        audio = audio.set_channels(2)       # 2 channels = Stereo
        
        # Extract the raw PCM byte string
        pcm_data = audio.raw_data
        print(f"Preprocessing complete! Total raw data: {len(pcm_data)} bytes.")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{AUDIO_FILE}'. Check the file path.")
        return
    except Exception as e:
        print(f"Audio processing error: {e}")
        return

    print(f"Connecting to ESP32 at {ESP32_IP}:{PORT}...")
    
    try:
        # Open a TCP socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ESP32_IP, PORT))
            print("Connected! Streaming audio to the robot...")
            
            # We send the data in chunks. 
            # TCP backpressure will naturally throttle this script so it 
            # only sends data as fast as the ESP32 can play it!
            chunk_size = 1024 
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                s.sendall(chunk)
            
            print("All data sent! Letting the robot finish playing...")
            time.sleep(1)  # <-- Add a 1-second delay here

            print("Finished streaming audio.")
            
    except ConnectionRefusedError:
        print("Connection refused. Make sure the ESP32 is powered on and the IP is correct.")
    except Exception as e:
        print(f"Network error: {e}")

if __name__ == "__main__":
    stream_audio()