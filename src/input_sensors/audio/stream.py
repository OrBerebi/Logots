import socket
import sounddevice as sd
import numpy as np

# ESP32 server details
ESP32_IP = "192.168.68.222"  # Replace with your ESP32's IP
ESP32_PORT = 12345          # Replace with your ESP32's port

# Audio settings
SAMPLE_RATE = 8000  # Sampling rate (match ESP32)
CHANNELS = 1        # Mono audio

def main():
    try:
        # Connect to the ESP32 server
        print(f"Connecting to ESP32 at {ESP32_IP}:{ESP32_PORT}...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ESP32_IP, ESP32_PORT))
        print("Connected!")

        # Stream audio
        print("Receiving audio...")
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
            while True:
                data = client_socket.recv(1024)  # Adjust buffer size if necessary
                if not data:
                    break
                stream.write(np.frombuffer(data, dtype='int16'))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing connection...")
        client_socket.close()

if __name__ == "__main__":
    main()
