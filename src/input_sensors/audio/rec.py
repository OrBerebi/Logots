import socket
import time
import numpy as np
from scipy.io.wavfile import write
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
ESP32_IP = '192.168.68.222'  # Update with your ESP32's IP address
ESP32_PORT = 12345           # Must match the port in your ESP32 code
RECORD_DURATION = 20         # Duration to record in seconds
sample_rate = int(8e3)       # Audio sampling rate, defined on the ESP32

WAV_FILE = 'audio_data.wav'  # Output WAV file
CSV_FILE = 'stg_audio_data.csv'  # Output CSV file
FPS = 5                      # Frames per second

print(f"Connecting to ESP32 at {ESP32_IP}:{ESP32_PORT} and recording for {RECORD_DURATION} seconds...")

# Connect to the ESP32 via TCP
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ESP32_IP, ESP32_PORT))
    print("Connection established!")
except Exception as e:
    print(f"Error connecting to ESP32: {e}")
    exit()

# Prepare to collect samples
samples = []
timestamps = []  # To store timestamps for each sample

start_time = time.time()

# Read audio data for the specified duration
print("Start!")
try:
    while time.time() - start_time < RECORD_DURATION:
        # Receive data from the ESP32
        data = client_socket.recv(1024)  # Buffer size of 1024 bytes
        if data:
            # Assume the ESP32 sends 16-bit PCM audio data as raw bytes
            audio_samples = np.frombuffer(data, dtype=np.int16)
            samples.extend(audio_samples)
            timestamps.extend(
                [datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')] * len(audio_samples)
            )  # Timestamp for each sample
except Exception as e:
    print(f"Error during data collection: {e}")
finally:
    # Close the socket connection
    client_socket.close()

# Estimate the sample rate
elapsed_time = time.time() - start_time

# Convert samples to a numpy array
audio_data = np.array(samples, dtype=np.int16)
#sample_rate = len(samples) / elapsed_time

# Calculate the frame duration in terms of samples
samples_per_frame = int(sample_rate / FPS)
#sample_rate = int(sample_rate)


# Normalize the audio data to the full 16-bit range
max_amplitude = np.max(np.abs(audio_data))  # Find the maximum absolute value in the audio data
if max_amplitude > 0:  # Avoid division by zero
    audio_data = (audio_data / max_amplitude) * 32767  # Scale to full range of int16
    audio_data = audio_data.astype(np.int16)  # Ensure data type is int16

# Group samples into frames
frames = []
for i in range(0, len(audio_data), samples_per_frame):
    frame = audio_data[i:i + samples_per_frame]
    timestamp = timestamps[i] if i < len(timestamps) else "N/A"
    frames.append((i // samples_per_frame, timestamp, frame.tolist()))

# Save the collected data to a CSV file
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame_id", "timestamp", "audio_samples"])
    for frame_id, timestamp, frame_samples in frames:
        writer.writerow([frame_id, timestamp, frame_samples])

# Save the audio data to a WAV file
write(WAV_FILE, sample_rate, audio_data)

print(f"Recording finished. Estimated sample rate: {sample_rate} Hz")
print(f"Data saved to '{CSV_FILE}' and audio exported to '{WAV_FILE}'.")

## Plot the audio waveform
#plt.figure(figsize=(10, 4))
#time_axis = np.arange(len(audio_data)) / sample_rate
#plt.plot(time_axis, audio_data)
#plt.title('Audio Signal')
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.grid(True)
#plt.show()


