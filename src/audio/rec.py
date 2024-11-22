import serial
import time
import numpy as np
from scipy.io.wavfile import write
import csv
import matplotlib.pyplot as plt
from datetime import datetime


# Configuration
SERIAL_PORT = '/dev/cu.usbserial-10'  # Update with your Arduino's serial port (e.g., 'COM3' on Windows)
BAUD_RATE = 1000000  # Match with Arduino's serial baud rate
RECORD_DURATION = 10  # Duration to record in seconds
WAV_FILE = 'audio_data.wav'  # Output WAV file
CSV_FILE = 'stg_audio_data.csv'  # Output CSV file
FPS = 5  # Frames per second


print(f"Opening the serial port and recording for {RECORD_DURATION} seconds...")

# Open the serial port
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for the Arduino to initialize

print("Start!")
# Prepare to collect samples
samples = []
timestamps = []  # To store timestamps for each sample

start_time = time.time()

# Read audio data for the specified duration
while time.time() - start_time < RECORD_DURATION:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()  # Read a line from the serial port
        try:
            sample_value = int(line)  # Convert the CSV value to an integer
            samples.append(sample_value)
            timestamps.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))  # Add timestamp
        except ValueError:
            # Skip lines that cannot be converted to integers
            continue

# Estimate the sample rate
elapsed_time = time.time() - start_time

# Close the serial connection
ser.close()

# Convert samples to a numpy array
audio_data = np.array(samples, dtype=np.int16)
sample_rate = len(samples) / elapsed_time

# Calculate the frame duration in terms of samples
samples_per_frame = int(sample_rate / FPS)
sample_rate = int(sample_rate)

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

# Plot the audio waveform
plt.figure(figsize=(10, 4))
time_axis = np.arange(len(audio_data)) / sample_rate
plt.plot(time_axis, audio_data)
plt.title('Audio Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
