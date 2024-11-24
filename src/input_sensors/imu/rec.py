import socket
import time
import csv
from datetime import datetime

# Configuration
ESP32_IP = '192.168.68.111'  # Update with your ESP32's IP address
ESP32_PORT = 12345           # Must match the port defined in the ESP32 code
RECORD_DURATION = 10         # Duration to record in seconds
FPS = 5                      # Frames per second
CSV_FILE = 'stg_imu_data.csv'  # Output CSV file

# Prepare to collect data
yaw_data = []
pitch_data = []
roll_data = []
timestamps = []
start_time = time.time()

# Function to parse yaw, pitch, and roll data from the incoming line
def parse_imu_data(line):
    # Expecting a line formatted as: y, p, r
    values = line.strip().split(",")
    if len(values) != 3:
        return None

    try:
        yaw = float(values[0])
        pitch = float(values[1])
        roll = float(values[2])
        return [yaw, pitch, roll]
    except ValueError:
        return None

# Connect to the ESP32
print(f"Connecting to ESP32 at {ESP32_IP}:{ESP32_PORT}...")
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ESP32_IP, ESP32_PORT))
    print("Connection established!")
except Exception as e:
    print(f"Error connecting to ESP32: {e}")
    exit()

# Read IMU data for the specified duration
print(f"Recording IMU data for {RECORD_DURATION} seconds...")
try:
    while time.time() - start_time < RECORD_DURATION:
        # Receive data from the ESP32
        data = client_socket.recv(1024).decode('utf-8')
        for line in data.split("\n"):
            imu_data_line = parse_imu_data(line.strip())
            if imu_data_line:
                timestamps.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
                yaw_data.append(imu_data_line[0])
                pitch_data.append(imu_data_line[1])
                roll_data.append(imu_data_line[2])
except Exception as e:
    print(f"Error during data collection: {e}")
finally:
    # Close the socket connection
    client_socket.close()

# Calculate the true sample rate (samples per second)
elapsed_time = time.time() - start_time
true_sample_rate = len(yaw_data) / elapsed_time
samples_per_frame = max(1, int(true_sample_rate / FPS))  # Ensure at least 1 sample per frame

# Group data into frames
frames = []
for i in range(0, len(yaw_data), samples_per_frame):
    frame_yaw = yaw_data[i:i + samples_per_frame]
    frame_pitch = pitch_data[i:i + samples_per_frame]
    frame_roll = roll_data[i:i + samples_per_frame]
    timestamp = timestamps[i] if i < len(timestamps) else "N/A"
    frames.append((i // samples_per_frame, timestamp, frame_yaw, frame_pitch, frame_roll))

# Save the collected data to a CSV file
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame_id", "timestamp", "yaw", "pitch", "roll"])
    for frame_id, timestamp, frame_yaw, frame_pitch, frame_roll in frames:
        writer.writerow([frame_id, timestamp, frame_yaw, frame_pitch, frame_roll])

# Output the true sample rate
print(f"Data collection finished. True sample rate: {true_sample_rate:.2f} Hz")
print(f"Data saved to '{CSV_FILE}'.")
