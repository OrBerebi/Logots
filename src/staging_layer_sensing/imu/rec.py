import serial
import time
import csv
from datetime import datetime

# Configuration
SERIAL_PORT = '/dev/tty.usbserial-110'  # Update with your Arduino's serial port (e.g., 'COM3' on Windows)
BAUD_RATE = 1000000  # Serial communication baud rate
RECORD_DURATION = 10  # Duration to record in seconds
FPS = 5  # Frames per second
CSV_FILE = 'stg_imu_data.csv'  # Output CSV file

# Open the serial port
print(f"Opening the serial port and recording IMU data for {RECORD_DURATION} seconds...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for the Arduino to initialize

# Prepare to collect data
yaw_data = []
pitch_data = []
roll_data = []
timestamps = []
start_time = time.time()

# Function to parse yaw, pitch, and roll data from the serial input
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

# Read IMU data for the specified duration
while time.time() - start_time < RECORD_DURATION:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        imu_data_line = parse_imu_data(line)
        if imu_data_line:
            timestamps.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
            yaw_data.append(imu_data_line[0])
            pitch_data.append(imu_data_line[1])
            roll_data.append(imu_data_line[2])

# Calculate the true sample rate (samples per second)
elapsed_time = time.time() - start_time
true_sample_rate = len(yaw_data) / elapsed_time
samples_per_frame = int(true_sample_rate / FPS)

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

# Close the serial connection
ser.close()
