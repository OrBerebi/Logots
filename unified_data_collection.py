import socket
import urllib.request
import numpy as np
import cv2
import time
import csv
from scipy.io.wavfile import write
from datetime import datetime
import threading

# Configuration
RECORD_DURATION = 20  # Duration to record in seconds
FPS = 5               # Frames per second

# Audio Configuration
AUDIO_ESP32_IP = '192.168.68.222'
AUDIO_ESP32_PORT = 12345
AUDIO_WAV_FILE = './recordings/12_02_25-imu-tests2/audio_data.wav'
AUDIO_CSV_FILE = './recordings/12_02_25-imu-tests2/stg_audio_data.csv'
AUDIO_SAMPLE_RATE = 8000  # Audio sampling rate defined on ESP32

# IMU Configuration
IMU_ESP32_IP = '192.168.68.111'
IMU_ESP32_PORT = 12345
IMU_CSV_FILE = './recordings/12_02_25-imu-tests2/stg_imu_data.csv'

# Video Configuration
VIDEO_ESP32_CAM_URL = "http://192.168.68.100/capture"
VIDEO_CSV_FILE = "./recordings/12_02_25-imu-tests2/stg_visual_data.csv"
VIDEO_FILE = "./recordings/12_02_25-imu-tests2/stg_visual_data.m4v"


# Function to collect audio data
def collect_audio():
    print("Starting audio collection...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((AUDIO_ESP32_IP, AUDIO_ESP32_PORT))
        print("Audio connection established!")

        samples = []
        timestamps = []
        start_time = time.time()

        while time.time() - start_time < RECORD_DURATION:
            data = client_socket.recv(1024)
            if data:
                audio_samples = np.frombuffer(data, dtype=np.int16)
                samples.extend(audio_samples)
                timestamps.extend(
                    [datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')] * len(audio_samples)
                )

        # Save audio data to files
        audio_data = np.array(samples, dtype=np.int16)

        # Normalize the audio data to the full 16-bit range
        max_amplitude = np.max(np.abs(audio_data))  # Find the maximum absolute value in the audio data
        if max_amplitude > 0:  # Avoid division by zero
            audio_data = (audio_data / max_amplitude) * 32767  # Scale to full range of int16
            audio_data = audio_data.astype(np.int16)  # Ensure data type is int16


        write(AUDIO_WAV_FILE, AUDIO_SAMPLE_RATE, audio_data)

        samples_per_frame = int(AUDIO_SAMPLE_RATE / FPS)
        with open(AUDIO_CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_id", "timestamp", "audio_samples"])
            for i in range(0, len(audio_data), samples_per_frame):
                frame = audio_data[i:i + samples_per_frame]
                timestamp = timestamps[i] if i < len(timestamps) else "N/A"
                writer.writerow([i // samples_per_frame, timestamp, frame.tolist()])

        print(f"Audio data saved to '{AUDIO_CSV_FILE}' and '{AUDIO_WAV_FILE}'.")

    except Exception as e:
        print(f"Error during audio collection: {e}")
    finally:
        client_socket.close()


# Function to collect IMU data
def collect_imu():
    print("Starting IMU collection...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((IMU_ESP32_IP, IMU_ESP32_PORT))
        print("IMU connection established!")

        yaw_data, pitch_data, roll_data, timestamps = [], [], [], []
        start_time = time.time()

        while time.time() - start_time < RECORD_DURATION:
            data = client_socket.recv(1024).decode('utf-8')
            for line in data.split("\n"):
                values = line.strip().split(",")
                if len(values) == 3:
                    try:
                        yaw, pitch, roll = map(float, values)
                        yaw_data.append(yaw)
                        pitch_data.append(pitch)
                        roll_data.append(roll)
                        timestamps.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
                    except ValueError:
                        continue

        samples_per_frame = max(1, int(len(yaw_data) / (FPS * RECORD_DURATION)))
        with open(IMU_CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_id", "timestamp", "yaw", "pitch", "roll"])
            for i in range(0, len(yaw_data), samples_per_frame):
                timestamp = timestamps[i] if i < len(timestamps) else "N/A"
                writer.writerow([i // samples_per_frame, timestamp, yaw_data[i:i + samples_per_frame],
                                 pitch_data[i:i + samples_per_frame], roll_data[i:i + samples_per_frame]])

        print(f"IMU data saved to '{IMU_CSV_FILE}'.")

    except Exception as e:
        print(f"Error during IMU collection: {e}")
    finally:
        client_socket.close()


# Function to collect video data
def collect_video():
    print("Starting video collection...")
    try:
        frames = []
        with open(VIDEO_CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_id", "timestamp", "frame_data"])

            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < RECORD_DURATION:
                with urllib.request.urlopen(VIDEO_ESP32_CAM_URL) as response:
                    img_data = response.read()
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    frames.append(gray_img)
                    writer.writerow([frame_count, timestamp, gray_img.flatten().tolist()])
                    frame_count += 1

        if frames:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            frame_height, frame_width = frames[0].shape
            video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, FPS, (frame_width, frame_height), isColor=False)
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

        print(f"Video data saved to '{VIDEO_CSV_FILE}' and '{VIDEO_FILE}'.")

    except Exception as e:
        print(f"Error during video collection: {e}")


# Main script
if __name__ == "__main__":
    # Create threads for each collection task
    audio_thread = threading.Thread(target=collect_audio)
    imu_thread = threading.Thread(target=collect_imu)
    video_thread = threading.Thread(target=collect_video)

    # Start threads
    audio_thread.start()
    imu_thread.start()
    video_thread.start()

    # Wait for all threads to complete
    audio_thread.join()
    imu_thread.join()
    video_thread.join()

    print("Data collection complete for all sources!")
