# recording_module.py

import socket
import urllib.request
import numpy as np
import cv2
import time
import csv
from scipy.io.wavfile import write
from datetime import datetime
import base64
import threading

# Configuration
FPS = 4  # Frames per second
AUDIO_SAMPLE_RATE = 8000  # Audio sampling rate defined on ESP32

# File paths
AUDIO_WAV_FILE = '../recordings/audio_data.wav'
AUDIO_CSV_FILE = '../recordings/stg_audio_data.csv'
IMU_CSV_FILE = '../recordings/stg_imu_data.csv'
MOTORS_CSV_FILE = '../recordings/stg_motor_data.csv'
VIDEO_CSV_FILE = "../recordings/stg_visual_data.csv"
VIDEO_FILE = "../recordings/stg_visual_data.m4v"

# IP addresses
AUDIO_ESP32_IP = '192.168.178.117'
AUDIO_ESP32_PORT = 12345
IMU_ESP32_IP = '192.168.178.111'
IMU_ESP32_PORT = 12345
MOTORS_ESP32_IP = '192.168.178.118'
MOTORS_ESP32_PORT = 12345
VIDEO_ESP32_CAM_URL = "http://192.168.178.100/capture"


def collect_audio(duration, stop_event):
    print("Starting audio collection...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((AUDIO_ESP32_IP, AUDIO_ESP32_PORT))
        print("Audio connection established!")

        samples = []
        timestamps = []
        start_time = time.time()

        while time.time() - start_time < duration and not stop_event.is_set():
            data = client_socket.recv(1024)
            if data:
                audio_samples = np.frombuffer(data, dtype=np.int16)
                samples.extend(audio_samples)
                timestamps.extend(
                    [datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')] * len(audio_samples)
                )

        audio_data = np.array(samples, dtype=np.int16)
        max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
        if max_amplitude > 0:
            audio_data = (audio_data / max_amplitude) * 32767
            audio_data = audio_data.astype(np.int16)

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


def collect_imu(duration, stop_event):
    print("Starting IMU collection...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((IMU_ESP32_IP, IMU_ESP32_PORT))
        print("IMU connection established!")

        yaw_data, pitch_data, roll_data, timestamps = [], [], [], []
        start_time = time.time()

        while time.time() - start_time < duration and not stop_event.is_set():
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

        samples_per_frame = max(1, int(len(yaw_data) / (FPS * duration)))
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


def collect_video(duration, stop_event):
    print("Starting video collection...")
    try:
        frames = []
        with open(VIDEO_CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_id", "timestamp", "frame_data"])

            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < duration and not stop_event.is_set():
                with urllib.request.urlopen(VIDEO_ESP32_CAM_URL) as response:
                    img_data = response.read()
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    resized_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
                    _, jpeg_buf = cv2.imencode('.jpg', resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    jpeg_base64 = base64.b64encode(jpeg_buf).decode('utf-8')

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    writer.writerow([frame_count, timestamp, jpeg_base64])
                    frames.append(resized_img)
                    frame_count += 1

        if frames:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, FPS, (640, 640), isColor=True)
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

        print(f"Video data saved to '{VIDEO_CSV_FILE}' and '{VIDEO_FILE}'.")

    except Exception as e:
        print(f"Error during video collection: {e}")


def run_data_collection(duration, stop_event):
    def safe_thread(func):
        return threading.Thread(target=lambda: func(duration, stop_event))

    threads = [
        safe_thread(collect_audio),
        safe_thread(collect_imu),
        safe_thread(collect_video),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("All tasks complete or stopped.")
