import cv2
import csv
import time
import requests
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from queue import Queue

# Configuration
ESP32_CAM_URL = "http://192.168.68.118/capture"  # Replace with your ESP32-CAM's URL
CSV_FILE = "stg_visual_data.csv"  # Output CSV file for black and white
VIDEO_FILE = "stg_visual_data.m4v"  # Output video file (still .m4v, but black and white)
FRAME_RATE = 5  # Target frame rate in frames per second
RECORD_DURATION = 10  # Duration to record in seconds

# Shared resources
frame_queue = Queue(maxsize=50)  # Limit the queue size to prevent memory overload
lock = Lock()
frames = []
frame_count = 0

# Producer thread: Fetch frames from ESP32-CAM
def fetch_frames():
    global frame_count
    session = requests.Session()
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > RECORD_DURATION:
            break

        try:
            fetch_start = time.perf_counter()
            response = session.get(ESP32_CAM_URL, stream=True, timeout=5)
            fetch_end = time.perf_counter()
            print(f"Frame fetch time: {fetch_end - fetch_start:.4f} seconds")

            img_data = response.content
            frame_queue.put((frame_count, img_data, time.time()))
            frame_count += 1
        except Exception as e:
            print(f"Error fetching frame: {e}")
            break


def process_frames():
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "timestamp", "frame_width", "frame_height", "frame_data"])

        while True:
            if not frame_queue.empty():
                frame_id, img_data, fetch_time = frame_queue.get()
                try:
                    decode_start = time.perf_counter()
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    print(img_data)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    decode_end = time.perf_counter()
                    print(f"Image decode time: {decode_end - decode_start:.4f} seconds")


                    if img is not None:
                        gray_start = time.perf_counter()
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray_end = time.perf_counter()
                        print(f"Grayscale conversion time: {gray_end - gray_start:.4f} seconds")

                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        gray_array = gray_img.flatten().tolist()

                        save_start = time.perf_counter()
                        writer.writerow([frame_id, timestamp, gray_array])
                        save_end = time.perf_counter()
                        print(f"CSV write time: {save_end - save_start:.4f} seconds")

                        with lock:
                            frames.append(gray_img)

                        process_time = time.time() - fetch_time
                        print(f"Frame {frame_id} processed. Frame rate: {1/process_time:.2f} fps")
                except Exception as e:
                    print(f"Error processing frame {frame_id}: {e}")

            if frame_count > 0 and frame_queue.empty() and frame_count >= RECORD_DURATION * FRAME_RATE:
                break


# Start threads
producer_thread = Thread(target=fetch_frames)
consumer_thread = Thread(target=process_frames)
producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()

# Save video
if frames:
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    with lock:
        video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, FRAME_RATE, (frames[0].shape[1], frames[0].shape[0]), isColor=False)
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

print(f"Data saved to '{CSV_FILE}' and video saved to '{VIDEO_FILE}'.")
