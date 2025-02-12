import urllib.request
import numpy as np
import cv2
import time
import csv
from datetime import datetime


# Configuration
ESP32_CAM_URL = "http://192.168.68.100/capture"  # Replace with your ESP32-CAM's URL
CSV_FILE = "stg_visual_data.csv"  # Output CSV file for black and white
VIDEO_FILE = "stg_visual_data.m4v"  # Output video file (still .m4v, but black and white)
FRAME_RATE = 5  # Target frame rate in frames per second
RECORD_DURATION = 20  # Duration to record in seconds

# Variables to track frame count and sample index
frame_count = 0
sample_index = 0

# List to store frames for the video
frames = []

# Open the CSV file for writing
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame_id", "timestamp", "frame_data"])

    start_time = time.time()

    while True:
        try:
            # Check if the recording duration has been reached
            elapsed_time = time.time() - start_time
            if elapsed_time > RECORD_DURATION:
                break

            start_frame_t = time.time()

            # Fetch a single frame from the ESP32-CAM
            fetch_start = time.perf_counter()
            with urllib.request.urlopen(ESP32_CAM_URL) as response:
                img_data = response.read()
            fetch_end = time.perf_counter()
            print(f"Frame fetch time: {fetch_end - fetch_start:.4f} seconds")

            # Convert the byte data to a numpy array
            img_array = np.frombuffer(img_data, dtype=np.uint8)

            # Decode the JPEG image
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                # Convert the frame to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

                # Add the grayscale frame to the list of frames
                frames.append(gray_img)

                # Get the frame dimensions (for grayscale, it's just height and width)
                frame_height, frame_width = gray_img.shape

                # Flatten the grayscale array to save in the CSV
                gray_array = gray_img.flatten().tolist()

                # Write the frame data to the CSV file
                writer.writerow([frame_count, timestamp, gray_array])

                # Display the image as a video frame (optional)
                # cv2.imshow("ESP32-CAM Video Stream", gray_img)

                # Press 'q' to exit the stream early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Increment the frame count and sample index
                frame_count += 1
                sample_index += 1

                # Calculate the frame rate
                frame_time = time.time() - start_frame_t
                actual_frame_rate = 1 / frame_time
                print(f"Frame {frame_count} collected. Frame rate: {actual_frame_rate:.2f} frames per second")

            else:
                print("Failed to decode image.")

        except Exception as e:
            print(f"Error capturing image: {e}")
            break

    # Clean up
    cv2.destroyAllWindows()

# Now write all collected frames to the video file (black and white)
if frames:
    # Use 'avc1' codec for H.264 in .m4v, but we need to set 1 channel for grayscale
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, round(actual_frame_rate), (frame_width, frame_height), isColor=False)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()

print(f"Data saved to '{CSV_FILE}' and video saved to '{VIDEO_FILE}'.")
