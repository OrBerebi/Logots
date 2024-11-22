# Logots

**Logots** is an open-source project dedicated to building AI-based robots. The first Demo is the catsitter. This robotic companion monitor entertains and interacts with your feline friend using advanced sensors and artificial intelligence. The repository contains hardware integration code and data collection scripts and will eventually house the AI-powered "brain" to bring the robotic cat-sitter to life.

---


## Features

- **Audio Monitoring**: ESP32 with INMP441 microphone module for capturing and analyzing sound data.
- **Video Monitoring**: ESP32-CAM module for real-time video capture and object recognition.
- **Motion Sensing**: Grove 9DOF IMU sensor with ESP32 for detecting movement and orientation.
- **Data Staging Layer**: Python scripts to collect sensor data and export it as `.csv` files for preprocessing and further analysis.
- **Motor Control** (Upcoming): Arduino Uno and L293D motor driver for physical movements, allowing the cat-sitter to interact dynamically.
- **AI Brain** (Upcoming): AI algorithms to process data, make decisions, and interact intelligently with your cat.
