# Logots - Claude Project Context

## What This Is
An AI-powered cat-sitter robot. Hardware (ESP32, Arduino) streams sensor data to a Python backend that runs perception, decision-making, and motor control pipelines.

## Entry Point
```
python run.py   # Launches the Tkinter GUI (gui_controller.py)
```

## Architecture

### Data Flow
```
Hardware Sensors
    → recording_module.py              (collect raw data)
    → transformation_mart_pipeline.py  (transform to features)
    → decision_engine.py               (fire rules → decisions, every ~3s)
    → execution_engine.py              (decisions → motor commands)
    → mrt_reflective_decisions.py      (LLM strategic layer, every ~20s)
    → recordings/*.csv                 (persisted output)
```

### Core Modules (`src/logots/core/`)
- **`recording_module.py`** — Streams data from all sensors, calls `run_data_collection()`
- **`transformation_mart_pipeline.py`** — Transforms raw sensor data into feature DataFrames:
  - `transform_audio()` — AST model (`mit/ast-finetuned-audioset-10-10-0.4593`) classifies cat meows vs. motor noise
  - `transform_imu()` — Computes movement intensity, rotation speed, balance state from YPR
  - `transform_visual()` — YOLOv8m (`yolov8m.pt`) detects cats (COCO class 15), extracts centroid/area
  - `transform_motor()` — Computes thrust/rotation velocity vectors
  - `build_mrt_experiences()` — Joins all modalities into a unified experience DataFrame (N=12 frame window)
- **`decision_engine.py`** — Rule-based system:
  - Rules: `rule_safety_stop` (priority 99), `rule_voice_command` (50), `rule_cat_greeting` (10), `rule_cat_gaze` (5)
  - Actions: `get_closer`, `back_off`, `center_gaze`, `play_arm`
  - Visual servoing constants: `FRAME_WIDTH=640`, `FOV_H_DEG=60`
- **`execution_engine.py`** — Converts action parameters into timestamped PWM motor commands (10Hz, `PWM_CRUISE=150`)
- **`mrt_reflective_decisions.py`** — LLM strategic layer (Claude Haiku, every ~20s):
  - Reads 80 rows of `master_mrt` + last 10 reactive decisions
  - Compresses into text summary, calls Anthropic API, injects motor commands via `BRAIN_COMMAND_QUEUE`
  - Interprets natural-language voice commands the keyword matcher misses
  - Logs all reasoning to `recordings/mrt_reflective_log.jsonl`
  - Graceful degradation: runs on rules only if API key or SDK is missing
- **`voice_command_module.py`** — Whisper STT (`tiny` model on MPS) for voice commands: "back off", "get closer", "play arm"

### GUI (`src/logots/gui/`)
- **`gui_controller.py`** — Tkinter app with D-pad controls, speed/arm sliders, start/stop recording
- Models preloaded in background thread on startup

### Firmware (`src/logots/firmware/`)
- `audio/esp32_audio` — ESP32 + INMP441 mic
- `imu/imu_ypr_v6` — Grove 9DOF IMU (outputs yaw/pitch/roll)
- `motors/arduino` — Arduino Uno + L293D motor driver
- `video/esp32`, `video/ESP32_CameraWebServer` — ESP32-CAM

### Models (`src/logots/models/`)
- `yolov8m.pt` — YOLOv8 medium, filtered to class 15 (cat), confidence threshold 0.10

## Hardware
- **Vision**: ESP32-CAM (640×640 JPEG, base64 encoded)
- **Audio**: ESP32 + INMP441, 16kHz, 3-second rolling buffer
- **IMU**: Grove 9DOF, outputs YPR lists per frame
- **Motors**: Arduino Uno + L293D, PWM range ~60–150, arm servo 0–180°

## Recordings (`recordings/`)
CSV outputs from pipeline runs — not source code, safe to discard/regenerate:
- `mrt_experience_data.csv` — Joined multi-modal experiences
- `mrt_immediate_decisions.csv` — Rule-fired decisions
- `mrt_decisions_to_actions.csv` — Decisions mapped to motor parameters
- `mrt_generated_motor.csv` — Final PWM command sequence
- `mrt_reflective_log.jsonl` — LLM reasoning trace (every 20s cycle, including no_action and failures)

## Key Constants
| Constant | Value | Location |
|---|---|---|
| `PWM_CRUISE` | 150 | `execution_engine.py` |
| `DT` | 0.1s (10Hz) | `execution_engine.py` |
| `V_SCALE` | 0.001 | `transformation_mart_pipeline.py` |
| `R_SCALE` | 0.005 | `transformation_mart_pipeline.py` |
| `FRAME_WIDTH` | 640px | `decision_engine.py` |
| `FOV_H_DEG` | 60° | `decision_engine.py` |
| `VISUAL_CONF_THR` | 0.10 | `transformation_mart_pipeline.py` |
| `N_FRAMES` | 12 | `build_mrt_experiences()` |

## Device
Runs on macOS with Apple Silicon. MPS is the preferred device for YOLO and AST models; falls back to CPU automatically.

## Dependencies
- `ultralytics` (YOLOv8)
- `transformers` (AST audio classification)
- `whisper` (voice commands)
- `torch` (MPS backend)
- `anthropic` (Claude Haiku API for reflective decisions)
- `python-dotenv` (loads `.env` for API key)
- `pandas`, `numpy`, `opencv-python`, `Pillow`
