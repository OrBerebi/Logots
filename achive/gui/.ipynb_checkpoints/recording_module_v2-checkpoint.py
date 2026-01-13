import json
import socket
import urllib.request
import numpy as np
import cv2
import time
import pandas as pd
from scipy.io.wavfile import write
from datetime import datetime
import base64
import threading
import os
import requests
import queue # Import queue for the Brain Buffer

# --- Custom Modules ---
import transformation_mart_pipeline as t_mart 
import visualization_module as viz
import decision_engine as d_engine 
import execution_engine as e_engine 

# --- Configuration ---
FPS = 4
AUDIO_SAMPLE_RATE = 16000
MART_WINDOW_SIZE = 12 
FRAMESIZE_VAL = 6
QUALITY_VAL = 4

# --- File paths ---
AUDIO_WAV_FILE = '../recordings/audio_data.wav'
MART_CSV_FILE = '../recordings/mrt_experience_data.csv'
VIDEO_FILE = "../recordings/stg_visual_data.mp4"
TIMING_LOG_FILE = '../recordings/pipeline_timing_stats.txt'
ANNOTATED_VIDEO_FILE = '../recordings/mrt_annotated_results.gif'
DECISION_CSV_FILE = '../recordings/mrt_immediate_decisions.csv'
ACTION_CSV_FILE = '../recordings/mrt_decisions_to_actions.csv'
GEN_MOTOR_CSV_FILE = '../recordings/mrt_generated_motor.csv'

# --- Network Configuration ---
AUDIO_ESP32_IP = '192.168.68.100'
AUDIO_ESP32_PORT = 12345
IMU_ESP32_IP = '192.168.68.123'
IMU_ESP32_PORT = 12345
MOTORS_ESP32_IP = '192.168.68.101'
MOTORS_ESP32_PORT = 12345
VIDEO_ESP32_CAM_IP_SET_RES = "http://192.168.68.116" 
VIDEO_ESP32_CAM_URL = "http://192.168.68.116/capture"

# --- GLOBAL CONTROL STATE ---
# Manual Control (From GUI)
current_motor_command = (0, 0, 90)
command_lock = threading.Lock()

# Auto Control (From Brain)
# We use a Queue to store the sequence of future actions
BRAIN_COMMAND_QUEUE = queue.Queue()

def send_motor(left_pwm, right_pwm, arm_angle=90):
    """Called by GUI to set manual command."""
    global current_motor_command
    with command_lock:
        current_motor_command = (left_pwm, right_pwm, arm_angle)

def normalize_to_rms(audio_data, target_rms=10000):
    """Normalizes audio data volume."""
    if len(audio_data) == 0:
        return audio_data
    rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
    if rms > 0:
        gain = target_rms / rms
        audio_data = audio_data.astype(np.float64) * gain
        audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
    return audio_data

def configure_camera(ip_address: str, framesize_val: int = 6, quality_val: int = 4):
    clean_ip = ip_address.replace("http://", "").replace("/", "")
    url_control = f"http://{clean_ip}/control"
    try:
        requests.get(url_control, params={'var': 'framesize', 'val': framesize_val}, timeout=5)
        time.sleep(0.2)
        requests.get(url_control, params={'var': 'quality', 'val': quality_val}, timeout=5)
        print(f"✅ Camera configured: Framesize {framesize_val}, Quality {quality_val}.")
    except Exception as e:
        print(f"🛑 WARNING: Could not set camera resolution: {e}")

# ==============================================================================
# STREAMING INFRASTRUCTURE
# ==============================================================================

class ThreadSafeBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.motor = []
        self.audio = [] 
        self.imu = []
        self.visual = []
        
        # Accumulators
        self.acc_motor = []
        self.acc_visual = [] 
        self.acc_imu = []
        self.acc_audio_raw = [] 
        
        # Pipeline State
        self.last_imu_raw_row = None
        self.ctx_trans_vis = pd.DataFrame()
        self.ctx_trans_aud = pd.DataFrame()
        self.ctx_trans_imu = pd.DataFrame()
        self.ctx_trans_mot = pd.DataFrame()
        
        # Results
        self.master_mrt = pd.DataFrame()
        self.master_visual_trans = pd.DataFrame()
        self.master_decisions = pd.DataFrame()
        self.master_actions = pd.DataFrame()
        self.master_gen_motor = pd.DataFrame()

    def add_motor(self, data):
        with self.lock:
            self.motor.append(data)
            self.acc_motor.append(data)

    def add_audio(self, samples, timestamp):
        with self.lock:
            self.audio.extend([(timestamp, s) for s in samples])
            self.acc_audio_raw.extend(samples)

    def add_imu(self, data):
        with self.lock:
            self.imu.append(data)
            self.acc_imu.append(data)

    def add_visual(self, data, raw_img_array):
        with self.lock:
            self.visual.append(data)
            self.acc_visual.append((data, raw_img_array)) 

    def get_chunk_if_ready(self, chunk_size):
        with self.lock:
            if len(self.visual) >= chunk_size:
                chunk_vis = self.visual[:chunk_size]
                self.visual = self.visual[chunk_size:] 
                start_t, end_t = chunk_vis[0]['timestamp'], chunk_vis[-1]['timestamp']
                
                chunk_mot = self.motor[:chunk_size]
                self.motor = self.motor[chunk_size:]
                
                chunk_imu = [x for x in self.imu if x['timestamp'] <= end_t]
                self.imu = [x for x in self.imu if x['timestamp'] > end_t] 
                
                target_audio_samples = int(chunk_size * (AUDIO_SAMPLE_RATE / FPS))
                chunk_aud_tuples = self.audio[:target_audio_samples]
                self.audio = self.audio[target_audio_samples:]

                return {'visual': chunk_vis, 'motor': chunk_mot, 'imu': chunk_imu, 'audio': chunk_aud_tuples}
            return None

# ==============================================================================
# COLLECTION THREADS (PRODUCERS)
# ==============================================================================

def stream_motors(stop_event, buffer):
    """
    THE MOTOR CORTEX: Arbitrates between Manual Input and Brain Queue.
    """
    print("[Stream] Motor thread started (Arbitrator Mode).")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_socket.connect((MOTORS_ESP32_IP, MOTORS_ESP32_PORT))
        frame_id = 0
        target_frame_duration = 1.0 / FPS

        while not stop_event.is_set():
            loop_start = time.time()
            
            # --- 1. Get Inputs ---
            
            # A. Manual Input
            with command_lock:
                manual_l, manual_r, manual_a = current_motor_command
                
            # B. Arbitration Logic
            if manual_l != 0 or manual_r != 0:
                # Case 1: Manual Override
                # If user is touching controls, they win.
                final_l, final_r, final_a = manual_l, manual_r, manual_a
                
                # Safety: Clear the brain queue so it doesn't execute old plans later
                with BRAIN_COMMAND_QUEUE.mutex:
                    BRAIN_COMMAND_QUEUE.queue.clear()
                    
            elif not BRAIN_COMMAND_QUEUE.empty():
                # Case 2: Auto Pilot
                # If manual is idle, check brain.
                final_l, final_r, final_a = BRAIN_COMMAND_QUEUE.get()
                
            else:
                # Case 3: Idle
                final_l, final_r, final_a = 0, 0, 90

            # --- 2. Execute ---
            try:
                client_socket.sendall(f"{final_l},{final_r},{final_a}\n".encode("utf-8"))
            except: pass

            # --- 3. Log (Proprioception) ---
            # We log what actually happened, so the pipeline sees the result of its own actions
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            buffer.add_motor({
                "frame_id": frame_id, "timestamp": ts, 
                "left_pwm": final_l, "right_pwm": final_r, "arm_angle": final_a
            })
            frame_id += 1
            
            # --- 4. Loop Timing ---
            elapsed = time.time() - loop_start
            sleep_t = target_frame_duration - elapsed
            if sleep_t > 0: time.sleep(sleep_t)
        
        client_socket.sendall("0,0,90\n".encode("utf-8"))
    except Exception as e:
        print(f"[Stream] Motor Error: {e}")
    finally:
        client_socket.close()

def stream_audio(stop_event, buffer):
    print("[Stream] Audio thread started.")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((AUDIO_ESP32_IP, AUDIO_ESP32_PORT))
        samples_per_frame = int(AUDIO_SAMPLE_RATE / FPS)
        accumulated_samples = []

        while not stop_event.is_set():
            data = client_socket.recv(4096)
            if not data: break
            new_samples = np.frombuffer(data, dtype=np.int16).tolist()
            accumulated_samples.extend(new_samples)
            
            while len(accumulated_samples) >= samples_per_frame:
                frame_batch = accumulated_samples[:samples_per_frame]
                accumulated_samples = accumulated_samples[samples_per_frame:]
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                buffer.add_audio(frame_batch, ts)
    except Exception as e:
        print(f"[Stream] Audio Error: {e}")
    finally:
        client_socket.close()

def stream_imu(stop_event, buffer):
    print("[Stream] IMU thread started.")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((IMU_ESP32_IP, IMU_ESP32_PORT))
        frame_id_est = 0
        while not stop_event.is_set():
            data = client_socket.recv(1024).decode('utf-8')
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            yaw, pitch, roll = [], [], []
            for line in data.split("\n"):
                parts = line.strip().split(",")
                if len(parts) == 3:
                    try:
                        y, p, r = map(float, parts)
                        yaw.append(y); pitch.append(p); roll.append(r)
                    except: continue
            if yaw:
                buffer.add_imu({
                    "frame_id": frame_id_est, "timestamp": ts,
                    "yaw": yaw, "pitch": pitch, "roll": roll
                })
                frame_id_est += 1
    except Exception as e:
        print(f"[Stream] IMU Error: {e}")
    finally:
        client_socket.close()

class VideoStreamProducer(threading.Thread):
    def __init__(self, url, stop_event, buffer):
        super().__init__()
        self.url = url
        self.stop_event = stop_event
        self.buffer = buffer
        self.daemon = True

    def run(self):
        print("[Stream] Video thread started.")
        frame_id = 0
        target_frame_duration = 1.0 / FPS
        
        while not self.stop_event.is_set():
            loop_start = time.time()
            try:
                with urllib.request.urlopen(self.url, timeout=1.0) as response:
                    img_bytes = response.read()
                
                if len(img_bytes) > 1024:
                    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
                        _, jpg = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        b64 = base64.b64encode(jpg).decode('utf-8')
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        
                        self.buffer.add_visual({
                            "frame_id": frame_id, "timestamp": ts, "frame_data": b64
                        }, resized) 
                        frame_id += 1
            except Exception: pass
            
            elapsed = time.time() - loop_start
            sleep_t = target_frame_duration - elapsed
            if sleep_t > 0: time.sleep(sleep_t)

# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

def process_chunk(chunk_data, buffer, chunk_index):
    try:
        t_start = time.time()
        
        # 1. Data Prep
        df_vis = pd.DataFrame(chunk_data['visual'])
        df_mot = pd.DataFrame(chunk_data['motor'])
        df_imu = pd.DataFrame(chunk_data['imu'])
        
        # Audio Prep
        samples_only = [x[1] for x in chunk_data['audio']]
        if len(df_vis) > 0 and len(samples_only) > 0:
            raw_arr = np.array(samples_only, dtype=np.int16)
            norm_arr = normalize_to_rms(raw_arr, target_rms=2000)
            norm_samples = norm_arr.tolist()
            
            s_per_f = len(norm_samples) // len(df_vis)
            audio_rows = []
            for i in range(len(df_vis)):
                start_i, end_i = i * s_per_f, (i+1) * s_per_f
                if i == len(df_vis) - 1: end_i = len(norm_samples)
                audio_rows.append({
                    "frame_id": df_vis.iloc[i]['frame_id'], 
                    "timestamp": df_vis.iloc[i]['timestamp'], 
                    "audio_samples": norm_samples[start_i:end_i]
                })
            df_aud = pd.DataFrame(audio_rows)
        else:
            df_aud = pd.DataFrame()

        # 2. Transformation
        trans_vis = t_mart.transform_visual(df_vis)
        trans_mot = t_mart.transform_motor(df_mot)
        trans_aud = t_mart.transform_audio(df_aud)
        
        if not df_imu.empty:
            if buffer.last_imu_raw_row is not None:
                imu_input = pd.concat([buffer.last_imu_raw_row, df_imu], ignore_index=True)
                trans_imu_full = t_mart.transform_imu(imu_input)
                trans_imu = trans_imu_full.iloc[1:].reset_index(drop=True)
            else:
                trans_imu = t_mart.transform_imu(df_imu)
            buffer.last_imu_raw_row = df_imu.iloc[[-1]]
        else:
            trans_imu = pd.DataFrame()

        buffer.master_visual_trans = pd.concat([buffer.master_visual_trans, trans_vis], ignore_index=True)

        # 3. Mart Layer
        def get_window(ctx, cur):
            if ctx.empty: return cur
            return pd.concat([ctx, cur], ignore_index=True)

        win_vis = get_window(buffer.ctx_trans_vis, trans_vis)
        win_aud = get_window(buffer.ctx_trans_aud, trans_aud)
        win_imu = get_window(buffer.ctx_trans_imu, trans_imu)
        win_mot = get_window(buffer.ctx_trans_mot, trans_mot)
        
        mart_full = t_mart.build_mrt_experiences(win_aud, win_imu, win_vis, win_mot, N_FRAMES=MART_WINDOW_SIZE)
        
        new_count = len(trans_vis)
        if not mart_full.empty and new_count > 0:
            mart_chunk = mart_full.tail(new_count).copy()
            buffer.master_mrt = pd.concat([buffer.master_mrt, mart_chunk], ignore_index=True)
            
            # 4. Decision & Execution
            imm_dec = d_engine.build_immediate_decisions(mart_chunk)
            if not imm_dec.empty:
                buffer.master_decisions = pd.concat([buffer.master_decisions, imm_dec], ignore_index=True)
                
                actions = d_engine.build_decisions_to_actions(imm_dec, mart_chunk)
                if not actions.empty:
                    buffer.master_actions = pd.concat([buffer.master_actions, actions], ignore_index=True)
                    
                    motor_gen = e_engine.build_mrt_motor(actions, start_frame_id=chunk_index*100)
                    if not motor_gen.empty:
                        buffer.master_gen_motor = pd.concat([buffer.master_gen_motor, motor_gen], ignore_index=True)
                        print(f"   ⚡ [Pipeline] ACTION TRIGGERED: {len(motor_gen)} frames generated.")
                        
                        # --- FEEDING THE CORTEX (The Closed Loop) ---
                        # Challenge: motor_gen assumes DT=0.1s (10Hz)
                        # Our Robot Loop runs at 1/FPS (4Hz = 0.25s)
                        # Strategy: Downsample 10Hz -> 4Hz to play back at correct speed
                        
                        execution_ratio = int((1.0/FPS) / 0.1) # e.g., 0.25 / 0.1 = 2.5 -> 2
                        if execution_ratio < 1: execution_ratio = 1
                        
                        # We take every Nth frame to approximate the speed
                        for i in range(0, len(motor_gen), execution_ratio):
                            row = motor_gen.iloc[i]
                            cmd = (int(row['left_pwm']), int(row['right_pwm']), int(row['arm_angle']))
                            BRAIN_COMMAND_QUEUE.put(cmd)

        buffer.ctx_trans_vis = trans_vis.tail(MART_WINDOW_SIZE)
        buffer.ctx_trans_aud = trans_aud.tail(MART_WINDOW_SIZE)
        buffer.ctx_trans_imu = trans_imu.tail(MART_WINDOW_SIZE)
        buffer.ctx_trans_mot = trans_mot.tail(MART_WINDOW_SIZE)

        dur = time.time() - t_start
        print(f"[Pipeline] Chunk {chunk_index} ({new_count}f) processed in {dur:.3f}s")
        
    except Exception as e:
        print(f"🛑 [Pipeline] Error in Chunk {chunk_index}: {e}")
        import traceback
        traceback.print_exc()

def pipeline_consumer(stop_event, buffer):
    print("[Pipeline] Consumer thread started.")
    chunk_index = 0
    while not stop_event.is_set():
        chunk = buffer.get_chunk_if_ready(chunk_size=MART_WINDOW_SIZE)
        if chunk:
            process_chunk(chunk, buffer, chunk_index)
            chunk_index += 1
        else:
            time.sleep(0.05)
    
    print("[Pipeline] Stop signal. Draining buffer...")
    while True:
        chunk = buffer.get_chunk_if_ready(chunk_size=MART_WINDOW_SIZE)
        if chunk:
            process_chunk(chunk, buffer, chunk_index)
            chunk_index += 1
        else:
            break

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_data_collection(duration_unused, stop_event):
    print("\n=== Starting Robot Stream Pipeline (Full Closed Loop) ===")
    
    stream_buffer = ThreadSafeBuffer()
    configure_camera(VIDEO_ESP32_CAM_IP_SET_RES, framesize_val=FRAMESIZE_VAL, quality_val=QUALITY_VAL)
    
    # Clear any old commands
    with BRAIN_COMMAND_QUEUE.mutex: BRAIN_COMMAND_QUEUE.queue.clear()
    
    threads = []
    t_mot = threading.Thread(target=stream_motors, args=(stop_event, stream_buffer), daemon=True)
    t_aud = threading.Thread(target=stream_audio, args=(stop_event, stream_buffer), daemon=True)
    t_imu = threading.Thread(target=stream_imu, args=(stop_event, stream_buffer), daemon=True)
    t_vis = VideoStreamProducer(VIDEO_ESP32_CAM_URL, stop_event, stream_buffer)
    t_pipe = threading.Thread(target=pipeline_consumer, args=(stop_event, stream_buffer), daemon=True)
    
    threads.extend([t_mot, t_aud, t_imu, t_vis, t_pipe])
    for t in threads: t.start()
    
    print("✅ System Running. Manual Override Active.")
    while not stop_event.is_set():
        time.sleep(0.5)
        
    print("\n🛑 Stop received. Closing threads...")
    for t in threads:
        t.join(timeout=2.0)
        
    print("=== Saving Data ===")
    try:
        if stream_buffer.acc_audio_raw:
            raw_audio = np.array(stream_buffer.acc_audio_raw, dtype=np.int16)
            norm_audio = normalize_to_rms(raw_audio, target_rms=2000)
            write(AUDIO_WAV_FILE, AUDIO_SAMPLE_RATE, norm_audio)
            print(f"Saved {AUDIO_WAV_FILE} (Normalized)")

        if not stream_buffer.master_mrt.empty:
            os.makedirs(os.path.dirname(MART_CSV_FILE), exist_ok=True)
            stream_buffer.master_mrt.to_csv(MART_CSV_FILE, index=False)
            print(f"Saved {MART_CSV_FILE}")

        def save_csv_safe(df, path, headers=None):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if df.empty and headers:
                pd.DataFrame(columns=headers).to_csv(path, index=False)
            else:
                df.to_csv(path, index=False)
        
        save_csv_safe(stream_buffer.master_decisions, DECISION_CSV_FILE, 
                      headers=['immed_id', 'timestamp', 'experience_id', 'rule_name', 'trigger_values', 'proposed_decision'])
        
        save_csv_safe(stream_buffer.master_actions, ACTION_CSV_FILE, 
                      headers=['decision_id', 'timestamp', 'source_module', 'source_event_id', 'experience_id', 'decision_type', 'parameters'])
        
        save_csv_safe(stream_buffer.master_gen_motor, GEN_MOTOR_CSV_FILE, 
                      headers=['frame_id', 'timestamp', 'left_pwm', 'right_pwm', 'arm_angle', 'source', 'decision_id'])

        if stream_buffer.acc_visual:
            print("Generating Video...")
            frames = [x[1] for x in stream_buffer.acc_visual]
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
            for f in frames: out.write(f)
            out.release()
            
            if not stream_buffer.master_mrt.empty and not stream_buffer.master_visual_trans.empty:
                print("Generating Annotated GIF...")
                viz.create_annotated_video(stream_buffer.master_mrt, stream_buffer.master_visual_trans, VIDEO_FILE, ANNOTATED_VIDEO_FILE)

    except Exception as e:
        print(f"Error during save: {e}")
        import traceback
        traceback.print_exc()

    print("=== Session Complete ===\n")