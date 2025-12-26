# transformation_mart_pipeline.py

import pandas as pd
import numpy as np
import ast
import time
import io
import base64
from PIL import Image
from datetime import datetime
import threading
from typing import Dict, Any, List, Tuple
import gc

# --- GLOBAL MODEL STATE & CONFIGURATION ---
_yolo_model = None
_yolo_lock = threading.Lock()
VISUAL_CONF_THR = 0.10
VISUAL_IMG_SIZE = 640

# --- GLOBALS ---
_audio_pipe = None

def get_visual_model(device: str = 'mps'): 
    """Initializes or returns the singleton YOLO model."""
    global _yolo_model
    # LAZY LOAD: We import YOLO here to prevent slow GUI startup
    try:
        from ultralytics import YOLO 
    except ImportError:
        print("Error: 'ultralytics' module not found. Please install it with 'pip install ultralytics'.")
        raise

    with _yolo_lock:
        if _yolo_model is None:
            print(f"Loading YOLOv8m model for visual transformation on device: {device}...")
            
            # Check for MPS availability only if requested
            if device == 'mps':
                 try:
                    import torch
                    if not torch.backends.mps.is_available():
                        # Fallback to CPU if MPS is requested but unavailable
                        print("🛑 MPS device not found/available. Falling back to CPU.")
                        device = 'cpu' 
                 except ImportError:
                    print("🛑 PyTorch not installed/found. Falling back to CPU.")
                    device = 'cpu'

            # The YOLO model is loaded and moved to the selected device
            model = YOLO('yolov8m.pt').to(device) 
            model.fuse()
            model.overrides['conf'] = VISUAL_CONF_THR
            model.overrides['classes'] = [15] # Class 15 is 'cat' in COCO dataset
            _yolo_model = model
            print(f"YOLO model loaded successfully on {device}.")
        return _yolo_model

# --- HELPER: Safe Eval ---
def safe_literal_eval(val):
    """
    Safely evaluates a string to a Python object. 
    If the value is already a list/dict (from memory), returns it as is.
    """
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val # Return raw string if parse fails
    return val

def get_audio_model(device='cpu'):
    """
    Initializes the AST Model.
    """
    global _audio_pipe
    
    if _audio_pipe is None:
        print(f"Loading AST Model on {device}...")
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("Please run: pip install transformers datasets")
        
        _audio_pipe = pipeline(
            "audio-classification", 
            model="mit/ast-finetuned-audioset-10-10-0.4593",
            device=device 
        )
        print("✅ AST Spectrogram Model loaded.")
        
    return _audio_pipe


# ==============================================================================
# 1. AUDIO TRANSFORMATION (GPU ENABLED + LOG-BASED VETO)
# ==============================================================================

def process_audio_frame(row: pd.Series, buffer: list, pipe) -> tuple:
    """
    3-Second Buffer + Log-Based Veto
    """
    # 1. Update Rolling Buffer
    raw_samples = row.get('audio_samples', [])
    if isinstance(raw_samples, str):
        try: raw_samples = ast.literal_eval(raw_samples)
        except: raw_samples = []
    
    if not isinstance(raw_samples, list): raw_samples = []
    buffer.extend(raw_samples)
    
    # 3 Seconds Buffer (Best for Context)
    INPUT_BUFFER_SIZE = 16000 
    
    if len(buffer) > INPUT_BUFFER_SIZE:
        buffer = buffer[-INPUT_BUFFER_SIZE:]
    
    # Defaults
    is_cat, is_human = 0, 0
    cat_prob, human_prob, motor_prob = 0.0, 0.0, 0.0
    loudness = 'none'

    # 2. Inference (Wait for buffer to fill > 80%)
    if len(buffer) >= (INPUT_BUFFER_SIZE * 0.8):
        waveform = np.array(buffer, dtype=np.float32)
        
        max_val = np.max(np.abs(waveform))
        if max_val > 0.01: waveform = waveform / max_val
            
        if max_val > 0.001: 
            try:
                outputs = pipe({"array": waveform, "sampling_rate": 16000}, top_k=10)
                
                # --- LOG ANALYSIS VETO LIST ---
                # Based on your logs, these are the sounds your robot makes:
                motor_labels = [
                    "Vehicle", "Engine", "Electric motor", "Mechanical fan", 
                    "Printer",        
                    "Sliding door",  
                    "Door",           
                    "Telephone",     
                    "Telephone bell ringing",
                    "Beep, bleep"
                ]
                
                cat_labels = ["Meow"]
                human_labels = ["Speech"]

                # Sum probabilities
                cat_prob = sum([x['score'] for x in outputs if x['label'] in cat_labels])
                human_prob = sum([x['score'] for x in outputs if x['label'] in human_labels])
                motor_prob = sum([x['score'] for x in outputs if x['label'] in motor_labels])
                
                # --- USER LOGIC ---
                
                # 1. Human (Speech)
                if human_prob > 0.50: 
                    is_human = 1
                
                # 2. Cat (Meow)
                if cat_prob > 0.09: 
                    is_cat = 1
                    
                # 3. Robot Noise Veto
                # If the sound is more "Printer/Door" than "Meow", ignore it.
                if motor_prob > cat_prob: 
                    is_cat = 0

                # Loudness
                if is_cat:
                    avg_energy = np.mean(np.abs(waveform))
                    if avg_energy < 0.1: loudness = 'low'
                    elif avg_energy < 0.3: loudness = 'medium'
                    else: loudness = 'high'
                
                # DEBUG: Print Raw Labels to monitor
                # if outputs[0]['score'] > 0.15:
                #      top_labels = [f"{x['label']} ({x['score']:.2f})" for x in outputs[:3]]
                #      print(f"Fr {row['frame_id']} RAW: {top_labels}")
                    
            except Exception as e:
                print(f"Inference Error: {e}")

    return {
        'frame_id': row['frame_id'], 
        'timestamp': row['timestamp'],
        'is_cat_voice': is_cat, 
        'is_human_voice': is_human,
        'cat_prob': cat_prob,
        'human_prob': human_prob,
        'motor_prob': motor_prob,
        'meow_loudness': loudness,
        'dominant_frequency': 0.0 
    }, buffer


def transform_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Main transformation for audio data."""
    import gc
    import torch
    
    # --- GPU TRY ---
    # We try to use MPS (Mac GPU). If it crashes, the user knows to revert to -1.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Attempting to run Audio AST on: {device}")
    
    pipe = get_audio_model(device=device)
    
    results = []
    audio_buffer = [] 
    
    print(f"Processing {len(df)} frames...")
    
    for i, row in df.iterrows():
        if i % 10 == 0: print(f"Processing {i}/{len(df)}...", end='\r')
        
        row_data = {
            'frame_id': row.get('frame_id', i), 
            'timestamp': row.get('timestamp', None),
            'audio_samples': row.get('audio_samples', [])
        }
        
        try:
            res, audio_buffer = process_audio_frame(row_data, audio_buffer, pipe)
            results.append(res)
        except Exception as e:
            print(f"\n⚠️ Skipped frame {i} due to error: {e}")
            results.append({
                'frame_id': row['frame_id'], 
                'timestamp': row['timestamp'], 
                'is_cat_voice': 0, 'is_human_voice': 0, 
                'meow_loudness': 'none', 'dominant_frequency': 0.0
            })
        
        if i % 500 == 0: gc.collect()

    return pd.DataFrame(results)


# ==============================================================================
# 2. IMU TRANSFORMATION (Cell 3)
# ==============================================================================

def unwrap_yaw(yaw_list: List[float]) -> np.ndarray:
    """Unwraps the yaw angle to handle 360-degree transitions."""
    arr = np.array(yaw_list, dtype=float)
    return np.degrees(np.unwrap(np.radians(arr)))


def avg_intra_yaw_diff(yaw_list: List[float]) -> float:
    """Computes the average difference between sequential yaw readings in a frame."""
    un = unwrap_yaw(yaw_list)
    return np.diff(un).mean() if un.size > 1 else 0.0


def compute_rotation_speed(yaw_list: List[float], prev_avg: float | None) -> Tuple[float, float, float]:
    """Computes instantaneous rotational change (speed) and yaw delta."""
    cur = avg_intra_yaw_diff(yaw_list)
    delta = cur - prev_avg if prev_avg is not None else 0.0
    return abs(delta), cur, delta 


def compute_movement_intensity(dy: float, dp: float, dr: float) -> float:
    """Calculates overall movement intensity using a Euclidean distance."""
    return np.sqrt(dy**2 + dp**2 + dr**2)


def compute_balance_state(pitch: float, roll: float, intensity: float) -> bool:
    """Determines if the robot is in a balanced state (low tilt, low movement)."""
    return abs(pitch) < 15 and abs(roll) < 15 and intensity < 0.3


def compute_cat_interaction(intensity: float) -> bool:
    """Heuristic for detecting potential cat interaction (high intensity)."""
    return intensity > 10


def compute_is_rest(intensity: float) -> bool:
    """Determines if the robot is resting (very low movement)."""
    return intensity < 1


def process_frame(row: pd.Series, state: Dict[str, Any]) -> Dict[str, Any]:
    """Processes a single IMU frame, maintaining state for deltas."""
    # Rotation speed and delta yaw
    rot, cur_avg, dy = compute_rotation_speed(row['yaw'], state.get('prev_avg_yaw'))
    
    # Pitch and Roll: If it's a list (from chunking), take the first value or mean.
    # The snippet assumes they are lists, so we take index 0.
    p = row['pitch'][0] if isinstance(row['pitch'], list) and row['pitch'] else row['pitch']
    r = row['roll'][0] if isinstance(row['roll'], list) and row['roll'] else row['roll']
    
    # Pitch delta
    prev_pitch = state.get('prev_pitch')
    dp = p - prev_pitch if prev_pitch is not None else 0.0
    
    # Roll delta
    prev_roll = state.get('prev_roll')
    dr = r - prev_roll if prev_roll is not None else 0.0
    
    # Movement intensity and states
    inten = compute_movement_intensity(dy, dp, dr)
    bal = compute_balance_state(p, r, inten)
    cat_int = compute_cat_interaction(inten)
    rest = compute_is_rest(inten)
    
    # Update state for next frame
    state['prev_avg_yaw'] = cur_avg
    state['prev_pitch'] = p
    state['prev_roll'] = r
    
    return {
        'frame_id': row['frame_id'], 'timestamp': row['timestamp'],
        'rotation_speed': rot, 'movement_intensity': inten,
        'balance_state': bal, 'cat_interaction_detected': cat_int,
        'is_rest': rest, 'delta_yaw': dy, 'delta_pitch': dp, 'delta_roll': dr
    }


def transform_imu(df: pd.DataFrame) -> pd.DataFrame:
    """Main transformation for IMU data, handling state sequentially."""
    
    # FIX: Use safe_literal_eval to handle both strings (CSV) and lists (Memory)
    df['yaw'] = df['yaw'].apply(safe_literal_eval)
    df['pitch'] = df['pitch'].apply(safe_literal_eval)
    df['roll'] = df['roll'].apply(safe_literal_eval)

    df_prepped = df[['frame_id', 'timestamp', 'yaw', 'pitch', 'roll']]
    
    # 2. Process frames sequentially
    state = {'prev_avg_yaw': None, 'prev_pitch': None, 'prev_roll': None}
    results = []
    
    for _, row in df_prepped.iterrows():
        results.append(process_frame(row, state))
        
    return pd.DataFrame(results)


# ==============================================================================
# 3. VISUAL TRANSFORMATION (Cell 4)
# ==============================================================================

def jpeg_b64_to_rgb_ndarray(b64: str, img_size: int = VISUAL_IMG_SIZE) -> np.ndarray:
    """Decodes a Base64 JPEG string to a resized RGB numpy array."""
    buf = base64.b64decode(b64)
    with Image.open(io.BytesIO(buf)) as im:
        return np.array(im.convert('RGB').resize((img_size, img_size), Image.LANCZOS))


def transform_visual(df: pd.DataFrame, device: str = 'cpu') -> pd.DataFrame:
    """Main transformation for visual data, using YOLO for cat detection."""
    
    if df.empty:
        return pd.DataFrame()
    
    model = get_visual_model(device)
    
    rows = []
    for _, r in df.iterrows():
        try:
            rgb = jpeg_b64_to_rgb_ndarray(r['frame_data'], VISUAL_IMG_SIZE)
            pil = Image.fromarray(rgb)
        except Exception as e:
            print(f"Warning: Skipping visual frame {r['frame_id']} due to decode error: {e}")
            continue
            
        # 2. Run inference
        res = model(pil, imgsz=VISUAL_IMG_SIZE, verbose=False)[0]
        boxes = res.boxes.cpu()
        
        # 3. Process detections
        det = pd.DataFrame({
            'xmin': boxes.xyxy[:,0].numpy(), 'ymin': boxes.xyxy[:,1].numpy(),
            'xmax': boxes.xyxy[:,2].numpy(), 'ymax': boxes.xyxy[:,3].numpy(),
            'confidence': boxes.conf.numpy(), 'class': boxes.cls.numpy().astype(int),
            'name': ['cat']*len(boxes)
        })

        primary_cat_area = np.nan
        primary_cat_centroid = np.nan
        
        if not det.empty:
            det['bounding_box_area'] = (det['xmax'] - det['xmin']) * (det['ymax'] - det['ymin'])
            cx = (det['xmin'] + det['xmax']) / 2.0
            cy = (det['ymin'] + det['ymax']) / 2.0
            det['bounding_box_centroid'] = list(zip(cx, cy))
            
            primary_det = det.sort_values('confidence', ascending=False).iloc[0]
            primary_cat_area = primary_det['bounding_box_area']
            primary_cat_centroid = primary_det['bounding_box_centroid']
        
        rows.append({
            'frame_id': int(r['frame_id']),
            'timestamp': r['timestamp'],
            'is_cat_detected': int(not det.empty),
            'cat_confidence': float(det['confidence'].max()) if not det.empty else 0.0,
            'bounding_box_area': primary_cat_area,
            'bounding_box_centroid': primary_cat_centroid,
            'inference_time': res.speed['inference'] if (res and hasattr(res, 'speed')) else 0.0,
            'raw_detection': det.to_dict('records')
        })
        
    return pd.DataFrame(rows)


# ==============================================================================
# 4. MOTOR TRANSFORMATION (Cell 5)
# ==============================================================================

def compute_motor_vectors(left_pwm: float, right_pwm: float) -> Tuple[float, float]:
    thrust_velocity = left_pwm + right_pwm
    rotation_velocity = left_pwm - right_pwm
    return thrust_velocity, rotation_velocity


def transform_motor(df: pd.DataFrame) -> pd.DataFrame:
    """Main transformation function for the motor data (stateless)."""
    features = df[['frame_id', 'timestamp']].copy()

    vectors = df.apply(
        lambda row: compute_motor_vectors(row['left_pwm'], row['right_pwm']),
        axis=1, result_type='expand'
    )
    
    features['thrust_velocity'] = vectors[0]
    features['rotation_velocity'] = vectors[1]
    features['arm_angle'] = df['arm_angle']
    
    return features

# ==============================================================================
# 5. MART EXPERIENCE LAYER (Cell 6)
# ==============================================================================

V_SCALE = 0.001 
R_SCALE = 0.005
LOUDNESS_RANK = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
RANK_TO_TEXT = {3: 'high', 2: 'medium', 1: 'low', 0: np.nan}

def build_mrt_experiences(aud_df: pd.DataFrame, imu_df: pd.DataFrame, vis_df: pd.DataFrame, mot_df: pd.DataFrame, N_FRAMES: int = 12) -> pd.DataFrame:
    """Assembles the final Data Mart (MRT)."""
    
    aud_df = aud_df.sort_values('frame_id').reset_index(drop=True)
    imu_df = imu_df.sort_values('frame_id').reset_index(drop=True)
    vis_df = vis_df.sort_values('frame_id').reset_index(drop=True)
    mot_df = mot_df.sort_values('frame_id').reset_index(drop=True)
    
    mot_df = mot_df.copy()
    mot_df['timestamp'] = pd.to_datetime(mot_df['timestamp'], errors='coerce')
    
    rows = []
    
    for fid in vis_df['frame_id'].unique():
        aud = aud_df[aud_df['frame_id'] <= fid].tail(N_FRAMES)
        imu = imu_df[imu_df['frame_id'] <= fid].tail(N_FRAMES)
        vis = vis_df[vis_df['frame_id'] <= fid].tail(N_FRAMES)
        mot = mot_df[mot_df['frame_id'] <= fid].tail(N_FRAMES).copy()

        current_vis_frame = vis[vis['frame_id'] == fid].iloc[0] if not vis[vis['frame_id'] == fid].empty else None
        if current_vis_frame is None: continue

        # Audio & IMU
        aud_is_cat = aud['is_cat_voice'].fillna(0).astype(bool)
        aud_is_human = aud['is_human_voice'].fillna(0).astype(bool)
        human_seq = aud.loc[aud_is_human, 'frame_id'].tolist()
        cat_seq = aud.loc[aud_is_cat, 'frame_id'].tolist()
        
        loudness_ranks = aud['meow_loudness'].astype(str).str.lower().map(LOUDNESS_RANK).fillna(0)
        max_rank = loudness_ranks.max()
        meow_loud = RANK_TO_TEXT.get(max_rank, np.nan)

        move_int = imu['movement_intensity'].mean() if not imu.empty else 0.0
        cat_int  = bool(imu['cat_interaction_detected'].any()) if not imu.empty else False

        # Vision
        cat_data_over_time = []
        for _, frame_data in vis.iterrows():
            detections = frame_data.get('raw_detection', [])
            centroid = frame_data.get('bounding_box_centroid') 
            area = frame_data.get('bounding_box_area')         

            if isinstance(detections, list) and len(detections) > 0 and centroid is not np.nan:
                if isinstance(centroid, (list, tuple)):
                     cat_data_over_time.append({'centroid': centroid, 'area': area})
                else:
                    cat_data_over_time.append(None)
            else:
                cat_data_over_time.append(None)

        detected_now = current_vis_frame['is_cat_detected'] == 1
        
        # Vision Feature: distance change
        area_changes = {'increases': 0, 'decreases': 0}
        valid_cat_data = [d for d in cat_data_over_time if d is not None]
        
        if len(valid_cat_data) >= 2:
            areas = [d['area'] for d in valid_cat_data]
            for i in range(1, len(areas)):
                if areas[i] > areas[i-1]:
                    area_changes['increases'] += 1
                elif areas[i] < areas[i-1]:
                    area_changes['decreases'] += 1
        
        if detected_now:
            if area_changes['increases'] > area_changes['decreases']:
                dist_change = 'closer'
            elif area_changes['decreases'] > area_changes['increases']:
                dist_change = 'farther'
            else:
                dist_change = 'no_change'
        else:
            dist_change = np.nan

        # Vision Feature: delta & sum position
        delta_pos = (np.nan, np.nan)
        total_dist_moved = 0.0
        
        if valid_cat_data:
            # sum_cat_position
            for i in range(1, len(valid_cat_data)):
                pos1 = np.array(valid_cat_data[i-1]['centroid'])
                pos2 = np.array(valid_cat_data[i]['centroid']) 
                total_dist_moved += np.linalg.norm(pos2 - pos1)
            
            # delta_cat_position
            if len(valid_cat_data) >= 2:
                 first_pos = np.array(valid_cat_data[0]['centroid'])
                 last_pos = np.array(valid_cat_data[-1]['centroid'])
                 delta_pos = tuple(last_pos - first_pos)

        cat_x, cat_y = current_vis_frame['bounding_box_centroid'] if detected_now and isinstance(current_vis_frame['bounding_box_centroid'], (list, tuple)) else (np.nan, np.nan)
        
        # Robot Motor Features
        delta_arm_angle, sum_arm_move = 0.0, 0.0
        delta_robot_pos, delta_robot_rot_deg, sum_robot_dist = (0.0, 0.0), 0.0, 0.0

        if len(mot) > 1:
            delta_arm_angle = mot['arm_angle'].iloc[-1] - mot['arm_angle'].iloc[0]
            sum_arm_move = mot['arm_angle'].diff().abs().sum()

            timestamps = mot['timestamp'].values
            dts = np.diff(timestamps).astype(float) / 1e9 
            dts = np.insert(dts, 0, 0.0)

            v = mot['thrust_velocity'].values * V_SCALE
            omega = mot['rotation_velocity'].values * R_SCALE

            delta_headings = omega * dts
            headings_at_end = np.cumsum(delta_headings)
            avg_headings = headings_at_end - (delta_headings / 2.0)

            dist_steps = v * dts
            x_steps = dist_steps * np.cos(avg_headings)
            y_steps = dist_steps * np.sin(avg_headings)

            local_x = np.sum(x_steps)
            local_y = np.sum(y_steps)
            delta_robot_rot_rad = headings_at_end[-1] - headings_at_end[0]
            delta_robot_rot_deg = np.degrees(delta_robot_rot_rad)
            sum_robot_dist = np.sum(np.abs(dist_steps))
            delta_robot_pos = (local_x, local_y)

        rows.append({
            'experience_id': fid,
            'last_experience_id_array': vis['frame_id'].tolist(),
            'timestamp': current_vis_frame['timestamp'],
            'is_cat_voice': bool(aud_is_cat.any()),
            'is_human_voice': bool(aud_is_human.any()),
            'human_voice_sequence': human_seq,
            'cat_voice_sequence': cat_seq,
            'meow_loudness': meow_loud,
            'cat_detected': detected_now,
            'cat_position_x': cat_x,
            'cat_position_y': cat_y,
            'cat_distance_change': dist_change,
            'delta_cat_position': delta_pos,
            'sum_cat_position': total_dist_moved,
            'movement_intensity': move_int,
            'cat_interaction_detected': cat_int,
            'delta_arm_angle': delta_arm_angle,
            'sum_arm_movement': sum_arm_move,
            'delta_robot_position': delta_robot_pos,
            'sum_robot_position': sum_robot_dist,
            'delta_robot_rotation': delta_robot_rot_deg
        })
        
    return pd.DataFrame(rows)