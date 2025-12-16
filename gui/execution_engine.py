import pandas as pd
import numpy as np
import json
from datetime import timedelta

# --- Configuration & Physics Constants ---
PWM_CRUISE = 150
DT = 0.1  # 10Hz Sample Rate for generated motor commands

# Scale factors (Must match transformation_mart_pipeline.py)
V_SCALE = 0.001 
R_SCALE = 0.005

# Derived speeds
SPEED_CM_PER_SEC = (PWM_CRUISE * 2) * V_SCALE * 100 
SPEED_DEG_PER_SEC = (PWM_CRUISE * 2) * R_SCALE * (180 / np.pi)

def build_mrt_motor(actions_df, start_frame_id=0):
    """
    Generates a sequence of motor commands based on the action parameters.
    Returns a DataFrame representing the 'Robot Intent' motor log.
    """
    motor_rows = []
    current_frame_id = start_frame_id
    
    if actions_df.empty:
        return pd.DataFrame(columns=['frame_id', 'timestamp', 'left_pwm', 'right_pwm', 'arm_angle', 'source', 'decision_id'])

    for _, action in actions_df.iterrows():
        try:
            params = json.loads(action['parameters'])
        except:
            params = {}
            
        start_time = action['timestamp']
        dec_id = action['decision_id']
        
        current_time = start_time
        arm_angle = 90 
        
        # Helper to append rows
        def add_row(l_pwm, r_pwm):
            nonlocal current_frame_id, current_time
            motor_rows.append({
                'frame_id': current_frame_id,
                'timestamp': current_time,
                'left_pwm': int(l_pwm),
                'right_pwm': int(r_pwm),
                'arm_angle': arm_angle,
                'source': 'mrt_decisions_to_actions',
                'decision_id': dec_id
            })
            current_frame_id += 1
            current_time += timedelta(seconds=DT)

        # --- 1. Rotation Phase ---
        rotate_deg = params.get('rotate_deg', 0.0)
        if abs(rotate_deg) > 1.0:
            duration_sec = abs(rotate_deg) / SPEED_DEG_PER_SEC
            steps = int(duration_sec / DT)
            
            if rotate_deg > 0: 
                l_pwm, r_pwm = -PWM_CRUISE, PWM_CRUISE # Turn Left
            else:
                l_pwm, r_pwm = PWM_CRUISE, -PWM_CRUISE # Turn Right
            
            for _ in range(max(1, steps)):
                add_row(l_pwm, r_pwm)

        # --- 2. Translation Phase ---
        move_cm = params.get('move_forward_cm', 0.0)
        if abs(move_cm) > 1.0:
            duration_sec = abs(move_cm) / SPEED_CM_PER_SEC
            steps = int(duration_sec / DT)
            
            if move_cm > 0:
                l_pwm, r_pwm = PWM_CRUISE, PWM_CRUISE # Forward
            else:
                l_pwm, r_pwm = -PWM_CRUISE, -PWM_CRUISE # Backward
                
            for _ in range(max(1, steps)):
                add_row(l_pwm, r_pwm)
                
        # --- 3. Idle / Stop Phase ---
        # If no movement calculated or action finished, add a stop frame
        if abs(rotate_deg) < 1.0 and abs(move_cm) < 1.0:
             add_row(0, 0)

    return pd.DataFrame(motor_rows)