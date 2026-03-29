import pandas as pd
import numpy as np
import uuid
import json
from datetime import datetime

# --- 1. Rule Definitions (Immediate Decisions) ---

def rule_cat_greeting(row):
    """Rule: If cat seen AND heard -> Get Closer."""
    # Safety: Handle NaN loudness
    loudness = row.get('meow_loudness')
    if pd.isna(loudness):
        loudness = 'unknown'

    # Check if cat detected and voice present
    if row.get('cat_detected') and row.get('is_cat_voice'):
        return {
            'rule_name': 'CAT_GREETING',
            'priority': 10,
            'proposed_decision': 'get_closer',
            'trigger_values': {
                'cat_detected': bool(row['cat_detected']),
                'meow_loudness': loudness
            }
        }
    return None

def rule_safety_stop(row):
    """Rule: If movement intensity is too high (collision) -> Back Off."""
    intensity = row.get('movement_intensity')
    if pd.isna(intensity):
        return None
        
    if intensity > 2.4:
        return {
            'rule_name': 'SAFETY_STOP',
            'priority': 99,
            'proposed_decision': 'back_off',
            'trigger_values': {'movement_intensity': float(intensity)}
        }
    return None

def rule_cat_gaze(row):
    """Rule: If cat is seen, keep it centered in the frame."""
    if row.get('cat_detected'):
        return {
            'rule_name': 'CAT_GAZE',
            'priority': 5, # Lower priority than Greeting (10) or Safety (99)
            'proposed_decision': 'center_gaze',
            'trigger_values': {
                'cat_x': row.get('cat_position_x')
            }
        }
    return None

def rule_voice_command(row):
    """Rule: Execute voice commands heard in the rolling audio window."""
    text = (row.get('voice_transcription') or '').lower()
    if not text:
        return None
    if 'back' in text and 'off' in text:
        decision = 'back_off'
    elif 'get' in text and 'closer' in text:
        decision = 'get_closer'
    elif 'play' in text and 'arm' in text:
        decision = 'play_arm'
    elif 'speak' in text:                   # <--- NEW: Catch the bark command
        decision = 'play_audio_bark'       # <--- NEW: Set the proposed decision
    else:
        return None
    return {
        'rule_name': 'VOICE_COMMAND',
        'priority': 50,  # Above cat rules (10), below safety stop (99)
        'proposed_decision': decision,
        'trigger_values': {'voice_transcription': text}
    }

# Registry of available rules
RULE_REGISTRY = [
    rule_safety_stop,
    rule_voice_command,
    rule_cat_greeting,
    rule_cat_gaze
]

# Tracks the currently-active voice command so we fire it only on the first
# experience row where it appears, then suppress until the utterance leaves
# the rolling window (i.e. no VOICE_COMMAND fires for a whole chunk).
_active_voice_decision = None  # e.g. "back_off", "play_arm", or None

def reset_voice_dedup():
    """Clear the voice-command deduplication state (call when recording restarts)."""
    global _active_voice_decision
    _active_voice_decision = None

def build_immediate_decisions(experiences_df):
    """Iterates through experiences and fires rules."""
    global _active_voice_decision
    decisions = []

    if experiences_df.empty:
        return pd.DataFrame()

    voice_cmd_seen_this_chunk = False

    for _, row in experiences_df.iterrows():
        # 1. Skip invalid timestamps
        if pd.isna(row['timestamp']):
            continue

        # 2. Run Registry
        potential_decisions = []
        for rule_func in RULE_REGISTRY:
            res = rule_func(row)
            if res:
                potential_decisions.append(res)

        # 3. Conflict Resolution (Highest Priority wins)
        if potential_decisions:
            best_decision = sorted(potential_decisions, key=lambda x: x['priority'], reverse=True)[0]

            # 4. Deduplicate voice commands using a latch:
            #    - First time a voice command fires → emit it, set the latch
            #    - Same decision still firing → suppress (same utterance in window)
            #    - Different voice decision → new utterance, emit it, update latch
            if best_decision['rule_name'] == 'VOICE_COMMAND':
                voice_cmd_seen_this_chunk = True
                proposed = best_decision['proposed_decision']
                if proposed == _active_voice_decision:
                    # Same utterance still in the rolling window — suppress
                    continue
                # New voice command (or first one) — allow and latch
                _active_voice_decision = proposed

            decisions.append({
                'immed_id': str(uuid.uuid4())[:8],
                'timestamp': datetime.now(),
                'experience_id': row['experience_id'],
                'rule_name': best_decision['rule_name'],
                'trigger_values': json.dumps(best_decision['trigger_values']),
                'proposed_decision': best_decision['proposed_decision']
            })

    # If no voice command fired in this entire chunk, the utterance has
    # left the rolling window — clear the latch so the next one can fire.
    if not voice_cmd_seen_this_chunk:
        _active_voice_decision = None

    return pd.DataFrame(decisions)


# --- 2. Action Definitions (Visual Servoing & Parameters) ---

# Visual Constants
FRAME_WIDTH = 640   
FOV_H_DEG = 60.0    
PIXELS_PER_DEG = FRAME_WIDTH / FOV_H_DEG

def def_get_closer(trigger_experience):
    """Goal: Center the cat and make it fill 30% of the frame."""
    target_area_pct = 0.30
    
    # Visual Servoing Logic
    curr_x = trigger_experience.get('cat_position_x')
    
    # Default Move
    est_move_cm = 30.0 
    
    # Calculate Turn (Centering)
    if pd.notna(curr_x):
        error_x_pixels = (FRAME_WIDTH / 2) - curr_x
        turn_deg = error_x_pixels / PIXELS_PER_DEG
    else:
        turn_deg = 0.0 

    return {
        "action": "move_sequence",
        "description": "Visual Servoing: Approach Target",
        "target_area_pct": target_area_pct,
        "parameters": {
            "move_forward_cm": float(est_move_cm),
            "rotate_deg": float(turn_deg)
        }
    }

def def_play_audio_bark(trigger_experience):
    """Goal: Play a pre-saved barking sound to deter or interact."""
    return {
        "action": "play_audio", # Identifies this as a network audio stream, not a motor command
        "description": "Audio: Playing dog bark",
        "parameters": {
            "filename": "dog-barking.wav" # The exact file we want to stream
        }
    }

def def_back_off(trigger_experience):
    """Goal: Retreat (Safety)."""
    return {
        "action": "move_sequence",
        "description": "Safety: Create Distance",
        "target_area_pct": 0.10,
        "parameters": {
            "move_forward_cm": -15.0, # Negative for backward
            "rotate_deg": 0.0
        }
    }

def def_play_arm(trigger_experience):
    """Goal: Wave the arm 4 times (left and right)."""
    # We define a list of angles for the wave sequence
    # Assuming 90 is center, 45 is left, 135 is right
    wave_sequence = [45, 135, 45, 135, 45, 135, 45, 135, 90]
    
    return {
        "action": "arm_sequence",
        "description": "Communication: Playful Waving",
        "parameters": {
            "arm_path": wave_sequence,
            "speed": 1.0,
            "move_forward_cm": 0.0,
            "rotate_deg": 0.0
        }
    }

def def_placeholder(trigger_experience):
    return {"action": "no_motor_action", "parameters": {}}

def def_center_gaze(trigger_experience):
    """Goal: Rotate the robot to center the cat without moving forward."""
    curr_x = trigger_experience.get('cat_position_x')
    
    if pd.notna(curr_x):
        # Calculate pixel error from center (640 / 2)
        error_x_pixels = (FRAME_WIDTH / 2) - curr_x
        # Convert pixel error to rotation degrees
        turn_deg = error_x_pixels / PIXELS_PER_DEG
    else:
        turn_deg = 0.0 

    return {
        "action": "move_sequence",
        "description": "Gaze: Pivoting to center cat",
        "parameters": {
            "move_forward_cm": 0.0, # Do not move forward
            "rotate_deg": float(turn_deg)
        }
    }

# Decision to Action Router
DECISION_DEFINITIONS = {
    'get_closer': def_get_closer,
    'back_off': def_back_off,
    'send_user_text': def_placeholder,
    'send_user_picture': def_placeholder,
    'play_arm': def_play_arm,
    'center_gaze': def_center_gaze,
    'play_audio_bark': def_play_audio_bark  # <--- NEW: Link decision to action
}

def get_action_parameters(decision_type, experience_row):
    func = DECISION_DEFINITIONS.get(decision_type, def_placeholder)
    return func(experience_row)

def build_decisions_to_actions(immediate_decisions_df, experiences_df):
    """Maps high-level decisions to specific action parameters."""
    actions = []
    
    if immediate_decisions_df.empty or experiences_df.empty:
        return pd.DataFrame()

    # Create a lookup for experiences
    exp_lookup = experiences_df.set_index('experience_id').to_dict('index')
    
    for _, decision in immediate_decisions_df.iterrows():
        exp_id = decision['experience_id']
        exp_context = exp_lookup.get(exp_id, {})
        
        decision_type = decision['proposed_decision']
        definition_result = get_action_parameters(decision_type, exp_context)
        
        actions.append({
            'decision_id': len(actions) + 1,
            'timestamp': datetime.now(),
            'source_module': 'mrt_immediate_decisions',
            'source_event_id': decision['immed_id'],
            'experience_id': exp_id,
            'decision_type': decision_type,
            'parameters': json.dumps(definition_result['parameters'])
        })
        
    return pd.DataFrame(actions)