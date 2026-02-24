import pandas as pd
import numpy as np

from . import transformation_mart_pipeline as t_mart

def get_stt_model():
    return t_mart.get_whisper_model()

def listen_and_propose_decision(audio_buffer_list):
    if not audio_buffer_list: return pd.DataFrame()
    samples = [x[1] for x in audio_buffer_list]
    audio_input = np.array(samples).astype(np.float32) / 32768.0
    model = get_stt_model()
    result = model.transcribe(audio_input, fp16=False)
    text = result['text'].lower().strip()

    # NEW DEBUG LINE: See what the robot actually 'hears' in your terminal
    if text:
        print(f"DEBUG: AI Transcribed: '{text}'")

    decision = None
    # Use "in" for partial matching to catch "Back off!" or "Please back off"
    if "back" in text and "off" in text: 
        decision = "back_off"
    elif "get" in text and "closer" in text: 
        decision = "get_closer"
    elif "play" in text and "arm" in text: 
        decision = "play_arm"

    if decision:
        return pd.DataFrame([{
            'decision_type': decision, # back_off, get_closer, or play_arm
            'text_heard': text,
            'source_module': 'mrt_communicate_user'
        }])
    return pd.DataFrame()