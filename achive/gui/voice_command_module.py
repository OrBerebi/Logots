import pandas as pd
import numpy as np
import ast
from datetime import datetime

# --- GLOBALS ---
_whisper_model = None

def get_stt_model():
    """Initializes OpenAI Whisper on Mac MPS."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        # Verify we have the correct OpenAI library
        if not hasattr(whisper, 'load_model'):
            raise AttributeError("Wrong 'whisper' library installed. Run: pip install openai-whisper")
        
        # Load the smallest model for the lowest latency in Berlin's local environment
        _whisper_model = whisper.load_model("tiny", device="mps")
    return _whisper_model

def listen_and_propose_decision(csv_path: str):
    """
    Standalone task to bridge raw audio to high-level decisions.
    """
    # 1. Extract raw samples from stg_audio_data.csv
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()
    
    # Grab the last 2 seconds of audio to detect recent commands
    recent = df.tail(8) 
    samples = []
    for val in recent['audio_samples']:
        # stg_audio_data stores samples as ARRAYS
        samples.extend(ast.literal_eval(val) if isinstance(val, str) else val)
    
    if not samples:
        return pd.DataFrame()

    # 2. Normalize for the model (-1.0 to 1.0)
    audio_input = np.array(samples).astype(np.float32) / 32768.0

    # 3. Transcribe via OpenAI Whisper
    model = get_stt_model()
    result = model.transcribe(audio_input, fp16=False)
    text = result['text'].lower().strip()

    # 4. Keyword Matching for established decisions
    decision = None
    if "back off" in text:
        decision = "back_off"
    elif "get closer" in text:
        decision = "get_closer"
    elif "play arm" in text:
        decision = "play_arm"

    # 5. Output directly to the Decision schema
    if decision:
        return pd.DataFrame([{
            'decision_id': int(datetime.now().timestamp()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'source_module': 'mrt_voice_command',
            'source_event_id': 0,
            'experience_id': 0, # Bypass experience sync
            'decision_type': 'play_arm' if decision == 'play_arm' else 'move',
            'parameters': str({'intent': decision})
        }])
    
    return pd.DataFrame()