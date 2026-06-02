"""
mrt_reflective_decisions — Strategic LLM decision layer.

Runs as a daemon thread alongside the reactive rule engine.
Every ~20 seconds, reviews recent experience and injects
strategic decisions into BRAIN_COMMAND_QUEUE.
"""

import os
import json
import time
import uuid
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from . import decision_engine as d_engine
from . import execution_engine as e_engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_ACTIONS = {'get_closer', 'back_off', 'play_arm', 'no_action'}
REFLECTIVE_LOG_PATH = 'recordings/mrt_reflective_log.jsonl'
DEFAULT_MODEL = 'claude-haiku-4-5-20251001'
DEFAULT_INTERVAL = 20
QUEUE_BUSY_THRESHOLD = 10
DEFAULT_TIMEOUT = 8.0
FPS = 4  # Must match recording_module.FPS
# Experience window: one row per visual frame at 4fps × 20s = 80 rows
EXPERIENCE_WINDOW = DEFAULT_INTERVAL * FPS  # 80 rows ≈ 20s of data

SYSTEM_PROMPT = """\
You are the strategic brain of a cat-sitter robot named Logots. Every 20 seconds \
you review the robot's recent sensory experience and decide whether to take a \
high-level action.

Available actions:
- "get_closer": Approach the cat. Use when cat is stationary, calm, and not stressed.
- "back_off": Retreat to create distance. Use when cat seems agitated, scared, or movement intensity is high.
- "play_arm": Wave the robot's arm playfully. Use when cat is nearby, engaged, and seems relaxed.
- "no_action": Do nothing — let the reactive rule system handle it. This is the default and preferred choice when the situation is already being handled.

Rules:
- Prefer "no_action" unless you see a clear opportunity the reactive rules are missing.
- Never override a safety stop (high movement_intensity).
- If a voice command from the user is present, prioritize it over sensor-based reasoning.
- Do not repeat an action the rule engine is already executing frequently.

Respond ONLY with a JSON object:
{"decision": "<action>", "reasoning": "<1-2 sentence explanation>"}"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_experience_context(mrt_df):
    """Compress recent experience rows into ~4 summary blocks for the LLM."""
    df = mrt_df.copy()
    if df.empty:
        return "No experience data available."

    df['ts'] = pd.to_datetime(df['timestamp'])
    t0 = df['ts'].min()
    df['rel_s'] = (df['ts'] - t0).dt.total_seconds()

    n_blocks = min(4, len(df))
    blocks = [chunk for _, chunk in df.groupby(np.arange(len(df)) // max(1, len(df) // n_blocks))]
    lines = []

    for block in blocks:
        if block.empty:
            continue
        t_start = block['rel_s'].iloc[0]
        t_end = block['rel_s'].iloc[-1]

        cat_count = int(block['cat_detected'].sum()) if 'cat_detected' in block else 0
        n = len(block)
        cat_pct = f"{cat_count}/{n}"

        avg_x = ''
        if cat_count > 0 and 'cat_position_x' in block:
            cx = block.loc[block['cat_detected'] == True, 'cat_position_x']
            if not cx.empty:
                avg_x = f", avg_cat_x={cx.mean():.0f}"

        dist_change = ''
        if 'cat_distance_change' in block:
            mode = block['cat_distance_change'].mode()
            if not mode.empty:
                dist_change = f", distance_trend={mode.iloc[0]}"

        meow_count = int(block['is_cat_voice'].sum()) if 'is_cat_voice' in block else 0
        meow_str = ''
        if meow_count > 0:
            loud = block.get('meow_loudness', pd.Series())
            max_loud = loud.dropna().max() if not loud.empty else ''
            meow_str = f", meows={meow_count}"
            if pd.notna(max_loud) and max_loud:
                meow_str += f" (max_loudness={max_loud})"

        move_int = ''
        if 'movement_intensity' in block:
            mi = block['movement_intensity'].mean()
            move_int = f", movement_intensity={mi:.2f}"

        lines.append(
            f"[t={t_start:.0f}–{t_end:.0f}s] cat_detected={cat_pct}"
            f"{avg_x}{dist_change}{meow_str}{move_int}"
        )

    # Deduplicated voice transcriptions
    if 'voice_transcription' in df.columns:
        vt = df[['voice_transcription', 'rel_s']].copy()
        vt['text_str'] = vt['voice_transcription'].astype(str).str.strip()
        vt = vt[vt['text_str'].str.len() > 0]
        if not vt.empty:
            unique_vt = vt.drop_duplicates(subset='text_str').sort_values('rel_s')
            lines.append("\nVoice transcriptions:")
            for _, row in unique_vt.iterrows():
                lines.append(f"  [t={row['rel_s']:.0f}s] \"{row['text_str']}\"")

    return '\n'.join(lines)


def format_decision_history(decisions_df, n_rows=10):
    """Summarize recent rule engine decisions."""
    df = decisions_df.tail(n_rows)
    if df.empty:
        return "No recent rule engine decisions."

    df = df.copy()
    df['ts'] = pd.to_datetime(df['timestamp'])
    now = df['ts'].max()
    total_span = (now - df['ts'].min()).total_seconds()

    counts = df['proposed_decision'].value_counts()
    lines = ["Recent rule engine decisions:"]
    for action, count in counts.items():
        lines.append(f"- {action} x{count}")
    lines.append(f"(over last {total_span:.0f}s)")

    missing = {'get_closer', 'back_off', 'play_arm'} - set(counts.index)
    if missing:
        lines.append(f"Not attempted: {', '.join(sorted(missing))}")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_reflective_prompt(experience_context, decision_history, voice_text=None):
    """Build the prompt bundle for the Anthropic API call."""
    user_parts = [
        "## Recent Experience\n" + experience_context,
        "\n## Rule Engine Decisions\n" + decision_history,
    ]
    if voice_text:
        user_parts.append(f'\n## Voice Command\nUser said: "{voice_text}"')

    user_parts.append("\nWhat should the robot do?")

    return {
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": '\n'.join(user_parts)}],
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt_bundle, model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT):
    """Call Claude via the Anthropic SDK. Returns parsed dict or None."""
    try:
        import anthropic
    except ImportError:
        print("[Reflective] anthropic SDK not installed. Skipping.")
        return None

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        t0 = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=200,
            system=prompt_bundle["system"],
            messages=prompt_bundle["messages"],
        )
        elapsed_ms = (time.time() - t0) * 1000
        print(f"[Reflective] API call: {elapsed_ms:.0f}ms")
        text = response.content[0].text.strip()

        # Strip markdown fences if the model wrapped the JSON
        if text.startswith('```'):
            lines = text.split('\n')
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith('```')]
            text = '\n'.join(lines).strip()

        if not text:
            print(f"[Reflective] LLM returned empty response. stop_reason={response.stop_reason}")
            return None

        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[Reflective] LLM returned non-JSON: {text[:200]}")
        return None
    except Exception as e:
        print(f"[Reflective] LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def parse_and_validate_response(raw_response):
    """Validate the LLM response. Returns a safe dict or None."""
    if raw_response is None:
        return None

    if isinstance(raw_response, str):
        try:
            raw_response = json.loads(raw_response)
        except json.JSONDecodeError:
            return {'decision': 'no_action', 'reasoning': 'Malformed LLM response.'}

    decision = raw_response.get('decision', 'no_action')
    if decision not in VALID_ACTIONS:
        decision = 'no_action'

    return {
        'decision': decision,
        'reasoning': raw_response.get('reasoning', ''),
    }


# ---------------------------------------------------------------------------
# Action injection
# ---------------------------------------------------------------------------

def inject_reflective_decision(buffer, decision_type, reasoning, experience_id):
    """Translate an LLM decision into motor commands and queue them."""
    from .recording_module import BRAIN_COMMAND_QUEUE

    with buffer.lock:
        last_row = buffer.master_mrt.iloc[-1].to_dict() if not buffer.master_mrt.empty else {}

    definition_result = d_engine.get_action_parameters(decision_type, last_row)

    action_row = {
        'decision_id': 1,
        'timestamp': datetime.now(),
        'source_module': 'mrt_reflective_decisions',
        'source_event_id': f'ref_{uuid.uuid4().hex[:8]}',
        'experience_id': experience_id,
        'decision_type': decision_type,
        'parameters': json.dumps(definition_result['parameters']),
    }

    actions_df = pd.DataFrame([action_row])
    motor_gen = e_engine.build_mrt_motor(actions_df, start_frame_id=900_000)

    with buffer.lock:
        buffer.master_actions = pd.concat([buffer.master_actions, actions_df], ignore_index=True)
        buffer.master_gen_motor = pd.concat([buffer.master_gen_motor, motor_gen], ignore_index=True)
        buffer.cap_masters()

    # Production mode: persist injected rows to disk (RAM mirror is capped above)
    if not getattr(buffer, 'debug', False):
        from .recording_module import ACTION_CSV_FILE, GEN_MOTOR_CSV_FILE
        buffer.append_csv(ACTION_CSV_FILE, actions_df)
        buffer.append_csv(GEN_MOTOR_CSV_FILE, motor_gen)

    # Downsample 10Hz motor frames to robot loop rate (4Hz)
    execution_ratio = int((1.0 / FPS) / 0.1)
    if execution_ratio < 1:
        execution_ratio = 1

    for i in range(0, len(motor_gen), execution_ratio):
        row = motor_gen.iloc[i]
        BRAIN_COMMAND_QUEUE.put((int(row['left_pwm']), int(row['right_pwm']), int(row['arm_angle'])))


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

def _log_to_jsonl(entry):
    try:
        with open(REFLECTIVE_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main thread loop
# ---------------------------------------------------------------------------

def reflective_decision_loop(stop_event, buffer, interval=DEFAULT_INTERVAL):
    """Daemon thread target: periodic LLM-based strategic decisions."""
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("[Reflective] No ANTHROPIC_API_KEY set. Reflective layer disabled.")
        return

    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("[Reflective] anthropic SDK not installed. Reflective layer disabled.")
        return

    from .recording_module import BRAIN_COMMAND_QUEUE

    print(f"[Reflective] Thread started (interval={interval}s).")
    suppression_until = 0.0

    while not stop_event.is_set():
        # Sleep in small increments so stop_event is responsive
        for _ in range(interval * 10):
            if stop_event.is_set():
                return
            time.sleep(0.1)

        # --- Guard: Queue busy ---
        qsize = BRAIN_COMMAND_QUEUE.qsize()
        if qsize > QUEUE_BUSY_THRESHOLD:
            _log_to_jsonl({"timestamp": datetime.now().isoformat(), "skipped": True, "reason": "queue_busy", "queue_size": qsize})
            continue

        # --- Guard: Suppression active ---
        if time.time() < suppression_until:
            _log_to_jsonl({"timestamp": datetime.now().isoformat(), "skipped": True, "reason": "suppressed"})
            continue

        # --- Snapshot buffer state ---
        # Take the full interval window so the LLM sees everything since its last cycle
        with buffer.lock:
            mrt_snapshot = buffer.master_mrt.tail(EXPERIENCE_WINDOW).copy() if not buffer.master_mrt.empty else pd.DataFrame()
            dec_snapshot = buffer.master_decisions.tail(10).copy() if not buffer.master_decisions.empty else pd.DataFrame()

        if mrt_snapshot.empty:
            continue

        # --- Extract best voice transcription (longest = most complete from sliding window) ---
        voice_text = None
        if 'voice_transcription' in mrt_snapshot.columns:
            vt = mrt_snapshot[['voice_transcription']].copy()
            vt['text_str'] = vt['voice_transcription'].astype(str).str.strip()
            vt = vt[vt['text_str'].str.len() > 0]
            if not vt.empty:
                voice_text = str(vt.loc[vt['text_str'].str.len().idxmax(), 'text_str'])

        # --- Build prompt and call LLM ---
        experience_ctx = format_experience_context(mrt_snapshot)
        decision_history = format_decision_history(dec_snapshot)
        prompt_bundle = build_reflective_prompt(experience_ctx, decision_history, voice_text)

        raw_response = call_llm(prompt_bundle)
        decision = parse_and_validate_response(raw_response)

        # --- Log ALL responses (including no_action and failures) ---
        experience_id = mrt_snapshot['experience_id'].iloc[-1] if 'experience_id' in mrt_snapshot.columns else -1
        _log_to_jsonl({
            "timestamp": datetime.now().isoformat(),
            "decision": decision['decision'] if decision else None,
            "reasoning": decision.get('reasoning', '') if decision else '',
            "experience_window_size": len(mrt_snapshot),
            "queue_size_at_time": qsize,
            "voice_text": voice_text,
            "experience_id": str(experience_id),
        })

        # --- Act on decision ---
        if decision is None or decision['decision'] == 'no_action':
            continue

        inject_reflective_decision(buffer, decision['decision'], decision.get('reasoning', ''), experience_id)
        print(f"[Reflective] Injected: {decision['decision']} | {decision.get('reasoning', '')}")
