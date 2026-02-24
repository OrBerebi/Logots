# Plan: `mrt_reflective_decisions`

---

## 1. Context

Products like Petcube and Furbo already do a decent job at remote monitoring and basic laser/treat interaction. What they all share is a fundamental limitation: they are **reactive**. They respond to a trigger and execute a fixed response. There is no memory of what happened 30 seconds ago, no understanding of a pattern building over time, and no judgment about whether the current response is actually the right one.

Our architecture already goes further than this — we have a multi-modal experience layer (`mrt_experience`) that fuses vision, audio, and motion into a continuous, timestamped picture of what the robot is perceiving. The reactive rule engine (`decision_engine.py`) reads from this and fires well-defined responses. That layer is solid and we're keeping it.

What we want to add is a layer **above** the rules — one that reads the same experience data but asks a different question: not *"what is happening right now?"* but *"what should we do about the situation we've been in for the last minute?"* This is the job of `mrt_reflective_decisions`: an LLM that sits on top of our pipeline, reviews recent experience periodically, and injects strategic decisions that the rule engine alone would never make.

---

## 2. Input, Output, and Connected Components

### Data Flow Position

```
mrt_experience_data (buffer.master_mrt)
        │
        ├──► decision_engine.py           ← EXISTING (reactive, every ~3s)
        │         └──► mrt_immediate_decisions
        │                   └──► mrt_decisions_to_actions ──► BRAIN_COMMAND_QUEUE
        │
        ├──► voice_command_module.py       ← EXISTING (keyword match, ~every 2s)
        │         └──► mrt_decisions_to_actions ──► BRAIN_COMMAND_QUEUE
        │
        └──► mrt_reflective_decisions      ← NEW (strategic, every ~20s)
                  └──► mrt_decisions_to_actions ──► BRAIN_COMMAND_QUEUE
```

### Inputs

| Input | Source | Why |
|---|---|---|
| `buffer.master_mrt` (last ~60 rows) | `ThreadSafeBuffer` | The robot's experience window — what it has seen, heard, and felt in the last ~60 seconds |
| `buffer.master_decisions` (last ~20 rows) | `ThreadSafeBuffer` | What the rule engine has already been deciding — prevents the reflective layer from redundantly repeating the same action |
| `buffer.audio_voice` (Whisper transcription) | `voice_command_module.py` | Raw transcription text, passed to the LLM for semantic interpretation instead of keyword matching |
| `BRAIN_COMMAND_QUEUE.qsize()` | `recording_module.py` | How full the action queue is — the LLM shouldn't add to an already-busy queue |

### Output

A single structured decision row, written in the exact same schema as `mrt_decisions_to_actions`, then injected into `BRAIN_COMMAND_QUEUE`:

```python
{
    'decision_id': <int>,
    'timestamp': <datetime>,
    'source_module': 'mrt_reflective_decisions',   # ← new source identifier
    'source_event_id': 'ref_<id>',
    'experience_id': <last experience_id in window>,
    'decision_type': <str>,       # e.g. 'get_closer', 'back_off', 'play_arm', 'no_action'
    'parameters': <json str>,
    'trigger_values': <json str>, # includes LLM reasoning text
    'reasoning': <str>            # LLM's explanation — logged only, not sent to robot
}
```

### Why These Components

**Reads from `mrt_experience`**, not raw sensor data, because the experience layer has already done the hard work — it fuses all modalities and produces a clean, human-readable row per frame. This is exactly the right abstraction to feed into a language model.

**Writes to `mrt_decisions_to_actions`**, because the execution engine and motor cortex already know how to handle that schema. The reflective layer doesn't need its own output path — it speaks the same language as the rest of the pipeline. `source_module = 'mrt_reflective_decisions'` is enough to trace it in the logs.

**Feeds `BRAIN_COMMAND_QUEUE` directly**, following the same pattern as voice commands today. The motor cortex (`stream_motors`) already arbitrates between manual, brain, and idle states — nothing needs to change there.

---

## 3. What the Component Does

### Operating Mode: Active Background Thread

`mrt_reflective_decisions` runs as a dedicated daemon thread, started alongside `pipeline_consumer` and `voice_command_consumer` in `run_data_collection()`. It runs on a timer, independently of the chunk pipeline.

It must **not** run inside `process_chunk()`. The LLM call takes 0.5–2 seconds, which would stall the 4fps pipeline. It lives on its own thread and reads from the shared `ThreadSafeBuffer` without blocking anything.

### Frequency: Every 20 Seconds (Adaptive)

A 20-second interval is the right default:

- The reactive rule engine fires every ~3 seconds (12 frames @ 4fps) — it handles real-time reflexes.
- The reflective layer looks at patterns. Patterns need time to emerge — 20 seconds of experience gives ~80 frames, ~3 rule-engine cycles, and a meaningful behavioral window.
- An API call every 20 seconds costs roughly $0.002–0.01 per call with Claude Haiku — negligible for hours of operation.

**Adaptive suppression**: If `BRAIN_COMMAND_QUEUE` already has more than 10 items pending (the robot is mid-action), the reflective layer skips its cycle and waits for the next one. It doesn't interrupt a sequence already in motion.

### LLM: Claude Haiku via Anthropic API

**Model: `claude-haiku-4-5-20251001`**

- **Latency**: ~0.5s response time, fits comfortably in a 20-second cycle
- **Cost**: ~10x cheaper than Sonnet, runs indefinitely without meaningful expense
- **Capability**: Reading a structured context block and returning a JSON decision is well within Haiku's abilities — this is a classification + judgment task, not deep reasoning
- **Future path**: if we later move to a Jetson Orin NX, we can swap the API call for a local model (`llama.cpp` + Mistral-7B or similar) and the rest of the architecture stays identical

The prompt uses **structured JSON output** — the system prompt instructs the model to respond only with a JSON object. This makes parsing deterministic.

### Voice Command Integration: Replace Keyword Matching

The current `voice_command_consumer` thread runs Whisper and then does string matching for "back off", "get closer", "play arm". It works but is brittle — anything outside those exact phrases is ignored.

The reflective layer absorbs this responsibility:

1. Whisper still runs on its existing cadence
2. Instead of keyword matching, it writes the transcribed text to a shared slot: `buffer.last_voice_text`
3. On every reflective cycle, the LLM receives the experience context **and** the last voice transcription (if any, within the last 30 seconds)
4. The LLM can now understand natural language in context: *"Leave her be for now"* → `no_action` + suppression flag for 2 minutes; *"Can you go check on her?"* → `get_closer`

The existing keyword-matching path stays as a **fast fallback** for clear commands when the reflective layer isn't mid-cycle. The two don't conflict.

---

## 4. Core Functions

### `format_experience_context(mrt_df, n_rows=40) → str`

Takes the last N rows of `buffer.master_mrt` and converts them into a compact, LLM-readable text block. Extracts only the fields that matter for decision-making:

```
[t=0s] cat_detected=True, cat_position_x=287, cat_distance_change=closer,
       is_cat_voice=False, movement_intensity=0.42, is_rest=False
[t=3s] cat_detected=True, cat_position_x=310, cat_distance_change=no_change,
       is_cat_voice=True, meow_loudness=medium, movement_intensity=0.18
...
```

Timestamps are expressed as relative seconds from the oldest row (not absolute datetimes), which is more meaningful to the model.

---

### `format_decision_history(decisions_df, n_rows=10) → str`

Summarizes what the rule engine has been doing recently:

```
Recent decisions (rule engine):
- center_gaze × 6 (last 18s)
- center_gaze × 4 (last 6s)
No get_closer or play_arm attempts in window.
```

This prevents the LLM from proposing an action the rule engine has already been handling, and gives it context about what hasn't been tried yet.

---

### `build_reflective_prompt(experience_context, decision_history, voice_text=None) → list[dict]`

Constructs the full message list for the Claude API call. The system prompt is fixed and defines:
- The robot's purpose and personality (friendly, non-threatening cat companion)
- The available action space with plain-language descriptions
- The output format (strict JSON)
- Priority rules (e.g., never override a safety stop)

The user message is dynamic and assembled from the formatted context blocks.

Example system prompt excerpt:

```
You are the strategic brain of a cat-sitter robot. Every 20 seconds you review
the robot's recent experience and decide whether to take a high-level action.

Available actions:
- "get_closer": Approach the cat (use when cat is stationary and not stressed)
- "back_off": Create distance (use when cat seems agitated or scared)
- "play_arm": Wave the arm (use when cat seems bored and engaged)
- "no_action": Do nothing and let the reactive system handle it (default)

Respond only with a JSON object:
{"decision": "<action>", "reasoning": "<1-2 sentence explanation>"}
```

---

### `call_llm(messages, model="claude-haiku-4-5-20251001", timeout=8.0) → dict | None`

Wraps the Anthropic API call with a hard timeout and error handling. Returns the parsed JSON dict or `None` on failure. Never raises — a failed LLM call means the reflective layer silently skips this cycle, and the reactive rules continue uninterrupted.

---

### `parse_and_validate_response(raw_response) → dict | None`

Parses the JSON from the LLM response and validates that `decision` is one of the known action types. Falls back to `no_action` if the response is malformed or unrecognized.

This is the safety gate. The LLM cannot hallucinate an unknown action into the motor cortex.

---

### `inject_reflective_decision(buffer, decision_type, reasoning, experience_id) → None`

Translates the LLM's decision into:
1. An action row (same schema as `mrt_decisions_to_actions`) — logged to `buffer.master_actions`
2. Motor commands via `e_engine.build_mrt_motor()` — put into `BRAIN_COMMAND_QUEUE`

Mirrors exactly what `voice_command_consumer` already does. Reuses `d_engine.get_action_parameters()` so action definitions stay in one place.

---

### `reflective_decision_loop(stop_event, buffer, interval=20)` — The Thread Target

The main loop. Pseudocode:

```python
while not stop_event.is_set():
    time.sleep(interval)

    # Skip if robot is already busy
    if BRAIN_COMMAND_QUEUE.qsize() > 10:
        continue

    # Skip if suppression is active (user said "leave it alone")
    if suppression_active and time.time() < suppression_until:
        continue

    # Snapshot shared state (brief lock)
    with buffer.lock:
        mrt_snapshot = buffer.master_mrt.tail(40).copy()
        dec_snapshot = buffer.master_decisions.tail(10).copy()
        voice_text = buffer.last_voice_text
        buffer.last_voice_text = None  # consume it

    if mrt_snapshot.empty:
        continue

    experience_ctx = format_experience_context(mrt_snapshot)
    decision_history = format_decision_history(dec_snapshot)
    messages = build_reflective_prompt(experience_ctx, decision_history, voice_text)

    response = call_llm(messages)
    if response is None:
        continue

    decision = parse_and_validate_response(response)
    if decision is None or decision['decision'] == 'no_action':
        continue

    inject_reflective_decision(buffer, decision['decision'], decision['reasoning'], ...)
```

---

## 5. Additional Considerations

### Suppression Flag

The loop maintains a `suppression_until` timestamp. If the LLM or a voice command implies the user wants the robot to hold off, the loop sets `suppression_until = time.time() + 120` and skips the next ~6 cycles. This prevents the LLM from overriding a deliberate human decision.

### Reasoning Log

Every LLM response — including `no_action` ones — is written to `recordings/mrt_reflective_log.jsonl`:

```json
{
  "timestamp": "...",
  "decision": "get_closer",
  "reasoning": "Cat has been stationary for 40s and meowed twice at medium loudness. Rule engine has only been centering gaze. Approaching is appropriate.",
  "experience_window_size": 40,
  "queue_size_at_time": 2
}
```

This log is not just for debugging — it's a record of the robot's reasoning over time, and a future dataset for fine-tuning.

### API Key Management

The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable at init time. If it's missing, the module logs a warning and the robot continues on rule-based decisions only.

### Graceful Degradation

If LLM calls fail consistently (network down, API outage), the reflective layer logs the error and keeps sleeping. The robot never stops because of a missing LLM response — the reactive rule engine is always the fallback.

### Thread Registration

One addition to `run_data_collection()` in `recording_module.py`:

```python
t_reflect = threading.Thread(
    target=mrd.reflective_decision_loop,
    args=(stop_event, stream_buffer),
    daemon=True
)
threads.append(t_reflect)
```

### Traceability

Every action row written by this component carries `source_module = 'mrt_reflective_decisions'`. We can always filter `mrt_decisions_to_actions.csv` to see exactly what the LLM decided vs. the rule engine — side by side, timestamped, with reasoning attached.

---

## Summary: Three-Tier Decision Architecture

| Layer | Component | Fires | Latency | Intelligence |
|---|---|---|---|---|
| Reactive | `decision_engine.py` | Every ~3s | <10ms | Rules |
| Voice | `voice_command_module.py` | On speech | ~0.5s | Keyword → LLM (semantic) |
| Reflective | `mrt_reflective_decisions.py` | Every ~20s | ~1s | LLM (strategic) |
