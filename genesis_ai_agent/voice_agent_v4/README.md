
# Genesis Voice Agent V4

## Overview

Genesis Voice Agent V2 is an offline AI-powered voice assistant designed for mission management. It allows a user to interact with a mission backend using natural voice commands. The assistant records speech, converts it to text, retrieves mission data, uses a local Large Language Model (LLM) to interpret the command, performs the requested operation, and responds with natural speech.

The entire pipeline runs locally using Whisper, Qwen2.5, and Piper TTS without relying on cloud-based LLM APIs.

---

# Architecture

```
                ┌────────────────────┐
                │   User Speaks      │
                └─────────┬──────────┘
                          │
                          ▼
                  Audio Recorder
                          │
                          ▼
             Faster Whisper (Speech → Text)
                          │
                          ▼
                 Mission API (GET Mission)
                          │
                          ▼
                  Prompt Builder
                          │
                          ▼
             Qwen2.5-3B-Instruct LLM
                          │
                          ▼
                Structured JSON Command
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
 Mission Resolver                  Mission Updater
(GET Requests)                  (PATCH Requests)
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 Response Formatter
                          │
                          ▼
                    Piper Text-to-Speech
                          │
                          ▼
                  Voice Response
```

---

# Pipeline

## Step 1 – Voice Recording

The assistant records the user's speech using the local microphone.

**Module**

```
recorder.py
```

Output:

```
audio/input.wav
```

---

## Step 2 – Speech Recognition

The recorded audio is transcribed using Faster Whisper.

**Model**

```
Whisper
```

Example

User says

> What is the current mission status?

Whisper returns

```
What is the current mission status?
```

---

## Step 3 – Retrieve Mission

Before reasoning, the assistant always requests the latest mission from the backend.

```
MissionAPI.get_mission()
```

The backend mission JSON is treated as the single source of truth.

Example mission

```json
{
    "mission_name": "Mission-001",
    "mission_type": "Surveillance",
    "status": "Active",
    "priority": "High",
    "loop_count": 2,
    "assigned_assets": [],
    "checkpoint": [],
    "objectives": []
}
```

---

## Step 4 – Prompt Builder

Prompt Builder combines

- Current mission JSON
- User transcript
- System prompt

into a single prompt for the LLM.

```
prompt_builder.py
```

---

## Step 5 – LLM Reasoning

The prompt is passed to

```
Qwen2.5-3B-Instruct
```

The model never answers in natural language.

Instead it generates structured JSON.

Example

```
User:
What is the mission priority?
```

LLM Output

```json
{
    "intent":"GET",
    "field":"priority",
    "target":""
}
```

Another example

```
User:
Set priority to High
```

Output

```json
{
    "intent":"PATCH",
    "field":"priority",
    "value":"High",
    "target":""
}
```

---

# Supported Intents

The assistant currently supports two intents.

## GET

Retrieves information from the mission.

Examples

- Mission details
- Priority
- Status
- Loop count
- Mission type
- Assets
- Objectives
- Checkpoints

---

## PATCH

Updates mutable mission fields.

Currently supported

- priority
- loop_count

---

# Mission Resolver

```
mission_resolver.py
```

Processes GET requests.

It resolves

- Mission fields
- Mission details
- Assigned assets
- Objectives
- Checkpoints
- Asset-specific queries
- Recursive asset lookup

---

## Supported GET Examples

### Mission Details

User

```
Give current mission details
```

Response

```
The current mission is Mission testing in site 4. It is a surveillance mission and is currently pending. The mission priority is medium with a loop count of 4. There is 1 assigned asset, 5 checkpoints, and 2 objectives.
```

---

### Assigned Assets

User

```
Give asset details
```

Response

```
The current assigned asset is Ranger-001,
which is an AGV playing the role of support.
```
### Objectives

User

```
Give objectives
```

Response

```
There are currently 2 objectives. Secure checkpoint A, which Secure and monitor checkpoint A. Scan perimeter, which Scan the surrounding perimeter for threats.
```

---

### Checkpoints

User

```
Give checkpoints
```

Response

```
There are 3 checkpoints.

Checkpoint A
Checkpoint B
Checkpoint C
```

---

# Mission Updater

```
mission_updater.py
```

Processes PATCH requests.

Current mutable fields

```
priority
loop_count
```

The updater

- validates fields
- updates local mission copy
- builds PATCH payload
- sends PATCH request
- returns updated mission

---

Example

User

```
Set priority to High
```

PATCH Payload

```json
{
    "priority":"High"
}
```

---

# Response Formatter

```
response_formatter.py
```

Converts resolver/updater outputs into conversational speech suitable for Piper TTS.

Instead of reading raw JSON, Genesis generates natural language.

Example

Instead of

```
priority = High
```

Genesis says

```
The current mission priority is High.
```

---

# Piper Text-to-Speech

Natural language responses are converted to speech using Piper.

The assistant finally speaks the generated response.

---

# Current Project Structure

```
Genesis_V2
│
├── audio/
│
├── models/
│
├── modules/
│   ├── mission_api.py
│   ├── mission_resolver.py
│   ├── mission_updater.py
│   ├── prompt_builder.py
│   ├── qwen_engine.py
│   ├── response_formatter.py
│
├── recorder.py
├── main.py
└── system_prompt.txt
```

---

# Performance Metrics

The pipeline records execution time for

- Recording
- Whisper transcription
- Mission retrieval
- Prompt generation
- LLM inference
- Mission processing
- Response formatting
- Piper TTS
- Total pipeline time

Qwen inference statistics include

- Input tokens
- Output tokens
- Total tokens
- Context usage
- Inference latency
- Tokens per second

---

# Current Capabilities

✔ Offline voice assistant

✔ Local Whisper speech recognition

✔ Local Qwen2.5 reasoning

✔ Structured JSON command generation

✔ Mission retrieval

✔ Mission updates

✔ Asset information retrieval

✔ Objective retrieval

✔ Checkpoint retrieval

✔ Mission overview

✔ Piper speech synthesis

✔ Execution timing

✔ LLM statistics

---

# Limitations

At the current stage, Genesis supports updates only for mission-level mutable fields:

- Priority
- Loop Count

Asset role updates and other asset modifications are under development and are intentionally excluded from the current implementation.

---

# Future Enhancements

Planned improvements include

- Asset role updates
- Asset assignment
- Asset position updates
- Mission creation and multiple mission handling 
- Objective description editing (optional)
- Multi-asset support
- Context-aware and multi command conversations 
- Streaming LLM responses and check for Qwen 2.5 replacement with better model
- Prompt enhancement 
- Voice interruption handling
- Naming the agent and checking for response (example : Hey, siri ....)
- caching the details of first get method api call for future use
---
# Execution 
```
https://github.com/khushinesari/DeepTerrain-internship-dailyprogress.git

cd DeepTerrain-internship-dailyprogress/genesis_ai_agent/voice_agent_v4

```
## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/khushinesari/DeepTerrain-internship-dailyprogress.git
```

### 2. Navigate to the project

```bash
cd DeepTerrain-internship-dailyprogress/genesis_ai_agent/voice_agent_v4
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure `config.py`

Update the API endpoints and verify the Piper executable and voice model paths.

### 5. Run the application

```bash
python main.py
```

