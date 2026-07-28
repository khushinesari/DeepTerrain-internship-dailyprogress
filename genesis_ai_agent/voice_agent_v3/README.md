## output : mission details being read out speciffically loop count and priority , mission name .
# Genesis Voice Agent V3

## Overview

Genesis Voice Agent V3 is an offline AI-powered voice assistant designed for mission management. This version focuses on retrieving and updating mission-level information using natural voice commands. It integrates local speech recognition, a locally hosted Large Language Model (LLM), backend mission APIs, and offline text-to-speech to provide a complete voice-controlled mission management experience.

Unlike later versions, V3 is limited to mission-level operations and does not support querying or modifying assets, checkpoints, or objectives.

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

The assistant records the user's speech through the system microphone.

**Module**

```
recorder.py
```

Output

```
audio/input.wav
```

---

## Step 2 – Speech Recognition

The recorded audio is transcribed locally using Faster Whisper.

**Speech-to-Text Model**

```
Faster Whisper
```

Example

User says

> What is the current mission priority?

Whisper returns

```
What is the current mission priority?
```

---

## Step 3 – Mission Retrieval

Before processing any request, Genesis retrieves the latest mission from the backend.

```
MissionAPI.get_mission()
```

The retrieved mission is treated as the single source of truth.

Example

```json
{
    "mission_name": "Mission-001",
    "mission_type": "Surveillance",
    "status": "Active",
    "priority": "Medium",
    "loop_count": 2
}
```

---

## Step 4 – Prompt Construction

The Prompt Builder combines

- Current mission JSON
- User command
- System prompt

into a single prompt for the LLM.

Module

```
prompt_builder.py
```

---

## Step 5 – LLM Reasoning

Genesis uses a locally hosted

```
Qwen2.5-3B-Instruct
```

to convert natural language into structured JSON commands.

Example

User

```
What is the mission priority?
```

LLM Output

```json
{
    "intent": "GET",
    "field": "priority",
    "target": ""
}
```

Another example

User

```
Set priority to High
```

LLM Output

```json
{
    "intent": "PATCH",
    "field": "priority",
    "value": "High",
    "target": ""
}
```

---

# Supported Intents

Genesis V3 supports two operations.

## GET

Retrieves mission-level information.

Supported fields

- Mission Name
- Mission Type
- Mission Status
- Priority
- Loop Count
- Mission Details

---

## PATCH

Updates mutable mission fields.

Supported fields

- Priority
- Loop Count

---

# Mission Resolver

```
mission_resolver.py
```

The Mission Resolver processes GET requests by extracting information from the mission JSON.

Supported mission fields

- mission_name
- mission_type
- status
- priority
- loop_count

---

## Mission Details

When the user requests the current mission details, Genesis returns the mission name.

Example

User

```
Give the current mission details.
```

Response

```
The current mission is Mission-001.
```

At this stage, the assistant does **not** summarize the mission type, mission status, priority, loop count, assets, checkpoints, or objectives as part of the mission details response. Those capabilities were introduced in later versions.
## Priority

User

```
What is the mission priority?
```

Response

```
The current mission priority is Medium.
```

---

## Loop Count

User

```
What is the loop count?
```

Response

```
The current loop count is 2.
```

---

## Mission Name

User

```
What is the mission name?
```

Response

```
The current mission name is Mission-001.
```

---

## Mission Status

User

```
What is the current mission status?
```

Response

```
The current mission status is Active.
```

---

## Mission Type

User

```
What type of mission is this?
```

Response

```
This is a Surveillance mission.
```

---

# Mission Updater

```
mission_updater.py
```

MissionUpdater processes all PATCH requests.

Supported mutable fields

```
priority
loop_count
```

The updater performs the following steps:

- Validates the requested field
- Checks whether the field is mutable
- Updates the local mission copy
- Builds the PATCH payload
- Sends the PATCH request to the backend
- Returns the updated mission

---

## Example

User

```
Set priority to High.
```

PATCH Payload

```json
{
    "priority": "High"
}
```

---

Another example

User

```
Set loop count to 5.
```

PATCH Payload

```json
{
    "loop_count": 5
}
```

---

# Response Formatter

```
response_formatter.py
```

The Response Formatter converts backend responses into conversational speech suitable for Piper TTS.

Examples

Instead of

```
priority = High
```

Genesis says

```
The current mission priority is High.
```

Instead of

```
loop_count = 3
```

Genesis says

```
The current loop count is 3.
```

Mission update example

```
Mission priority has been updated successfully.
```

---

# Piper Text-to-Speech

The formatted response is synthesized into speech using Piper.

This enables Genesis to provide fully offline voice feedback.

---

# Project Structure

```
Genesis_V3
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

# Current Features

- Offline speech recognition using Faster Whisper
- Local LLM reasoning using Qwen2.5-3B-Instruct
- Structured JSON command generation
- Mission retrieval from backend
- Mission name retrieval
- Mission details (mission name only)
- Mission status retrieval
- Priority retrieval
- Loop count retrieval
- Priority update
- Loop count update
- Offline speech synthesis using Piper

---

# Limitations

Genesis Voice Agent V3 is limited to mission-level operations.

The following features are **not** available in this version:

- Asset information retrieval
- Asset updates
- Checkpoint retrieval
- Checkpoint updates
- Objective retrieval
- Objective updates
- Asset location queries
- Asset role queries
- Multi-asset reasoning
- Performance matrices 

These capabilities were introduced in later versions of the Genesis Voice Agent.

---

# Future Work

Planned enhancements for subsequent versions include:

- Asset information retrieval : name , role, type 
- Checkpoint summaries: name , order 
- Objective summaries: name , description 
- Asset role updates
- Multi-asset mission management
- Continuous voice interaction
- Mutable fields enhancement: loop_count, priority, asset_role 
