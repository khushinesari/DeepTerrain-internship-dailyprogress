# Genesis Voice Agent V2

## Overview

Genesis Voice Agent V2 is a fully local AI-powered voice assistant for mission management.

Unlike Version 1, the LLM is **not responsible for retrieving or maintaining mission state**.

Instead,

- Python communicates with the backend APIs.
- Python selects the appropriate mission.
- The selected mission is provided to the LLM.
- The LLM performs reasoning and generates structured mission updates.
- Python validates the update and patches it back to the backend.
- The response is converted into human-readable speech.

This significantly improves reliability, scalability, and maintainability.

---

# Technology Stack

Speech-to-Text
- Faster Whisper

LLM
- Qwen2.5-3B-Instruct

Text-to-Speech
- Piper

Backend
- REST APIs

Programming Language
- Python

---

# High-Level Pipeline

```text
                 User Voice
                     │
             Recorder Module
                     │
            Faster Whisper STT
                     │
              User Transcript
                     │
          Conversation Context
                     │
        Mission Cache / Selector
                     │
         Selected Mission JSON
                     │
            Prompt Builder
                     │
         System Prompt
         Mission Schema
         Command Schema
         Selected Mission
         Transcript
                     │
                 Qwen LLM
                     │
            Structured JSON
                     │
                  Parser
                     │
          ┌──────────┴──────────┐
          │                     │
         GET                  PATCH
          │                     │
 Mission Resolver       Mission Updater
          │                     │
 Response Formatter     PATCH API
          │                     │
             Piper Voice Output
```

---

# System Architecture

```
modules/

recorder.py

whisper_engine.py

conversation_context.py

mission_api.py

mission_cache.py

mission_selector.py

prompt_builder.py

qwen_engine.py

parser.py

mission_resolver.py

mission_updater.py

response_formatter.py

tts_engine.py
```

---

# Module Responsibilities

## Recorder

Records microphone audio.

Output

```
audio.wav
```

---

## Whisper Engine

Converts speech into text.

Example

Input

```
Where is Vehicle 1?
```

Output

```
Where is Vehicle 1?
```

---

## Conversation Context

Maintains conversational memory.

Stores

- current mission
- last referenced asset
- last intent

This allows

```
Move Vehicle 1

↓

Increase its speed

↓

Where is it now?
```

without repeating the mission name.

---

## Mission Cache

Responsible for maintaining all available missions.

Startup

```
GET /missions
```

stores

```
Mission A

Mission B

Mission C

...
```

The cache may also be refreshed on demand.

---

## Mission Selector

Receives

```
Transcript

Mission Cache
```

Finds

```
Which mission contains Vehicle 1?
```

Returns

```
Mission ID

Mission JSON
```

Only this mission is forwarded to the LLM.

---

## Prompt Builder

Builds the prompt.

Prompt contains

- System Prompt
- Mission Schema
- Command Schema
- Selected Mission JSON
- User Transcript

The LLM never receives every mission.

---

## Qwen

Performs reasoning.

Determines

GET

PATCH

Requested fields

Mission updates

---

## Parser

Converts LLM response into Python objects.

Example

```json
{
    "intent":"PATCH",
    "mission_update":{
        "priority":"LOW"
    }
}
```

---

## Mission Resolver

Used only for GET.

Extracts requested information from the selected mission.

Example

Operator

```
Where is Vehicle 1?
```

Response Data

```json
{
    "latitude":12.97,
    "longitude":77.59
}
```

---

## Mission Updater

Used only for PATCH.

Pipeline

```
Mission JSON

+

mission_update

↓

Validation

↓

Mutable Field Check

↓

Apply Update

↓

PATCH API
```

Immutable fields are never modified.

---

## Response Formatter

Converts structured data into natural language.

Example

JSON

```json
{
    "latitude":12.9715,
    "longitude":77.5946
}
```

Voice Output

```
Vehicle 1 is currently located at latitude
12.9715 and longitude 77.5946.
```

Another example

JSON

```json
{
    "priority":"LOW"
}
```

Voice

```
Mission priority has been updated to LOW.
```

---

## Piper

Converts the formatted response into speech.

---

# Startup Flow

```
Application Starts

↓

GET /missions

↓

Mission Cache Created

↓

Conversation Context Reset

↓

Ready
```

---

# GET Workflow

```
User Voice

↓

Whisper

↓

Transcript

↓

Mission Selector

↓

Selected Mission

↓

Prompt Builder

↓

Qwen

↓

Parser

↓

Mission Resolver

↓

Response Formatter

↓

Piper
```

---

# PATCH Workflow

```
User Voice

↓

Whisper

↓

Transcript

↓

Mission Selector

↓

Selected Mission

↓

Prompt Builder

↓

Qwen

↓

Parser

↓

Mission Updater

↓

PATCH API

↓

Response Formatter

↓

Piper
```

---

# Show All Missions Workflow

Operator

```
Show all missions
```

Pipeline

```
Mission Cache

↓

List Missions

↓

Formatter

↓

Piper
```

Voice Output

```
There are three active missions.

Mission Alpha

Mission Bravo

Mission Charlie
```

The operator may then say

```
Open Mission Bravo
```

which sets

```
Current Mission Context = Mission Bravo
```

Subsequent commands operate on this mission unless the user changes the context.

---

# Immutable Fields

Examples

- mission_id
- mission_name
- asset_id
- asset_name
- asset_type
- creation_time

These fields cannot be modified.

---

# Mutable Fields

Examples

- priority
- objectives
- loop_count
- asset_position
- convoy_position
- status
- route
- speed

Only these fields are eligible for updates.

---

# Advantages of Version 2

- Only one mission is provided to the LLM.
- The LLM focuses on language understanding rather than mission discovery.
- Mission retrieval and updates remain deterministic in Python.
- Follow-up commands are supported through conversation context.
- API communication is centralized and easy to maintain.
- Immutable fields are protected by Python validation.
- Natural-language responses are generated independently of the raw JSON returned by the backend.

---

# Future Enhancements

- Automatic mission cache refresh.
- Multi-operator support.
- Authentication and role-based access control.
- Streaming Whisper and streaming Piper.
- Retrieval-augmented mission documentation.
- Multi-agent coordination across multiple active missions.
