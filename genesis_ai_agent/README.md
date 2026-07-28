# AIM
### Building an AI agent add on to the existing genesis control system where the manual entry for filling the form is replaced by the agent which works on voice commands 
## Architecture 
```
Microphone
      │
      ▼
Open Source ASR
(Whisper / Parakeet)
      │
      ▼
     Text
      │
      ▼
     LLM
(Qwen3 / GLM-4 / Llama3)
      │
      ▼
Intent Classification
      +
  Form Filling
      +
Prompt Creation
      │
      ▼
    Form
      │
      ▼
Control System API
```
---
## Best Open Source Speech-to-Text Models
1. Whisper Large-v3 Turbo
2. NVIDIA Parakeet
3. Canary-multilingual
## LLM for Intent Understanding
1. Qwen3
2. GLM-4
3. DeepSeek-R1
## Prompting 
- Example system prompt:
```
You are an industrial control assistant.

Your job is to classify every command.

Possible intents:

1. FETCH
2. POST

If FETCH:

Return

{
 "intent":"FETCH",
 "target":"",
 "filters":{},
 "form_updates":{}
}

If POST:

Return

{
 "intent":"POST",
 "target":"",
 "updates":{},
 "form_updates":{}
}

Never answer in natural language.

Return valid JSON only.
```
## Handling the Form
Rather than hard-coding mappings, create a Form Context describing the schema. The LLM receives:

- The user's command
- The operation (FETCH or POST)
- The current form template/schema 
The LLM's task is to fill only the relevant fields and preserve everything else. This makes the agent adaptable when the form schema changes.
## Intent classification
The agent asks:
```Is the user trying to retrieve information or change something?```
The AI agent needs to determine which operation should be executed.
Natural language can express the same intent in many different ways.
- Examples:```Intent = FETCH/GET```
```
Show mission priority.

Get current checkpoints.

Display convoy position.

What is the loop count?

Where is UAV_01?

Show all objectives.
```
- Examples:```Intent = POST/PATCH```
```
Increase priority.

Move UAV_01.

Change loop count to 5.

Update objective 2.

Modify convoy position.

```
## Entity Extraction
Finds the important information from the text 
- Supoose the speech-to-text output is
```Increase Motor M12 speed to 250 RPM.```
- Entity classification
```
{
    "device":"Motor",
    "id":"M12",
    "parameter":"speed",
    "value":"250 RPM"
}
```
## Form Filling
- Taking the extracted information from the user's command and inserting it into the correct fields of the control system's expected data structure.
- Suppose the operator says:
```
Increase Vehicle 3 speed to 40 km/hr
```
- suppose the control system requires this JSON:
```
{
    "screen":"MotorControl",
    "deviceId":"",
    "parameter":"",
    "newValue":"",
    "priority":"",
    "timestamp":""
}
```
- Form filling produces:
```
{
    "screen":"MotorControl",
    "deviceId":"3",
    "parameter":"speed",
    "newValue":40,
    "priority":"Normal",
    "timestamp":"2026-07-09T10:30:00"
}
```
- Temporary JSON Structure
```
{
  "mission": {
    "mission_name": "Mission_001",
    "mission_type": "Reconnaissance",
    "priority": "HIGH",
    "loop_count": 3,

    "assigned_assets": [
      {
        "asset_name": "UAV_01",
        "asset_type": "Drone",

        "asset_position": {
          "latitude": 12.9716,
          "longitude": 77.5946,
          "altitude": 120
        },

        "convoy_position": {
          "latitude": 12.9732,
          "longitude": 77.5981,
          "altitude": 0
        }
      },

      {
        "asset_name": "UGV_02",
        "asset_type": "GroundVehicle",

        "asset_position": {
          "latitude": 12.9720,
          "longitude": 77.5960,
          "altitude": 0
        },

        "convoy_position": {
          "latitude": 12.9732,
          "longitude": 77.5981,
          "altitude": 0
        }
      }
    ],

    "checkpoints": [
      {
        "id": 1,
        "latitude": 12.9751,
        "longitude": 77.6025
      },
      {
        "id": 2,
        "latitude": 12.9780,
        "longitude": 77.6062
      }
    ],

    "objectives": [
      {
        "id": 1,
        "description": "Monitor Warehouse 001",
        "status": "Pending"
      }
    ]
  }
}
```
- The user never mentioned priority or timestamp, but the agent fills them using defaults or system-generated values.

- This completed form is what gets sent to the control system.
## Where does the form come from?
1. Method 1: Fixed Template
- Each operation has a predefined template.
- The agent selects the correct template and fills in the values.
2. Method 2: Dynamic Template
- The LLM receives the current form and updates only the relevant fields.
- This is ideal when the form structure evolves over time.
## Prompt Creation
- The LLM should not receive only the user's sentence.

- Instead, it should receive:

  - The user's command,
  - The detected intent,
  - The extracted entities,
  - The current form or schema,
  - clear instructions on what output is expected.
The prompt acts as the LLM's specification.
- Suppose the user says:
 ``` Open Valve V12.```
- Without context, the model doesn't know:

  - Which form should be updated?
  - Which fields correspond to the valve state?
  - Should it return JSON, XML, or plain text?
  - Should existing values be preserved?
  - Are there mandatory fields that need defaults?
- The prompt supplies all of this context.
### NOTE :
Prompt creation and form filling are closely linked. Prompt creation is the instruction-building stage: it tells the LLM exactly what task to perform, provides the current form, the extracted information, and the constraints. Form filling is the result of that process: the LLM returns a completed version of the black-box form with the appropriate fields populated.
## Improvement 
Rather than asking the LLM to directly output the final black-box form, consider a two-stage architecture:
1. Structured command generation: The LLM produces a standardized intermediate representation
2. Form adapter: A deterministic Python module maps this intermediate representation to the current black-box form required by the control system.
- Changes to the control system's form schema require updates only in the adapter, not in the LLM prompts.
- Reduces the chance of malformed outputs reaching the control system.
