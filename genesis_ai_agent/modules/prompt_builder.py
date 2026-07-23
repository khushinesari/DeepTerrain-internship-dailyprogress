import json

from config import (
    PROMPT_DIR,
    SCHEMA_DIR,
    PROMPT_FILE,
    DATA_DIR,
)

SYSTEM_PROMPT = PROMPT_DIR / "system_prompt.txt"

MISSION_SCHEMA = SCHEMA_DIR / "mission_schema.json"

COMMAND_SCHEMA = SCHEMA_DIR / "command_schema.json"

MISSION_CURRENT = DATA_DIR / "mission_current.json"

# --------------------------------------------------
# Load System Prompt
# --------------------------------------------------

def load_system_prompt():

    with open(
        SYSTEM_PROMPT,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read().strip()

def load_mission_current():

    with open(
        MISSION_CURRENT,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)
# --------------------------------------------------
# Load Mission Schema
# --------------------------------------------------

def load_mission_schema():

    with open(
        MISSION_SCHEMA,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# --------------------------------------------------
# Load Command Schema
# --------------------------------------------------

def load_command_schema():

    with open(
        COMMAND_SCHEMA,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# --------------------------------------------------
# Prompt Builder
# --------------------------------------------------

def build_prompt(transcript: str):

    system_prompt = load_system_prompt()

    mission_schema = load_mission_schema()

    command_schema = load_command_schema()
    mission_current = load_mission_current()
    prompt = f"""
    {system_prompt}

    ==================================================
    MISSION SCHEMA
    ==================================================

    {json.dumps(mission_schema, indent=4)}

    ==================================================
    COMMAND SCHEMA
    ==================================================

    {json.dumps(command_schema, indent=4)}
    ==================================================
    CURRENT SCHEMA
    ==================================================

    {json.dumps(mission_current, indent=4)}

    ==================================================
    USER TRANSCRIPT
    ==================================================

    {transcript}

    ==================================================
    TASK
    ==================================================

    You are given:

    1. Mission Schema
    2. Command Schema
    3. Operator Transcript

    Your objective is to convert the operator transcript into a structured command.

    Instructions:

    1. Determine whether the operator intent is GET or PATCH.

    2. Follow the Command Schema exactly.

    3. If the intent is GET:

    - Understand the operator request.

    - Use semantic understanding.

    - Find the corresponding field(s) in the Mission Schema.

    - Retrieve the values ONLY from the Current Mission.

    - Populate query.requested_fields.

    - Populate response_data with the values found in Current Mission.

    - Leave mission_update empty.

    - Leave mission_llmoutput empty.

    - Do not invent values..
    4. If the intent is PATCH:
   - Populate "mission_update" with ONLY the fields that change.
   - Populate "mission_llmoutput" with the COMPLETE updated mission.
   - "mission_llmoutput" MUST strictly follow the Mission Schema.
   - Preserve all fields that are not modified.

    5. Do not invent new fields.

6. Do not remove required fields.

7. Use the exact field names from the schemas.

8. Return ONLY a single valid JSON object.

9. Do not include explanations.

10. Do not include Markdown.

11. Do not wrap the JSON inside triple backticks.
"""

    return prompt.strip()


# --------------------------------------------------
# Save Prompt
# --------------------------------------------------

def save_prompt(prompt: str):

    with open(
        PROMPT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(prompt)