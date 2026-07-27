"""
prompt_builder.py

Builds the complete prompt for Qwen.

The prompt consists of:
1. System Prompt
2. Current Mission JSON
3. User Command
"""

import json

from config import SYSTEM_PROMPT_FILE


class PromptBuilder:

    def __init__(self):

        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    ############################################################

    def build_prompt(
        self,
        transcript: str,
        current_mission: dict
    ) -> str:

        mission_json = json.dumps(
            current_mission,
            indent=4,
            ensure_ascii=False
        )

        prompt = f"""{self.system_prompt}

============================================================
CURRENT MISSION
============================================================

{mission_json}

============================================================
USER COMMAND
============================================================

{transcript}

============================================================
OUTPUT
============================================================

Return ONLY one valid JSON object.

"""

        return prompt


prompt_builder = PromptBuilder()