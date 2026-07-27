"""
parser.py

Parses the JSON returned by the LLM into a structured object.

Expected LLM output

GET
{
    "intent":"GET",
    "field":"battery",
    "target":"Rigenjur"
}

PATCH
{
    "intent":"PATCH",
    "field":"priority",
    "target":"",
    "value":"high"
}
"""

import json
from dataclasses import dataclass


@dataclass
class ParsedCommand:

    intent: str = ""

    field: str = ""

    target: str = ""

    value: object = None

    raw: dict = None


class Parser:

    #########################################################
    # Parse LLM Response
    #########################################################

    def parse(self, response):

        if response is None:

            return ParsedCommand()

        # ---------------------------------------------
        # Remove Markdown Code Blocks
        # ---------------------------------------------

        response = response.strip()

        if response.startswith("```"):

            lines = response.splitlines()

            lines = [
                line
                for line in lines
                if not line.startswith("```")
            ]

            response = "\n".join(lines)

        # ---------------------------------------------
        # Decode JSON
        # ---------------------------------------------

        try:

            data = json.loads(response)

        except Exception:

            print("\nParser Error")
            print("---------------------")
            print(response)

            return ParsedCommand()

        # ---------------------------------------------
        # Build Parsed Command
        # ---------------------------------------------

        cmd = ParsedCommand()

        cmd.intent = str(
            data.get("intent", "")
        ).upper()

        cmd.field = data.get("field", "")

        cmd.target = data.get("target", "")

        cmd.value = data.get("value", None)

        cmd.raw = data

        return cmd


#########################################################
# Singleton
#########################################################

parser = Parser()