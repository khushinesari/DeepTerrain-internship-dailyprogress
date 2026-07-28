import json

from config import (
    RAW_RESPONSE_FILE,
    MISSION_LLM_OUTPUT
)


class ResponseParser:

    def clean_response(self, response: str):

        response = response.strip()

        response = response.replace("```json", "")
        response = response.replace("```", "")

        return response.strip()

    def extract_json(self, response: str):

        start = response.find("{")

        if start == -1:
            raise ValueError("No JSON found.")

        braces = 0

        end = -1

        for i in range(start, len(response)):

            if response[i] == "{":
                braces += 1

            elif response[i] == "}":

                braces -= 1

                if braces == 0:
                    end = i + 1
                    break

        if end == -1:
            raise ValueError("Invalid JSON.")

        return response[start:end]

    def parse(self, response):

        cleaned = self.clean_response(response)

        print("\n----- Cleaned Response -----\n")
        print(cleaned)
        print("\n----------------------------\n")

        try:
            parsed = json.loads(cleaned)

        except json.JSONDecodeError as e:

            print("\nJSON ERROR")
            print(e)

            raise

        self.save_response(parsed)

        self.save_mission_output(parsed)

        return parsed

    def save_response(self, parsed):

        with open(
            RAW_RESPONSE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                parsed,
                f,
                indent=4
            )

    def save_mission_output(self, parsed):

        if "mission_llmoutput" not in parsed:
            return

        with open(
            MISSION_LLM_OUTPUT,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                parsed["mission_llmoutput"],
                f,
                indent=4
            )