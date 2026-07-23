import json


class ResponseFormatter:

    def format(self, parsed: dict) -> str:
        """
        Convert parsed LLM response into operator-friendly speech.
        """

        intent = parsed.get("intent", "").upper()

        if intent == "GET":
            return self._format_get(parsed)

        elif intent == "PATCH":
            return self._format_patch(parsed)

        return "I could not understand the command."

    # --------------------------------------------------
    # GET Formatting
    # --------------------------------------------------

    def _format_get(self, parsed):

        data = parsed.get("response_data", {})

        if not data:
            return "No information found."

        sentences = []

        for field, value in data.items():

            name = field.split(".")[-1].replace("_", " ").title()

            if isinstance(value, list):

                sentences.append(f"{name}:")

                for item in value:

                    sentences.append(str(item))

            elif isinstance(value, dict):

                sentences.append(f"{name} is {value}")

            else:

                sentences.append(f"{name} is {value}.")

        return " ".join(sentences)

    # --------------------------------------------------
    # PATCH Formatting
    # --------------------------------------------------

    def _format_patch(self, parsed: dict) -> str:

        updates = parsed.get("mission_update", {})

        if not updates:
            return "No mission parameters were updated."

        messages = []

        self._flatten_updates(updates, "", messages)

        return "Mission updated successfully. " + " ".join(messages)

    # --------------------------------------------------
    # Recursive formatter
    # --------------------------------------------------

    def _flatten_updates(self, obj, prefix, messages):

        if isinstance(obj, dict):

            for key, value in obj.items():

                new_prefix = f"{prefix}.{key}" if prefix else key

                self._flatten_updates(value, new_prefix, messages)

        elif isinstance(obj, list):

            value = ", ".join(map(str, obj))
            field = prefix.replace("_", " ").replace(".", " ")
            messages.append(f"{field} updated to {value}.")

        else:

            field = prefix.replace("_", " ").replace(".", " ")
            messages.append(f"{field} updated to {obj}.")