"""
response_formatter.py

Converts structured mission execution results into
natural language responses suitable for TTS.
"""


class ResponseFormatter:

    #########################################################

    def format(self, result: dict) -> str:

        """
        Convert MissionUpdater/MissionResolver output
        into a spoken response.
        """

        if not result.get("success", False):

            return self._format_error(result)

        # ----------------------------------------
        # GET Response
        # ----------------------------------------

        if "value" in result:

            return self._format_get(result)

        # ----------------------------------------
        # PATCH Response
        # ----------------------------------------

        return self._format_patch(result)

    #########################################################

    def _format_get(self, result):

        field = result["field"]
        value = result["value"]

        target = result.get("target", "")

        if target:

            return (
                f"The {field} of {target} is {value}."
            )

        return (
            f"The {field} is {value}."
        )

    #########################################################

    def _format_patch(self, result):

        message = result.get("message", "")

        if message:

            return message

        return "Mission updated successfully."

    #########################################################

    def _format_error(self, result):

        return result.get(
            "message",
            "Sorry, I couldn't complete your request."
        )


#########################################################

response_formatter = ResponseFormatter()