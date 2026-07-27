"""
response_formatter.py

Generates natural spoken responses.
"""


class ResponseFormatter:

    FIELD_NAMES = {

        "loop_count": "loop count",

        "priority": "priority",

        "status": "status",

        "mission_name": "mission name",

        "mission_type": "mission type",

        "speed": "speed"
    }

    # -------------------------------------------------

    def pretty(self, field):

        return self.FIELD_NAMES.get(
            field,
            field.replace("_", " ")
        )

    # -------------------------------------------------

    def format(self, result, command):

        intent = command["intent"]

        field = command["field"]

        # ====================================================
        # PATCH
        # ====================================================

        if intent == "PATCH":

            pretty = self.pretty(field)

            if result["success"]:

                return (
                    f"{pretty.capitalize()} has been updated "
                    f"to {command['value']}. "
                    f"Mission updated successfully."
                )

            return (
                f"Could not update the {pretty} field. "
                f"Mission update failed."
            )

        # ====================================================
        # GET
        # ====================================================

        if intent == "GET":

            if not result["success"]:

                return "Unable to retrieve the requested information."

            # ------------------------------------------------
            # COMPLETE MISSION SUMMARY
            # ------------------------------------------------

            if field == "mission_details":

                mission = result["value"]

                mission_name = (mission.get("mission_name") or mission.get("name") or "Unknown")

                type = mission.get("type", "Unknown")

                status = mission.get("status", "Unknown")

                priority = mission.get("priority", "Unknown")

                loop_count = mission.get("loop_count", "Unknown")

                assigned_assets = mission.get("assigned_assets", [])

                checkpoint = mission.get("checkpoint", [])

                objectives = mission.get("objectives", [])

                return (
                    f"The current mission is {mission_name}. "
                    f"Mission type is {type}. "
                    f"Current status is {status}. "
                    f"Priority is {priority}. "
                    f"Loop count is {loop_count}. "
                    f"There are {len(assigned_assets)} assigned assets. "
                    f"There are {len(checkpoint)} checkpoints. "
                    f"There are {len(objectives)} objectives."
                )

            # ------------------------------------------------
            # Normal GET
            # ------------------------------------------------

            pretty = self.pretty(field)

            return (
                f"The current {pretty} "
                f"is {result['value']}."
            )

        # ====================================================

        return "Operation completed."

response_formatter=ResponseFormatter()