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
        "speed": "speed",
        "role": "role"
    }

    def pretty(self, field):
        return self.FIELD_NAMES.get(field, field.replace("_", " "))

    def format(self, result, command):

        intent = command["intent"]
        field = command["field"]

        if intent == "PATCH":
            if field == "role" and command.get("target"):
                if result["success"]:
                    return (f"The role of {command['target']} has been updated "
                        f"to {command['value']} successfully.")
                return(f"I couldn't update the role of {command['target']}.")
            pretty = self.pretty(field)

            if result["success"]:
                return (
                    f"{pretty.capitalize()} has been updated to "
                    f"{command['value']}. Mission updated successfully."
                )

            return (
                f"Could not update the {pretty} field. "
                f"Mission update failed."
            )

        if intent != "GET":
            return "Operation completed."

        if not result["success"]:
            return "Unable to retrieve the requested information."

        # Mission summary
        if field == "mission_details":

            mission = result["value"]
            mission_name = mission.get("mission_name", "Unknown")
            type = mission.get("type", "Unknown")
            status = mission.get("status", "Unknown")
            priority = mission.get("priority", "Unknown")
            loop_count = mission.get("loop_count", "Unknown")

            asset_count = len(mission.get("assigned_assets", []))
            checkpoint_count = len(mission.get("checkpoint", []))
            objective_count = len(mission.get("objectives", []))

            return (
                f"The current mission is {mission_name}. "
                f"It is a {type} mission and is currently {status}. "
                f"The mission priority is {priority} with a loop count of {loop_count}. "
                f"There {'is' if asset_count == 1 else 'are'} {asset_count} assigned "
                f"{'asset' if asset_count == 1 else 'assets'}, "
                f"{checkpoint_count} checkpoint{'s' if checkpoint_count != 1 else ''}, "
                f"and {objective_count} objective{'s' if objective_count != 1 else ''}."
            )
        # Assigned assets
        if field.startswith("assigned_assets"):

            assets = result["value"]

            if not assets:
                return "There are currently no assigned assets."

            speech = [
                f"There {'is' if len(assets)==1 else 'are'} currently "
                f"{len(assets)} assigned "
                f"{'asset' if len(assets)==1 else 'assets'}."
            ]
            asset = assets[0]
            for i, asset in enumerate(assets,1):
                speech.append(
                    f"Asset {i} is {asset.get('asset_name','Unknown')}, "
                    f"a {asset.get('asset_type','Unknown')} type asset "
                    f"assigned the role of {asset.get('role','Unknown')}."
                )

            return " ".join(speech)
        if field == "asset_position":

            position = result["value"]

            coords = position.get("coordinates", [])

            if len(coords) == 2:
                return (
                f"{command['target']} is currently located at "
                f"{coords[0]}, {coords[1]}."
                )

            return f"The location of {command['target']} is unavailable."
        # Checkpoints
        if field == "checkpoint":

            checkpoints = result["value"]

            if not checkpoints:
                return "There are currently no checkpoints."

            speech = [
                f"There {'is' if len(checkpoints)==1 else 'are'} currently "
                f"{len(checkpoints)} checkpoint"
                f"{'' if len(checkpoints)==1 else 's'}."
            ]

            for cp in checkpoints:
                name = cp.get("name") or "Unnamed checkpoint"
                order = cp.get("order","Unknown")
                speech.append(
                    f"{name} is checkpoint number {order}."
                )

            return " ".join(speech)

        # Objectives
        if field == "objectives":

            objectives = result["value"]

            if not objectives:
                return "There are currently no objectives."

            speech = [
                f"There {'is' if len(objectives)==1 else 'are'} currently "
                f"{len(objectives)} objective"
                f"{'' if len(objectives)==1 else 's'}."
            ]

            for obj in objectives:
                speech.append(
                    f"{obj.get('objectiveName','Unknown')}, "
                    f"which {obj.get('description','has no description')}."
                )

            return " ".join(speech)

        pretty = self.pretty(field)

        return f"The current {pretty} is {result['value']}."


response_formatter = ResponseFormatter()
