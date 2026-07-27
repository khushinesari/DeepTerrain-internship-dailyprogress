"""
mission_updater.py

Applies updates to the current mission and PATCHes only
the mutable mission fields to the backend.
"""

import json
from copy import deepcopy

from modules.mission_api import mission_api


class MissionUpdater:

    # Only these mission fields are allowed to be PATCHed
    MUTABLE_FIELDS = {
        "priority",
        "loop_count"
    }

    # =========================================================

    def update(self, mission: dict, command: dict):

        updated = deepcopy(mission)

        field = command["field"]
        value = command["value"]
        target = command.get("target", "")

        print("\n" + "=" * 80)
        print("CURRENT MISSION")
        print("=" * 80)
        print(json.dumps(mission, indent=4, ensure_ascii=False))
        print("=" * 80)

        # -----------------------------------------------------
        # Asset updates are currently not supported
        # -----------------------------------------------------

        if target != "":

            return {
                "success": False,
                "message": (
                    "Asset updates are currently disabled. "
                    "Only mission fields can be updated."
                )
            }

        # -----------------------------------------------------
        # Validate field exists
        # -----------------------------------------------------

        if field not in updated:

            return {
                "success": False,
                "message": f"Mission field '{field}' not found."
            }

        # -----------------------------------------------------
        # Validate field is mutable
        # -----------------------------------------------------

        if field not in self.MUTABLE_FIELDS:

            return {
                "success": False,
                "message": (
                    f"'{field}' is immutable and cannot be PATCHed."
                )
            }

        # -----------------------------------------------------
        # Update local mission
        # -----------------------------------------------------

        updated[field] = value

        print("\n" + "=" * 80)
        print("UPDATED MISSION")
        print("=" * 80)
        print(json.dumps(updated, indent=4, ensure_ascii=False))
        print("=" * 80)

        # -----------------------------------------------------
        # Build PATCH payload
        # -----------------------------------------------------

        patch_payload = {
            field: value
        }

        print("\n" + "=" * 80)
        print("PATCH PAYLOAD")
        print("=" * 80)
        print(json.dumps(patch_payload, indent=4, ensure_ascii=False))
        print("=" * 80)

        # -----------------------------------------------------
        # Send PATCH request
        # -----------------------------------------------------

        response = mission_api.patch_mission(patch_payload)

        print("\n" + "=" * 80)
        print("PATCH RESPONSE")
        print("=" * 80)
        print(json.dumps(response, indent=4, ensure_ascii=False))
        print("=" * 80)

        return {
            "success": response.get("success", False),
            "message": response.get(
                "message",
                "Mission updated successfully."
            ),
            "mission": updated
        }


# ============================================================

mission_updater = MissionUpdater()