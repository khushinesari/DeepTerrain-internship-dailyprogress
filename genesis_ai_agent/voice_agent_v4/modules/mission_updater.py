"""
mission_updater.py
Updated for session-cache architecture.

Changes:
- Uses cached mission passed from main().
- Sends only PATCH payload.
- Uses backend returned mission.
- Caller should replace mission_cache with returned mission.
"""

import json
from copy import deepcopy

from modules.mission_api import mission_api


class MissionUpdater:

    MUTABLE_FIELDS = {
        "priority",
        "loop_count",
        "role",
    }

    def _find_asset(self, assets, asset_name):
        for asset in assets:
            if asset.get("asset_name", "").lower() == asset_name.lower():
                return asset
        return None

    def update(self, mission: dict, command: dict):

        updated = deepcopy(mission)

        field = command["field"]
        value = command["value"]
        target = command.get("target", "")

        # Asset role update
        if target:

            if field != "role":
                return {
                    "success": False,
                    "message": "Only asset role can be updated."
                }

            asset = self._find_asset(
                updated.get("assigned_assets", []),
                target
            )

            if asset is None:
                return {
                    "success": False,
                    "message": f"Asset '{target}' not found."
                }

            patch_payload = {
                "asset_name": target,
                "role": value
            }

        else:

            if field not in updated:
                return {
                    "success": False,
                    "message": f"Mission field '{field}' not found."
                }

            if field not in self.MUTABLE_FIELDS:
                return {
                    "success": False,
                    "message": f"'{field}' is immutable."
                }

            patch_payload = {
                field: value
            }

        print("=" * 60)
        print("PATCH PAYLOAD")
        print(json.dumps(patch_payload, indent=4))
        print("=" * 60)

        response = mission_api.patch_mission(patch_payload)

        print("=" * 60)
        print("PATCH RESPONSE")
        print(json.dumps(response, indent=4))
        print("=" * 60)

        if not response.get("success", False):
            return response

        data = response.get("data", {})

        if not data.get("success", False):
            return {
                "success": False,
                "message": data.get("message", "PATCH failed.")
            }

        updated_mission = data.get("mission")

        if updated_mission is None:
            return {
                "success": False,
                "message": "Backend did not return updated mission."
            }

        return {
            "success": True,
            "message": data.get(
                "message",
                "Mission updated successfully."
            ),
            "mission": updated_mission
        }


mission_updater = MissionUpdater()