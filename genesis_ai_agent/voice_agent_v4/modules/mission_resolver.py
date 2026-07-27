"""
mission_resolver.py

Handles all GET operations.
"""


class MissionResolver:

    def resolve(self, mission: dict, command: dict):

        field = command["field"]

        target = command.get("target", "")

        # -------------------------------------------------
        # Complete Mission Summary
        # -------------------------------------------------

        if field == "mission_details":

            return {
                "success": True,
                "field": "mission_details",
                "value": mission
            }

        # -------------------------------------------------
        # Mission-level field
        # -------------------------------------------------

        if target == "":

            if field not in mission:

                return {
                    "success": False,
                    "message": f"{field} not found."
                }

            return {
                "success": True,
                "field": field,
                "value": mission[field]
            }

        # -------------------------------------------------
        # Asset lookup
        # -------------------------------------------------

        asset = self._find_asset(mission, target)

        if asset is None:

            return {
                "success": False,
                "message": f"Asset '{target}' not found."
            }

        if field not in asset:

            return {
                "success": False,
                "message": f"'{field}' not found in '{target}'."
            }

        return {
            "success": True,
            "field": field,
            "target": target,
            "value": asset[field]
        }

    # -------------------------------------------------

    def _find_asset(self, node, target):

        if not isinstance(target, str):
            return None

        target = target.lower()

        if isinstance(node, dict):

            if isinstance(node.get("name"), str):

                if node["name"].lower() == target:
                    return node

            for value in node.values():

                result = self._find_asset(value, target)

                if result is not None:
                    return result

        elif isinstance(node, list):

            for item in node:

                result = self._find_asset(item, target)

                if result is not None:
                    return result

        return None


mission_resolver = MissionResolver()