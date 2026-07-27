"""
mission_resolver.py

Resolves GET requests from the structured command
returned by Qwen.

Responsibilities
----------------
1. Read mission-level fields.
2. Read asset-level fields.
3. Return structured response.
"""

from copy import deepcopy


class MissionResolver:

    #########################################################

    def resolve(self, mission: dict, command: dict):

        field = command["field"]
        target = command.get("target", "")

        # --------------------------------------------------
        # Mission-level GET
        # --------------------------------------------------

        if target == "":

            if field not in mission:

                return {
                    "success": False,
                    "message": f"Mission field '{field}' not found."
                }

            return {
                "success": True,
                "field": field,
                "value": deepcopy(mission[field])
            }

        # --------------------------------------------------
        # Asset-level GET
        # --------------------------------------------------

        asset = self._find_asset(
            mission,
            target
        )

        if asset is None:

            return {
                "success": False,
                "message": f"Asset '{target}' not found."
            }

        if field not in asset:

            return {
                "success": False,
                "message": f"Field '{field}' not found in asset '{target}'."
            }

        return {
            "success": True,
            "target": target,
            "field": field,
            "value": deepcopy(asset[field])
        }

    #########################################################

    def _find_asset(self, node, target):

        if not isinstance(target, str):
            return None

        target = target.lower()

        if isinstance(node, dict):

            name = node.get("name")

            if isinstance(name, str):

                if name.lower() == target:
                    return node

            for value in node.values():

                result = self._find_asset(
                    value,
                    target
                )

                if result is not None:
                    return result

        elif isinstance(node, list):

            for item in node:

                result = self._find_asset(
                    item,
                    target
                )

                if result is not None:
                    return result

        return None


mission_resolver = MissionResolver()