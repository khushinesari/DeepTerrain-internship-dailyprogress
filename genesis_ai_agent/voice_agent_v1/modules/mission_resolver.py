import json

from config import MISSION_CURRENT


class MissionResolver:

    def __init__(self):

        self.mission = self.load_mission()

    # -------------------------------------------------
    # Load Current Mission
    # -------------------------------------------------

    def load_mission(self):

        with open(
            MISSION_CURRENT,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    # -------------------------------------------------
    # Resolve GET fields
    # -------------------------------------------------

    def resolve(self, parsed):

        if parsed.get("intent", "").upper() != "GET":
            return parsed

        requested = parsed.get("query", {}).get(
            "requested_fields",
            []
        )

        response = {}

        for field in requested:

            value = self.get_value(field)

            response[field] = value

        parsed["response_data"] = response

        return parsed

    # -------------------------------------------------
    # Traverse JSON
    # -------------------------------------------------

    def get_value(self, field_path):

        keys = field_path.split(".")

        current = self.mission

        for key in keys:

            if isinstance(current, dict):

                current = current.get(key)

            elif isinstance(current, list):

                values = []

                for item in current:

                    value = self.walk(item, keys[keys.index(key):])

                    values.append(value)

                return values

            else:

                return None

        return current

    # -------------------------------------------------
    # Recursive list traversal
    # -------------------------------------------------

    def walk(self, obj, remaining_keys):

        current = obj

        for key in remaining_keys:

            if isinstance(current, dict):

                current = current.get(key)

            else:

                return None

        return current