"""
mission_api.py

Handles communication with the Mission Backend.

Responsibilities
----------------
1. Fetch the latest mission.
2. Fetch a specific mission.
3. Patch a mission.
4. Hide backend response format from the rest of the application.
"""

import requests

from config import (
    API_BASE_URL,
    GET_ALL_MISSIONS,
    GET_SINGLE_MISSION,
    PATCH_SINGLE_MISSION,
    API_TIMEOUT,
    API_HEADERS,
)


class MissionAPI:

    #########################################################
    # Constructor
    #########################################################

    def __init__(self):

        self.base_url = API_BASE_URL.rstrip("/")

    #########################################################
    # Internal Request Helper
    #########################################################

    def _request(self, method, endpoint, payload=None):

        if endpoint.startswith("http"):
            url = endpoint
        else:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:

            response = requests.request(
                method=method.upper(),
                url=url,
                headers=API_HEADERS,
                json=payload,
                timeout=API_TIMEOUT
            )

            response.raise_for_status()

            return {
                "success": True,
                "data": response.json()
            }

        except requests.exceptions.RequestException as e:

            return {
                "success": False,
                "message": str(e)
            }

        except ValueError:

            return {
                "success": False,
                "message": "Backend returned invalid JSON."
            }

    #########################################################
    # Get Current Mission
    #########################################################

    def get_current_mission(self):

        """
        Returns ONLY the mission dictionary.

        Raises ValueError if the backend response
        does not contain a mission.
        """

        response = self._request(
            "GET",
            GET_ALL_MISSIONS
        )

        if not response["success"]:
            return response

        data = response["data"]

        if "mission" not in data:

            return {
                "success": False,
                "message": "Mission field missing in backend response."
            }

        return {
            "success": True,
            "mission": data["mission"]
        }

    #########################################################
    # Get Single Mission
    #########################################################

    def get_mission(self, mission_id):

        endpoint = GET_SINGLE_MISSION.format(
            mission_id=mission_id
        )

        response = self._request(
            "GET",
            endpoint
        )

        if not response["success"]:
            return response

        return {
            "success": True,
            "mission": response["data"]
        }

    #########################################################
    # Patch Mission
    #########################################################

    def patch_mission(self, updated_mission):

        response = self._request(
            "PATCH",
            PATCH_SINGLE_MISSION,
            payload=updated_mission
        )

        return response

    #########################################################
    # Connectivity Test
    #########################################################

    def ping(self):

        response = self.get_current_mission()

        return response["success"]


#########################################################
# Singleton
#########################################################

mission_api = MissionAPI()


#########################################################
# Standalone Test
#########################################################

if __name__ == "__main__":

    response = mission_api.get_current_mission()

    if response["success"]:

        print(response["mission"])

    else:

        print(response["message"])