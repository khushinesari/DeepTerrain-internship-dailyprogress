"""
mission_api.py

All communication with the backend REST API.

No other module should use requests directly.
"""

import requests

from config import(
    API_BASE_URL,
    GET_ALL_MISSIONS,
    GET_SINGLE_MISSION,
    PATCH_SINGLE_MISSION,
    API_HEADERS,
    API_TIMEOUT
)


class MissionAPI:

    def __init__(self):

        self.base_url = API_BASE_URL

    ###########################################################
    # GET ALL MISSIONS
    ###########################################################

    def get_all_missions(self):

        url = self.base_url + GET_ALL_MISSIONS

        response = requests.get(
            url,
            headers=API_HEADERS,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    ###########################################################
    # GET SINGLE MISSION
    ###########################################################

    def get_mission(self, mission_id):

        endpoint = GET_SINGLE_MISSION.format(
            mission_id=mission_id
        )

        url = self.base_url + endpoint

        response = requests.get(
            url,
            headers=API_HEADERS,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    ###########################################################
    # PATCH MISSION
    ###########################################################

    def patch_mission(
        self,
        mission_id,
        mission_json
    ):

        endpoint = PATCH_SINGLE_MISSION.format(
            mission_id=mission_id
        )

        url = self.base_url + endpoint

        response = requests.patch(
            url,
            json=mission_json,
            headers=API_HEADERS,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()

        return response.json()
# ==========================================================
# Singleton
# ==========================================================

mission_api = MissionAPI()
if __name__ == "__main__":

        api = MissionAPI()

        try:

            missions = api.get_all_missions()

            print("\nConnected Successfully\n")

            print(type(missions))

            if isinstance(missions, list):

                print(f"Number of missions : {len(missions)}")

            else:

                print(missions)

        except Exception as e:

            print(e)