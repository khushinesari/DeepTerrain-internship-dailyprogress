"""
=========================================================
mission_cache.py
=========================================================

Session-level mission cache.

Responsibilities
----------------
1. Store mission after initial GET.
2. Serve mission for all GET requests.
3. Replace mission after every PATCH.
4. Clear cache on exit.
5. Optional timeout support.
=========================================================
"""

import time
from copy import deepcopy


class MissionCache:

    def __init__(self, timeout=None):
        """
        timeout : seconds
        None -> cache never expires during session
        """

        self._mission = None
        self._timestamp = None
        self.timeout = timeout

    # ==========================================================
    # Initialize Cache
    # ==========================================================

    def initialize(self, mission: dict):

        self._mission = deepcopy(mission)
        self._timestamp = time.time()

    # ==========================================================
    # Get Mission
    # ==========================================================

    def get(self):

        if self._mission is None:
            raise RuntimeError(
                "Mission cache is empty. "
                "Initialize the cache before using it."
            )

        return deepcopy(self._mission)

    # ==========================================================
    # Replace Mission
    # ==========================================================

    def replace(self, mission: dict):

        self._mission = deepcopy(mission)
        self._timestamp = time.time()

    # ==========================================================
    # Clear Cache
    # ==========================================================

    def clear(self):

        self._mission = None
        self._timestamp = None

    # ==========================================================
    # Cache Exists?
    # ==========================================================

    def is_initialized(self):

        return self._mission is not None

    # ==========================================================
    # Cache Expired?
    # ==========================================================

    def is_expired(self):

        if self.timeout is None:
            return False

        if self._timestamp is None:
            return True

        return (time.time() - self._timestamp) > self.timeout

    # ==========================================================
    # Cache Age
    # ==========================================================

    def age(self):

        if self._timestamp is None:
            return None

        return time.time() - self._timestamp

    # ==========================================================
    # Pretty Print
    # ==========================================================

    def info(self):

        print("\n" + "=" * 80)
        print("MISSION CACHE")
        print("=" * 80)

        if self._mission is None:
            print("EMPTY")
        else:
            print(f"Mission : {self._mission.get('mission_name', 'Unknown')}")
            print(f"Cached  : {self.age():.2f} sec ago")

        print("=" * 80)


# ==========================================================
# Singleton
# ==========================================================

mission_cache = MissionCache(timeout=None)