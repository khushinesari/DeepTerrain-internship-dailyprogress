"""
conversation_context.py

Maintains the runtime conversation state.

This module DOES NOT cache missions permanently.
It only stores information relevant to the current
conversation turn.
"""


class ConversationContext:

    #########################################################
    # Constructor
    #########################################################

    def __init__(self):

        self.clear()

    #########################################################
    # Current Mission
    #########################################################

    def set_current_mission(self, mission):

        self.current_mission = mission

    def get_current_mission(self):

        return self.current_mission

    #########################################################
    # Current Asset
    #########################################################

    def set_current_asset(self, asset):

        self.current_asset = asset

    def get_current_asset(self):

        return self.current_asset

    #########################################################
    # Last Transcript
    #########################################################

    def set_last_transcript(self, transcript):

        self.last_transcript = transcript

    def get_last_transcript(self):

        return self.last_transcript

    #########################################################
    # Last Intent
    #########################################################

    def set_last_intent(self, intent):

        self.last_intent = intent

    def get_last_intent(self):

        return self.last_intent

    #########################################################
    # Clear Runtime Context
    #########################################################

    def clear(self):

        self.current_mission = None
        self.current_asset = None
        self.last_transcript = ""
        self.last_intent = ""

    #########################################################
    # Debug
    #########################################################

    def print_state(self):

        print("\n========== Conversation Context ==========")

        if self.current_mission:

            print(
                "Mission :",
                self.current_mission.get(
                    "mission_name",
                    "Unknown"
                )
            )

        else:

            print("Mission : None")

        print("Asset :", self.current_asset)
        print("Intent :", self.last_intent)
        print("Transcript :", self.last_transcript)

        print("==========================================\n")


#########################################################
# Singleton
#########################################################

conversation_context = ConversationContext()