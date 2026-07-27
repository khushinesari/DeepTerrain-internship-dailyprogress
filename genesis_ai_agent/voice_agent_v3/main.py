"""
=========================================================
Genesis Voice Agent V2
=========================================================

Pipeline

Voice
    ↓
Whisper
    ↓
MissionAPI
    ↓
PromptBuilder
    ↓
QwenEngine
    ↓
MissionUpdater / MissionResolver
    ↓
ResponseFormatter
    ↓
Piper
=========================================================
"""

from modules.recorder import recorder
from modules.whisper_engine import whisper_engine
from modules.mission_api import mission_api
from modules.prompt_builder import prompt_builder
from modules.qwen_engine import qwen_engine
from modules.mission_updater import mission_updater
from modules.mission_resolver import mission_resolver
from modules.response_formatter import response_formatter
from modules.tts_engine import tts
from config import INPUT_AUDIO


def main():

    print("=" * 70)
    print("Genesis Voice Agent V2")
    print("=" * 70)

    while True:

        try:

            # -------------------------------------------------
            # Record Audio
            # -------------------------------------------------

            print("\nListening...")

            audio_path = recorder.record(INPUT_AUDIO)
            # -------------------------------------------------
            # Speech → Text
            # -------------------------------------------------

            transcript = whisper_engine.transcribe(audio_path)

            if not transcript.strip():

                print("No speech detected.")

                continue

            print(f"\nUser : {transcript}")

            # -------------------------------------------------
            # Fetch Latest Mission
            # -------------------------------------------------
            import json

            mission_response = mission_api.get_current_mission()

            if not mission_response["success"]:

                print(mission_response["message"])

                continue

            current_mission = mission_response["mission"]

            print("\n" + "=" * 80)
            print("CURRENT MISSION LOADED FROM BACKEND")
            print("=" * 80)

            print(
                json.dumps(
                current_mission,
                indent=4,
                ensure_ascii=False)
            )

            print("=" * 80 + "\n")
            

            # -------------------------------------------------
            # Build Prompt
            # -------------------------------------------------

            prompt = prompt_builder.build_prompt(
                transcript=transcript,
                current_mission=current_mission
            )

            # -------------------------------------------------
            # Qwen Inference
            # -------------------------------------------------

            command, latency = qwen_engine.infer(prompt)

            print("\nLLM Command")
            print(command)

            print(f"\nInference Time : {latency} sec")

            # -------------------------------------------------
            # Execute Intent
            # -------------------------------------------------

            intent = command["intent"].upper()

            if intent == "PATCH":

                result = mission_updater.update(
                    current_mission,
                    command
                )

            elif intent == "GET":

                result = mission_resolver.resolve(
                    current_mission,
                    command
                )

            else:

                result = {
                    "success": False,
                    "message": f"Unsupported intent '{intent}'."
                }

            # -------------------------------------------------
            # Format Response
            # -------------------------------------------------

            reply = response_formatter.format(result)

            print("\nGenesis :", reply)

            # -------------------------------------------------
            # Speak
            # -------------------------------------------------

            tts.speak(reply)

        except KeyboardInterrupt:

            print("\nExiting Genesis.")

            break

        except Exception as e:

            print("\nERROR")

            print(e)


if __name__ == "__main__":

    main()