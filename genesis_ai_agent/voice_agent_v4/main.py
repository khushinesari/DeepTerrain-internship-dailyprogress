"""
Updated main.py for Genesis Voice Agent V5.1
Includes complete pipeline statistics.
"""

import time

from modules.recorder import recorder
from modules.whisper_engine import whisper_engine
from modules.mission_api import mission_api
from modules.mission_cache import mission_cache
from modules.prompt_builder import prompt_builder
from modules.qwen_engine import qwen_engine
from modules.mission_updater import mission_updater
from modules.mission_resolver import mission_resolver
from modules.response_formatter import response_formatter
from modules.tts_engine import tts
from config import INPUT_AUDIO


def main():

    print("=" * 70)
    print("Genesis Voice Agent V5.1")
    print("=" * 70)

    # ---------------------- Startup ---------------------- #

    startup_start = time.perf_counter()

    response = mission_api.get_current_mission()

    startup_time = time.perf_counter() - startup_start

    if not response["success"]:
        raise RuntimeError(response["message"])

    mission_cache.initialize(response["mission"])

    print(f"Mission loaded successfully.")
    print(f"Startup Time : {startup_time:.3f} sec")
    print("=" * 70)

    interaction = 1

    try:

        while True:

            print(f"\nInteraction #{interaction}")

            total_start = time.perf_counter()

            # -------------------------------------------------
            # Recording
            # -------------------------------------------------

            t0 = time.perf_counter()

            audio_path = recorder.record(INPUT_AUDIO)

            record_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Whisper
            # -------------------------------------------------

            t0 = time.perf_counter()

            transcript = whisper_engine.transcribe(audio_path)

            whisper_time = time.perf_counter() - t0

            if not transcript.strip():
                print("No speech detected.")
                continue

            # -------------------------------------------------
            # Cache Read
            # -------------------------------------------------

            t0 = time.perf_counter()

            current_mission = mission_cache.get()

            cache_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Prompt Builder
            # -------------------------------------------------

            t0 = time.perf_counter()

            prompt = prompt_builder.build_prompt(
                transcript=transcript,
                current_mission=current_mission
            )

            prompt_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Qwen
            # -------------------------------------------------

            t0 = time.perf_counter()

            command, _ = qwen_engine.infer(prompt)

            qwen_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Mission Processing
            # -------------------------------------------------

            t0 = time.perf_counter()

            if command["intent"].upper() == "GET":

                result = mission_resolver.resolve(
                    current_mission,
                    command
                )

            else:

                result = mission_updater.update(
                    current_mission,
                    command
                )

                if result.get("success") and result.get("mission"):
                    mission_cache.replace(result["mission"])

            mission_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Response Formatter
            # -------------------------------------------------

            t0 = time.perf_counter()

            reply = response_formatter.format(result, command)

            formatter_time = time.perf_counter() - t0

            print("\nAssistant:", reply)

            # -------------------------------------------------
            # Piper TTS
            # -------------------------------------------------

            t0 = time.perf_counter()

            tts.speak(reply)

            tts_time = time.perf_counter() - t0

            total_time = time.perf_counter() - total_start

            # -------------------------------------------------
            # Pipeline Statistics
            # -------------------------------------------------

            print("\n" + "=" * 70)
            print("PIPELINE STATISTICS")
            print("=" * 70)

            print(f"Recording            : {record_time:8.3f} sec")
            print(f"Whisper STT          : {whisper_time:8.3f} sec")
            print(f"Mission Cache Read   : {cache_time:8.3f} sec")
            print(f"Prompt Builder       : {prompt_time:8.3f} sec")
            print(f"Qwen Inference       : {qwen_time:8.3f} sec")
            print(f"Mission Processing   : {mission_time:8.3f} sec")
            print(f"Response Formatter   : {formatter_time:8.3f} sec")
            print(f"Piper TTS            : {tts_time:8.3f} sec")

            print("-" * 70)
            print(f"TOTAL PIPELINE       : {total_time:8.3f} sec")
            print("-" * 70)

            print(f"Transcript           : {transcript}")
            print(f"Intent               : {command['intent']}")
            print(f"Field                : {command.get('field', '-')}")
            print(f"Target               : {command.get('target', '-')}")
            print("=" * 70)

            interaction += 1

    except KeyboardInterrupt:

        print("\nShutting down...")

        mission_cache.clear()

        print("Mission cache cleared.")


if __name__ == "__main__":
    main()