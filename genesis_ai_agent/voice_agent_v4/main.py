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

import json
import time

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


# =========================================================

def print_pipeline_stats(stats):

    print("\n")
    print("=" * 80)
    print("GENESIS PIPELINE METRICS")
    print("=" * 80)

    print(f"Recording           : {stats['record']:.2f} sec")
    print(f"Whisper STT         : {stats['whisper']:.2f} sec")
    print(f"Mission GET API     : {stats['mission_get']:.2f} sec")
    print(f"Prompt Builder      : {stats['prompt']:.2f} sec")
    print(f"LLM Inference       : {stats['llm']:.2f} sec")
    print(f"Mission Processing  : {stats['mission_process']:.2f} sec")
    print(f"Response Formatter  : {stats['formatter']:.2f} sec")
    print(f"Piper TTS           : {stats['tts']:.2f} sec")

    print("-" * 80)

    print(f"Total Pipeline Time : {stats['total']:.2f} sec")

    print("=" * 80)


# =========================================================

def main():

    print("=" * 70)
    print("Genesis Voice Agent V2")
    print("=" * 70)

    while True:

        try:

            pipeline_start = time.perf_counter()

            # -------------------------------------------------
            # Record Audio
            # -------------------------------------------------

            print("\nListening...")

            t0 = time.perf_counter()

            audio_path = recorder.record(INPUT_AUDIO)

            record_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Speech → Text
            # -------------------------------------------------

            t0 = time.perf_counter()

            transcript = whisper_engine.transcribe(audio_path)

            whisper_time = time.perf_counter() - t0

            if not transcript.strip():

                print("No speech detected.")

                continue

            print(f"\nUser : {transcript}")

            # -------------------------------------------------
            # Fetch Mission
            # -------------------------------------------------

            t0 = time.perf_counter()

            mission_response = mission_api.get_current_mission()

            mission_get_time = time.perf_counter() - t0

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
                    ensure_ascii=False,
                )
            )

            print("=" * 80)

            # -------------------------------------------------
            # Prompt Builder
            # -------------------------------------------------

            t0 = time.perf_counter()

            prompt = prompt_builder.build_prompt(
                transcript=transcript,
                current_mission=current_mission,
            )

            prompt_time = time.perf_counter() - t0

            # -------------------------------------------------
            # LLM
            # -------------------------------------------------

            print("\nGenerating response using Qwen2.5...\n")

            command, llm_stats = qwen_engine.infer(prompt)

            llm_time = llm_stats["latency"]

            print("\nLLM Command")
            print(command)

            # -------------------------------------------------
            # Execute Intent
            # -------------------------------------------------

            t0 = time.perf_counter()

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

            mission_process_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Response Formatter
            # -------------------------------------------------

            t0 = time.perf_counter()

            reply = response_formatter.format(
                result,
                command
            )

            formatter_time = time.perf_counter() - t0

            print("\nGenesis :", reply)

            # -------------------------------------------------
            # Piper
            # -------------------------------------------------

            t0 = time.perf_counter()

            tts.speak(reply)

            tts_time = time.perf_counter() - t0

            # -------------------------------------------------
            # Pipeline Statistics
            # -------------------------------------------------

            total_pipeline = (
                time.perf_counter()
                - pipeline_start
            )

            pipeline_stats = {
                "record": record_time,
                "whisper": whisper_time,
                "mission_get": mission_get_time,
                "prompt": prompt_time,
                "llm": llm_time,
                "mission_process": mission_process_time,
                "formatter": formatter_time,
                "tts": tts_time,
                "total": total_pipeline,
            }

            print_pipeline_stats(pipeline_stats)

        except KeyboardInterrupt:

            print("\nExiting Genesis.")

            break

        except Exception as e:

            print("\nERROR")
            print(e)


# =========================================================

if __name__ == "__main__":

    main()