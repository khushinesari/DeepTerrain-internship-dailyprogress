from modules.recorder import Recorder
from modules.whisper_engine import WhisperEngine
from modules.prompt_builder import build_prompt, save_prompt
from modules.qwen_engine import QwenEngine
from modules.parser import ResponseParser
from modules.response_formatter import ResponseFormatter
from modules.tts_engine import TTSEngine
from config import INPUT_AUDIO

def main():

    print("=" * 60)
    print("GENESIS VOICE AGENT")
    print("=" * 60)

    # -------------------------------------------------
    # Initialize Modules
    # -------------------------------------------------

    recorder = Recorder()
    whisper = WhisperEngine()
    llm = QwenEngine()
    parser = ResponseParser()
    formatter = ResponseFormatter()
    tts = TTSEngine()

    # -------------------------------------------------
    # Record Audio
    # -------------------------------------------------

    print("\nRecording...\n")

    audio_path = recorder.record(INPUT_AUDIO)

    print(f"Audio saved to:\n{audio_path}")

    # -------------------------------------------------
    # Speech to Text
    # -------------------------------------------------

    print("\nTranscribing...\n")

    transcript = whisper.transcribe(audio_path)

    print("\nUSER:")
    print(transcript)

    if not transcript or not transcript.strip():
        print("\nNo speech detected.")
        return

    # -------------------------------------------------
    # Build Prompt
    # -------------------------------------------------

    print("\nBuilding Prompt...\n")

    prompt = build_prompt(transcript)

    save_prompt(prompt)

    print("Prompt Saved.")

    # -------------------------------------------------
    # Generate LLM Response
    # -------------------------------------------------

    print("\nGenerating Response...\n")

    raw_response, inference_time = llm.generate(prompt)

    print("\nRAW RESPONSE:\n")
    print(raw_response)

    print(f"\nInference Time: {inference_time:.2f} seconds")

    # -------------------------------------------------
    # Parse Response
    # -------------------------------------------------

    print("\nParsing Response...\n")

    parsed = parser.parse(raw_response)

    print("Parsed Successfully.")

    # -------------------------------------------------
    # Format Response for Speech
    # -------------------------------------------------

    speech = formatter.format(parsed)

    print("\nASSISTANT:")
    print(speech)

    # -------------------------------------------------
    # Text-to-Speech
    # -------------------------------------------------

    print("\nSpeaking...\n")

    tts.speak(speech)

    print("\nDone.")


if __name__ == "__main__":
    main()