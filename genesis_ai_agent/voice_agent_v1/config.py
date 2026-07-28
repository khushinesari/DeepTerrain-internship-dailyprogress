"""
===========================================================
Genesis Voice Agent - Global Configuration
===========================================================
All project-wide settings are defined here.
No hardcoded values should exist in any other file.
===========================================================
"""

from pathlib import Path

# =========================================================
# Project Directories
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = ROOT_DIR / "audio"
OUTPUT_DIR = ROOT_DIR / "outputs"
PROMPT_DIR = ROOT_DIR / "prompts"
SCHEMA_DIR = ROOT_DIR / "schemas"
LOG_DIR = ROOT_DIR / "logs"

# Create directories automatically

for directory in [
    AUDIO_DIR,
    OUTPUT_DIR,
    PROMPT_DIR,
    SCHEMA_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# =========================================================
# Audio Files
# =========================================================

INPUT_AUDIO = AUDIO_DIR / "input.wav"

# =========================================================
# Output Files
# =========================================================

TRANSCRIPT_FILE = OUTPUT_DIR / "transcript.txt"

PROMPT_FILE = OUTPUT_DIR / "prompt.txt"

RAW_RESPONSE_FILE = OUTPUT_DIR / "response.json"

MISSION_LLM_OUTPUT = OUTPUT_DIR / "mission_llmoutput.json"
# =========================================================
# Whisper Configuration
# =========================================================

WHISPER_MODEL = "small"

DEVICE = "cpu"

COMPUTE_TYPE = "int8"

LANGUAGE = "en"

BEAM_SIZE = 5

# Automatically remove long silence
VAD_FILTER = True

# Avoid hallucinations
CONDITION_ON_PREVIOUS_TEXT = False

TEMPERATURE = 0.0

# =========================================================
# Audio Recording
# =========================================================

SAMPLE_RATE = 16000

CHANNELS = 1

RECORD_DURATION = 5      # seconds (temporary)

# =====================================================
# Qwen Configuration
# =====================================================

HF_TOKEN = None      # Uses environment variable if None

QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"

MAX_NEW_TOKENS = 1024

TEMPERATURE = 0.1

TOP_P = 0.9

TOP_K = 50
# =========================================================
# Logging
# =========================================================

LOG_LEVEL = "INFO"

LOG_FILE = LOG_DIR / "voice_agent.log"

# =========================================================
# API (Future)
# =========================================================

SERVER_HOST = "127.0.0.1"

SERVER_PORT = 8000

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROMPTS_DIR = BASE_DIR / "prompts"
SCHEMA_DIR = BASE_DIR / "schemas"

SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"

MISSION_SCHEMA_FILE = SCHEMA_DIR / "mission_schema.json"

COMMAND_SCHEMA_FILE = SCHEMA_DIR / "command_schema.json"
DATA_DIR = BASE_DIR / "data"

MISSION_CURRENT = DATA_DIR / "mission_current.json"
# =====================================================
# Piper
# =====================================================

PIPER_EXE = r"C:\Users\KHUSHI\Documents\deepterrain_internship\genesis_voice_agent\piper\piper\piper.exe"

VOICE_MODEL = ROOT_DIR / "voices" / "en_US-amy-medium.onnx"

OUTPUT_AUDIO = AUDIO_DIR / "output" / "output.wav"