import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))
from modules.tts_engine import TTSEngine

tts = TTSEngine()

tts.speak(

    "Hello. I am Genesis. Your mission assistant is now online."

)