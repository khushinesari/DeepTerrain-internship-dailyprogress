"""
==========================================================
Piper TTS Engine
==========================================================
"""

import subprocess
import sounddevice as sd
import soundfile as sf

from config import (
    PIPER_EXE,
    VOICE_MODEL,
    OUTPUT_AUDIO,
)


class TTSEngine:

    def __init__(self):

        self.piper = str(PIPER_EXE)

        self.voice = str(VOICE_MODEL)

        self.output = str(OUTPUT_AUDIO)

    # --------------------------------------------------

    def synthesize(self, text):

        command = [

            self.piper,

            "--model",

            self.voice,

            "--output_file",

            self.output

        ]

        process = subprocess.Popen(

            command,

            stdin=subprocess.PIPE,

            text=True

        )

        process.communicate(text)

    # --------------------------------------------------

    def play(self):

        data, samplerate = sf.read(self.output)

        sd.play(data, samplerate)

        sd.wait()

    # --------------------------------------------------

    def speak(self, text):

        self.synthesize(text)

        self.play()

tts=TTSEngine()