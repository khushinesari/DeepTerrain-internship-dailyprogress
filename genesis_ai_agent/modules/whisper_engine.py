from pathlib import Path
import time

from faster_whisper import WhisperModel

from config import *

class WhisperEngine:

    def __init__(self):

        print("Loading Whisper...")

        self.model = WhisperModel(

            WHISPER_MODEL,

            device=DEVICE,

            compute_type=COMPUTE_TYPE

        )

        print("Whisper Ready\n")

    def transcribe(self, audio_path: Path):

     start = time.time()

     segments, info = self.model.transcribe(

        str(audio_path),

        beam_size=BEAM_SIZE,

        language=LANGUAGE

     )

     transcript = " ".join(

        segment.text.strip()

        for segment in segments

     )

     transcript = " ".join(transcript.split())

     processing_time = round(

        time.time() - start,

        2

     )

     with open(

        TRANSCRIPT_FILE,

        "w"

     ) as f:

        f.write(transcript)

     print()

     print("Transcript")

     print("----------------")

     print(transcript)

     print()

     return transcript