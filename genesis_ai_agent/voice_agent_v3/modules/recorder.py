from pathlib import Path

import sounddevice as sd
import soundfile as sf


class Recorder:

    def __init__(

        self,

        sample_rate=16000,

        channels=1

    ):

        self.sample_rate = sample_rate

        self.channels = channels

    def record(

        self,

        output_path: Path,

        duration: int = 5

    ) -> Path:

        output_path.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        print("\n============================")

        print(" Speak Now")

        print("============================\n")

        audio = sd.rec(

            int(duration * self.sample_rate),

            samplerate=self.sample_rate,

            channels=self.channels,

            dtype="float32"

        )

        sd.wait()

        sf.write(

            output_path,

            audio,

            self.sample_rate

        )

        print("Recording Finished\n")

        return output_path

recorder = Recorder()