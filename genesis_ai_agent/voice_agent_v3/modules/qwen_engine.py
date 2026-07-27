"""
=========================================================
Qwen Engine
=========================================================
Loads Qwen2.5 and performs structured inference.

Responsibilities
----------------
1. Load Qwen model.
2. Accept complete prompt.
3. Perform inference.
4. Extract JSON from LLM output.
5. Validate JSON.
6. Return Python dictionary.
=========================================================
"""

import os
import time
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from config import (
    HF_TOKEN,
    QWEN_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    RAW_RESPONSE_FILE,
)


class QwenEngine:

    #########################################################

    def __init__(self):

        print("\nLoading Qwen...")

        if torch.cuda.is_available():

            self.device = torch.device("cuda")

            print(f"Using GPU : {torch.cuda.get_device_name(0)}")

        else:

            self.device = torch.device("cpu")

            print("Using CPU")

        token = HF_TOKEN or os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL,
            token=token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            token=token,
            torch_dtype=torch.float16
            if self.device.type == "cuda"
            else torch.float32,
            device_map="auto"
            if self.device.type == "cuda"
            else None,
        )

        if self.device.type == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        print("Qwen Loaded Successfully.\n")

    #########################################################

    def _extract_json(self, response: str) -> str:

        """
        Extract the JSON object returned by Qwen.
        """

        response = response.strip()

        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL
        )

        if match is None:

            raise ValueError(
                "Qwen did not return a valid JSON object."
            )

        return match.group()

    #########################################################

    def _validate(self, command: dict) -> dict:

        """
        Validate command returned by Qwen.
        """

        if "intent" not in command:

            raise ValueError(
                "Missing required field: intent"
            )

        intent = command["intent"].upper()

        if intent not in ["GET", "PATCH"]:

            raise ValueError(
                f"Unsupported intent: {intent}"
            )

        if intent == "PATCH":

            required = [
                "field",
                "value"
            ]

        else:

            required = [
                "field"
            ]

        missing = [
            key
            for key in required
            if key not in command
        ]

        if missing:

            raise ValueError(
                f"Missing required fields: {missing}"
            )

        # --------------------------------------------------
        # Optional metadata
        # --------------------------------------------------

        command.setdefault("target", "")
        command.setdefault("field_found", True)
        command.setdefault("target_found", True)
        command.setdefault("confidence", 1.0)

        return command

    #########################################################

    def infer(self, prompt: str):

        """
        Perform inference and return structured command.
        """

        start = time.time()

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = generated_ids[
            :,
            model_inputs.input_ids.shape[1]:
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        latency = round(
            time.time() - start,
            2
        )

        # --------------------------------------------------
        # Save raw response for debugging
        # --------------------------------------------------

        try:

            with open(
                RAW_RESPONSE_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(response)

        except Exception:

            pass

        # --------------------------------------------------
        # Extract JSON
        # --------------------------------------------------

        json_string = self._extract_json(response)

        try:

            command = json.loads(json_string)

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Invalid JSON returned by Qwen:\n{e}"
            )

        command = self._validate(command)

        return command, latency


#########################################################
# Singleton
#########################################################

qwen_engine = QwenEngine()


#########################################################
# Standalone Test
#########################################################

if __name__ == "__main__":

    prompt = """
Return only JSON.

User:
Set loop count to 3.
"""

    command, latency = qwen_engine.infer(prompt)

    print("\nLatency :", latency, "seconds\n")

    print(json.dumps(
        command,
        indent=4
    ))