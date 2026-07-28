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
        self.model_name = QWEN_MODEL
        print("Qwen Loaded Successfully.\n")

    #########################################################

    def _extract_json(self, response:str):

        # Remove markdown if present
        response = response.replace("```json", "")
        response = response.replace("```", "").strip()

        # Extract first JSON object
        match = re.search(r"\{[\s\S]*\}", response)

        if not match:
            raise ValueError("No JSON object found.")

        json_text = match.group(0)

        # ----------------------------------------------------
        # Remove trailing commas
        # ----------------------------------------------------

        json_text = re.sub(
            r",\s*}",
            "}",
            json_text
        )

        json_text = re.sub(
            r",\s*]",
            "]",
            json_text
        )

        return json.loads(json_text)

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
        # command.setdefault("confidence", 1.0)

        return command
    #########################################################

    def _print_llm_stats(
        self,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        ):

        total_tokens = input_tokens + output_tokens

        # Qwen2.5 supports a very large context window.
        # Adjust if you use another model.
        context_window = 2097152

        context_percent = (
            total_tokens /
            context_window
        ) * 100

        tokens_per_second = (
            output_tokens / latency
            if latency > 0
            else 0
        )

        print("\n" + "=" * 80)
        print("GENESIS LLM STATISTICS")
        print("=" * 80)

        print(f"Model               : {self.model_name}")
        print(f"Device              : {self.device}")

        if self.device.type == "cuda":
            print(
                f"GPU                 : {torch.cuda.get_device_name(0)}"
            )

        print(f"Prompt Tokens       : {input_tokens}")
        print(f"Completion Tokens   : {output_tokens}")
        print(f"Total Tokens        : {total_tokens}")
        print(f"Inference Time      : {latency:.2f} sec")
        print(f"Generation Speed    : {tokens_per_second:.2f} tokens/sec")
        print(f"Context Used        : {context_percent:.5f}%")

        print("=" * 80)
    #########################################################

    def infer(self, prompt: str):

        """
        Perform inference and return structured command.
        """

        start = time.perf_counter()

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
        input_tokens = model_inputs.input_ids.shape[1]
        with torch.no_grad():

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=TEMPERATURE > 0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = generated_ids[
            :,
            model_inputs.input_ids.shape[1]:
        ]
        output_tokens = generated_ids.shape[1]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        latency = time.perf_counter() - start

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


        try:

            command = self._extract_json(response)

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Invalid JSON returned by Qwen:\n{e}"
            )

        command = self._validate(command)
        self._print_llm_stats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        )

        stats = {
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tokens_per_second": (
                output_tokens / latency
                if latency > 0
                else 0
            ),
        }

        return command, stats
        
    


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

    command, stats = qwen_engine.infer(prompt)

    print()

    for key, value in stats.items():
        print(f"{key:20} : {value}")

    print(json.dumps(
        command,
        indent=4
    ))