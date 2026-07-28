"""
=========================================================
Qwen Engine
=========================================================
Loads Qwen2.5 and generates responses.
=========================================================
"""

import os
import time
import torch
from config import RAW_RESPONSE_FILE

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
)


class QwenEngine:

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

    # --------------------------------------------------

    def generate(self, prompt: str):

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

        return response, latency
    # def save_response(response: str):
    #     with open(
    #     RAW_RESPONSE_FILE,
    #     "w",
    #     encoding="utf-8"
    #     ) as f:
    #         f.write(response)