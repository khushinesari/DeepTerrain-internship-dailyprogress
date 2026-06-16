#!/usr/bin/env python3

import os
import re
import json
import torch
from google.colab import userdata
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SUMMARY_JSON = "/content/summary.json"

OUTPUT_DIR = "/content/strategy_output"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =====================================================
# LOAD SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("LOADING SUMMARY")
print("=" * 60)

with open(
    SUMMARY_JSON,
    "r"
) as f:

    summary = json.load(f)

print(
    f"\nRoutes Remaining: {summary.get('routes_remaining', 0)}"
)

# =====================================================
# LOAD MODEL
# =====================================================

print("\n" + "=" * 60)
print("LOADING QWEN3")
print("=" * 60)
hf_token = userdata.get('HF_TOKEN')
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

print("\nModel Loaded Successfully")

# =====================================================
# PROMPT
# =====================================================

prompt = f"""
You are a surveillance strategy agent.

Your task is to analyze intrusion route statistics
and produce a surveillance strategy.

DO NOT RETURN JSON.

RETURN EXACTLY IN THIS FORMAT:

STRATEGY: route_elimination or coverage_expansion

PRIORITY_CORRIDOR: WEST or CENTER or EAST

ROUTE_WEIGHT: number between 0 and 1

COVERAGE_WEIGHT: number between 0 and 1

BOTTLENECKS: comma separated bottleneck IDs

REASONING: one concise sentence

Terrain Summary:

{json.dumps(summary, indent=2)}

Remember:

Only output the fields above.

Do not explain anything else.
"""

# =====================================================
# TOKENIZE
# =====================================================

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

# =====================================================
# GENERATE
# =====================================================

print("\nGenerating Strategy...\n")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.0,
    do_sample=False
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# =====================================================
# REMOVE THINKING BLOCKS
# =====================================================

if "</think>" in response:

    response = response.split(
        "</think>"
    )[-1]

response = response.strip()

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

print(response)

# =====================================================
# FIELD EXTRACTION
# =====================================================

def extract(pattern, text, default=""):

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return default

strategy_name = extract(
    r"STRATEGY:\s*(.*)",
    response
)

priority_corridor = extract(
    r"PRIORITY_CORRIDOR:\s*(.*)",
    response
)

route_weight = extract(
    r"ROUTE_WEIGHT:\s*(.*)",
    response,
    "0.5"
)

coverage_weight = extract(
    r"COVERAGE_WEIGHT:\s*(.*)",
    response,
    "0.5"
)

bottlenecks = extract(
    r"BOTTLENECKS:\s*(.*)",
    response
)

reasoning = extract(
    r"REASONING:\s*(.*)",
    response
)

# =====================================================
# CLEAN VALUES
# =====================================================

try:

    route_weight = float(
        route_weight
    )

except:

    route_weight = 0.5

try:

    coverage_weight = float(
        coverage_weight
    )

except:

    coverage_weight = 0.5

priority_bottlenecks = []

if bottlenecks:

    priority_bottlenecks = [

        b.strip()

        for b in bottlenecks.split(",")

        if len(b.strip())
    ]

# =====================================================
# BUILD JSON
# =====================================================

strategy = {

    "strategy":
        strategy_name,

    "priority_corridor":
        priority_corridor,

    "priority_bottlenecks":
        priority_bottlenecks,

    "route_weight":
        route_weight,

    "coverage_weight":
        coverage_weight,

    "reasoning":
        reasoning
}

# =====================================================
# SAVE JSON
# =====================================================

strategy_path = os.path.join(
    OUTPUT_DIR,
    "strategy.json"
)

with open(
    strategy_path,
    "w"
) as f:

    json.dump(
        strategy,
        f,
        indent=2
    )

# =====================================================
# DISPLAY
# =====================================================

print("\n" + "=" * 60)
print("GENERATED STRATEGY")
print("=" * 60)

print(
    json.dumps(
        strategy,
        indent=2
    )
)

print(
    f"\nSaved:\n{strategy_path}"
)

print("\n" + "=" * 60)
print("AGENT 2 COMPLETE")
print("=" * 60)
