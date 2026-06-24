"""
Agent 2: Strategy Generator - IMPROVED VERSION
With flexible JSON structure handling and auto-recovery
(Adapted for local VS Code execution on Windows — no Colab dependency)
"""

import json
import os
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai
import os

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

# def setup_device():
#     """Configure device for inference"""
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print(f" Using GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         device = torch.device("cpu")
#         print(" GPU not available, using CPU (slower)")
#     return device

# def load_qwen_model(device):
#     """Load Qwen 2.5 3B Instruct model"""
#     HF_TOKEN = os.getenv("HF_TOKEN")
#     print("\n Loading Qwen 2.5 3B Instruct model...")

#     model_id = "Qwen/Qwen2.5-3B-Instruct"

#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_id,token=HF_TOKEN)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             token=HF_TOKEN,
#             torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
#             device_map="auto" if device.type == "cuda" else None
#         )
#         if device.type == "cpu":
#             model = model.to(device)
#         model.eval()
#         print(" Qwen 2.5 3B Instruct loaded successfully")
#         return tokenizer, model
#     except Exception as e:
#         print(f" Error loading model: {e}")
#         raise

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if API_KEY is None:
    raise ValueError(
        "GEMINI_API_KEY not found"
    )

genai.configure(
    api_key=API_KEY.strip()
)

# ============================================================================
# DATA LOADING AND ANALYSIS
# ============================================================================
# NOTE: this module-level file_path is unused by load_summary_json (which takes
# its own argument) — kept as-is since logic was not to be changed. Update this
# to your actual local path if anything else in your pipeline references it.
file_path = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\terrain_intelligence_output\summary.json"

def load_summary_json(file_path: str) -> Dict[str, Any]:
    """Load summary.json file"""
    print(f"\nLoading summary data from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Successfully loaded summary data")
        print(f"  - Routes remaining: {data['routes_remaining']}")
        print(f"  - Bottlenecks identified: {len(data['top_bottlenecks'])}")
        return data
    except FileNotFoundError:
        print(f" File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f" Invalid JSON format in {file_path}")
        raise

def analyze_corridor_distribution(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-analyze corridor distribution for context"""
    total_usage = (
        summary_data['west_corridor_usage'] +
        summary_data['center_corridor_usage'] +
        summary_data['east_corridor_usage']
    )

    analysis = {
        "total_usage": total_usage,
        "west_percentage": round(summary_data['west_corridor_usage'] / total_usage * 100, 2),
        "center_percentage": round(summary_data['center_corridor_usage'] / total_usage * 100, 2),
        "east_percentage": round(summary_data['east_corridor_usage'] / total_usage * 100, 2),
        "most_used_corridor": max(
            [("west", summary_data['west_corridor_usage']),
             ("center", summary_data['center_corridor_usage']),
             ("east", summary_data['east_corridor_usage'])],
            key=lambda x: x[1]
        )[0],
        "bottleneck_hotspots": [
            {
                "location": f"({bn['grid_row']}, {bn['grid_col']})",
                "frequency": bn['frequency']
            }
            for bn in summary_data['top_bottlenecks'][:3]
        ]
    }
    return analysis

# ============================================================================
# LLM PROMPTING AND ANALYSIS
# ============================================================================
def create_analysis_prompt(summary_data, corridor_analysis):
    prompt = f"""
    You are a surveillance strategy planner.
    Your job is to analyze the route distribution and determine:
    1. Which corridor should be prioritized.
    2. Whether the next camera should focus on:
     - route elimination
     - coverage expansion
    3. Appropriate route and coverage weights.

    IMPORTANT:

    The weights MUST be derived from the input data.
    You MUST select the 5 most important bottlenecks from the input data for the next camera placement.Return them in:"priority_bottlenecks" ordered from highest priority to lowest priority.
    Return ONLY JSON.

    Use:

    - bottleneck frequency
    - corridor importance
    - strategy decision

    when choosing them.

    Guidelines:

    - If one corridor dominates(>60% usage ):
     prioritize route elimination.

    - If routes_remaining is very high:
     increase route_weights.

    - If routes_remaining is low but coverage is poor:
     increase coverage_weights.

    - route_weights + coverage_weights MUST equal 1.0

    - Adjust weights according to the terrain summary.

    Priority corridor should primarily be determined from:
    1. The corridor containing the highest-frequency bottleneck.
    2. Corridor usage percentages.

    INPUT DATA
    routes_remaining = {summary_data['routes_remaining']}

    west_corridor_usage = {summary_data['west_corridor_usage']}

    center_corridor_usage = {summary_data['center_corridor_usage']}

    east_corridor_usage = {summary_data['east_corridor_usage']}

    top_bottlenecks =
    {json.dumps(corridor_analysis['bottleneck_hotspots'], indent=2)}

    Return ONLY valid JSON.

    JSON FORMAT:{{
    "location": "<top bottleneck location>",
    "frequency": <top bottleneck frequency>,

    "strategy_decision":
        "<route_elimination or coverage_expansion>",

    "reasoning":
        "<explain why>",

    "priority_corridor":
        "<west or center or east>",

    
    "priority_bottlenecks":
        "<give top 5 bottlenecks>",

    "priority_reasoning":
        "<explain corridor choice>",    
    "route_weights":
        <float>,

    "coverage_weights":
        <float>,

    "implementation_notes":
        "<camera placement guidance>",

    "expected_impact":
        "<predicted effect>"
    "weight_formula":
        "<formula used>"}}

Return ONLY JSON.
If the highest-frequency bottleneck lies in CENTER,CENTER should generally be prioritized.

"""
    return prompt
def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Robust JSON extraction"""
    response = response.strip()

    json_start = response.find("{")
    if json_start == -1:
        raise ValueError("No JSON found")

    brace_count = 0
    json_end = -1

    for i in range(json_start, len(response)):
        if response[i] == "{":
            brace_count += 1
        elif response[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break

    if json_end == -1:
        raise ValueError("JSON object not closed")

    return json.loads(response[json_start:json_end])

# def query_llm(tokenizer, model, prompt: str, device) -> str:
#     """Query Qwen model and get response"""
#     print("\n Querying Qwen 2.5 3B for strategy analysis...")

#     try:
#         # Prepare input
#         messages = [
#             {"role": "user", "content": prompt}
#         ]
#         print("\n========== DEBUG PROMPT ==========")
#         print(type(prompt))
#         print(prompt)
#         print("==================================")
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         model_inputs = tokenizer(text, return_tensors="pt").to(device)

#         # Generate response with optimized parameters
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 **model_inputs,
#                 max_new_tokens=1000,
#                 temperature=0.1,          # Very low for JSON consistency
#                 top_p=0.9,
#                 top_k=50,
#                 do_sample=True,          # No randomness for JSON
#                 pad_token_id=tokenizer.eos_token_id
#             )

#         # Decode response
#         generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]

#         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#         # Extract JSON
#         strategy = extract_json_from_response(response)
#         print(" Strategy analysis completed successfully")
#         return strategy

#     except Exception as e:
#         print(f" Error querying LLM: {e}")
#         print(f"Response preview: {response[:200] if 'response' in locals() else 'N/A'}")
#         raise

def query_llm(prompt):

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    response = model.generate_content(
        prompt
    )

    raw_text = response.text.strip()

    print(
        "\nLLM RAW RESPONSE:\n"
    )
    print(raw_text)

    strategy = extract_json_from_response(
        raw_text
    )

    return strategy


# ============================================================================
# STRATEGY VALIDATION AND RECOVERY
# ============================================================================

def normalize_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize strategy - handle nested or alternative structures
    """
    # Check if strategy is wrapped in another object
    if 'strategy' in strategy and isinstance(strategy['strategy'], dict):
        strategy = strategy['strategy']

    # Check if we have the main keys
    required_keys = [
    'location',
    'frequency',
    'strategy_decision',
    'reasoning',
    'priority_corridor',
    'priority_reasoning',
    'priority_bottlenecks',
    'route_weights',
    'coverage_weights',
    'implementation_notes',
    'expected_impact'
]

    # If missing some keys, try to find them in nested structures
    for key in required_keys:
        if key not in strategy:
            # Search in all nested dicts
            for k, v in strategy.items():
                if isinstance(v, dict) and key in v:
                    strategy[key] = v[key]
                    break

    return strategy

def validate_and_repair_strategy(strategy: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    """
    Validate strategy and attempt repair
    Returns: (is_valid, repaired_strategy)
    """
    # Try to normalize first
    strategy = normalize_strategy(strategy)

    required_keys = [
        'strategy_decision',
        'reasoning',
        'priority_corridor',
        'priority_reasoning',
        'priority_bottlenecks',
        'route_weights',
        'coverage_weights',
        'implementation_notes',
        'expected_impact'
    ]

    missing_keys = [k for k in required_keys if k not in strategy]

    if not missing_keys:
        return True, strategy

    # Try to repair missing keys with defaults
    print(f"\nMissing keys: {missing_keys}")
    print("Attempting repair with sensible defaults...")

    defaults = {
    'location': '(2, 4)',
    'frequency': 164,
    'strategy_decision': 'route_elimination',
    'reasoning': 'Based on corridor analysis',
    'priority_corridor': 'center',
    'priority_reasoning': 'Highest usage concentration',
    'priority_bottlenecks': [],
    'route_weights': None,
    'coverage_weights': None,
    'implementation_notes': 'Standard implementation approach',
    'expected_impact': '20-25% efficiency improvement'
}

    for key in missing_keys:
        strategy[key] = defaults[key]
        print(f"   Repaired '{key}' with default value")

    return False, strategy  # Valid structure but used defaults

def enhance_strategy(strategy: Dict[str, Any], summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add metadata and enhance strategy with additional computed values"""
    top_bn = summary_data['top_bottlenecks'][0]
    enhanced = {
        "metadata": {
            "agent": "agent_2_strategy_generator",
            "iteration": summary_data['iteration'],
            "routes_remaining": summary_data['routes_remaining'],
            "status": "completed"
        },
        "corridor_status": {
            "west": {
                "usage_percentage": summary_data['west_corridor_usage'],
                "status": "underutilized" if summary_data['west_corridor_usage'] < 20 else "normal"
            },
            "center": {
                "usage_percentage": summary_data['center_corridor_usage'],
                "status": "overutilized" if summary_data['center_corridor_usage'] > 60 else "normal"
            },
            "east": {
                "usage_percentage": summary_data['east_corridor_usage'],
                "status": "normal"
            }
        },
        "strategy": {
    "location": f"({top_bn['grid_row']}, {top_bn['grid_col']})",
    "frequency": top_bn["frequency"],

    "strategy_decision": strategy["strategy_decision"],
    "reasoning": strategy["reasoning"],

    "priority_corridor": strategy["priority_corridor"],
    "priority_reasoning": strategy["priority_reasoning"],
    "priority_bottlenecks": strategy["priority_bottlenecks"],
    "route_weights": strategy["route_weights"],
    "coverage_weights": strategy["coverage_weights"],

    "implementation_notes": strategy["implementation_notes"],
    "expected_impact": strategy["expected_impact"]
},
        "next_stage": "camera_scoring",
        "input_summary": summary_data
    }

    return enhanced

def save_strategy(strategy: Dict[str, Any], output_path: str = "strategy.json"):
    """Save strategy to JSON file"""
    print(f"\nSaving strategy to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(strategy, f, indent=2)
        print(f" Strategy saved successfully")

        if 'strategy' in strategy:
            strat = strategy['strategy']
            print(f"\nStrategy Summary:")
            print(f"  - Decision: {strat.get('strategy_decision', 'N/A')}")
            print(f"  - Priority Corridor: {strat.get('priority_corridor', 'N/A')}")
            print(f"  - Route Weights: {strat.get('route_weights', 'N/A')}")
            print(f"  - Coverage Weights: {strat.get('coverage_weights', 'N/A')}")

        return True
    except Exception as e:
        print(f" Error saving strategy: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def get_latest_summary_file(summary_dir):

    summary_files = []

    for file in os.listdir(summary_dir):

        if (
            file.startswith("summary_iter_")
            and file.endswith(".json")
        ):

            try:
                iteration = int(
                    file.replace("summary_iter_", "")
                        .replace(".json", "")
                )

                summary_files.append(
                    (iteration, file)
                )

            except:
                pass

    if not summary_files:
        raise FileNotFoundError(
            "No summary_iter_*.json files found"
        )

    summary_files.sort(
        key=lambda x: x[0]
    )

    latest_file = summary_files[-1][1]

    return os.path.join(
        summary_dir,
        latest_file
    )
# ==========================================================
# AUTO DETECT LATEST SUMMARY FILE
# ==========================================================

def get_latest_summary_file(summary_dir):

    default_summary = os.path.join(
        summary_dir,
        "summary.json"
    )

    summary_iters = []

    for file in os.listdir(summary_dir):

        if (
            file.startswith("summary_iter_")
            and file.endswith(".json")
        ):

            try:

                iteration = int(
                    file.replace(
                        "summary_iter_",
                        ""
                    ).replace(
                        ".json",
                        ""
                    )
                )

                summary_iters.append(
                    (iteration, file)
                )

            except:
                pass

    # No iteration summaries yet
    if len(summary_iters) == 0:

        if os.path.exists(default_summary):

            print(
                "\nUsing summary.json "
                "(Iteration 0)"
            )

            return default_summary

        raise FileNotFoundError(
            "No summary files found."
        )

    summary_iters.sort(
        key=lambda x: x[0]
    )

    latest_file = summary_iters[-1][1]

    latest_path = os.path.join(
        summary_dir,
        latest_file
    )

    print(
        f"\nUsing latest summary: "
        f"{latest_file}"
    )

    return latest_path
def main(summary_json_path: str = "summary.json", output_path: str = "strategy.json"):
    """Main execution function"""

    print("=" * 70)
    print("AGENT 2: STRATEGY GENERATOR (IMPROVED)")
    print("=" * 70)

    try:
        # 1. Setup
        #device = setup_device()
        #tokenizer, model = load_qwen_model(device)

        # 2. Load data
        summary_data = load_summary_json(summary_json_path)
        corridor_analysis = analyze_corridor_distribution(summary_data)

        # 3. Create prompt and query LLM
        prompt = create_analysis_prompt(summary_data, corridor_analysis)
        #strategy = query_llm(tokenizer, model, prompt, device)
        strategy = query_llm(prompt)
        # 4. Validate and enhance
        is_valid, strategy = validate_and_repair_strategy(strategy)

        if not is_valid:
            print(" Strategy was repaired with defaults")
        else:
            print(" Strategy validation passed")

        enhanced_strategy = enhance_strategy(strategy, summary_data)

        # 5. Save output
        save_strategy(enhanced_strategy, output_path)

        print("\n" + "=" * 70)
        print(" AGENT 2 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nOutput file: {output_path}")
        print("Ready for Agent 3: Camera Scoring")

        return enhanced_strategy

    except Exception as e:
        print(f"\n AGENT 2 FAILED: {str(e)}")
        raise

# ============================================================================
# LOCAL (VS CODE) ENTRY POINT
# ============================================================================
if __name__ == "__main__":

    SUMMARY_DIR = (
        r"C:\Users\KHUSHI\Documents"
        r"\deepterrain_internship"
        r"\poleplacement_codes"
        r"\DeepTerrain"
        r"\agentic_ai"
        r"\terrain_intelligence_output"
    )

    output_dir = (
        r"C:\Users\KHUSHI\Documents"
        r"\deepterrain_internship"
        r"\poleplacement_codes"
        r"\DeepTerrain"
        r"\agentic_ai"
        r"\agent2_output"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    summary_file = get_latest_summary_file(
        SUMMARY_DIR
    )

    output_file = os.path.join(
        output_dir,
        "strategy.json"
    )

    strategy_result = main(
        summary_json_path=summary_file,
        output_path=output_file
    )