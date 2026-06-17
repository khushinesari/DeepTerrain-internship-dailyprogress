!pip install torch transformers accelerate -q

"""
Agent 2: Strategy Generator - IMPROVED VERSION
With flexible JSON structure handling and auto-recovery
"""

import json
import os
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

def setup_device():
    """Configure device for inference"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ GPU not available, using CPU (slower)")
    return device

def load_qwen_model(device):
    """Load Qwen 2.5 3B Instruct model"""
    print("\n📥 Loading Qwen 2.5 3B Instruct model...")
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        if device.type == "cpu":
            model = model.to(device)
        model.eval()
        print("✓ Qwen 2.5 3B Instruct loaded successfully")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# ============================================================================
# DATA LOADING AND ANALYSIS
# ============================================================================

def load_summary_json(file_path: str) -> Dict[str, Any]:
    """Load summary.json file"""
    print(f"\n📂 Loading summary data from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Successfully loaded summary data")
        print(f"  - Routes remaining: {data['routes_remaining']}")
        print(f"  - Bottlenecks identified: {len(data['top_bottlenecks'])}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in {file_path}")
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

def create_analysis_prompt(summary_data: Dict[str, Any], corridor_analysis: Dict[str, Any]) -> str:
    """Create detailed prompt for LLM analysis - STRICT JSON-ONLY OUTPUT"""
    
    prompt = f"""CRITICAL INSTRUCTION: Respond with ONLY valid JSON. Zero additional text.

Analyze this routing network and provide strategy:

NETWORK DATA:
- Routes remaining: {summary_data['routes_remaining']}
- West: {summary_data['west_corridor_usage']}%
- Center: {summary_data['center_corridor_usage']}%
- East: {summary_data['east_corridor_usage']}%
- Top bottlenecks: {json.dumps(corridor_analysis['bottleneck_hotspots'][:2])}

JSON OUTPUT REQUIRED - NO OTHER TEXT:
{{
  "strategy_decision": "route_elimination",
  "reasoning": "explanation",
  "priority_corridor": "center",
  "priority_reasoning": "explanation",
  "route_weights": {{
    "bottleneck_multiplier": 1.5,
    "corridor_concentration": 0.5,
    "utilization_factor": 1.0
  }},
  "coverage_weights": {{
    "priority_corridor_weight": 0.5,
    "secondary_corridor_weight": 0.3,
    "tertiary_corridor_weight": 0.2,
    "bottleneck_coverage": 1.0
  }},
  "implementation_notes": "actions",
  "expected_impact": "improvement"
}}"""

    return prompt

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Robust JSON extraction with multiple fallback strategies
    """
    # Strategy 1: Brace counting (most reliable)
    json_start = response.find('{')
    if json_start >= 0:
        brace_count = 0
        json_end = -1
        for i in range(json_start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > 0:
            json_str = response[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass  # Try next strategy
    
    # Strategy 2: Find largest valid JSON object
    json_start = response.find('{')
    if json_start >= 0:
        for json_end in range(len(response), json_start, -1):
            try:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
            except:
                continue
    
    raise ValueError("Could not extract valid JSON from response")

def query_llm(tokenizer, model, prompt: str, device) -> str:
    """Query Qwen model and get response"""
    print("\n🤖 Querying Qwen 2.5 3B for strategy analysis...")
    
    try:
        # Prepare input
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Generate response with optimized parameters
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1000,
                temperature=0.1,          # Very low for JSON consistency
                top_p=0.9,
                top_k=50,
                do_sample=False,          # No randomness for JSON
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract JSON
        strategy = extract_json_from_response(response)
        print("✓ Strategy analysis completed successfully")
        return strategy
            
    except Exception as e:
        print(f"❌ Error querying LLM: {e}")
        print(f"Response preview: {response[:200] if 'response' in locals() else 'N/A'}")
        raise

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
        'strategy_decision',
        'reasoning',
        'priority_corridor',
        'priority_reasoning',
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
        'route_weights',
        'coverage_weights',
        'implementation_notes',
        'expected_impact'
    ]
    
    missing_keys = [k for k in required_keys if k not in strategy]
    
    if not missing_keys:
        return True, strategy
    
    # Try to repair missing keys with defaults
    print(f"\n⚠ Missing keys: {missing_keys}")
    print("Attempting repair with sensible defaults...")
    
    defaults = {
        'strategy_decision': 'route_elimination',
        'reasoning': 'Based on corridor analysis',
        'priority_corridor': 'center',
        'priority_reasoning': 'Highest usage concentration',
        'route_weights': {
            'bottleneck_multiplier': 1.5,
            'corridor_concentration': 0.5,
            'utilization_factor': 1.0
        },
        'coverage_weights': {
            'priority_corridor_weight': 0.5,
            'secondary_corridor_weight': 0.3,
            'tertiary_corridor_weight': 0.2,
            'bottleneck_coverage': 1.0
        },
        'implementation_notes': 'Standard implementation approach',
        'expected_impact': '20-25% efficiency improvement'
    }
    
    for key in missing_keys:
        strategy[key] = defaults[key]
        print(f"  ✓ Repaired '{key}' with default value")
    
    return False, strategy  # Valid structure but used defaults

def enhance_strategy(strategy: Dict[str, Any], summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add metadata and enhance strategy with additional computed values"""
    
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
        "strategy": strategy,
        "next_stage": "camera_scoring",
        "input_summary": summary_data
    }
    
    return enhanced

def save_strategy(strategy: Dict[str, Any], output_path: str = "strategy.json"):
    """Save strategy to JSON file"""
    print(f"\n💾 Saving strategy to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(strategy, f, indent=2)
        print(f"✓ Strategy saved successfully")
        
        if 'strategy' in strategy:
            strat = strategy['strategy']
            print(f"\nStrategy Summary:")
            print(f"  - Decision: {strat.get('strategy_decision', 'N/A')}")
            print(f"  - Priority Corridor: {strat.get('priority_corridor', 'N/A')}")
            print(f"  - Route Weights: {strat.get('route_weights', 'N/A')}")
            print(f"  - Coverage Weights: {strat.get('coverage_weights', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving strategy: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(summary_json_path: str = "summary.json", output_path: str = "strategy.json"):
    """Main execution function"""
    
    print("=" * 70)
    print("AGENT 2: STRATEGY GENERATOR (IMPROVED)")
    print("=" * 70)
    
    try:
        # 1. Setup
        device = setup_device()
        tokenizer, model = load_qwen_model(device)
        
        # 2. Load data
        summary_data = load_summary_json(summary_json_path)
        corridor_analysis = analyze_corridor_distribution(summary_data)
        
        # 3. Create prompt and query LLM
        prompt = create_analysis_prompt(summary_data, corridor_analysis)
        strategy = query_llm(tokenizer, model, prompt, device)
        
        # 4. Validate and enhance
        is_valid, strategy = validate_and_repair_strategy(strategy)
        
        if not is_valid:
            print("⚠ Strategy was repaired with defaults")
        else:
            print("✓ Strategy validation passed")
        
        enhanced_strategy = enhance_strategy(strategy, summary_data)
        
        # 5. Save output
        save_strategy(enhanced_strategy, output_path)
        
        print("\n" + "=" * 70)
        print("✓ AGENT 2 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nOutput file: {output_path}")
        print("Ready for Agent 3: Camera Scoring")
        
        return enhanced_strategy
        
    except Exception as e:
        print(f"\n❌ AGENT 2 FAILED: {str(e)}")
        raise

# ============================================================================
# GOOGLE COLAB ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # For Google Colab - upload summary.json first
    
    summary_file = "/content/summary.json"  # Colab upload location
    if not os.path.exists(summary_file):
        summary_file = "summary.json"  # Local fallback
    
    output_file = "strategy.json"
    
    strategy_result = main(summary_json_path=summary_file, output_path=output_file)
