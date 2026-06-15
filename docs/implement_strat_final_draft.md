# Implementation Roadmap (Based on Final Architecture v1.0)

## Recommended Tech Stack

### Core Environment

Python 3.11+

Recommended virtual environment:

```bash
python -m venv crew_env

source crew_env/bin/activate
```

or on Windows:

```bash
crew_env\Scripts\activate
```

---

## Required Packages

### Existing Pipeline

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install pandas
pip install networkx
pip install open3d
pip install scikit-image
pip install opencv-python
pip install shapely
pip install tqdm
```

---

### CrewAI

```bash
pip install crewai
pip install crewai-tools
```

---

### LLM Integration

#### Option 1 (Recommended Initially)

Qwen3 Local

```bash
pip install ollama
```

Install Ollama:

```bash
https://ollama.com
```

Download model:

```bash
ollama pull qwen3:8b
```

Advantages:

* No API cost
* Local execution
* Easy experimentation
* Works well for strategy tasks

---

#### Option 2

GPT-4o Mini

```bash
pip install openai
```

Environment variable:

```bash
OPENAI_API_KEY=xxxxx
```

Advantages:

* Better reasoning consistency
* No local GPU required

Disadvantages:

* API costs

---

# Recommended Folder Structure

```text
project/

│
├── data/
│
├── terrain_processing/
│   ├── segmentation.py
│   ├── masking.py
│   ├── raycasting.py
│
├── path_generation/
│   ├── msmg.py
│   ├── multi_astar.py
│
├── optimization/
│   ├── scoring.py
│   ├── greedy_optimizer.py
│
├── analytics/
│   ├── heatmap.py
│   ├── bottlenecks.py
│   ├── route_statistics.py
│
├── crew/
│   ├── agents.py
│   ├── tasks.py
│   ├── crew_setup.py
│
├── config/
│   ├── weights.json
│
└── main.py
```

---

# Development Phases

## Phase 1

### Goal

Run the complete pipeline without CrewAI.

Pipeline:

```text
Terrain
      ↓

Visibility

      ↓

Route Generation

      ↓

Greedy Optimization

      ↓

Camera Placement
```

Deliverable:

```text
Static Camera Placement System
```

---

## Phase 2

### Build Terrain Intelligence Agent

This is NOT an LLM.

Implement:

```python
compute_coverage()

compute_route_heatmap()

compute_corridor_usage()

compute_bottlenecks()

compute_routes_remaining()
```

Output:

```json
{
  "coverage": 91,
  "routes_remaining": 18,
  "west_corridor_usage": 72,
  "east_corridor_usage": 15,
  "top_bottlenecks": [
    "ridge_1",
    "valley_2"
  ]
}
```

Deliverable:

```text
Terrain Intelligence Agent
```

---

## Phase 3

### Integrate CrewAI

Create:

```text
Terrain Intelligence Agent
```

and

```text
Camera Placement Strategy Agent
```

Crew Process:

```python
Process.sequential
```

Workflow:

```text
Analytics
    ↓
Strategy
```

Deliverable:

```text
CrewAI Integration
```

---

## Phase 4

### Connect Qwen3

Recommended first model:

```text
Qwen3 8B
```

via:

```python
Ollama
```

Example:

```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="qwen3:8b"
)
```

Deliverable:

```text
Local LLM Strategy Agent
```

---

## Phase 5

### Generate Weight Updates

Input:

```json
{
  "coverage": 91,
  "routes_remaining": 18,
  "west_corridor_usage": 72
}
```

Expected Output:

```json
{
  "coverage_weight": 1,
  "route_weight": 4,
  "bottleneck_weight": 8
}
```

Store:

```text
config/weights.json
```

Deliverable:

```text
Adaptive Weight Updates
```

---

## Phase 6

### Integrate With Greedy Optimizer

Current:

```python
score = coverage_gain
```

Replace with:

```python
score = (
    coverage_weight * coverage_gain
    +
    route_weight * routes_removed
    +
    bottleneck_weight * bottleneck_score
)
```

Deliverable:

```text
Route-Aware Camera Placement
```

---

## Phase 7

### Periodic Strategy Reviews

Do NOT call the LLM every iteration.

Recommended:

```text
50 Simulations
        ↓
Analytics Summary
        ↓
Strategy Review
        ↓
Weight Update
```

or

```text
100 Simulations
        ↓
Analytics Summary
        ↓
Strategy Review
        ↓
Weight Update
```

Deliverable:

```text
Low-Cost CrewAI System
```

---

# Final Runtime Workflow

```text
Terrain
        ↓

Segmentation
        ↓

Masks
        ↓

Candidate Poles
        ↓

Raycasting
        ↓

Coverage Maps
        ↓

MSMG / Multi-A*
        ↓

Generated Routes
        ↓

Terrain Intelligence Agent
(No LLM)
        ↓

Route Statistics
        ↓

Camera Placement Strategy Agent
(Qwen3 / GPT-4o Mini)
        ↓

Weight Updates
        ↓

Greedy Optimizer
        ↓

Best Camera Placement
        ↓

Update Environment
        ↓

Repeat
```

---

# Model Recommendation

## Development Stage

Use:

```text
Qwen3 8B
```

Reason:

* Free
* Local
* Sufficient for strategic reasoning
* Easy CrewAI integration

---

## Production Stage

Option A:

```text
Continue with Qwen3
```

Cost:

```text
~ $0 API Cost
```

---

Option B:

```text
GPT-4o Mini
```

Only for:

```text
Strategy Reviews
```

Never for:

```text
Path Generation
Visibility
Coverage
Camera Placement
```

This keeps token usage extremely low while preserving the existing architecture.
