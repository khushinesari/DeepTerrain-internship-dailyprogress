# LLM-Assisted Route-Aware Camera Placement Optimization

## Objective

Develop a camera placement optimization framework that combines traditional visibility analysis with periodic AI-assisted strategy refinement.

The goal is to improve camera placement decisions by considering not only terrain coverage, but also the routes most likely to be used by an intruder.

Unlike a fully agent-based system, the proposed framework uses an LLM as a strategic advisor rather than a decision-maker during every iteration.

This significantly reduces computational cost and LLM token usage while retaining the ability to discover useful surveillance patterns.

---

# Motivation

Current pole placement methods typically optimize for:

* Visibility coverage
* Terrain coverage
* Camera count

However, high visibility coverage does not necessarily imply strong surveillance effectiveness.

Example:

```text
Coverage = 92%

Remaining Intrusion Routes = 15
```

Although coverage is high, several viable intrusion paths may still exist.

This suggests that camera placement quality should also consider:

* Route visibility
* Route frequency
* Route elimination capability
* Detection probability

---

# Core Principle

Existing algorithms continue to handle path generation.

Examples:

* A*
* MSMG
* K-Shortest Paths

The proposed framework does not replace these algorithms.

Instead:

```text
Path Generation
        ↓
Route Analysis
        ↓
Camera Scoring
        ↓
Periodic LLM Review
        ↓
Improved Camera Scoring
```

The LLM acts as a strategic advisor rather than participating in every decision.

---

# System Components

## Environment

The environment maintains:

* Terrain
* Obstacles
* Existing cameras
* Visibility maps
* Route history
* Camera placement history

Responsibilities:

* Generate routes
* Update visibility
* Calculate coverage
* Store simulation statistics

---

## Route Generation Layer

Generate candidate intrusion routes using existing algorithms.

Possible methods:

* A*
* MSMG
* K-Shortest Paths

Output:

```text
Route 1
Route 2
Route 3
...
Route N
```

Each route contains metadata such as:

* Length
* Exposure percentage
* Hidden percentage
* Corridor information

---

# Route Heatmap Generation

The system records how frequently each terrain cell appears across generated routes.

Example:

```python
route_heatmap = np.zeros(grid.shape)

for path in paths:
    for cell in path:
        route_heatmap[cell] += 1
```

Result:

```text
1  1  2  1
2  5  8  2
1  4 15  1
```

Higher values indicate terrain regions frequently used by intrusion routes.

---

# Camera Scoring

Traditional visibility-based scoring:

```python
score = coverage_gain
```

Proposed route-aware scoring:

```python
score = (
    coverage_gain
    + route_importance
)
```

Example:

```python
def score_pole(pole):

    visible_cells = visibility(pole)

    coverage_gain = len(visible_cells)

    route_score = sum(
        route_heatmap[cell]
        for cell in visible_cells
    )

    return (
        coverage_gain
        + 5 * route_score
    )
```

This rewards cameras that observe frequently used routes rather than only maximizing raw visibility.

---

# Simulation Phase

Run repeated simulations.

Example:

```text
Generate Routes
      ↓
Evaluate Cameras
      ↓
Place Camera
      ↓
Update Visibility
      ↓
Repeat
```

Collect:

* Coverage gain
* Route reductions
* Camera placements
* Intruder success rates

---

# Metrics Collection

For each simulation record:

```python
{
    "camera_id": camera_id,
    "coverage_gain": coverage_gain,
    "routes_removed": routes_removed,
    "intruder_success": success_rate
}
```

Example:

```text
Pole 31

Coverage Gain = 5%
Routes Removed = 12
```

```text
Pole 52

Coverage Gain = 8%
Routes Removed = 1
```

This allows evaluation beyond coverage alone.

---

# Periodic LLM Analysis

Instead of querying the LLM every round, the system periodically generates a compact summary.

Example:

```json
{
    "west_corridor_usage": 78,
    "east_corridor_usage": 15,
    "best_camera_locations": [31,18,7],
    "average_intruder_success": 0.24
}
```

The summary is sent to the LLM for strategic analysis.

Example prompt:

```text
Analyze the following surveillance statistics.

West corridor usage: 78
East corridor usage: 15

Most successful camera:
Pole 31

Suggest improvements to camera prioritization.
```

Example output:

```text
West corridor appears to be a critical bottleneck.

Increase priority for cameras covering
the western ridge.

Reduce emphasis on open-field coverage.
```

---

# Knowledge Base

Store LLM recommendations as reusable rules.

Example:

```python
knowledge_base = {

    "west_corridor_weight": 2.5,

    "open_field_weight": 0.5,

    "ridge_priority": 3.0
}
```

These rules are applied during future camera scoring.

---

# Updated Camera Scoring

The scoring function evolves over time.

Example:

```python
def score_pole(pole):

    coverage_gain = ...

    route_score = ...

    bottleneck_score = ...

    return (
        coverage_gain
        + 5 * route_score
        + 10 * bottleneck_score
    )
```

The LLM influences the weights but does not directly place cameras.

---

# Cost Optimization

Traditional LLM Agent Design:

```text
Route Selection
      ↓
LLM

Camera Selection
      ↓
LLM

Every Round
```

Result:

```text
Thousands of LLM calls
```

---

Proposed Design:

```text
Simulation
      ↓
Statistics Collection
      ↓
Periodic LLM Review
```

Result:

```text
Tens of LLM calls
instead of
thousands
```

This significantly reduces token usage and operational cost.

---

# Open-Source LLM Variant

The framework can also use local open-source models such as:

* Qwen3
* Llama 3
* DeepSeek

Advantages:

* No API costs
* Unlimited analysis runs
* Full local deployment

In this setup, strategic analysis can be performed more frequently without significant cost concerns.

---

# Development Workflow

```text
Terrain
      ↓

Visibility Analysis
      ↓

Path Generation
(A*, MSMG, KSP)

      ↓

Route Heatmap

      ↓

Route-Aware Camera Scoring

      ↓

Simulation Metrics

      ↓

Periodic LLM Analysis

      ↓

Knowledge Base Update

      ↓

Improved Camera Scoring

      ↓

Repeat
```

---

# Current Working Hypothesis

The most practical approach is to retain the existing visibility and path-planning infrastructure while introducing a route-aware scoring framework supported by periodic LLM analysis.

This enables AI-assisted optimization without requiring continuous LLM inference, reinforcement learning, or large-scale training pipelines.

The result is a lower-cost, explainable, and scalable camera placement optimization system that can evolve over time using accumulated surveillance statistics.
