# CrewAI-Assisted Route-Aware Camera Placement Framework

## Objective

Develop an AI-assisted camera placement framework that improves surveillance effectiveness by combining:

* Existing terrain analysis
* Existing visibility computation
* Existing route generation algorithms
* Lightweight CrewAI orchestration

The objective is to:

* Maximize terrain coverage
* Minimize surviving intrusion routes
* Minimize the number of cameras deployed
* Maintain low LLM usage and operational cost

The proposed framework does **not** replace existing visibility or path-planning algorithms. Instead, it introduces a strategic reasoning layer that continuously improves camera placement priorities.

---

# Core Design Principle

The system separates responsibilities between algorithms and AI.

```text
Algorithms
=
Geometry + Visibility + Path Planning

CrewAI
=
Strategy + Decision Prioritization
```

Existing algorithms remain responsible for:

* Terrain analysis
* Visibility computation
* Coverage estimation
* Route generation
* Camera placement optimization

CrewAI is responsible for:

* Pattern analysis
* Route behavior analysis
* Bottleneck identification
* Camera scoring refinement

---

# Complete Workflow

```text
Point Cloud / DEM
        ↓

Ground & Obstacle Segmentation
        ↓

Terrain Mask
Obstacle Mask
        ↓

Valid Pole Generation
        ↓

Visibility Analysis
(Raycasting)
        ↓

FoV Mask
Coverage Maps
        ↓

Route Generation
(MSMG / Multi-A*)
        ↓

Generated Intrusion Routes
        ↓

Terrain Intelligence Agent
(Non-LLM)
        ↓

Route Statistics
Heatmaps
Bottleneck Analysis
        ↓

Camera Placement Strategy Agent
(LLM)
        ↓

Scoring Weight Updates
        ↓

Greedy Camera Optimizer
        ↓

Best Camera Placement
        ↓

Update Visibility
        ↓

Generate New Routes
        ↓

Repeat
```

---

# Phase 1: Terrain Processing

## Inputs

* Point Cloud (.pcd)
* Map
* Terrain Data

The system first separates:

```text
Ground
Obstacle
```

Output:

```text
Terrain Mask
Obstacle Mask
```

These masks define:

* Where intruders can move
* Where cameras can be placed
* Which regions are blocked

---

# Phase 2: Candidate Pole Generation

The framework identifies valid camera locations.

Example:

```text
Valid Pole Locations

P1
P2
P3
...
Pn
```

Requirements:

* Stable terrain
* Valid placement area
* Outside obstacle regions

---

# Phase 3: Visibility Analysis

Each candidate pole is evaluated using raycasting.

For every candidate pole:

```text
Pole
        ↓

Raycasting
        ↓

Visible Cells
```

Output:

```text
Coverage Gain
Visibility Map
FoV Mask
```

The result indicates how much terrain can be observed from each candidate location.

---

# Phase 4: Route Generation

The intrusion planning layer generates possible routes.

Possible algorithms:

* MSMG
* Multi-A*
* K-Shortest Paths

Example:

```text
Route 1
Route 2
Route 3
...
Route 100
```

Each route represents a potential intruder path through the terrain.

---

# Terrain Intelligence Agent (Agent 1)

## Type

Non-LLM Agent

## Purpose

Transform large numbers of generated routes into meaningful statistics.

The agent uses existing Python code and analytical functions.

No LLM calls are required.

---

## Why Agent 1 Exists

Suppose route generation produces:

```text
P1
P2
P3
...
P100
```

Each route may contain hundreds of coordinates.

Example:

```python
[(10,20),
 (11,20),
 (12,21),
 ...]
```

Sending all route coordinates to an LLM would be:

* Expensive
* Slow
* Unnecessary

Instead, Agent 1 summarizes route behavior.

---

## Heatmap Generation

Agent 1 creates a route heatmap.

For every route:

```python
for cell in route:
    heatmap[cell] += 1
```

Meaning:

```text
Cell Value = Number of Routes
Passing Through That Cell
```

Example:

```text
1   2   5   8

2  15  40  10

3  20  80  12
```

A value of:

```text
80
```

means:

```text
80 routes passed here
```

This reveals areas frequently used by intruders.

---

## Statistics Computed

Agent 1 computes:

```python
coverage

routes_remaining

route_heatmap

corridor_usage

route_frequency

bottlenecks
```

Example output:

```json
{
  "coverage": 91,
  "routes_remaining": 18,
  "west_corridor_usage": 72,
  "east_corridor_usage": 15,
  "north_corridor_usage": 9,
  "top_bottlenecks": [
    "ridge_1",
    "valley_2"
  ]
}
```

This converts:

```text
100 Routes
        ↓
1 Statistical Summary
```

---

# Camera Placement Strategy Agent (Agent 2)

## Type

LLM Agent

## Purpose

Analyze route statistics and recommend placement priorities.

This agent does not perform:

* Path generation
* Visibility analysis
* Raycasting
* Camera placement

Instead, it acts as a strategic reviewer.

---

## What Agent 2 Receives

Agent 2 never receives:

* Route coordinates
* Map data
* Visibility maps
* Raw path geometry

Instead, it receives:

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

This keeps token usage extremely low.

---

## What Agent 2 Actually Analyzes

Agent 2 does not analyze:

```text
Route 17
Route 42
Route 93
```

Instead it analyzes:

```text
Patterns Across All Routes
```

Examples:

### Pattern 1

Observation:

```text
72% of successful routes
use the western corridor
```

Recommendation:

```text
Increase priority for
western bottlenecks
```

---

### Pattern 2

Observation:

```text
Coverage = 95%

Routes Remaining = 12
```

Recommendation:

```text
Reduce coverage emphasis

Increase route elimination emphasis
```

---

### Pattern 3

Observation:

```text
Pole 31 repeatedly blocks
large numbers of routes
```

Recommendation:

```text
Prioritize terrain features
similar to Pole 31
```

---

## Strategy Agent Output

Example:

```json
{
  "coverage_weight": 1,
  "route_weight": 4,
  "bottleneck_weight": 8
}
```

These values are used to update camera scoring.

---

# Camera Placement Optimization

The optimizer remains algorithmic.

The LLM never directly places cameras.

---

## Traditional Scoring

```python
score = coverage_gain
```

This prioritizes only visibility.

---

## Route-Aware Scoring

```python
score = (
    coverage_weight * coverage_gain
    +
    route_weight * routes_removed
    +
    bottleneck_weight * bottleneck_score
)
```

This balances:

* Coverage
* Route elimination
* Bottleneck protection

---

# Camera Selection

Every candidate pole receives a score.

Example:

```python
best_pole = argmax(candidate_scores)
```

The highest scoring pole is selected.

The camera is then placed.

---

# Environment Update

After placement:

```text
Visibility Updates
        ↓

Coverage Updates
        ↓

Route Regeneration
```

New intrusion routes are generated.

The process repeats.

---

# Cost Optimization Strategy

The LLM should not run every iteration.

Avoid:

```text
Every Route
        ↓
LLM Analysis
```

Preferred:

```text
50–100 Simulations
        ↓
Analytics Summary
        ↓
Strategy Review
        ↓
Weight Update
```

This dramatically reduces token usage while preserving strategic adaptation.

---

# Final Output

Example:

```json
{
  "camera_locations": [
    [320,110],
    [540,220],
    [700,340]
  ],

  "coverage": 96.2,

  "routes_remaining": 1,

  "cameras_used": 3
}
```

---

# Current Working Hypothesis

The proposed framework uses existing terrain, visibility, and route-planning algorithms for all geometric computations while employing CrewAI as a lightweight strategic layer.

The Terrain Intelligence Agent converts route behavior into compact statistics, while the Camera Placement Strategy Agent analyzes these statistics and updates camera-scoring priorities.

This enables route-aware camera placement with minimal LLM usage, low operational cost, and strong compatibility with the existing DeepTerrain pipeline.
