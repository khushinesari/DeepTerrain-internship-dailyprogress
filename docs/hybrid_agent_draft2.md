# Incremental Development Framework for Intruder-Defender Camera Placement Optimization

## Objective

Develop an intelligent camera placement system that can progressively evolve from rule-based decision making to AI-assisted optimization while maintaining compatibility with the existing visibility and path-planning infrastructure.

The proposed development path consists of four phases:

1. Rule-Based Agents
2. Open-Source LLM Agents
3. RL Training Using LLM Gameplay
4. Fully RL-Based Deployment

This approach minimizes development risk while allowing gradual introduction of AI components.

---

# Core Principle

Path generation remains independent from the agents.

Existing algorithms continue to generate valid routes:

* A*
* MSMG
* K-Shortest Paths

Agents are responsible only for decision making.

```text
Path Generator
      ↓
Candidate Routes
      ↓
Agent Decision
```

---

# Phase 1: Rule-Based Intruder and Defender

## Goal

Build the complete game loop without introducing AI models.

This phase validates:

* Environment design
* Visibility updates
* Route evaluation
* Camera placement pipeline
* Scoring framework

---

## Rule-Based Intruder

The Intruder evaluates generated routes using predefined scoring functions.

Example:

```python
route_score = (
    hidden_percentage
    - exposure_percentage
    - distance_penalty
)
```

Select:

```python
best_route = max(candidate_routes)
```

Objective:

* Minimize exposure
* Minimize route cost

---

## Rule-Based Defender

The Defender evaluates candidate camera locations.

Example:

```python
camera_score = (
    coverage_gain
    + routes_removed
    - camera_cost
)
```

Select:

```python
best_camera = max(candidate_locations)
```

Objective:

* Increase coverage
* Reduce intrusion opportunities
* Minimize camera count

---

## Expected Outcome

A complete working simulation:

```text
Generate Routes
      ↓
Choose Route
      ↓
Place Camera
      ↓
Update Visibility
      ↓
Repeat
```

No AI models required.

---

# Phase 2: Open-Source LLM-Based Agents

## Goal

Replace rule-based decision making with reasoning-based agents.

Recommended Model:

Qwen3

Reasons:

* Strong reasoning capability
* Open source
* Local deployment possible
* No API costs

---

## LLM Intruder

Input:

* Candidate routes
* Route metadata
* Visibility statistics
* Previous outcomes

Example:

```json
{
  "route_1": {
    "exposure": 75,
    "length": 100
  },
  "route_2": {
    "exposure": 20,
    "length": 130
  }
}
```

Output:

```json
{
  "chosen_route": 2
}
```

---

## LLM Defender

Input:

* Current visibility map
* Existing cameras
* Candidate camera locations
* Intruder selection

Output:

```json
{
  "camera_location": "C17"
}
```

---

## Expected Benefit

The agents begin making strategic decisions rather than following fixed scoring rules.

Examples:

* Identifying bottlenecks
* Protecting likely future routes
* Adapting to previous outcomes

---

# Phase 3: Gameplay Data Collection

## Goal

Use LLM-generated decisions to create a training dataset.

For every decision store:

```python
(
 state,
 action,
 reward,
 next_state
)
```

Example:

```python
{
  "visibility": visibility_map,
  "routes": candidate_routes,
  "action": selected_route,
  "reward": reward_value
}
```

Collected data represents expert gameplay.

---

## Data Sources

Intruder:

* Route selections
* Success rates
* Exposure statistics

Defender:

* Camera placements
* Coverage gains
* Route reductions

---

# Phase 4: Reinforcement Learning

## Goal

Train RL agents using gameplay collected from previous phases.

---

## RL Intruder

Learns:

* Route selection
* Risk assessment
* Adaptive behavior

Example reward:

```python
reward =
+100 if goal_reached
-10 for exposure
-100 if detected
```

---

## RL Defender

Learns:

* Camera placement strategies
* Coverage optimization
* Route denial

Example reward:

```python
reward =
+50 routes_removed
+20 coverage_gain
-10 camera_added
```

---

# RL Self-Play

After initial training:

```text
RL Intruder
      vs
RL Defender
```

Run thousands of simulations.

Agents continue improving through experience.

---

# Final Deployment Architecture

Once RL performance exceeds LLM performance:

```text
Terrain
      ↓

Path Generation
(A*, MSMG, KSP)

      ↓

RL Intruder

      ↓

RL Defender

      ↓

Visibility Update

      ↓

Repeat
```

The LLM layer is removed entirely.

---

# Advantages of the Incremental Approach

## Phase 1

* Fast implementation
* Easy debugging
* Validates environment design

## Phase 2

* Introduces strategic reasoning
* No expensive API dependency
* Local deployment using Qwen3

## Phase 3

* Generates high-quality training data
* Captures expert-like decisions

## Phase 4

* Near-zero runtime cost
* Fast inference
* Continuous learning
* Scalable simulations

---

# Development Workflow

```text
PHASE 1

Rule-Based Intruder
+
Rule-Based Defender

        ↓

Validate Environment

        ↓

PHASE 2

Qwen3 Intruder
+
Qwen3 Defender

        ↓

Generate Expert Gameplay

        ↓

PHASE 3

Collect
(State, Action, Reward)

        ↓

Train RL Agents

        ↓

PHASE 4

RL Intruder
+
RL Defender

        ↓

Fully Autonomous Optimization
```

---

# Current Working Hypothesis

The most practical path is to begin with rule-based agents, introduce open-source LLM reasoning through Qwen3, collect expert gameplay data, and eventually transition to Reinforcement Learning agents for long-term deployment.

This allows rapid prototyping in the early stages while enabling the development of a scalable and cost-effective intelligent surveillance optimization system.
