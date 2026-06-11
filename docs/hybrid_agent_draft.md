# Hybrid LLM + RL Framework for Intruder-Defender Camera Placement Optimization

## Goal

Develop a camera placement optimization framework that combines the reasoning capabilities of Large Language Models (LLMs) with the learning capabilities of Reinforcement Learning (RL).

The framework models surveillance planning as a game between:

* Intruder
* Defender

The objective is to identify camera placements that maximize surveillance effectiveness while minimizing deployment cost.

---

# Motivation

Traditional camera placement methods primarily focus on visibility coverage.

However, high coverage does not necessarily imply strong security.

An environment may achieve high coverage while still allowing multiple hidden traversal routes.

This framework introduces route-aware optimization by evaluating:

* Coverage
* Intrusion opportunities
* Camera efficiency
* Detection capability

---

# Core Idea

The proposed approach combines:

### LLMs

Used as expert decision-makers during the initial stages.

Responsibilities:

* Route selection
* Camera placement selection
* Strategic reasoning

### RL Agents

Used to learn from LLM decisions and eventually replace them.

Responsibilities:

* Fast decision making
* Adaptive strategies
* Large-scale simulation

---

# System Components

## Environment

The environment maintains:

* Terrain
* Obstacles
* Existing cameras
* Visibility maps
* Candidate routes
* Route history
* Camera history

Responsibilities:

* Generate routes
* Update visibility
* Validate camera placements
* Calculate rewards
* Store gameplay data

---

## Path Generation Layer

Path generation remains independent from the agents.

Possible algorithms:

* A*
* MSMG
* K-Shortest Paths

Output:

```text
P1
P2
P3
...
Pn
```

along with route metadata.

---

## LLM Intruder

The Intruder does not generate routes.

Instead, it selects from existing routes.

Input:

* Candidate routes
* Route metadata
* Visibility information
* Previous outcomes

Output:

```json
{
  "chosen_path": 2
}
```

Objective:

* Reach destination
* Minimize visibility exposure
* Avoid surveillance regions

---

## LLM Defender

The Defender receives:

* Visibility map
* Existing cameras
* Chosen route
* Candidate camera locations

Output:

```json
{
  "camera_location": "C17"
}
```

Objective:

* Increase coverage
* Reduce intrusion routes
* Improve detection probability
* Minimize camera count

---

# Data Collection Phase

During LLM gameplay, store:

```python
(
 state,
 action,
 reward,
 next_state
)
```

for both agents.

Example:

```python
state = current_environment

action = selected_camera

reward = route_reduction_score
```

The collected data forms the initial training dataset.

---

# RL Training Phase

Two RL agents are trained:

## RL Intruder

Learns:

* Route selection
* Risk assessment
* Adaptive traversal strategies

Reward examples:

```python
+100  Goal Reached
-10   Visibility Exposure
-100  Detected
```

---

## RL Defender

Learns:

* Camera placement strategies
* Route elimination
* Camera efficiency

Reward examples:

```python
+50  Route Eliminated
+20  Coverage Increase
-10  Additional Camera
```

---

# Training Workflow

```text
Terrain

    ↓

Path Generation

    ↓

LLM Intruder

    ↓

LLM Defender

    ↓

Gameplay Data Collection

    ↓

RL Training

    ↓

RL vs RL Simulation

    ↓

Optimized Policies
```

---

# Deployment Workflow

After sufficient training:

```text
Terrain

    ↓

Path Generator

    ↓

RL Intruder

    ↓

RL Defender

    ↓

Environment Update

    ↓

Repeat
```

The LLM is no longer required during deployment.

---

# Advantages

## LLM Phase

* Immediate expert reasoning
* Explainable decisions
* No initial training required

## RL Phase

* Faster inference
* Lower operational cost
* Continuous learning
* Scalable simulation

---

# Current Working Hypothesis

The most practical development path is:

```text
Path Generator

      ↓

LLM Intruder
      +
LLM Defender

      ↓

Generate Expert Gameplay

      ↓

Train RL Agents

      ↓

RL Intruder
      +
RL Defender

      ↓

Efficient Long-Term Surveillance Optimization
```

This approach combines the reasoning capabilities of LLMs with the efficiency and adaptability of Reinforcement Learning while preserving the existing path-planning infrastructure.
