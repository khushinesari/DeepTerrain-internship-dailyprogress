#  V2: Terrain-Aware Multi-Agent Surveillance Optimization

## Overview

V2 Architecture is an autonomous surveillance optimization framework that incrementally places surveillance cameras over complex terrain using a combination of classical computer vision, path planning, 3D visibility analysis, and Large Language Model (LLM)-based strategic reasoning.

Unlike conventional camera placement systems that optimize coverage in a single step, v2 operates iteratively. After each camera placement, the environment is updated, detected intrusion routes are removed, and the remaining terrain is reanalyzed before selecting the next camera.

---

# Pipeline

## 1. Terrain Segmentation

Input:

* Point Cloud (.pcd)

Output:

* Ground Point Cloud
* Obstacle Point Cloud

---

## 2. Digital Elevation Model Generation

The segmented ground is converted into a Digital Elevation Model (DEM) together with slope and terrain-normal information. These products enable terrain-aware visibility computation.

Outputs:

* DEM
* Slope Map
* Surface Normals

---

## 3. Camera Candidate Generation

Potential camera installation locations are generated over valid terrain.

Camera parameters:

* Pole Height: 10 m
* Tilt: 5°
* Horizontal FoV: 30°
* Maximum Range: 150 m

Approximately 80,000 candidate locations are produced.

---

## 4. Route Generation

Potential intrusion routes are generated using:

* Multi-Source Multi-Goal A*
* MSMG
* K-Shortest Paths

Output:

* paths_1.json

---

# Multi-Agent Optimization Loop

Each iteration consists of the following stages.

## Agent 1: Terrain Intelligence

Inputs:

* Remaining routes

Outputs:

* Route density heatmap
* Corridor statistics
* Bottleneck regions
* Summary JSON

---

## Agent 2: Strategic Planner

Inputs:

* Agent 1 summary

Outputs:

* Route importance weight
* Coverage importance weight
* Priority corridor
* Camera placement strategy

The LLM does not invent terrain information. It only reasons over structured outputs produced by Agent 1.

---

## Candidate Selector

Filters approximately 80,000 candidate poles using:

* Priority corridor
* Bottleneck regions
* Grid metadata

Result:

* Approximately 500 candidates

---

## Camera Scoring

Each candidate is evaluated using:

* Distance to bottleneck
* Corridor priority
* Route importance
* Coverage importance

Top 50 candidates are retained.

---

## 3D Visibility Engine

Every candidate is evaluated using terrain-aware viewshed analysis.

The visibility engine considers:

* Pole height
* Camera tilt
* Terrain elevation
* Obstacles
* Line-of-sight occlusion

The highest-scoring camera is selected.

---

## Environment Update

After selecting a camera:

* Camera is added to history
* Global visibility map is updated
* Detected routes are removed
* Remaining routes are written to paths_iter_k.json

The optimization continues until:

* Route threshold reached
* Coverage threshold achieved
* Camera budget exhausted

---

# Key Features

* Terrain-aware surveillance optimization
* Multi-agent decision making
* LLM-guided strategic reasoning
* Physics-based 3D visibility
* Incremental environment updates
* Dynamic route elimination

---

# Future goals 

* Multi-camera cooperative optimization
* Dynamic moving intruders and defender(camera)
* Uncertainty-aware path prediction

---

# Technologies

* Python
* NumPy
* OpenCV
* Open3D
* Matplotlib
* A*
* Hugging Face Transformers
* Digital Elevation Models
* Viewshed Analysis

The proposed framework combines:

* Computer Vision
* Terrain Analysis
* Path Planning
* Multi-Agent Systems
* Large Language Models
* 3D Geometric Visibility

to create a closed-loop surveillance optimization system capable of adaptive camera placement over complex terrains.
