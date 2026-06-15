#!/usr/bin/env python3

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ==========================================================
# CONFIG
# ==========================================================

PATH_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\output\multi_astar_v2\paths_1.json"

COVERAGE_STATS = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\output\coverage_stats.txt"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\terrain_intelligence_output"

os.makedirs(OUT_DIR, exist_ok=True)

GRID_ROWS = 10
GRID_COLS = 10

ITERATION = 0
# later:
# ITERATION = 1,2,3...

# ==========================================================
# LOAD ROUTES
# ==========================================================

print("\n" + "="*60)
print("TERRAIN INTELLIGENCE AGENT")
print("="*60)

with open(PATH_JSON, "r") as f:
    routes = json.load(f)

if len(routes) == 0:
    raise Exception("No routes found.")

print(f"Routes Loaded : {len(routes)}")

# ==========================================================
# MAP SIZE
# ==========================================================

max_r = 0
max_c = 0

for route in routes:

    for r, c in route:

        max_r = max(max_r, r)
        max_c = max(max_c, c)

H = max_r + 1
W = max_c + 1

print(f"Map Height : {H}")
print(f"Map Width  : {W}")

# ==========================================================
# GRID BOTTLENECK ANALYSIS
# ==========================================================

grid_h = H // GRID_ROWS + 1
grid_w = W // GRID_COLS + 1

grid_counter = Counter()

for route in routes:

    visited_grids = set()

    for r, c in route:

        gr = r // grid_h
        gc = c // grid_w

        visited_grids.add((gr, gc))

    for cell in visited_grids:

        grid_counter[cell] += 1

heatmap = np.zeros(
    (GRID_ROWS, GRID_COLS),
    dtype=np.float32
)

for (gr, gc), freq in grid_counter.items():

    if (
        gr < GRID_ROWS and
        gc < GRID_COLS
    ):
        heatmap[gr, gc] = freq

# ==========================================================
# SAVE GRID HEATMAP
# ==========================================================

plt.figure(figsize=(8,8))

plt.imshow(
    heatmap,
    cmap="hot"
)

plt.colorbar(
    label="Routes Passing Through Grid"
)

plt.title(
    "Grid Route Density Heatmap"
)

plt.savefig(
    os.path.join(
        OUT_DIR,
        "grid_heatmap.png"
    ),
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# TOP BOTTLENECKS
# ==========================================================

top_grids = sorted(
    grid_counter.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

bottlenecks = []

for (gr, gc), freq in top_grids:

    bottlenecks.append({

        "grid": f"{gr}_{gc}",

        "grid_row": int(gr),

        "grid_col": int(gc),

        "frequency": int(freq)
    })

# ==========================================================
# CORRIDOR ANALYSIS
# ==========================================================

west_limit = W / 3
east_limit = 2 * W / 3

west_count = 0
center_count = 0
east_count = 0

for route in routes:

    for _, c in route:

        if c < west_limit:

            west_count += 1

        elif c < east_limit:

            center_count += 1

        else:

            east_count += 1

total_visits = (
    west_count +
    center_count +
    east_count
)

west_pct = round(
    100 * west_count / total_visits,
    2
)

center_pct = round(
    100 * center_count / total_visits,
    2
)

east_pct = round(
    100 * east_count / total_visits,
    2
)

# ==========================================================
# CORRIDOR HEATMAP
# ==========================================================

corridor_heatmap = np.array([
    [
        west_pct,
        center_pct,
        east_pct
    ]
])

plt.figure(figsize=(8,3))

plt.imshow(
    corridor_heatmap,
    cmap="hot",
    aspect="auto"
)

plt.xticks(
    [0,1,2],
    ["WEST","CENTER","EAST"]
)

plt.yticks([])

plt.colorbar(
    label="Route Density (%)"
)

plt.title(
    "Corridor Route Density"
)

plt.savefig(
    os.path.join(
        OUT_DIR,
        "corridor_heatmap.png"
    ),
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

summary = {

    "iteration": ITERATION,

    "routes_remaining":
        int(len(routes)),

    "west_corridor_usage":
        west_pct,

    "center_corridor_usage":
        center_pct,

    "east_corridor_usage":
        east_pct,

    "top_bottlenecks":
        bottlenecks
}

# ==========================================================
# ADD COVERAGE IF AVAILABLE
# ==========================================================

if ITERATION > 0 and os.path.exists(COVERAGE_STATS):

    print("Loading coverage stats...")

    txt = open(
        COVERAGE_STATS,
        "r"
    ).read()

    coverage_match = re.search(
        r"Covered by cones\s*:\s*\d+\s*\(([\d.]+)%\)",
        txt
    )

    blind_match = re.search(
        r"Blind spot area\s*:\s*\d+\s*\(([\d.]+)%\)",
        txt
    )

    if coverage_match:

        summary[
            "coverage_percent"
        ] = float(
            coverage_match.group(1)
        )

    if blind_match:

        summary[
            "blind_spot_percent"
        ] = float(
            blind_match.group(1)
        )

# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary_path = os.path.join(
    OUT_DIR,
    "summary.json"
)

with open(summary_path, "w") as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print("\nSaved summary.json")

print("\nSummary:\n")

print(
    json.dumps(
        summary,
        indent=2
    )
)

print("\n" + "="*60)
print("AGENT 1 COMPLETE")
print("="*60)
