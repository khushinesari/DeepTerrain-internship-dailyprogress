#!/usr/bin/env python3

import os
import json
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# CONFIG

TERRAIN_NPY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\terrain_mask.npy"

OBSTACLE_NPY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\obstacle_mask.npy"

SELECT_MASK = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\start_end_selection\top_bottom_selected_mask_1.npy"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\output\multi_astar_v2"

os.makedirs(OUT_DIR, exist_ok=True)

MAX_STARTS = 25
MAX_GOALS = 25

ALLOW_DIAG = True
DIAG_COST = 1.414

START_BAND_LIMIT = 200
GOAL_OFFSET = 370

RANDOM_SEED = 42

# LOAD MASKS

print("\n" + "=" * 60)
print("MULTI A* ROUTE GENERATION STARTED")
print("=" * 60)

terrain = np.load(TERRAIN_NPY).astype(bool)
obstacle = np.load(OBSTACLE_NPY).astype(bool)
select = np.load(SELECT_MASK).astype(bool)

H = max(
    terrain.shape[0],
    obstacle.shape[0],
    select.shape[0]
)

W = max(
    terrain.shape[1],
    obstacle.shape[1],
    select.shape[1]
)

def pad(arr):
    return np.pad(
        arr,
        (
            (0, H - arr.shape[0]),
            (0, W - arr.shape[1])
        ),
        constant_values=False
    )

terrain = pad(terrain)
obstacle = pad(obstacle)
select = pad(select)

print(f"Terrain Shape  : {terrain.shape}")
print(f"Obstacle Shape : {obstacle.shape}")
print(f"Select Shape   : {select.shape}")

# WALKABLE REGION

walkable = terrain & (~obstacle)

print(f"\nWalkable Cells : {np.sum(walkable):,}")

# START / GOAL EXTRACTION

start_cells = np.array([
    (r, c)
    for r, c in zip(*np.where(select))
    if r < START_BAND_LIMIT and walkable[r, c]
])

goal_cells = np.array([
    (r, c)
    for r, c in zip(*np.where(select))
    if r >= H - GOAL_OFFSET and walkable[r, c]
])

print(f"\nAvailable Starts : {len(start_cells)}")
print(f"Available Goals  : {len(goal_cells)}")

# RANDOM SAMPLING

rng = np.random.default_rng(RANDOM_SEED)

if len(start_cells) > MAX_STARTS:

    idx = rng.choice(
        len(start_cells),
        MAX_STARTS,
        replace=False
    )

    start_cells = start_cells[idx]

if len(goal_cells) > MAX_GOALS:

    idx = rng.choice(
        len(goal_cells),
        MAX_GOALS,
        replace=False
    )

    goal_cells = goal_cells[idx]

print(f"\nSampled Starts : {len(start_cells)}")
print(f"Sampled Goals  : {len(goal_cells)}")

total_jobs = len(start_cells) * len(goal_cells)

print(f"Total A* Runs  : {total_jobs}")

# A*

MOVES = [
    (-1,0),
    (1,0),
    (0,-1),
    (0,1)
]

if ALLOW_DIAG:

    MOVES += [
        (-1,-1),
        (-1,1),
        (1,-1),
        (1,1)
    ]

def heuristic(a, b):

    return (
        abs(a[0] - b[0])
        +
        abs(a[1] - b[1])
    )

def astar(start, goal):

    pq = []

    heapq.heappush(
        pq,
        (0, start)
    )

    parent = {
        start: None
    }

    gscore = {
        start: 0
    }

    while pq:

        _, current = heapq.heappop(pq)

        if current == goal:

            path = []

            node = current

            while node is not None:

                path.append(node)

                node = parent[node]

            return path[::-1]

        for dr, dc in MOVES:

            nr = current[0] + dr
            nc = current[1] + dc

            if not (
                0 <= nr < H
                and
                0 <= nc < W
            ):
                continue

            if not walkable[nr, nc]:
                continue

            step = (
                1.0
                if dr == 0 or dc == 0
                else DIAG_COST
            )

            neighbour = (nr, nc)

            new_g = gscore[current] + step

            if (
                neighbour not in gscore
                or
                new_g < gscore[neighbour]
            ):

                gscore[neighbour] = new_g

                parent[neighbour] = current

                f = (
                    new_g
                    +
                    heuristic(
                        neighbour,
                        goal
                    )
                )

                heapq.heappush(
                    pq,
                    (f, neighbour)
                )

    return None

# GENERATE ROUTES

print("\nGenerating Routes...\n")

routes = []

job_counter = 0

for start in start_cells:

    for goal in goal_cells:

        job_counter += 1

        if job_counter % 25 == 0:

            print(
                f"A* Progress: "
                f"{job_counter}/{total_jobs}"
            )

        path = astar(
            tuple(start),
            tuple(goal)
        )

        if path is not None:

            routes.append(path)

print(
    f"\nCandidate Routes Found: "
    f"{len(routes)}"
)

# REMOVE DUPLICATES

unique_routes = {}

for route in routes:

    key = tuple(route)

    if key not in unique_routes:

        unique_routes[key] = route

routes = list(unique_routes.values())

print(
    f"Unique Routes: "
    f"{len(routes)}"
)

# SAVE ROUTES

routes_py = [
    [
        [int(r), int(c)]
        for r, c in route
    ]
    for route in routes
]

with open(
    os.path.join(
        OUT_DIR,
        "paths_1.json"
    ),
    "w"
) as f:

    json.dump(
        routes_py,
        f,
        indent=2
    )

np.save(
    os.path.join(
        OUT_DIR,
        "paths_1.npy"
    ),
    np.array(
        routes_py,
        dtype=object
    ),
    allow_pickle=True
)

print("Saved paths_1.json")
print("Saved paths_1.npy")

# VISUALIZATION

print("\nGenerating Visualization...")

bg = np.full(
    (H, W, 3),
    [128,128,128],
    np.uint8
)

bg[terrain] = [0,255,0]
bg[obstacle] = [255,0,0]

canvas = bg.copy()

for route in routes_py:

    for r, c in route:

        canvas[r, c] = [0,0,0]

for route in routes_py:

    sr, sc = route[0]
    gr, gc = route[-1]

    canvas[sr, sc] = [0,255,255]
    canvas[gr, gc] = [255,255,0]

fig, ax = plt.subplots(
    figsize=(10,10)
)

ax.imshow(canvas)

ax.axis("off")

legend = [

    Patch(
        color="green",
        label="Terrain"
    ),

    Patch(
        color="red",
        label="Obstacle"
    ),

    Patch(
        color="black",
        label="Routes"
    ),

    Patch(
        color="cyan",
        label="Start"
    ),

    Patch(
        color="yellow",
        label="Goal"
    )
]

ax.legend(
    handles=legend,
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "paths_colored_1.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved paths_colored_1.png")

print("\n" + "=" * 60)
print("MULTI A* COMPLETED")
print(f"Routes Generated : {len(routes)}")
print("=" * 60)
