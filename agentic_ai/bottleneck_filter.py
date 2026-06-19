#!/usr/bin/env python3

import os
import json
import numpy as np
import open3d as o3d

# ==========================================================
# CONFIG
# ==========================================================

SUMMARY_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\terrain_intelligence_output\summary.json"

VALID_PLY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step2\valid_camera_positions.ply"

TERRAIN_MASK = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\terrain_mask.npy"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\bottleneck_filter_output"

os.makedirs(OUT_DIR, exist_ok=True)

GRID_ROWS = 10
GRID_COLS = 10

# ==========================================================
# LOAD SUMMARY
# ==========================================================

print("\n" + "="*60)
print("BOTTLENECK FILTER STARTED")
print("="*60)

with open(SUMMARY_JSON, "r") as f:
    summary = json.load(f)

top_bottlenecks = summary["top_bottlenecks"]

print(
    f"\nLoaded {len(top_bottlenecks)} bottlenecks"
)

# ==========================================================
# LOAD TERRAIN MASK
# ==========================================================

terrain = np.load(TERRAIN_MASK)

H, W = terrain.shape

print(
    f"Terrain Size : {H} x {W}"
)

grid_h = H // GRID_ROWS + 1
grid_w = W // GRID_COLS + 1

# ==========================================================
# LOAD VALID CAMERA POSITIONS
# ==========================================================

print("\nLoading valid poles...")

pcd = o3d.io.read_point_cloud(
    VALID_PLY
)

points = np.asarray(
    pcd.points
)

print(
    f"Total poles : {len(points)}"
)

# ==========================================================
# WORLD COORDINATE BOUNDS
# ==========================================================

min_x = points[:,0].min()
max_x = points[:,0].max()

min_y = points[:,1].min()
max_y = points[:,1].max()

print(
    f"\nWorld Bounds:"
)

print(
    f"X : {min_x:.2f} -> {max_x:.2f}"
)

print(
    f"Y : {min_y:.2f} -> {max_y:.2f}"
)

# ==========================================================
# WORLD -> MASK MAPPING
# ==========================================================

def world_to_mask(x, y):

    col = int(
        ((x - min_x) /
        (max_x - min_x))
        * (W - 1)
    )

    row = int(
        ((y - min_y) /
        (max_y - min_y))
        * (H - 1)
    )

    return row, col

# ==========================================================
# BUILD TARGET GRID SET
# ==========================================================

target_grids = set()

for b in top_bottlenecks:

    gr = b["grid_row"]
    gc = b["grid_col"]

    # include neighbors

    for dr in [-1, 0, 1]:

        for dc in [-1, 0, 1]:

            ngr = gr + dr
            ngc = gc + dc

            if (
                0 <= ngr < GRID_ROWS
                and
                0 <= ngc < GRID_COLS
            ):
                target_grids.add(
                    (ngr, ngc)
                )

print(
    f"\nTarget grids : {len(target_grids)}"
)

# ==========================================================
# FILTER POLES
# ==========================================================

filtered = []

for idx, (x, y, z) in enumerate(points):

    row, col = world_to_mask(x, y)

    gr = row // grid_h
    gc = col // grid_w

    if (gr, gc) in target_grids:

        filtered.append({

            "pole_id": int(idx),

            "x": float(x),

            "y": float(y),

            "z": float(z),

            "grid_row": int(gr),

            "grid_col": int(gc)
        })

print(
    f"\nFiltered poles : {len(filtered)}"
)

# ==========================================================
# SAVE JSON
# ==========================================================

json_path = os.path.join(
    OUT_DIR,
    "candidate_poles.json"
)

with open(json_path, "w") as f:

    json.dump(
        filtered,
        f,
        indent=2
    )

print(
    f"\nSaved:"
)

print(json_path)

# ==========================================================
# SAVE NPY
# ==========================================================

pole_array = np.array([

    [
        p["pole_id"],
        p["x"],
        p["y"],
        p["z"]
    ]

    for p in filtered

])

npy_path = os.path.join(
    OUT_DIR,
    "candidate_poles.npy"
)

np.save(
    npy_path,
    pole_array
)

print(npy_path)

print("\n" + "="*60)
print("BOTTLENECK FILTER COMPLETE")
print("="*60)
