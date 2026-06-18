#!/usr/bin/env python3

import os
import json
import numpy as np
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

TOP50_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\camera_scoring_output\top50_candidates.json"

PATHS_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\multi_astar_v2\paths_1.json"

TERRAIN_MASK = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\terrain_mask.npy"

OBSTACLE_MASK = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\obstacle_mask.npy"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\visibility_output"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

MAX_RANGE = 150

FOV_DEG = 30

AZIMUTH_STEPS = 12

# ==========================================================
# LOAD
# ==========================================================

print("\n" + "="*60)
print("VISIBILITY EVALUATOR")
print("="*60)

with open(TOP50_JSON) as f:
    candidates = json.load(f)

with open(PATHS_JSON) as f:
    routes = json.load(f)

terrain_mask = np.load(TERRAIN_MASK)

obstacle_mask = np.load(OBSTACLE_MASK)

H, W = terrain_mask.shape

# ==========================================================
# SIMPLE FOV
# ==========================================================

def build_fov_mask(
    row,
    col,
    azimuth_deg
):

    mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    azimuth = np.radians(
        azimuth_deg
    )

    half_fov = np.radians(
        FOV_DEG / 2
    )

    for r in range(
        max(0, row-MAX_RANGE),
        min(H, row+MAX_RANGE)
    ):

        for c in range(
            max(0, col-MAX_RANGE),
            min(W, col+MAX_RANGE)
        ):

            dr = r-row
            dc = c-col

            dist = np.sqrt(
                dr*dr + dc*dc
            )

            if dist > MAX_RANGE:
                continue

            angle = np.arctan2(
                dr,
                dc
            )

            diff = np.arctan2(
                np.sin(angle-azimuth),
                np.cos(angle-azimuth)
            )

            if abs(diff) <= half_fov:

                if obstacle_mask[
                    r,
                    c
                ] == 0:

                    mask[
                        r,
                        c
                    ] = 1

    return mask

# ==========================================================
# ROUTE IMPACT
# ==========================================================

results = []

for pole in tqdm(candidates):

    row = pole["bev_row"]
    col = pole["bev_col"]

    best_score = -1

    best_result = None

    for az in range(
        0,
        360,
        int(360/AZIMUTH_STEPS)
    ):

        fov_mask = build_fov_mask(
            row,
            col,
            az
        )

        coverage_gain = int(
            np.sum(fov_mask)
        )

        routes_intersected = 0

        for route in routes:

            hit = False

            for r,c in route:

                if (
                    0 <= r < H
                    and
                    0 <= c < W
                ):

                    if fov_mask[
                        r,
                        c
                    ]:

                        hit = True
                        break

            if hit:
                routes_intersected += 1

        score = (
            routes_intersected
            +
            0.001*coverage_gain
        )

        if score > best_score:

            best_score = score

            best_result = {

                "pole_id":
                    pole["pole_id"],

                "world_x":
                    pole["world_x"],

                "world_y":
                    pole["world_y"],

                "world_z":
                    pole["world_z"],

                "bev_row":
                    row,

                "bev_col":
                    col,

                "azimuth":
                    az,

                "coverage_gain":
                    coverage_gain,

                "routes_intersected":
                    routes_intersected,

                "score":
                    round(
                        score,
                        3
                    )
            }

    results.append(
        best_result
    )

# ==========================================================
# SORT
# ==========================================================

results.sort(
    key=lambda x:
    x["score"],
    reverse=True
)

# ==========================================================
# SAVE
# ==========================================================

ranked_path = os.path.join(
    OUT_DIR,
    "visibility_ranked.json"
)

with open(
    ranked_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )

best_path = os.path.join(
    OUT_DIR,
    "best_camera.json"
)

with open(
    best_path,
    "w"
) as f:

    json.dump(
        results[0],
        f,
        indent=2
    )

# ==========================================================
# REPORT
# ==========================================================

print(
    "\nBEST CAMERA\n"
)

print(
    json.dumps(
        results[0],
        indent=2
    )
)

print(
    f"\nSaved:\n{best_path}"
)

print("\n" + "="*60)
print("VISIBILITY EVALUATION COMPLETE")
print("="*60)