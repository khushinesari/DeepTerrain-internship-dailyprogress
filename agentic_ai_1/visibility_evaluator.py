#!/usr/bin/env python3

import os
import json
import numpy as np
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

TOP50_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\camera_scoring_output\top50_candidates.json"

#PATHS_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\multi_astar_v2\paths_1.json"

TERRAIN_MASK = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\terrain_mask.npy"

OBSTACLE_MASK = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\obstacle_mask.npy"

#STRATEGY_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\agent2_output\strategy.json"
STRATEGY_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent2_output"
)
#===helper===
def get_latest_strategy():

    strategy_files = []

    for file in os.listdir(STRATEGY_DIR):

        if (
            file.startswith("strategy_iter_")
            and file.endswith(".json")
        ):

            idx = int(
                file.replace(
                    "strategy_iter_",
                    ""
                ).replace(
                    ".json",
                    ""
                )
            )

            strategy_files.append(
                (idx, file)
            )

    strategy_files.sort()

    return os.path.join(
        STRATEGY_DIR,
        strategy_files[-1][1]
    )


OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\visibility_output"

os.makedirs(OUT_DIR, exist_ok=True)

MAX_RANGE = 150
FOV_DEG = 30
AZIMUTH_STEPS = 12

# ==========================================================
# AUTO DETECT CURRENT ROUTE FILE
# ==========================================================

RUNTIME_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\runtime_state"
)

ORIGINAL_PATHS = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\multi_astar_v2"
    r"\paths_1.json"
)

PATHS_JSON = ORIGINAL_PATHS

if os.path.exists(RUNTIME_DIR):

    route_files = []

    for file in os.listdir(RUNTIME_DIR):

        if (
            file.startswith("paths_iter_")
            and
            file.endswith(".json")
        ):

            try:

                idx = int(
                    file.replace(
                        "paths_iter_",
                        ""
                    ).replace(
                        ".json",
                        ""
                    )
                )

                route_files.append(
                    (idx, file)
                )

            except:
                pass

    if len(route_files) > 0:

        route_files.sort(
            key=lambda x: x[0]
        )

        latest_file = route_files[-1][1]

        PATHS_JSON = os.path.join(
            RUNTIME_DIR,
            latest_file
        )

print(
    f"\nUsing Route File:\n{PATHS_JSON}"
)

# ==========================================================
# LOAD
# ==========================================================

print("\n" + "=" * 60)
print("VISIBILITY EVALUATOR")
print("=" * 60)

with open(TOP50_JSON, "r") as f:
    candidates = json.load(f)

with open(PATHS_JSON, "r") as f:
    routes = json.load(f)

STRATEGY_JSON = get_latest_strategy()
with open(STRATEGY_JSON, "r") as f:
    strategy = json.load(f)

route_weight = strategy["strategy"]["route_weights"]
coverage_weight = strategy["strategy"]["coverage_weights"]

print(f"\nRoute Weight    : {route_weight}")
print(f"Coverage Weight : {coverage_weight}")

terrain_mask = np.load(TERRAIN_MASK)
obstacle_mask = np.load(OBSTACLE_MASK)

H, W = terrain_mask.shape

# ==========================================================
# SIMPLE FOV
# ==========================================================

def build_fov_mask(row, col, azimuth_deg):

    mask = np.zeros((H, W), dtype=np.uint8)

    azimuth = np.radians(azimuth_deg)
    half_fov = np.radians(FOV_DEG / 2)

    for r in range(max(0, row - MAX_RANGE),
                   min(H, row + MAX_RANGE)):

        for c in range(max(0, col - MAX_RANGE),
                       min(W, col + MAX_RANGE)):

            dr = r - row
            dc = c - col

            dist = np.sqrt(dr * dr + dc * dc)

            if dist > MAX_RANGE:
                continue

            angle = np.arctan2(dr, dc)

            diff = np.arctan2(
                np.sin(angle - azimuth),
                np.cos(angle - azimuth)
            )

            if abs(diff) <= half_fov:

                if obstacle_mask[r, c] == 0:
                    mask[r, c] = 1

    return mask

# ==========================================================
# ROUTE IMPACT EVALUATION
# ==========================================================

results = []

for pole in tqdm(candidates):

    row = pole["bev_row"]
    col = pole["bev_col"]

    best_raw_score = -1
    best_result = None

    for az in range(
        0,
        360,
        int(360 / AZIMUTH_STEPS)
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

            for r, c in route:

                if (
                    0 <= r < H
                    and
                    0 <= c < W
                ):

                    if fov_mask[r, c]:

                        hit = True
                        break

            if hit:
                routes_intersected += 1

        raw_score = (
            routes_intersected
            +
            0.001 * coverage_gain
        )

        if raw_score > best_raw_score:

            best_raw_score = raw_score

            best_result = {

                "pole_id":
                    pole["pole_id"],

                "grid":
                    pole["grid"],

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

                "raw_score":
                    round(
                        raw_score,
                        3
                    )
            }

    results.append(
        best_result
    )

# ==========================================================
# NORMALIZATION
# ==========================================================

max_routes = max(
    r["routes_intersected"]
    for r in results
)

max_coverage = max(
    r["coverage_gain"]
    for r in results
)

print(
    f"\nMax Routes Intersected: "
    f"{max_routes}"
)

print(
    f"Max Coverage Gain: "
    f"{max_coverage}"
)

# ==========================================================
# AGENT-AWARE FINAL SCORE
# ==========================================================

for result in results:

    route_metric = (

        result[
            "routes_intersected"
        ]

        /

        max_routes

        if max_routes > 0

        else 0
    )

    coverage_metric = (

        result[
            "coverage_gain"
        ]

        /

        max_coverage

        if max_coverage > 0

        else 0
    )

    final_score = (

        route_weight
        *
        route_metric

        +

        coverage_weight
        *
        coverage_metric

    )

    result[
        "route_metric"
    ] = round(
        route_metric,
        4
    )

    result[
        "coverage_metric"
    ] = round(
        coverage_metric,
        4
    )

    result[
        "final_score"
    ] = round(
        final_score,
        4
    )

# ==========================================================
# SORT USING AGENT STRATEGY
# ==========================================================

results.sort(

    key=lambda x:
    x["final_score"],

    reverse=True
)

# ==========================================================
# DEBUG TOP 5
# ==========================================================

print(
    "\nTOP 5 CAMERAS\n"
)

for cam in results[:5]:

    print(

        f"Pole={cam['pole_id']} | "

        f"Grid={cam['grid']} | "

        f"Routes={cam['routes_intersected']} | "

        f"Coverage={cam['coverage_gain']} | "

        f"RouteMetric={cam['route_metric']} | "

        f"CoverageMetric={cam['coverage_metric']} | "

        f"FinalScore={cam['final_score']}"

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
    f"\nSaved:\n{ranked_path}"
)

print(
    f"\nSaved:\n{best_path}"
)

print("\n" + "=" * 60)
print("VISIBILITY EVALUATION COMPLETE")
print("=" * 60)