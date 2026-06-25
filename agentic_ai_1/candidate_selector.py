#!/usr/bin/env python3

import os
import json
import numpy as np
import open3d as o3d

# ==========================================================
# CONFIG
# ==========================================================

#STRATEGY_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\agent2_output\strategy.json"
STRATEGY_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent2_output"
)
#=====helper====
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

GRID_METADATA_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\terrain_intelligence_output\grid_metadata.json"

BEV_METADATA_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\bev_metadata.json"

VALID_CAMERA_PLY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step2\valid_camera_positions.ply"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\candidate_selector_output"

TOP_BOTTLENECKS = 5

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

# ==========================================================
# LOAD STRATEGY
# ==========================================================

print("\n" + "=" * 60)
print("CANDIDATE SELECTOR")
print("=" * 60)
STRATEGY_JSON = get_latest_strategy()

with open(
    STRATEGY_JSON,
    "r"
) as f:

    strategy_json = json.load(f)

raw_bottlenecks = strategy_json[
    "strategy"
][
    "priority_bottlenecks"
]

priority_grids = []

for item in raw_bottlenecks:

    # Case 1:
    # "5_7"
    if isinstance(item, str):

        priority_grids.append(item)

    # Case 2:
    # {"location":"(5, 7)", ...}
    elif isinstance(item, dict):

        if "grid" in item:

            priority_grids.append(
                item["grid"]
            )

        elif "location" in item:

            loc = item[
                "location"
            ]

            loc = (
                loc.replace("(", "")
                   .replace(")", "")
                   .replace(" ", "")
            )

            row, col = loc.split(",")

            priority_grids.append(
                f"{row}_{col}"
            )

print("\nPriority Grids:")

for g in priority_grids:

    print(" ", g)

print("\nPriority Grids:")

for g in priority_grids:

    print(" ", g)

# ==========================================================
# LOAD GRID METADATA
# ==========================================================

with open(
    GRID_METADATA_JSON,
    "r"
) as f:

    grid_metadata = json.load(f)

# ==========================================================
# LOAD BEV METADATA
# ==========================================================

with open(
    BEV_METADATA_JSON,
    "r"
) as f:

    bev = json.load(f)

min_x = bev["min_x"]
min_y = bev["min_y"]
grid_res = bev["grid_res"]

print("\nBEV Parameters")

print(f"min_x = {min_x}")
print(f"min_y = {min_y}")
print(f"grid_res = {grid_res}")

# ==========================================================
# BUILD SEARCH REGION
# ==========================================================

selected_grids = set()

for grid_id in priority_grids:

    gr = int(
        grid_id.split("_")[0]
    )

    gc = int(
        grid_id.split("_")[1]
    )

    for dr in [-1, 0, 1]:

        for dc in [-1, 0, 1]:

            ngr = gr + dr
            ngc = gc + dc

            key = f"{ngr}_{ngc}"

            if key in grid_metadata:

                selected_grids.add(
                    key
                )

print(
    f"\nSelected Search Grids: "
    f"{len(selected_grids)}"
)

# ==========================================================
# LOAD CAMERA POSITIONS
# ==========================================================

pcd = o3d.io.read_point_cloud(
    VALID_CAMERA_PLY
)

points = np.asarray(
    pcd.points
)

print(
    f"\nValid Camera Positions:"
    f" {len(points)}"
)

# ==========================================================
# FILTER CANDIDATES
# ==========================================================

candidate_poles = []

for pole_id, pt in enumerate(points):

    x = float(pt[0])
    y = float(pt[1])
    z = float(pt[2])

    # -----------------------------------------
    # WORLD -> BEV
    # -----------------------------------------

    bev_col = int(
        (x - min_x)
        /
        grid_res
    )

    bev_row = int(
        (y - min_y)
        /
        grid_res
    )

    selected_grid = None

    for grid_id in selected_grids:

        meta = grid_metadata[
            grid_id
        ]

        if (

            meta["row_start"]
            <= bev_row
            <= meta["row_end"]

            and

            meta["col_start"]
            <= bev_col
            <= meta["col_end"]

        ):

            selected_grid = grid_id

            break

    if selected_grid is None:

        continue

    candidate_poles.append({

        "pole_id":
            int(pole_id),

        "world_x":
            x,

        "world_y":
            y,

        "world_z":
            z,

        "bev_row":
            int(bev_row),

        "bev_col":
            int(bev_col),

        "grid":
            selected_grid
    })

# ==========================================================
# DISTANCE TO GRID CENTER
# ==========================================================

for pole in candidate_poles:

    meta = grid_metadata[
        pole["grid"]
    ]

    center_row = (
        meta["row_start"]
        +
        meta["row_end"]
    ) / 2

    center_col = (
        meta["col_start"]
        +
        meta["col_end"]
    ) / 2

    dist = np.sqrt(

        (
            pole["bev_row"]
            -
            center_row
        ) ** 2

        +

        (
            pole["bev_col"]
            -
            center_col
        ) ** 2
    )

    pole[
        "distance_to_center"
    ] = float(dist)

candidate_poles.sort(
    key=lambda x:
    x["distance_to_center"]
)

# ==========================================================
# SAVE JSON
# ==========================================================

json_path = os.path.join(
    OUT_DIR,
    "candidate_poles.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        candidate_poles,
        f,
        indent=2
    )

# ==========================================================
# SAVE NPY
# ==========================================================

npy_path = os.path.join(
    OUT_DIR,
    "candidate_poles.npy"
)

np.save(
    npy_path,
    candidate_poles
)

# ==========================================================
# SUMMARY
# ==========================================================

print(
    f"\nCandidate Poles Selected:"
    f" {len(candidate_poles)}"
)

if len(candidate_poles):

    print(
        "\nTop Candidate:\n"
    )

    print(
        json.dumps(
            candidate_poles[0],
            indent=2
        )
    )

print(
    f"\nSaved:\n{json_path}"
)

print(
    f"\nSaved:\n{npy_path}"
)

print("\n" + "=" * 60)
print("CANDIDATE SELECTION COMPLETE")
print("=" * 60)