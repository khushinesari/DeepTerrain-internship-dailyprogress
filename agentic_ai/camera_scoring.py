#!/usr/bin/env python3

import os
import json
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================

CANDIDATE_POLES_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\candidate_selector_output\candidate_poles.json"

STRATEGY_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\agent2_output\strategy.json"

SUMMARY_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\terrain_intelligence_output\summary.json"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\camera_scoring_output"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

# ==========================================================
# LOAD FILES
# ==========================================================

print("\n" + "=" * 60)
print("CAMERA SCORING")
print("=" * 60)

with open(
    CANDIDATE_POLES_JSON,
    "r"
) as f:

    candidate_poles = json.load(f)

with open(
    STRATEGY_JSON,
    "r"
) as f:

    strategy = json.load(f)

with open(
    SUMMARY_JSON,
    "r"
) as f:

    summary = json.load(f)

print(
    f"\nCandidate Poles: "
    f"{len(candidate_poles)}"
)

# ==========================================================
# LOAD WEIGHTS
# ==========================================================

route_weight = strategy["strategy"]["route_weights"]

coverage_weight = strategy["strategy"]["coverage_weights"]

print(
    f"\nRoute Weight    : {route_weight}"
)

print(
    f"Coverage Weight : {coverage_weight}"
)

# ==========================================================
# BOTTLENECK FREQUENCIES
# ==========================================================

bottleneck_freq = {}

for b in summary["top_bottlenecks"]:

    bottleneck_freq[
        b["grid"]
    ] = b["frequency"]

max_freq = max(
    bottleneck_freq.values()
)

# ==========================================================
# SCORE POLES
# ==========================================================

ranked = []

for pole in candidate_poles:

    grid = pole["grid"]

    freq = bottleneck_freq.get(
        grid,
        1
    )

    # ------------------------------------------
    # ROUTE SCORE
    #
    # higher bottleneck frequency
    # = higher route score
    # ------------------------------------------

    route_score = (
        freq /
        max_freq
    )

    # ------------------------------------------
    # COVERAGE SCORE
    #
    # closer to grid center
    # = better
    # ------------------------------------------

    dist = pole[
        "distance_to_center"
    ]

    coverage_score = (
        1.0 /
        (1.0 + dist)
    )

    # ------------------------------------------
    # FINAL SCORE
    # ------------------------------------------

    score = (

        route_weight
        *
        route_score

        +

        coverage_weight
        *
        coverage_score

    )

    ranked.append({

    "pole_id":
        pole["pole_id"],

    # ----------------------------------
    # WORLD COORDS
    # ----------------------------------

    "world_x":
        pole["world_x"],

    "world_y":
        pole["world_y"],

    "world_z":
        pole["world_z"],

    # ----------------------------------
    # BEV COORDS
    # ----------------------------------

    "bev_row":
        pole["bev_row"],

    "bev_col":
        pole["bev_col"],

    # ----------------------------------
    # GRID INFO
    # ----------------------------------

    "grid":
        grid,

    "distance_to_center":
        pole["distance_to_center"],

    # ----------------------------------
    # SCORES
    # ----------------------------------

    "route_score":
        round(
            route_score,
            4
        ),

    "coverage_score":
        round(
            coverage_score,
            4
        ),

    "final_score":
        round(
            score,
            4
        )
})

# ==========================================================
# SORT
# ==========================================================

ranked.sort(

    key=lambda x:
    x["final_score"],

    reverse=True
)

# ==========================================================
# TOP 50
# ==========================================================

top_50 = ranked[:50]

# ==========================================================
# SAVE
# ==========================================================

ranked_path = os.path.join(
    OUT_DIR,
    "ranked_candidates.json"
)

with open(
    ranked_path,
    "w"
) as f:

    json.dump(
        ranked,
        f,
        indent=2
    )

top50_path = os.path.join(
    OUT_DIR,
    "top50_candidates.json"
)

with open(
    top50_path,
    "w"
) as f:

    json.dump(
        top_50,
        f,
        indent=2
    )

# ==========================================================
# REPORT
# ==========================================================

print(
    f"\nRanked Candidates:"
    f" {len(ranked)}"
)

print(
    f"\nTop Candidate:\n"
)

print(
    json.dumps(
        top_50[0],
        indent=2
    )
)

print(
    f"\nSaved:\n{ranked_path}"
)

print(
    f"\nSaved:\n{top50_path}"
)

print("\n" + "=" * 60)
print("CAMERA SCORING COMPLETE")
print("=" * 60)