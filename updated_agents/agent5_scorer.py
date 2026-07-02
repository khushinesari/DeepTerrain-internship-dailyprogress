#!/usr/bin/env python3
"""
Agent 5 — Camera Scorer and Placer
Reads:
  - fov_results_iter_N.json     (from Agent 4)
  - strategy_iter_N.json        (from Agent 2, for weights)
  - current routes JSON
  - runtime_state/placed_cameras.json  (existing placements)

Scores every candidate:
    score = route_weight × route_score + coverage_weight × coverage_score

  route_score    = routes eliminated by this FOV / total_routes_remaining
  coverage_score = unique walkable cells in FOV  / total_walkable_cells

  Both scores come directly from the raycaster FOV — nothing random.

Picks the best camera, eliminates all routes intersecting its FOV,
writes the surviving routes to runtime_state/paths_iter_N.json.

Writes:
  - runtime_state/placed_cameras.json     (cumulative)
  - runtime_state/paths_iter_N.json       (surviving routes)
  - agent5_output/placement_result_iter_N.json
"""

import os
import json
import numpy as np

# ===========================================================
# CONFIG
# ===========================================================

FOV_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent4_output"
)

STRATEGY_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent2_output"
)

ORIGINAL_PATHS = (
    r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\inetractive_multi_astar_output\paths_1.json"
)

RUNTIME_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\runtime_state"
)

OUT_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent5_output"
)

os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================
# HELPERS
# ===========================================================

def get_latest_file(directory, prefix, suffix=".json"):
    files = []
    for fn in os.listdir(directory):
        if fn.startswith(prefix) and fn.endswith(suffix):
            try:
                it = int(fn[len(prefix):-len(suffix)])
                files.append((it, fn))
            except ValueError:
                pass
    if not files:
        raise FileNotFoundError(f"No {prefix}*{suffix} found in {directory}")
    files.sort(key=lambda x: x[0])
    return files[-1][0], os.path.join(directory, files[-1][1])


def load_routes(runtime_dir, original_path):
    """Load the latest surviving routes."""
    iters = []
    for fn in os.listdir(runtime_dir):
        if fn.startswith("paths_iter_") and fn.endswith(".json"):
            try:
                it = int(fn.replace("paths_iter_", "").replace(".json", ""))
                iters.append((it, fn))
            except ValueError:
                pass
    if iters:
        iters.sort(key=lambda x: x[0])
        path = os.path.join(runtime_dir, iters[-1][1])
    else:
        path = original_path
    print(f"Routes from: {path}")
    with open(path, "r") as f:
        return json.load(f)


# ===========================================================
# MAIN
# ===========================================================

print("\n" + "=" * 60)
print("AGENT 5 — CAMERA SCORER AND PLACER")
print("=" * 60)

# ---- Load FOV results -------------------------------------
iteration, fov_path = get_latest_file(FOV_DIR, "fov_results_iter_")
print(f"FOV results : {fov_path}")

with open(fov_path, "r") as f:
    fov_data = json.load(f)

candidates   = fov_data["candidates"]
total_routes = fov_data["total_routes"]
total_walk   = fov_data["total_walkable"]

# ---- Load strategy weights --------------------------------
_, strat_path = get_latest_file(STRATEGY_DIR, "strategy_iter_")
print(f"Strategy    : {strat_path}")

with open(strat_path, "r") as f:
    strategy = json.load(f)

route_w    = float(strategy["strategy"]["route_weights"])
coverage_w = float(strategy["strategy"]["coverage_weights"])

print(f"Weights     : route={route_w}  coverage={coverage_w}")

# ---- Load current routes ----------------------------------
routes = load_routes(RUNTIME_DIR, ORIGINAL_PATHS)
print(f"Routes remaining: {len(routes)}")

# ---- Score every candidate --------------------------------
for cand in candidates:
    score = (route_w    * cand["route_score"] +
             coverage_w * cand["coverage_score"])
    cand["final_score"] = float(score)

candidates.sort(key=lambda c: c["final_score"], reverse=True)
best = candidates[0]

print(f"\nBest candidate id={best['candidate_id']}")
print(f"  row={best['row']:.1f}  col={best['col']:.1f}")
print(f"  azimuth={best['best_azimuth_deg']:.1f}°")
print(f"  route_score={best['route_score']:.4f}  "
      f"coverage_score={best['coverage_score']:.4f}  "
      f"final={best['final_score']:.4f}")
print(f"  FOV cells={best['fov_cell_count']}  "
      f"routes_hit={best['routes_hit']}")

# ---- Eliminate routes whose cells intersect FOV -----------
fov_set = frozenset(map(tuple, best["fov_cells"]))

surviving = []
eliminated = 0

for route in routes:
    route_cells = set(map(tuple, route))
    if route_cells.isdisjoint(fov_set):
        surviving.append(route)
    else:
        eliminated += 1

print(f"\nRoutes eliminated : {eliminated}")
print(f"Routes surviving  : {len(surviving)}")

# ---- Persist placed cameras -------------------------------
placed_path = os.path.join(RUNTIME_DIR, "placed_cameras.json")

if os.path.exists(placed_path):
    with open(placed_path, "r") as f:
        placed_cameras = json.load(f)
else:
    placed_cameras = []

placed_cameras.append({
    "iteration"      : iteration,
    "candidate_id"   : best["candidate_id"],
    "row"            : best["row"],
    "col"            : best["col"],
    "z_ground"       : best["z_ground"],
    "cam_z"          : best["cam_z"],
    "azimuth_deg"    : best["best_azimuth_deg"],
    "route_score"    : best["route_score"],
    "coverage_score" : best["coverage_score"],
    "final_score"    : best["final_score"],
    "routes_eliminated": eliminated,
    "fov_cell_count" : best["fov_cell_count"]
})

with open(placed_path, "w") as f:
    json.dump(placed_cameras, f, indent=2)

print(f"placed_cameras.json updated ({len(placed_cameras)} cameras total)")

# ---- Write surviving routes for next iteration ------------
next_iter   = iteration + 1
routes_path = os.path.join(RUNTIME_DIR, f"paths_iter_{next_iter}.json")

with open(routes_path, "w") as f:
    json.dump(surviving, f, indent=2)

print(f"Saved surviving routes → {routes_path}")

# ---- Write placement result summary -----------------------
result = {
    "iteration"          : iteration,
    "route_weight"       : route_w,
    "coverage_weight"    : coverage_w,
    "camera_placed"      : {
        "candidate_id"   : best["candidate_id"],
        "row"            : best["row"],
        "col"            : best["col"],
        "z_ground"       : best["z_ground"],
        "cam_z"          : best["cam_z"],
        "azimuth_deg"    : best["best_azimuth_deg"]
    },
    "scores"             : {
        "route_score"    : best["route_score"],
        "coverage_score" : best["coverage_score"],
        "final_score"    : best["final_score"]
    },
    "routes_before"      : len(routes),
    "routes_eliminated"  : eliminated,
    "routes_after"       : len(surviving),
    "fov_cell_count"     : best["fov_cell_count"],
    "total_cameras_so_far": len(placed_cameras),
    "all_scores_ranked"  : [
        {
            "candidate_id" : c["candidate_id"],
            "final_score"  : c["final_score"],
            "route_score"  : c["route_score"],
            "coverage_score": c["coverage_score"]
        }
        for c in candidates[:10]
    ]
}

out_path = os.path.join(OUT_DIR, f"placement_result_iter_{iteration}.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved result → {out_path}")

print("\n" + "=" * 60)
print("AGENT 5 COMPLETE")
print("=" * 60)