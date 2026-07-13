#!/usr/bin/env python3
"""
Agent 4 — Raycasting FOV Engine

Reads:
  - shortlisted_candidates_iter_N.json   (from Agent 3)
  - terrain_mask.npy + obstacle_mask.npy
  - current routes JSON

FOV is computed with the SAME ray-casting method used in Agent 5:
  For each azimuth being tested, rays are cast at even angular steps across
  [azimuth-FOV_DEG, azimuth+FOV_DEG]. Each ray marches outward from
  R_near to R_far (derived from camera height + tilt/FOV geometry) in
  fixed pixel steps. Marching a ray STOPS the instant it hits an obstacle
  cell — nothing beyond that point on the ray is ever marked visible, so
  the camera genuinely cannot see behind an obstacle.

  This replaces the old NUM_RAYS=10000 Fibonacci-sampled 3D-cone approach,
  which (a) was stochastic rather than deterministic, (b) used a flat-
  terrain z-tracking approximation to decide "ground hit", and (c) was far
  slower for no accuracy benefit over a direct 2D angular sweep.

Camera model (unchanged from before):
  z_ground is real elevation in metres from the 3D reconstruction.
  The camera is mounted at z_ground + POLE_HEIGHT metres.
  PIXEL_SCALE_M (metres per pixel) is derived from the world extents
  recorded by Agent 3 in shortlisted_candidates_iter_N.json, so Agent 4's
  scoring geometry and Agent 5's final placement geometry always agree.

FOV params:
  FOV_DEG       = 30      half-angle of cone
  TILT_DEG      = 5       downward tilt below horizontal
  MAX_RANGE     = 150.0   metres
  POLE_HEIGHT   = 10.0    metres above ground
  AZIMUTH_STEPS = 12      azimuths tested per candidate; best (by routes hit) is kept

Writes:
  - agent4_output/fov_results_iter_N.json
"""

import os
import json
import numpy as np

# ===========================================================
# CONFIG
# ===========================================================

FOV_DEG        = 30
TILT_DEG       = 5
MAX_RANGE      = 150.0
POLE_HEIGHT    = 10.0
AZIMUTH_STEPS  = 12

# Ray casting resolution knobs (must match Agent 5 for consistent geometry)
RAY_STEP_PX      = 0.5   # marching step size along each ray, in pixels
MAX_ANGULAR_STEP = 0.5   # cap on angular spacing between rays, in degrees

# Metres per pixel in the terrain grid.
# Set to None -> computed automatically from shortlisted_candidates world_extents.
PIXEL_SCALE_M  = None

TERRAIN_NPY = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\Static_scripts\step4\terrain_mask.npy"
)

OBSTACLE_NPY = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\Static_scripts\step4\obstacle_mask.npy"
)

CANDIDATES_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent3_output"
)

ORIGINAL_PATHS = (
    r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\multi_astar_v2\paths_1.json"
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
    r"\agent4_output"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================
# LOAD TERRAIN
# ===========================================================

terrain  = np.load(TERRAIN_NPY).astype(bool)
obstacle = np.load(OBSTACLE_NPY).astype(bool)
walkable = terrain & (~obstacle)
H, W     = walkable.shape
total_walkable = int(np.sum(walkable))

print("\n" + "=" * 60)
print("AGENT 4 — RAYCASTING FOV ENGINE")
print("=" * 60)
print(f"Map: {H}×{W}  walkable cells: {total_walkable}")

# ===========================================================
# RESOLVE LATEST ROUTES
# ===========================================================

PATH_JSON = ORIGINAL_PATHS
if os.path.exists(RUNTIME_DIR):
    iters = []
    for fn in os.listdir(RUNTIME_DIR):
        if fn.startswith("paths_iter_") and fn.endswith(".json"):
            try:
                it = int(fn.replace("paths_iter_", "").replace(".json", ""))
                iters.append((it, fn))
            except ValueError:
                pass
    if iters:
        iters.sort(key=lambda x: x[0])
        PATH_JSON = os.path.join(RUNTIME_DIR, iters[-1][1])

print(f"Routes file: {PATH_JSON}")
with open(PATH_JSON, "r") as f:
    routes = json.load(f)

total_routes = len(routes)
print(f"Routes remaining: {total_routes}")

# Build cell→route index for fast intersection
cell_to_routes = {}
for ri, route in enumerate(routes):
    for cell in route:
        key = (int(cell[0]), int(cell[1]))
        cell_to_routes.setdefault(key, []).append(ri)

# ===========================================================
# RESOLVE CANDIDATES
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
        raise FileNotFoundError(f"No {prefix}*{suffix} in {directory}")
    files.sort(key=lambda x: x[0])
    return files[-1][0], os.path.join(directory, files[-1][1])

iteration, cand_path = get_latest_file(CANDIDATES_DIR, "shortlisted_candidates_iter_")
print(f"Candidates: {cand_path}")

with open(cand_path, "r") as f:
    cand_data = json.load(f)

candidates = cand_data["candidates"]

# Resolve pixel scale from world extents stored by Agent 3
if PIXEL_SCALE_M is None:
    ext = cand_data.get("world_extents", {})
    x_span = ext.get("x_max", 690) - ext.get("x_min", -120)   # fallback ~690m
    PIXEL_SCALE_M = x_span / max(W - 1, 1)
    print(f"Computed PIXEL_SCALE_M = {PIXEL_SCALE_M:.4f} m/px  "
          f"(world_x_span={x_span:.1f}m / {W} cols)")

print(f"Processing {len(candidates)} candidates × {AZIMUTH_STEPS} azimuths")

# ===========================================================
# RAY CASTING (deterministic angular sweep, obstacle-stopping)
# ===========================================================

def fov_sector_cells(cam_row, cam_col, cam_z, azimuth_deg,
                     H, W, obstacle, pixel_scale):
    """
    FOV via ray casting — identical method to Agent 5's fov_sector_cells,
    kept in sync so scoring geometry (here) and placement geometry (Agent 5)
    never disagree.

    R_near / R_far come from the camera height and tilt/FOV cone geometry:
        R_far  = min(cam_z / tan(tilt-fov), MAX_RANGE)   [shallowest ray]
        R_near = cam_z / tan(tilt+fov)                   [steepest ray]
    converted metres -> pixels via pixel_scale.

    Rays are cast at even angular steps across [azimuth-FOV, azimuth+FOV].
    Each ray marches from R_near to R_far in RAY_STEP_PX increments; the
    march stops the instant it enters an obstacle cell (occlusion — nothing
    farther along that ray is ever marked visible).
    """
    az_rad   = np.deg2rad(azimuth_deg)
    fov_rad  = np.deg2rad(FOV_DEG)
    tilt_rad = np.deg2rad(TILT_DEG)

    steep   = tilt_rad + fov_rad
    R_far_m = min(cam_z / np.tan(steep), MAX_RANGE)
    R_far   = R_far_m / pixel_scale

    shallow = tilt_rad - fov_rad
    R_near  = (cam_z / np.tan(shallow) / pixel_scale) if shallow > 1e-4 else 0.0

    if R_far <= R_near:
        return frozenset()

    if R_far > 0:
        d_rad = min(np.deg2rad(MAX_ANGULAR_STEP), 1.0 / R_far)
    else:
        d_rad = np.deg2rad(MAX_ANGULAR_STEP)
    d_rad = max(d_rad, 1e-4)

    num_rays = max(int(np.ceil((2 * fov_rad) / d_rad)), 1)

    cells = set()
    for i in range(num_rays + 1):
        ray_az = (az_rad - fov_rad) + (2 * fov_rad) * (i / num_rays)
        ddr = -np.cos(ray_az)   # north = -row
        ddc =  np.sin(ray_az)   # east  = +col

        dist = R_near
        while dist <= R_far:
            pr = cam_row + ddr * dist
            pc = cam_col + ddc * dist
            r_i, c_i = int(round(pr)), int(round(pc))

            if not (0 <= r_i < H and 0 <= c_i < W):
                break

            if obstacle[r_i, c_i]:
                break  # occluded — camera cannot see behind this obstacle

            cells.add((r_i, c_i))
            dist += RAY_STEP_PX

    return frozenset(cells)


def score_candidate(cam_row, cam_col, cam_z, azimuth_deg):
    fov_cells = fov_sector_cells(cam_row, cam_col, cam_z, azimuth_deg,
                                  H, W, obstacle, PIXEL_SCALE_M)

    hit_routes = set()
    for cell in fov_cells:
        if cell in cell_to_routes:
            hit_routes.update(cell_to_routes[cell])

    route_score    = len(hit_routes) / max(total_routes, 1)
    coverage_score = len(fov_cells)  / max(total_walkable, 1)
    return fov_cells, hit_routes, route_score, coverage_score


# ===========================================================
# MAIN LOOP
# ===========================================================

azimuths = [360.0 * k / AZIMUTH_STEPS for k in range(AZIMUTH_STEPS)]
results  = []

for i, cand in enumerate(candidates):
    if i % 10 == 0:
        print(f"  {i}/{len(candidates)}")

    cam_row = cand["row"]
    cam_col = cand["col"]
    cam_z   = cand["z_ground"] + POLE_HEIGHT    # height above local ground, drives tilt geometry

    best_az     = 0.0
    best_cells  = set()
    best_routes = set()
    best_rs     = 0.0
    best_cs     = 0.0

    for az in azimuths:
        fov_cells, hit_routes, rs, cs = score_candidate(cam_row, cam_col, cam_z, az)
        # Primary criterion: routes hit. Tiebreak (and sole criterion when
        # there are no routes at all, so every azimuth ties at 0 hits):
        # coverage_score. Without this tiebreak, a 0-routes run would just
        # keep whichever azimuth happened to be tested first (0°) instead
        # of the one that actually sees the most terrain.
        better = (len(hit_routes), cs) > (len(best_routes), best_cs)
        if better:
            best_az     = az
            best_cells  = fov_cells
            best_routes = hit_routes
            best_rs     = rs
            best_cs     = cs

    results.append({
        "candidate_id"     : cand["candidate_id"],
        "row"              : cam_row,
        "col"              : cam_col,
        "z_ground"         : cand["z_ground"],
        "cam_z"            : cam_z,
        "world_x"          : cand.get("world_x", 0.0),
        "world_y"          : cand.get("world_y", 0.0),
        "best_azimuth_deg" : best_az,
        "fov_cell_count"   : len(best_cells),
        "fov_cells"        : [[int(r), int(c)] for r, c in best_cells],
        "routes_hit"       : len(best_routes),
        "route_score"      : float(best_rs),
        "coverage_score"   : float(best_cs)
    })

# ===========================================================
# SAVE
# ===========================================================

output = {
    "iteration"      : iteration,
    "total_routes"   : total_routes,
    "total_walkable" : total_walkable,
    "pixel_scale_m"  : float(PIXEL_SCALE_M),
    "fov_params"     : {
        "FOV_DEG"          : FOV_DEG,
        "TILT_DEG"         : TILT_DEG,
        "MAX_RANGE"        : MAX_RANGE,
        "POLE_HEIGHT"      : POLE_HEIGHT,
        "AZIMUTH_STEPS"    : AZIMUTH_STEPS,
        "RAY_STEP_PX"      : RAY_STEP_PX,
        "MAX_ANGULAR_STEP" : MAX_ANGULAR_STEP,
        "method"           : "deterministic angular ray march, obstacle-stopping"
    },
    "candidates"     : results
}

out_path = os.path.join(OUT_DIR, f"fov_results_iter_{iteration}.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved → {out_path}")
print("\n" + "=" * 60)
print("AGENT 4 COMPLETE")
print("=" * 60)