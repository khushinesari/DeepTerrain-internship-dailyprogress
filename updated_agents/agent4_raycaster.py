#!/usr/bin/env python3
"""
Agent 4 — Raycasting FOV Engine
Reads:
  - shortlisted_candidates_iter_N.json   (from Agent 3)
  - terrain_mask.npy + obstacle_mask.npy
  - current routes JSON

Camera model (confirmed from PLY inspection):
  z_ground is real elevation in metres from the 3D reconstruction.
  The camera is mounted at z_ground + POLE_HEIGHT metres.
  Raycasting uses pixel-space (row, col) for terrain lookup,
  with metres as the distance unit.

  PIXEL_SCALE_M: metres per pixel in the terrain grid.
  Set this to your terrain's GSD (ground sampling distance).
  Example: if your terrain_mask covers 690m × 401m and is 690×401 pixels,
  PIXEL_SCALE_M = 1.0 m/px.  If it's 1380×802 pixels, PIXEL_SCALE_M = 0.5.

FOV params:
  FOV_DEG       = 30      half-angle of cone
  TILT_DEG      = 5       downward tilt below horizontal
  MAX_RANGE     = 150.0   metres
  NUM_RAYS      = 10000   rays per azimuth test
  POLE_HEIGHT   = 10.0    metres above ground
  RAY_STEP      = 0.5     metres per march step
  HIT_THRESHOLD = 0.3     metres ground-hit tolerance
  AZIMUTH_STEPS = 12      azimuths tested; best is kept

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
NUM_RAYS       = 10_000
POLE_HEIGHT    = 10.0
RAY_STEP       = 0.5
HIT_THRESHOLD  = 0.3
AZIMUTH_STEPS  = 12

# Metres per pixel in the terrain grid.
# CRITICAL: set this to match your terrain_mask resolution.
# If unknown: terrain world span / terrain pixel span.
# From the PLY: world X span ~690m, world Y span ~402m.
# You must check terrain_mask.npy shape to compute this.
# Set to None → computed automatically from shortlisted_candidates metadata.
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
        if key not in cell_to_routes:
            cell_to_routes[key] = []
        cell_to_routes[key].append(ri)

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

# Resolve pixel scale from world extents stored in candidates file
if PIXEL_SCALE_M is None:
    ext = cand_data.get("world_extents", {})
    x_span = ext.get("x_max", 690) - ext.get("x_min", -120)   # fallback ~690m
    PIXEL_SCALE_M = x_span / max(W - 1, 1)
    print(f"Computed PIXEL_SCALE_M = {PIXEL_SCALE_M:.4f} m/px  "
          f"(world_x_span={x_span:.1f}m / {W} cols)")

# MAX_RANGE in pixels (for stepping)
MAX_RANGE_PX = MAX_RANGE / PIXEL_SCALE_M
STEP_PX      = RAY_STEP  / PIXEL_SCALE_M
max_steps    = int(MAX_RANGE_PX / STEP_PX) + 1

print(f"PIXEL_SCALE_M={PIXEL_SCALE_M:.4f}  MAX_RANGE_PX={MAX_RANGE_PX:.1f}  steps={max_steps}")
print(f"Processing {len(candidates)} candidates × {AZIMUTH_STEPS} azimuths")

# ===========================================================
# RAYCASTING
# ===========================================================

def build_ray_dirs_2d(azimuth_deg: float) -> tuple:
    """
    Build NUM_RAYS 2D direction vectors (drow, dcol) spread within FOV_DEG
    of azimuth_deg, using Fibonacci disk sampling for uniform distribution.

    Azimuth convention: 0° = North (row decreases), clockwise.
    Returns (drow, dcol, dz_norm) all shape (NUM_RAYS,), unit vectors in 3D
    but we use drow/dcol for pixel stepping and dz_norm for z tracking.
    """
    az_rad   = np.deg2rad(azimuth_deg)
    tilt_rad = np.deg2rad(TILT_DEG)
    fov_rad  = np.deg2rad(FOV_DEG)

    # Central direction in (east, north, up):
    #   azimuth 0 → north → (east=0, north=1)
    #   azimuth 90 → east → (east=1, north=0)
    east_c  =  np.sin(az_rad) * np.cos(tilt_rad)
    north_c =  np.cos(az_rad) * np.cos(tilt_rad)
    up_c    = -np.sin(tilt_rad)   # tilted down

    center  = np.array([east_c, north_c, up_c])
    center /= np.linalg.norm(center)

    # Build orthonormal basis
    world_up = np.array([0., 0., 1.])
    right = np.cross(center, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    up2   = np.cross(right, center)
    up2  /= np.linalg.norm(up2)

    # Fibonacci cone sampling (uniform solid angle within FOV half-angle)
    n   = NUM_RAYS
    phi = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(n, dtype=np.float64)

    cos_max = np.cos(fov_rad)
    tz  = 1.0 - idx / (n - 1) * (1.0 - cos_max)
    r_  = np.sqrt(np.maximum(1.0 - tz ** 2, 0.0))
    theta = phi * idx

    dx = r_ * np.cos(theta)
    dy = r_ * np.sin(theta)
    dz = tz

    # Local → world (east, north, up)
    dirs = (dx[:, None] * right[None, :]
          + dy[:, None] * up2[None, :]
          + dz[:, None] * center[None, :])

    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= np.maximum(norms, 1e-12)

    # Convert (east, north, up) → pixel (drow, dcol):
    #   east → +col, north → -row  (image row 0 is top = north)
    drow = -dirs[:, 1] / PIXEL_SCALE_M   # north component → row (inverted)
    dcol =  dirs[:, 0] / PIXEL_SCALE_M   # east component  → col
    dz_n =  dirs[:, 2]                   # up component (for z tracking)

    return drow.astype(np.float32), dcol.astype(np.float32), dz_n.astype(np.float32)


def raycast_fov(cam_row: float, cam_col: float, cam_z: float,
                azimuth_deg: float) -> set:
    """
    March NUM_RAYS from the camera position.
    Returns set of (row, col) walkable terrain cells hit.

    z tracking: camera is at cam_z metres elevation.
    Ground elevation is approximated as z_ground of the camera (flat-terrain
    assumption within the FOV cone). For sloped terrain, replace
    ground_z with a DEM lookup — see comment inside.
    """
    drow, dcol, dz_n = build_ray_dirs_2d(azimuth_deg)
    hit_cells = set()

    for i in range(NUM_RAYS):
        dr = float(drow[i])
        dc = float(dcol[i])
        dz = float(dz_n[i])

        cur_z = cam_z  # metres, starts at camera height

        for step in range(1, max_steps + 1):
            t    = step * RAY_STEP                      # metres marched
            nr   = int(round(cam_row + dr * step))
            nc   = int(round(cam_col + dc * step))
            cur_z = cam_z + dz * t                     # elevation at this point

            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                break

            # ---- Ground hit check ----
            # For flat-terrain approximation: ground is at cam_z - POLE_HEIGHT.
            # For DEM: replace ground_z with dem[nr, nc].
            ground_z = cam_z - POLE_HEIGHT              # flat-terrain fallback
            # ground_z = dem[nr, nc]                   # uncomment if DEM npy available

            if cur_z <= ground_z + HIT_THRESHOLD:
                if walkable[nr, nc]:
                    hit_cells.add((nr, nc))
                break

            # Obstacle occlusion terminates ray
            if obstacle[nr, nc]:
                break

    return hit_cells


def score_candidate(cam_row, cam_col, cam_z, azimuth_deg):
    fov_cells = raycast_fov(cam_row, cam_col, cam_z, azimuth_deg)

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
    cam_z   = cand["z_ground"] + POLE_HEIGHT    # actual elevation in metres

    best_az     = 0.0
    best_cells  = set()
    best_routes = set()
    best_rs     = 0.0
    best_cs     = 0.0

    for az in azimuths:
        fov_cells, hit_routes, rs, cs = score_candidate(cam_row, cam_col, cam_z, az)
        if len(hit_routes) > len(best_routes):
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
        "FOV_DEG"      : FOV_DEG,
        "TILT_DEG"     : TILT_DEG,
        "MAX_RANGE"    : MAX_RANGE,
        "NUM_RAYS"     : NUM_RAYS,
        "POLE_HEIGHT"  : POLE_HEIGHT,
        "RAY_STEP"     : RAY_STEP,
        "HIT_THRESHOLD": HIT_THRESHOLD,
        "AZIMUTH_STEPS": AZIMUTH_STEPS
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
