#!/usr/bin/env python3
"""
Agent 6 — Coverage Expansion Planner
Reads:
  - agent4_output/fov_results_iter_N.json   (all candidate FOVs)
  - runtime_state/placed_cameras.json       (FOV cells already covered)
  - terrain_mask.npy + obstacle_mask.npy

Produces:
  - A blind-spot grid: walkable cells NOT yet covered by any placed camera
  - Re-ranks remaining candidates purely by NEW coverage they add
    (no LLM, no route score — pure spatial coverage maximisation)
  - Suggests the best next location for coverage expansion
  - Writes coverage_expansion_iter_N.json

Why no LLM for coverage selection?
  Coverage is a geometric problem: find the camera position that illuminates
  the most un-seen walkable pixels.  The optimal answer comes from the
  raycasted FOV footprints already computed by Agent 4 — we simply sort
  candidates by |fov_cells ∩ blind_spot| / |blind_spot|.
  This is deterministic, fast, and explainable.

When to use LLM for coverage:
  Use it only to decide STRATEGY — e.g. "should we keep eliminating routes
  or shift entirely to coverage expansion?" (that stays in Agent 2).
  For geometric decisions — which cell, which direction — pure spatial
  scoring is more reliable than an LLM.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ===========================================================
# CONFIG
# ===========================================================

# FOV/ray-casting params — must match Agent 4/5 so a camera's counted
# coverage here is the same cone that was actually placed.
FOV_DEG           = 30
TILT_DEG          = 5
MAX_RANGE         = 150.0
RAY_STEP_PX       = 0.5
MAX_ANGULAR_STEP  = 0.5

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

FOV_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent4_output"
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
    r"\agent6_output"
)

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
        raise FileNotFoundError(f"No {prefix}*{suffix} in {directory}")
    files.sort(key=lambda x: x[0])
    return files[-1][0], os.path.join(directory, files[-1][1])


def fov_sector_cells(cam_row, cam_col, cam_z, azimuth_deg,
                     H, W, obstacle, pixel_scale):
    """
    FOV via ray casting — identical method to Agent 4/5's fov_sector_cells.
    Used here to recompute each PLACED camera's FOV directly from its
    stored geometry, rather than looking it up in this iteration's
    candidate list (which won't contain cameras placed in earlier
    iterations, since Agent 3 reshortlists candidates every iteration).
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
        ddr = -np.cos(ray_az)
        ddc =  np.sin(ray_az)

        dist = R_near
        while dist <= R_far:
            pr = cam_row + ddr * dist
            pc = cam_col + ddc * dist
            r_i, c_i = int(round(pr)), int(round(pc))

            if not (0 <= r_i < H and 0 <= c_i < W):
                break
            if obstacle[r_i, c_i]:
                break  # occluded — cannot see behind this obstacle

            cells.add((r_i, c_i))
            dist += RAY_STEP_PX

    return frozenset(cells)

# ===========================================================
# MAIN
# ===========================================================

print("\n" + "=" * 60)
print("AGENT 6 — COVERAGE EXPANSION PLANNER")
print("=" * 60)

# ---- Terrain --------------------------------------------------
terrain  = np.load(TERRAIN_NPY).astype(bool)
obstacle = np.load(OBSTACLE_NPY).astype(bool)
walkable = terrain & (~obstacle)
H, W     = walkable.shape
total_walkable = int(np.sum(walkable))
print(f"Map {H}×{W}  walkable cells: {total_walkable}")

# ---- Load FOV results ----------------------------------------
iteration, fov_path = get_latest_file(FOV_DIR, "fov_results_iter_")
print(f"FOV results : {fov_path}")

with open(fov_path, "r") as f:
    fov_data = json.load(f)

candidates = fov_data["candidates"]

# Pixel scale — read the same value Agent 4/5 used, so ray-casting placed
# cameras here produces the identical cone that was actually placed.
PIXEL_SCALE = float(fov_data.get("pixel_scale_m", 690.0 / (W - 1)))

# Build per-candidate FOV as frozenset (used later for ranking NEW candidates)
fov_sets = {}
for cand in candidates:
    fov_sets[cand["candidate_id"]] = frozenset(
        map(tuple, cand["fov_cells"])
    )

# ---- Build already-covered set from placed cameras -----------
# FIXED: previously this cross-referenced each placed camera's candidate_id
# against THIS iteration's freshly reshortlisted candidate list. Agent 3
# reshortlists candidates every iteration (different bottlenecks -> mostly
# different candidate_ids), so a camera placed in an earlier iteration was
# almost never found here and its coverage silently vanished from the
# total. Now every placed camera's FOV is recomputed directly from its own
# stored (row, col, cam_z, azimuth_deg) via the same ray-casting used to
# place it — correct regardless of which iteration placed it or whether it
# reappears in any later shortlist.
placed_path = os.path.join(RUNTIME_DIR, "placed_cameras.json")
covered_cells = set()
placed_cameras = []

if os.path.exists(placed_path):
    with open(placed_path, "r") as f:
        placed_cameras = json.load(f)

    for cam in placed_cameras:
        covered_cells |= fov_sector_cells(
            cam["row"], cam["col"], cam["cam_z"], cam["azimuth_deg"],
            H, W, obstacle, PIXEL_SCALE
        )

print(f"Cells already covered by {len(placed_cameras)} cameras: {len(covered_cells)}")

# ---- Blind-spot map ------------------------------------------
blind_mask = np.zeros((H, W), dtype=np.uint8)
walkable_coords = np.argwhere(walkable)

for r, c in walkable_coords:
    if (r, c) not in covered_cells:
        blind_mask[r, c] = 1

blind_total = int(np.sum(blind_mask))
coverage_pct = 100.0 * len(covered_cells) / max(total_walkable, 1)

print(f"Blind-spot cells : {blind_total}")
print(f"Coverage so far  : {coverage_pct:.2f}%")

# ---- Re-rank candidates by marginal new coverage -------------
#  Marginal coverage = cells in candidate FOV that are NOT yet covered
#  (i.e. in the blind spot).

blind_set = frozenset(zip(*np.argwhere(blind_mask).T.tolist()))   # frozenset of (r,c)

ranked = []
for cand in candidates:
    cid   = cand["candidate_id"]
    fov_s = fov_sets[cid]

    marginal       = fov_s - covered_cells          # new cells this camera adds
    marginal_count = len(marginal)
    marginal_score = marginal_count / max(blind_total, 1)

    ranked.append({
        "candidate_id"    : cid,
        "row"             : cand["row"],
        "col"             : cand["col"],
        "z_ground"        : cand["z_ground"],
        "best_azimuth_deg": cand["best_azimuth_deg"],
        "fov_cell_count"  : cand["fov_cell_count"],
        "new_cells"       : marginal_count,
        "marginal_score"  : float(marginal_score),
        "route_score"     : cand["route_score"],
        "coverage_score"  : cand["coverage_score"]
    })

ranked.sort(key=lambda x: x["marginal_score"], reverse=True)

best_cov = ranked[0] if ranked else None

if best_cov:
    print(f"\nBest coverage-expansion camera:")
    print(f"  candidate_id : {best_cov['candidate_id']}")
    print(f"  row={best_cov['row']:.1f}  col={best_cov['col']:.1f}")
    print(f"  azimuth      : {best_cov['best_azimuth_deg']:.1f}°")
    print(f"  new_cells    : {best_cov['new_cells']}  "
          f"marginal_score={best_cov['marginal_score']:.4f}")

# ---- Grid-level blind-spot summary (10×10) -------------------
GRID_ROWS, GRID_COLS = 10, 10
grid_h = H // GRID_ROWS + 1
grid_w = W // GRID_COLS + 1

grid_blind   = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
grid_total   = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

for r, c in walkable_coords:
    gr = min(r // grid_h, GRID_ROWS - 1)
    gc = min(c // grid_w, GRID_COLS - 1)
    grid_total[gr, gc] += 1
    if blind_mask[r, c]:
        grid_blind[gr, gc] += 1

# Fraction of each grid cell that is blind
with np.errstate(divide="ignore", invalid="ignore"):
    grid_blind_frac = np.where(
        grid_total > 0,
        grid_blind / grid_total,
        0
    )

# Find top uncovered grids
grid_flat = [(grid_blind_frac[gr, gc], gr, gc)
             for gr in range(GRID_ROWS)
             for gc in range(GRID_COLS)]
grid_flat.sort(reverse=True)
top_blind_grids = [
    {"grid_row": int(gr), "grid_col": int(gc), "blind_fraction": float(bf)}
    for bf, gr, gc in grid_flat[:10]
    if bf > 0
]

# ---- Save blind-spot heatmap image ---------------------------
plt.figure(figsize=(8, 8))
plt.imshow(grid_blind_frac, cmap="hot", vmin=0, vmax=1)
plt.colorbar(label="Blind-spot fraction")
plt.title(f"Blind-spot grid (iter {iteration}) — coverage {coverage_pct:.1f}%")
heatmap_path = os.path.join(OUT_DIR, f"blind_spot_heatmap_iter_{iteration}.png")
plt.savefig(heatmap_path, bbox_inches="tight")
plt.close()
print(f"Saved blind-spot heatmap → {heatmap_path}")

# ---- Write output JSON ---------------------------------------
output = {
    "iteration"          : iteration,
    "total_walkable"     : total_walkable,
    "covered_cells"      : len(covered_cells),
    "blind_cells"        : blind_total,
    "coverage_percent"   : float(coverage_pct),
    "placed_cameras"     : len(placed_cameras),
    "top_blind_grids"    : top_blind_grids,

    "best_coverage_camera": best_cov,

    "top10_candidates_by_marginal_coverage": ranked[:10],

    "method_note": (
        "Marginal coverage score = new walkable cells added by this camera FOV "
        "divided by total remaining blind-spot cells. Fully deterministic — "
        "no LLM involved. Use Agent 2 (LLM) only to decide WHEN to switch "
        "strategy; use this agent to decide WHERE to place the next camera "
        "for maximum coverage gain."
    )
}

out_path = os.path.join(OUT_DIR, f"coverage_expansion_iter_{iteration}.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved → {out_path}")

print("\n" + "=" * 60)
print("AGENT 6 COMPLETE")
print("=" * 60)