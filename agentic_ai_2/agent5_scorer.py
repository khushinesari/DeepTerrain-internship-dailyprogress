#!/usr/bin/env python3
"""
Agent 5 — Camera Scorer, Placer and Visualizer

FOV is computed with RAY CASTING:
  For angular steps spanning the camera's azimuth sweep (az ± FOV_DEG), a ray
  is marched outward from the camera in fixed-size steps from R_near to R_far.
  Every cell the ray passes through is marked visible; marching along that
  ray STOPS the instant it hits an obstacle cell (everything beyond that
  point on the ray is occluded / shadowed and is not marked visible).
  This is the classic per-ray "march and stop at first obstacle" approach,
  as opposed to filling the whole wedge region and then checking a straight
  line-of-sight per pixel.

  Angular resolution is chosen automatically so that neighboring rays are
  no more than ~1 pixel apart at R_far (the widest part of the wedge),
  which avoids leaving unmarked gaps between rays.

Why the very first raycaster (before this one) was wrong:
  - It only recorded the terminal hit point of each ray → thin ring
  - It treated dz as a z-velocity component which caused rays to barely
    drop (flat cone) so all hits clustered at max range

Writes per iteration:
  - agent5_output/placement_result_iter_N.json
  - agent5_output/map_iter_N.png          (this iteration: FOV + elim + surviving)
  - agent5_output/map_cumulative_N.png    (all cameras so far)
  - runtime_state/placed_cameras.json
  - runtime_state/paths_iter_{N+1}.json
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

# ===========================================================
# CONFIG
# ===========================================================

FOV_DEG    = 30
TILT_DEG   = 5
MAX_RANGE  = 150.0
POLE_H     = 10.0

# Ray casting resolution knobs
RAY_STEP_PX      = 0.5   # marching step size along each ray, in pixels
MAX_ANGULAR_STEP = 0.5   # cap on angular spacing between rays, in degrees

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
    r"\agentic_ai\agent4_output"
)
STRATEGY_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai\agent2_output"
)
ORIGINAL_PATHS = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai\multi_astar_v2\paths_1.json"
)
RUNTIME_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai\runtime_state"
)
OUT_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai\agent5_output"
)

os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CAM_COLORS = [
    "#FF4444","#FF8C00","#FFD700","#00E5FF","#B44FFF",
    "#00FF99","#FF69B4","#7FFF00","#FF6347","#40E0D0",
]

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


def load_routes(runtime_dir, original_path):
    iters = []
    for fn in os.listdir(runtime_dir):
        if fn.startswith("paths_iter_") and fn.endswith(".json"):
            try:
                it = int(fn.replace("paths_iter_","").replace(".json",""))
                iters.append((it, fn))
            except ValueError:
                pass
    path = os.path.join(runtime_dir, sorted(iters)[-1][1]) if iters else original_path
    print(f"Routes from: {path}")
    with open(path) as f:
        return json.load(f)


def fov_sector_cells(cam_row, cam_col, cam_z, azimuth_deg,
                     H, W, obstacle, pixel_scale):
    """
    FOV via ray casting.

    The camera at height cam_z looks toward azimuth_deg (0=N, clockwise).
    The cone has half-angle FOV_DEG around a central ray tilted TILT_DEG
    below horizontal, giving ground bounds:
        R_near = cam_z / tan(tilt + fov)   [steepest ray → nearest ground hit]
        R_far  = min(cam_z / tan(tilt-fov), MAX_RANGE)  [shallowest → farthest]
    (converted from meters to pixels via pixel_scale).

    Rays are cast at even angular steps across [azimuth-FOV, azimuth+FOV].
    Each ray is marched from R_near to R_far in RAY_STEP_PX increments; every
    cell the march passes through is marked visible, and marching along that
    ray stops as soon as it enters an obstacle cell (occlusion — anything
    farther along that ray is in shadow and is never marked visible).

    Returns a frozenset of (row, col) visible ground cells.
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

    # Choose angular step so neighboring rays are <= ~1px apart at R_far
    # (arc length between two rays spaced `d` radians apart at radius R_far
    # is R_far * d; solve for d given a 1px target, then convert to degrees).
    if R_far > 0:
        d_rad = min(np.deg2rad(MAX_ANGULAR_STEP), 1.0 / R_far)
    else:
        d_rad = np.deg2rad(MAX_ANGULAR_STEP)
    d_rad = max(d_rad, 1e-4)  # guard against zero/negative steps

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
            r_i = int(round(pr))
            c_i = int(round(pc))

            if not (0 <= r_i < H and 0 <= c_i < W):
                break  # ray left the map — nothing further to see on it

            if obstacle[r_i, c_i]:
                break  # occluded — stop marching this ray, no cells beyond

            cells.add((r_i, c_i))
            dist += RAY_STEP_PX

    return frozenset(cells)


def make_base(terrain, obstacle):
    H, W = terrain.shape
    c = np.zeros((H, W, 3), dtype=np.float32)
    c[:] = [0.06, 0.06, 0.06]
    c[terrain & ~obstacle] = [0.14, 0.32, 0.14]
    c[obstacle.astype(bool)] = [0.26, 0.15, 0.07]
    return c


def routes_layer(route_list, H, W, rgb, alpha):
    L = np.zeros((H, W, 4), dtype=np.float32)
    for route in route_list:
        for r, c in route:
            if 0 <= r < H and 0 <= c < W:
                L[r, c] = [*rgb, alpha]
    return L


def fov_layer_img(cells, H, W, rgb, alpha):
    L = np.zeros((H, W, 4), dtype=np.float32)
    for r, c in cells:
        if 0 <= r < H and 0 <= c < W:
            L[r, c] = [*rgb, alpha]
    return L


def draw_cam(ax, cam, hex_c, label, arrow_len=26):
    r, c  = cam["row"], cam["col"]
    az    = np.deg2rad(cam["azimuth_deg"])
    dr_a  = -np.cos(az) * arrow_len
    dc_a  =  np.sin(az) * arrow_len
    ax.add_patch(plt.Circle((c, r), 18, color=hex_c, fill=False,
                             lw=1.2, alpha=0.30, zorder=6))
    ax.add_patch(plt.Circle((c, r), 10, color=hex_c, fill=True,
                             lw=2, ec="white", zorder=7))
    ax.annotate("", xy=(c+dc_a, r+dr_a), xytext=(c, r),
                arrowprops=dict(arrowstyle="-|>", color=hex_c,
                                lw=2.2, mutation_scale=15), zorder=8)
    ax.text(c+20, r-20, label, fontsize=7.5, color=hex_c,
            fontweight="bold", va="bottom", zorder=9,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])


# ===========================================================
# MAIN
# ===========================================================

print("\n" + "="*60)
print("AGENT 5 — CAMERA SCORER AND PLACER")
print("="*60)

terrain  = np.load(TERRAIN_NPY).astype(bool)
obstacle = np.load(OBSTACLE_NPY).astype(bool)
H, W     = terrain.shape
walkable = terrain & ~obstacle

with open(ORIGINAL_PATHS) as f:
    all_routes_original = json.load(f)

base_canvas = make_base(terrain, obstacle)

# ---- Load FOV results from Agent 4 -----------------------
iteration, fov_path = get_latest_file(FOV_DIR, "fov_results_iter_")
print(f"FOV results : {fov_path}")
with open(fov_path) as f:
    fov_data = json.load(f)
candidates   = fov_data["candidates"]
total_routes = fov_data["total_routes"]

# Pixel scale: read the exact value Agent 4 used for its FOV scoring, so
# Agent 5's final placement geometry is never out of sync with the scores
# that picked this candidate. (Previously this was a separately hardcoded
# 690/(W-1), which could silently disagree with Agent 4's own PIXEL_SCALE_M
# whenever the terrain reconstruction's world extents didn't match 690m.)
PIXEL_SCALE = fov_data.get("pixel_scale_m", 690.0 / (W - 1))
print(f"Map {H}×{W}  pixel_scale={PIXEL_SCALE:.4f} m/px  (from Agent 4 output)")

# ---- Load strategy weights & decision ---------------------
_, strat_path = get_latest_file(STRATEGY_DIR, "strategy_iter_")
with open(strat_path) as f:
    strategy = json.load(f)
route_w           = float(strategy["strategy"]["route_weights"])
coverage_w        = float(strategy["strategy"]["coverage_weights"])
strategy_decision = strategy["strategy"].get("strategy_decision", "route_elimination")
print(f"Weights: route={route_w}  coverage={coverage_w}")
print(f"LLM strategy_decision: {strategy_decision}")

# ---- Load current surviving routes -----------------------
routes = load_routes(RUNTIME_DIR, ORIGINAL_PATHS)
print(f"Routes remaining: {len(routes)}")

# ---- Load cameras placed in earlier iterations -------------
# Needed BEFORE scoring so coverage-expansion mode can rank candidates by
# genuinely NEW coverage (cells no previously placed camera already sees),
# and so coverage-so-far is always computed from every placed camera
# directly (by ray-casting its stored geometry) rather than only the ones
# that happen to reappear in this iteration's freshly reshortlisted
# candidate list — Agent 3 reshortlists every iteration, so a camera placed
# two iterations ago will usually NOT be present in today's candidate list,
# and cross-referencing by candidate_id would silently drop its coverage.
placed_path = os.path.join(RUNTIME_DIR, "placed_cameras.json")
if os.path.exists(placed_path):
    with open(placed_path) as f:
        placed_cameras = json.load(f)
else:
    placed_cameras = []

total_walkable_cells = int(np.sum(walkable))
covered_cells_so_far = set()
for cam in placed_cameras:
    covered_cells_so_far |= fov_sector_cells(
        cam["row"], cam["col"], cam["cam_z"], cam["azimuth_deg"],
        H, W, obstacle, PIXEL_SCALE
    )
coverage_so_far_pct = 100.0 * len(covered_cells_so_far) / max(total_walkable_cells, 1)
print(f"Coverage from {len(placed_cameras)} previously placed camera(s): "
      f"{len(covered_cells_so_far)} cells ({coverage_so_far_pct:.1f}%)")

# ---- Score candidates ------------------------------------
if strategy_decision == "coverage_expansion":
    # Rank purely by marginal NEW coverage this candidate would add beyond
    # what's already seen — ignores route_score entirely, since the LLM has
    # judged that plugging blind spots matters more than blocking more
    # routes right now.
    print("Coverage-expansion mode: ranking candidates by marginal new coverage.")
    for cand in candidates:
        cand_fov = frozenset(map(tuple, cand.get("fov_cells", [])))
        marginal = cand_fov - covered_cells_so_far
        cand["marginal_new_cells"] = len(marginal)
        cand["final_score"] = len(marginal) / max(total_walkable_cells, 1)
    candidates.sort(key=lambda c: c["final_score"], reverse=True)
else:
    print("Route-elimination mode: ranking candidates by LLM-weighted score.")
    for cand in candidates:
        cand["marginal_new_cells"] = None
        cand["final_score"] = route_w * cand["route_score"] + coverage_w * cand["coverage_score"]
    candidates.sort(key=lambda c: c["final_score"], reverse=True)

best = candidates[0]

print(f"\nBest candidate id={best['candidate_id']}")
print(f"  row={best['row']:.1f}  col={best['col']:.1f}  az={best['best_azimuth_deg']:.1f}°")
if strategy_decision == "coverage_expansion":
    print(f"  marginal_new_cells={best['marginal_new_cells']}  final(marginal_score)={best['final_score']:.4f}")
else:
    print(f"  route_score={best['route_score']:.4f}  coverage_score={best['coverage_score']:.4f}  final={best['final_score']:.4f}")

# ---- Compute FOV via ray casting --------------------------
print("\nComputing FOV via ray casting...")
cam_z_val = best["z_ground"] + POLE_H
fov_set   = fov_sector_cells(
    best["row"], best["col"], cam_z_val,
    best["best_azimuth_deg"],
    H, W, obstacle, PIXEL_SCALE
)
print(f"FOV visible cells: {len(fov_set)}")

# ---- Eliminate routes ------------------------------------
surviving   = []
elim_routes = []
for route in routes:
    rc = set(map(tuple, route))
    (elim_routes if not rc.isdisjoint(fov_set) else surviving).append(route)

eliminated = len(elim_routes)
print(f"Routes eliminated: {eliminated}  surviving: {len(surviving)}")

# ---- Persist placed camera (placed_cameras was already loaded above) -----
new_entry = {
    "iteration"          : iteration,
    "candidate_id"       : best["candidate_id"],
    "row"                : best["row"],
    "col"                : best["col"],
    "z_ground"           : best["z_ground"],
    "cam_z"              : cam_z_val,
    "azimuth_deg"        : best["best_azimuth_deg"],
    "route_score"        : best["route_score"],
    "coverage_score"     : best["coverage_score"],
    "final_score"        : best["final_score"],
    "strategy_decision"  : strategy_decision,
    "marginal_new_cells" : best.get("marginal_new_cells"),
    "routes_eliminated"  : eliminated,
    "fov_cell_count"     : len(fov_set),
}
placed_cameras.append(new_entry)

with open(placed_path, "w") as f:
    json.dump(placed_cameras, f, indent=2)
print(f"placed_cameras.json updated ({len(placed_cameras)} cameras total)")

# ---- Write surviving routes ------------------------------
next_iter = iteration + 1
with open(os.path.join(RUNTIME_DIR, f"paths_iter_{next_iter}.json"), "w") as f:
    json.dump(surviving, f, indent=2)

# ---- Write result JSON -----------------------------------
result = {
    "iteration"          : iteration,
    "strategy_decision"  : strategy_decision,
    "route_weight"       : route_w,
    "coverage_weight"    : coverage_w,
    "coverage_before_pct": coverage_so_far_pct,
    "camera_placed"      : new_entry,
    "routes_before"     : len(routes),
    "routes_eliminated" : eliminated,
    "routes_after"      : len(surviving),
    "fov_cell_count"    : len(fov_set),
    "total_cameras"     : len(placed_cameras),
    "all_scores_ranked" : [
        {"candidate_id": c["candidate_id"], "final_score": c["final_score"],
         "route_score": c["route_score"], "coverage_score": c["coverage_score"]}
        for c in candidates[:10]
    ]
}
with open(os.path.join(OUT_DIR, f"placement_result_iter_{iteration}.json"), "w") as f:
    json.dump(result, f, indent=2)

# ===========================================================
# VISUALIZATION
# ===========================================================

print("\nRendering visualizations...")

hex_c   = CAM_COLORS[iteration % len(CAM_COLORS)]
rgb_c   = mcolors.to_rgb(hex_c)
cam_idx = len(placed_cameras) - 1

# -----------------------------------------------------------
# MAP 1 — This iteration
# -----------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(20, 13), dpi=130)
fig1.patch.set_facecolor("#0d0d0d")
ax1.set_facecolor("#0d0d0d")

ax1.imshow(base_canvas, interpolation="nearest", zorder=1)
ax1.imshow(fov_layer_img(fov_set, H, W, rgb_c, 0.45),
           interpolation="nearest", zorder=2)
if elim_routes:
    ax1.imshow(routes_layer(elim_routes, H, W, rgb_c, 0.88),
               interpolation="nearest", zorder=3)
if surviving:
    ax1.imshow(routes_layer(surviving, H, W, (1.,1.,1.), 0.75),
               interpolation="nearest", zorder=4)

for prev in placed_cameras[:-1]:
    ax1.add_patch(plt.Circle((prev["col"], prev["row"]), 6,
                              color="white", fill=True, alpha=0.28, zorder=5))

draw_cam(ax1, new_entry, hex_c,
         f'CAM {cam_idx+1}\naz={new_entry["azimuth_deg"]:.0f}°')

stats1 = (
    f'Iteration      : {iteration}\n'
    f'Camera #{cam_idx+1}       : id={best["candidate_id"]}\n'
    f'Azimuth        : {new_entry["azimuth_deg"]:.0f}°\n'
    f'FOV cells      : {len(fov_set)}\n'
    f'Routes before  : {len(routes)}\n'
    f'Eliminated NOW : {eliminated}\n'
    f'Surviving      : {len(surviving)}\n'
    f'Route score    : {best["route_score"]:.4f}\n'
    f'Coverage score : {best["coverage_score"]:.4f}\n'
    f'Final score    : {best["final_score"]:.4f}'
)
ax1.text(0.01, 0.99, stats1, transform=ax1.transAxes,
         fontsize=8.5, color="white", va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", fc="#000000", alpha=0.72, ec="#444"), zorder=10)

leg1 = [
    mpatches.Patch(color=[0.14,0.32,0.14], label="Walkable terrain"),
    mpatches.Patch(color=[0.26,0.15,0.07], label="Obstacle"),
    mpatches.Patch(color=(*rgb_c, 0.45),   label=f"FOV sector ({len(fov_set)} cells)"),
    mpatches.Patch(color=(*rgb_c, 0.88),   label=f"Eliminated routes ({eliminated})"),
    mpatches.Patch(color=(1.,1.,1.,0.75),  label=f"Surviving routes ({len(surviving)})"),
]
ax1.legend(handles=leg1, loc="lower right", fontsize=8,
           facecolor="#111", edgecolor="#444", labelcolor="white", framealpha=0.85)
ax1.set_title(
    f'Iteration {iteration}  |  Camera {cam_idx+1}  |  az={new_entry["azimuth_deg"]:.0f}°  |  '
    f'{eliminated} routes eliminated  →  {len(surviving)} surviving',
    color=hex_c, fontsize=12, fontweight="bold", pad=8)
ax1.axis("off")
fig1.tight_layout(pad=0.3)
p1 = os.path.join(OUT_DIR, f"map_iter_{iteration}.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
plt.close(fig1)
print(f"  Saved → {p1}")

# -----------------------------------------------------------
# MAP 2 — Cumulative: rebuild all FOV sectors for all placed cameras
# -----------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(20, 13), dpi=130)
fig2.patch.set_facecolor("#0d0d0d")
ax2.set_facecolor("#0d0d0d")
ax2.imshow(base_canvas, interpolation="nearest", zorder=1)

# Rebuild elimination sequence from original routes
remaining_idx = list(range(len(all_routes_original)))
cam_elim_sets = []

for cam in placed_cameras:
    cam_fov = fov_sector_cells(
        cam["row"], cam["col"], cam["cam_z"],
        cam["azimuth_deg"],
        H, W, obstacle, PIXEL_SCALE
    )
    elim_now = []
    alive    = []
    for ri in remaining_idx:
        rc = set(map(tuple, all_routes_original[ri]))
        (elim_now if not rc.isdisjoint(cam_fov) else alive).append(ri)
    cam_elim_sets.append((cam_fov, elim_now))
    remaining_idx = alive

for i, (cam, (cam_fov, elim_idxs)) in enumerate(zip(placed_cameras, cam_elim_sets)):
    c_hex = CAM_COLORS[i % len(CAM_COLORS)]
    c_rgb = mcolors.to_rgb(c_hex)
    if cam_fov:
        ax2.imshow(fov_layer_img(cam_fov, H, W, c_rgb, 0.25),
                   interpolation="nearest", zorder=2)
    if elim_idxs:
        ax2.imshow(routes_layer([all_routes_original[ri] for ri in elim_idxs],
                                H, W, c_rgb, 0.80),
                   interpolation="nearest", zorder=3)

if remaining_idx:
    ax2.imshow(routes_layer([all_routes_original[ri] for ri in remaining_idx],
                            H, W, (1.,1.,1.), 0.80),
               interpolation="nearest", zorder=4)

for i, cam in enumerate(placed_cameras):
    draw_cam(ax2, cam, CAM_COLORS[i % len(CAM_COLORS)],
             f'C{i+1}\n{cam["azimuth_deg"]:.0f}°\n−{cam["routes_eliminated"]}r',
             arrow_len=22)

total_elim = len(all_routes_original) - len(remaining_idx)
pct        = 100.0 * total_elim / max(len(all_routes_original), 1)

stats2 = (
    f'Cameras placed   : {len(placed_cameras)}\n'
    f'Total routes     : {len(all_routes_original)}\n'
    f'Eliminated total : {total_elim}  ({pct:.1f}%)\n'
    f'Surviving        : {len(remaining_idx)}'
)
ax2.text(0.01, 0.99, stats2, transform=ax2.transAxes,
         fontsize=9.5, color="white", va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", fc="#000000", alpha=0.72, ec="#444"), zorder=10)

leg2 = [
    mpatches.Patch(color=[0.14,0.32,0.14], label="Walkable terrain"),
    mpatches.Patch(color=[0.26,0.15,0.07], label="Obstacle"),
    mpatches.Patch(color=(1.,1.,1.,0.8),   label=f"Surviving routes ({len(remaining_idx)})"),
]
for i, cam in enumerate(placed_cameras):
    leg2.append(mpatches.Patch(
        color=CAM_COLORS[i % len(CAM_COLORS)],
        label=f'Cam {i+1}  iter={cam["iteration"]}  az={cam["azimuth_deg"]:.0f}°  −{cam["routes_eliminated"]}r'
    ))
ax2.legend(handles=leg2, loc="lower right", fontsize=7.5,
           facecolor="#111", edgecolor="#444", labelcolor="white", framealpha=0.85)
ax2.set_title(
    f'Cumulative — {len(placed_cameras)} cameras  |  '
    f'{total_elim}/{len(all_routes_original)} routes eliminated ({pct:.1f}%)  |  '
    f'{len(remaining_idx)} surviving',
    color="white", fontsize=12, fontweight="bold", pad=8)
ax2.axis("off")
fig2.tight_layout(pad=0.3)
p2 = os.path.join(OUT_DIR, f"map_cumulative_{iteration}.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
plt.close(fig2)
print(f"  Saved → {p2}")

print("\n" + "="*60)
print("AGENT 5 COMPLETE")
print("="*60)