#!/usr/bin/env python3
"""
Agent 3 — Candidate Shortlister
Reads:
  - strategy_iter_N.json           (from Agent 2)
  - valid_camera_positions.ply     (Open3D binary, double x/y/z + uchar rgb)
  - grid_metadata.json             (pixel-space grid bounds)
  - terrain_mask.npy               (to know map H, W)

Writes:
  - shortlisted_candidates_iter_N.json   (top 100 candidates)

PLY coordinate system (confirmed from file inspection):
  x : world metres, range -120 to +570  (≈ col direction)
  y : world metres, range -122 to +280  (≈ row direction)
  z : elevation in metres, range -30 to +103

The terrain mask pixel grid uses (row, col). To map PLY world coords
to pixel space we normalise x→col and y→row using the world extents
baked into X_MIN/X_MAX/Y_MIN/Y_MAX below.

IMPORTANT: set these four values to match your actual terrain reconstruction
extents (the same bounding box used when generating terrain_mask.npy).
If you do not know them exactly, the fallback uses the PLY min/max which
gives a correct *relative* ordering — close enough for shortlisting.
"""

import os
import json
import struct
import numpy as np

# ===========================================================
# CONFIG
# ===========================================================

TOP_N             = 100
CANDIDATES_PER_BN = 25          # per-bottleneck pool before de-dup

# Blind-spot augmentation: pull candidates near Agent 6's top uncovered
# grid cells too, not just route bottlenecks. This is what lets coverage
# keep improving even when there are no routes (route bottlenecks are
# undefined with 0 routes) or when routes simply don't pass through real
# blind spots (a route sample was never a guarantee of full coverage).
CANDIDATES_PER_BLIND_GRID = 15
AGENT6_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\claude"
    r"\agent6_output"
)

# World→pixel mapping.
# Set these to your terrain reconstruction bounding box.
# If unknown, leave as None → derived automatically from PLY extents
# (gives correct relative ordering, may be slightly off at edges).
WORLD_X_MIN = None   # metres corresponding to col=0
WORLD_X_MAX = None   # metres corresponding to col=W-1
WORLD_Y_MIN = None   # metres corresponding to row=0  (check Y-flip below!)
WORLD_Y_MAX = None   # metres corresponding to row=H-1
FLIP_Y      = True   # True if world Y increases downward (same as image row)
                     # False if world Y increases upward (common in 3D tools)

VALID_PLY = (
    r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step2\valid_camera_positions.ply"
)

TERRAIN_NPY = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\Static_scripts\step4\terrain_mask.npy"
)

STRATEGY_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent2_output"
)

GRID_METADATA_PATH = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\terrain_intelligence_output"
    r"\grid_metadata.json"
)

OUT_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\agent3_output"
)

RUNTIME_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\runtime_state"
)

# Minimum distance (metres) a new candidate must keep from every already-
# placed camera. This is a coarse pre-filter, not a substitute for Agent 5's
# exact FOV-overlap check — two cameras this far apart CAN still overlap if
# aimed at each other, and two cameras closer than this CAN sometimes avoid
# overlap if aimed apart. It exists purely to stop Agent 3 from repeatedly
# handing Agent 5 a pool of candidates clustered right next to existing
# cameras, which is what was making the strict no-overlap constraint fail
# so often. Tune down for dense/small sites, up for sparse/large ones.
MIN_CAMERA_SEPARATION_M = 20.0

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================
# PLY READER  (binary_little_endian, double x/y/z + uchar rgb)
# ===========================================================

def read_ply(path: str) -> tuple:
    """
    Returns (x, y, z) as three float64 numpy arrays, shape (N,).
    Handles Open3D binary_little_endian format with property double x/y/z.
    """
    with open(path, "rb") as f:
        raw = f.read()

    # ---- parse header ----
    idx = 0
    lines = []
    while True:
        nl = raw.find(b"\n", idx)
        if nl == -1:
            break
        line = raw[idx:nl].decode("utf-8", errors="replace").strip()
        lines.append(line)
        idx = nl + 1
        if line == "end_header":
            data_start = nl + 1
            break

    n_vertices = 0
    format_    = "ascii"
    props      = []    # list of (dtype_char, name)

    dtype_map = {
        "double" : ("d", 8),
        "float"  : ("f", 4),
        "int"    : ("i", 4),
        "uint"   : ("I", 4),
        "short"  : ("h", 2),
        "ushort" : ("H", 2),
        "char"   : ("b", 1),
        "uchar"  : ("B", 1),
    }

    for line in lines:
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("format"):
            format_ = line.split()[1]
        elif line.startswith("property"):
            parts = line.split()
            ptype = parts[1]
            pname = parts[2]
            if ptype in dtype_map:
                props.append((dtype_map[ptype][0], dtype_map[ptype][1], pname))

    stride = sum(sz for _, sz, _ in props)

    endian = "<" if "little" in format_ else ">"
    fmt    = endian + "".join(dc for dc, _, _ in props)

    print(f"  PLY: {n_vertices} vertices  stride={stride}B  format={format_}")

    data = np.frombuffer(raw, dtype=np.uint8)[data_start : data_start + n_vertices * stride]
    data = data.reshape(n_vertices, stride)

    # Decode each property
    prop_arrays = {}
    offset = 0
    for dc, sz, name in props:
        col_bytes = data[:, offset : offset + sz]
        arr = np.frombuffer(col_bytes.tobytes(), dtype=f"{endian}{dc}")
        prop_arrays[name] = arr
        offset += sz

    return (prop_arrays["x"].astype(np.float64),
            prop_arrays["y"].astype(np.float64),
            prop_arrays["z"].astype(np.float64))


# ===========================================================
# HELPERS
# ===========================================================

def get_latest_strategy(directory: str):
    files = []
    for fn in os.listdir(directory):
        if fn.startswith("strategy_iter_") and fn.endswith(".json"):
            try:
                it = int(fn.replace("strategy_iter_", "").replace(".json", ""))
                files.append((it, fn))
            except ValueError:
                pass
    if not files:
        raise FileNotFoundError("No strategy_iter_*.json found in " + directory)
    files.sort(key=lambda x: x[0])
    return files[-1][0], os.path.join(directory, files[-1][1])


def world_to_pixel(wx, wy, H, W,
                   x_min, x_max, y_min, y_max, flip_y):
    """
    Convert world (x, y) coordinates to pixel (row, col).
    wx, wy : float or np.ndarray
    Returns (row, col) as float arrays.
    """
    col = (wx - x_min) / (x_max - x_min) * (W - 1)
    if flip_y:
        row = (wy - y_min) / (y_max - y_min) * (H - 1)
    else:
        row = (1.0 - (wy - y_min) / (y_max - y_min)) * (H - 1)
    return row, col


def grid_to_pixel_center(bn, grid_meta):
    """
    Convert a bottleneck entry to pixel-space centre (row_c, col_c).

    Handles every format the LLM can produce:
      - dict with 'grid_row' and 'grid_col' keys   e.g. {"grid_row":7,"grid_col":4}
      - dict with a 'location' string key           e.g. {"location":"(7, 4)","frequency":215}
      - dict with a 'grid' string key               e.g. {"grid":"7_4","frequency":215}
      - bare string                                  e.g. "(7, 4)" or "7_4"
      - list/tuple                                   e.g. [7, 4]
    """
    import re

    gr, gc = None, None

    if isinstance(bn, (list, tuple)) and len(bn) >= 2:
        gr, gc = int(bn[0]), int(bn[1])

    elif isinstance(bn, dict):
        if "grid_row" in bn and "grid_col" in bn:
            gr, gc = int(bn["grid_row"]), int(bn["grid_col"])
        elif "location" in bn:
            # e.g. "(7, 4)"
            nums = re.findall(r"\d+", str(bn["location"]))
            if len(nums) >= 2:
                gr, gc = int(nums[0]), int(nums[1])
        elif "grid" in bn:
            # e.g. "7_4"
            nums = re.findall(r"\d+", str(bn["grid"]))
            if len(nums) >= 2:
                gr, gc = int(nums[0]), int(nums[1])
        else:
            # Last resort: scan all string values for two numbers
            for v in bn.values():
                nums = re.findall(r"\d+", str(v))
                if len(nums) >= 2:
                    gr, gc = int(nums[0]), int(nums[1])
                    break

    elif isinstance(bn, str):
        nums = re.findall(r"\d+", bn)
        if len(nums) >= 2:
            gr, gc = int(nums[0]), int(nums[1])

    if gr is None or gc is None:
        return None, None

    key = f"{gr}_{gc}"
    if key not in grid_meta:
        return None, None

    meta = grid_meta[key]
    r_c  = (meta["row_start"] + meta["row_end"]) / 2.0
    c_c  = (meta["col_start"] + meta["col_end"]) / 2.0
    return r_c, c_c


# ===========================================================
# MAIN
# ===========================================================

print("\n" + "=" * 60)
print("AGENT 3 — CANDIDATE SHORTLISTER")
print("=" * 60)

# ---- Load terrain to get H, W ----------------------------
terrain = np.load(TERRAIN_NPY).astype(bool)
H, W    = terrain.shape
print(f"Terrain grid: H={H}  W={W}")

# ---- Load strategy ---------------------------------------
iteration, strategy_path = get_latest_strategy(STRATEGY_DIR)
print(f"Strategy file: {strategy_path}")

with open(strategy_path, "r") as f:
    strategy = json.load(f)

priority_bottlenecks = strategy["strategy"]["priority_bottlenecks"]
print(f"Priority bottlenecks from LLM: {len(priority_bottlenecks)}")
print(f"  Raw sample: {priority_bottlenecks[:2]}")

# ---- Load grid metadata ----------------------------------
with open(GRID_METADATA_PATH, "r") as f:
    grid_metadata = json.load(f)

# ---- Read PLY -------------------------------------------
print(f"\nReading PLY: {VALID_PLY}")
ply_x, ply_y, ply_z = read_ply(VALID_PLY)
N = len(ply_x)
print(f"Loaded {N} candidates")
print(f"  World X: {ply_x.min():.2f} → {ply_x.max():.2f}")
print(f"  World Y: {ply_y.min():.2f} → {ply_y.max():.2f}")
print(f"  World Z: {ply_z.min():.2f} → {ply_z.max():.2f}")

# ---- Resolve world extents --------------------------------
x_min = WORLD_X_MIN if WORLD_X_MIN is not None else float(ply_x.min())
x_max = WORLD_X_MAX if WORLD_X_MAX is not None else float(ply_x.max())
y_min = WORLD_Y_MIN if WORLD_Y_MIN is not None else float(ply_y.min())
y_max = WORLD_Y_MAX if WORLD_Y_MAX is not None else float(ply_y.max())

print(f"\nWorld→Pixel mapping extents:")
print(f"  X: [{x_min:.2f}, {x_max:.2f}]  → col [0, {W-1}]")
print(f"  Y: [{y_min:.2f}, {y_max:.2f}]  → row [0, {H-1}]  (flip_y={FLIP_Y})")

# Convert all PLY points to pixel space
ply_rows, ply_cols = world_to_pixel(
    ply_x, ply_y, H, W, x_min, x_max, y_min, y_max, FLIP_Y
)

# Clip to valid pixel range
ply_rows = np.clip(ply_rows, 0, H - 1)
ply_cols = np.clip(ply_cols, 0, W - 1)

# ---- Find top candidates near each bottleneck ------------
selected_indices = set()

for bn in priority_bottlenecks:
    rc, cc = grid_to_pixel_center(bn, grid_metadata)
    if rc is None:
        print(f"  [WARN] Could not resolve bottleneck: {bn}")
        continue

    dist    = np.sqrt((ply_rows - rc) ** 2 + (ply_cols - cc) ** 2)
    nearest = np.argsort(dist)[:CANDIDATES_PER_BN]
    selected_indices.update(nearest.tolist())

selected_indices = sorted(selected_indices)
print(f"\nCandidates after per-bottleneck union: {len(selected_indices)}")

# ---- Augment with candidates near blind-spot grids (Agent 6) ----------
# Uses the SAME grid_metadata pixel centers as the bottleneck lookup above,
# just keyed by Agent 6's top_blind_grids instead of the LLM's
# priority_bottlenecks. Silently skipped if Agent 6 hasn't produced output
# yet (e.g. iteration 0) — this is additive, not a replacement.
def get_latest_agent6_output(directory):
    if not os.path.isdir(directory):
        return None
    files = []
    for fn in os.listdir(directory):
        if fn.startswith("coverage_expansion_iter_") and fn.endswith(".json"):
            try:
                it = int(fn.replace("coverage_expansion_iter_", "").replace(".json", ""))
                files.append((it, fn))
            except ValueError:
                pass
    if not files:
        return None
    files.sort(key=lambda x: x[0])
    with open(os.path.join(directory, files[-1][1])) as f:
        return json.load(f)

coverage_doc = get_latest_agent6_output(AGENT6_DIR)
blind_added = 0
if coverage_doc and coverage_doc.get("top_blind_grids"):
    for bg in coverage_doc["top_blind_grids"]:
        key = f"{bg['grid_row']}_{bg['grid_col']}"
        if key not in grid_metadata:
            continue
        meta = grid_metadata[key]
        rc = (meta["row_start"] + meta["row_end"]) / 2.0
        cc = (meta["col_start"] + meta["col_end"]) / 2.0

        dist    = np.sqrt((ply_rows - rc) ** 2 + (ply_cols - cc) ** 2)
        nearest = np.argsort(dist)[:CANDIDATES_PER_BLIND_GRID]
        before  = len(selected_indices)
        selected_indices = sorted(set(selected_indices) | set(nearest.tolist()))
        blind_added += len(selected_indices) - before
    print(f"Candidates added from {len(coverage_doc['top_blind_grids'])} blind-spot grids: {blind_added}")
else:
    print("No Agent 6 output yet — shortlist is route-bottleneck-only this iteration.")

print(f"Candidates after blind-spot augmentation: {len(selected_indices)}")

# ---- Drop candidates too close to already-placed cameras --------------
placed_path = os.path.join(RUNTIME_DIR, "placed_cameras.json")
if os.path.exists(placed_path):
    with open(placed_path) as f:
        placed_cameras = json.load(f)
else:
    placed_cameras = []

if placed_cameras:
    # Placed cameras only store pixel row/col, not world x/y, so convert
    # the candidate pool's world coords to the same pixel space to compare
    # (row, col) directly — avoids needing the inverse pixel->world mapping.
    approx_scale_m_per_px = (x_max - x_min) / max(W - 1, 1)
    min_sep_px = MIN_CAMERA_SEPARATION_M / approx_scale_m_per_px

    idx_arr = np.array(selected_indices)
    keep_mask = np.ones(len(idx_arr), dtype=bool)
    for cam in placed_cameras:
        d = np.sqrt((ply_rows[idx_arr] - cam["row"]) ** 2 +
                    (ply_cols[idx_arr] - cam["col"]) ** 2)
        keep_mask &= (d >= min_sep_px)

    before = len(selected_indices)
    selected_indices = idx_arr[keep_mask].tolist()
    print(f"Dropped {before - len(selected_indices)} candidate(s) within "
          f"{MIN_CAMERA_SEPARATION_M}m of an existing camera "
          f"({len(placed_cameras)} placed so far). "
          f"{len(selected_indices)} remain.")

    if not selected_indices:
        print("[WARNING] Separation filter removed every candidate — "
              "falling back to the unfiltered pool so Agent 4/5 still have "
              "something to evaluate (Agent 5's exact overlap check will "
              "have the final say).")
        selected_indices = idx_arr.tolist()

# ---- Trim to TOP_N if needed (by distance to NEAREST area of interest) ---
# Nearest-to-ANY-center, not nearest-to-the-mean-center: a candidate near a
# blind-spot grid but far from the bottleneck cluster should still survive
# the trim, since it's serving a different (and equally valid) goal.
if len(selected_indices) > TOP_N:
    centers = []
    for bn in priority_bottlenecks:
        rc, cc = grid_to_pixel_center(bn, grid_metadata)
        if rc is not None:
            centers.append((rc, cc))
    if coverage_doc and coverage_doc.get("top_blind_grids"):
        for bg in coverage_doc["top_blind_grids"]:
            key = f"{bg['grid_row']}_{bg['grid_col']}"
            if key in grid_metadata:
                meta = grid_metadata[key]
                centers.append(((meta["row_start"] + meta["row_end"]) / 2.0,
                                 (meta["col_start"] + meta["col_end"]) / 2.0))

    if centers:
        idx_arr    = np.array(selected_indices)
        center_r   = np.array([c[0] for c in centers])
        center_c   = np.array([c[1] for c in centers])
        # distance from every candidate to every center -> (n_candidates, n_centers)
        d_all      = np.sqrt(
            (ply_rows[idx_arr][:, None] - center_r[None, :]) ** 2 +
            (ply_cols[idx_arr][:, None] - center_c[None, :]) ** 2
        )
        d_nearest  = d_all.min(axis=1)          # distance to closest area of interest
        order      = np.argsort(d_nearest)
        selected_indices = idx_arr[order[:TOP_N]].tolist()
    else:
        selected_indices = selected_indices[:TOP_N]

print(f"Final shortlist: {len(selected_indices)} candidates")

# ---- Build output ----------------------------------------
candidates = []
for idx in selected_indices:
    candidates.append({
        "candidate_id" : int(idx),
        "row"          : float(ply_rows[idx]),
        "col"          : float(ply_cols[idx]),
        "z_ground"     : float(ply_z[idx]),    # elevation in metres
        "world_x"      : float(ply_x[idx]),
        "world_y"      : float(ply_y[idx]),
        "world_z"      : float(ply_z[idx])
    })

output = {
    "iteration"            : iteration,
    "total_ply_candidates" : N,
    "shortlisted_count"    : len(candidates),
    "world_extents"        : {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "flip_y": FLIP_Y
    },
    "terrain_grid"         : {"H": H, "W": W},
    "candidates"           : candidates
}

out_path = os.path.join(OUT_DIR, f"shortlisted_candidates_iter_{iteration}.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved → {out_path}")
print("\n" + "=" * 60)
print("AGENT 3 COMPLETE")
print("=" * 60)