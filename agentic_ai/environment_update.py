#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

BEST_CAMERA_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\visibility_output\best_camera.json"
CURRENT_ROUTES_JSON = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\multi_astar_v2\paths_1.json"

TERRAIN_MASK_PATH = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\terrain_mask.npy"
OBSTACLE_MASK_PATH = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\obstacle_mask.npy"

RUNTIME_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\runtime_state"

MAX_RANGE = 150
FOV_DEG = 30

# ==========================================================
# HELPERS
# ==========================================================

def build_fov_mask(row, col, azimuth_deg, H, W, obstacle_mask):

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
# MAIN
# ==========================================================

os.makedirs(RUNTIME_DIR, exist_ok=True)

terrain_mask = np.load(TERRAIN_MASK_PATH)
obstacle_mask = np.load(OBSTACLE_MASK_PATH)

H, W = terrain_mask.shape

with open(BEST_CAMERA_JSON, "r") as f:
    best_camera = json.load(f)

with open(CURRENT_ROUTES_JSON, "r") as f:
    routes = json.load(f)

fov_mask_path = os.path.join(RUNTIME_DIR, "fov_mask.npy")
placed_cameras_path = os.path.join(RUNTIME_DIR, "placed_cameras.json")

if os.path.exists(fov_mask_path):
    runtime_fov_mask = np.load(fov_mask_path)
else:
    runtime_fov_mask = np.zeros_like(terrain_mask, dtype=np.uint8)

if os.path.exists(placed_cameras_path):
    with open(placed_cameras_path, "r") as f:
        placed_cameras = json.load(f)
else:
    placed_cameras = []

camera_id = len(placed_cameras) + 1

iteration_dir = os.path.join(
    RUNTIME_DIR,
    f"iteration_{camera_id}"
)
os.makedirs(iteration_dir, exist_ok=True)

row = best_camera["bev_row"]
col = best_camera["bev_col"]
azimuth = best_camera["azimuth"]

camera_mask = build_fov_mask(
    row=row,
    col=col,
    azimuth_deg=azimuth,
    H=H,
    W=W,
    obstacle_mask=obstacle_mask
)

np.save(
    os.path.join(
        iteration_dir,
        f"camera_{camera_id}_mask.npy"
    ),
    camera_mask
)

camera_record = {
    "camera_id": camera_id,
    "pole_id": best_camera["pole_id"],
    "bev_row": row,
    "bev_col": col,
    "azimuth": azimuth,
    "world_x": best_camera.get("world_x"),
    "world_y": best_camera.get("world_y"),
    "world_z": best_camera.get("world_z")
}

placed_cameras.append(camera_record)

with open(placed_cameras_path, "w") as f:
    json.dump(placed_cameras, f, indent=2)

runtime_fov_mask = np.maximum(
    runtime_fov_mask,
    camera_mask
)

np.save(
    fov_mask_path,
    runtime_fov_mask
)

blocked_routes = []
remaining_routes = []

for route in routes:

    visible = False

    for r, c in route:

        if (
            0 <= r < H and
            0 <= c < W and
            camera_mask[r, c]
        ):
            visible = True
            break

    if visible:
        blocked_routes.append(route)
    else:
        remaining_routes.append(route)

with open(
    os.path.join(iteration_dir, "blocked_routes.json"),
    "w"
) as f:
    json.dump(blocked_routes, f)

with open(
    os.path.join(iteration_dir, "remaining_routes.json"),
    "w"
) as f:
    json.dump(remaining_routes, f)

covered_cells = np.sum(
    (runtime_fov_mask == 1) & (terrain_mask == 1)
)

terrain_cells = np.sum(
    terrain_mask == 1
)

coverage_percent = (
    100.0 * covered_cells / terrain_cells
    if terrain_cells > 0 else 0
)

environment_state = {
    "iteration": camera_id,
    "placed_cameras": len(placed_cameras),
    "routes_before": len(routes),
    "routes_removed": len(blocked_routes),
    "routes_remaining": len(remaining_routes),
    "coverage_percent": round(coverage_percent, 2),
    "blind_spot_percent": round(
        100 - coverage_percent,
        2
    )
}

with open(
    os.path.join(
        iteration_dir,
        "environment_state.json"
    ),
    "w"
) as f:
    json.dump(
        environment_state,
        f,
        indent=2
    )

img = np.zeros((H, W, 3), dtype=np.uint8)

img[terrain_mask == 1] = [0, 255, 0]
img[obstacle_mask == 1] = [255, 0, 0]
img[runtime_fov_mask == 1] = [255, 255, 0]

for route in remaining_routes:
    for r, c in route:
        if 0 <= r < H and 0 <= c < W:
            img[r, c] = [0, 0, 255]

plt.figure(figsize=(14, 10))
plt.imshow(img)

for cam in placed_cameras:

    plt.scatter(
        cam["bev_col"],
        cam["bev_row"],
        s=80,
        c="magenta"
    )

    plt.text(
        cam["bev_col"] + 5,
        cam["bev_row"],
        f"C{cam['camera_id']}",
        color="white",
        fontsize=8
    )

plt.title(
    f"Updated Environment Iteration {camera_id}"
)

plt.axis("off")

plt.savefig(
    os.path.join(
        iteration_dir,
        f"updated_environment_iter_{camera_id}.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Environment update complete.")
print(environment_state)
