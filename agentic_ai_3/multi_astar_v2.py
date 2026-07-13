#!/usr/bin/env python3

"""
DeepTerrain Interactive Multi-A*

Pipeline
--------
Terrain
    ↓
Interactive Start Corridor
    ↓
Interactive Goal Corridor
    ↓
Corridor Band Generation
    ↓
Obstacle Removal
    ↓
Farthest Point Sampling
    ↓
Large Scale Multi-A*
    ↓
Route Database
"""

import os
import cv2
import json
import heapq
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

# ============================================================
# CONFIG
# ============================================================

TERRAIN_NPY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\terrain_mask.npy"

OBSTACLE_NPY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step3\output\obstacle_mask.npy"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\multi_astar_v2"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# Corridor Parameters
# ------------------------------------------------------------

CORRIDOR_WIDTH = 60

# ------------------------------------------------------------
# Number of FPS Samples
# ------------------------------------------------------------

MAX_START_POINTS = 50
MAX_GOAL_POINTS = 50

# ------------------------------------------------------------
# A*
# ------------------------------------------------------------

ALLOW_DIAGONAL = True

DIAGONAL_COST = 1.414

# ------------------------------------------------------------
# FPS Seed
# ------------------------------------------------------------

RANDOM_SEED = 42
print()
print("="*60)
print("INTERACTIVE MULTI A*")
print("="*60)

terrain = np.load(
    TERRAIN_NPY
).astype(bool)

obstacle = np.load(
    OBSTACLE_NPY
).astype(bool)

H = max(

    terrain.shape[0],

    obstacle.shape[0]

)

W = max(

    terrain.shape[1],

    obstacle.shape[1]

)
def pad(mask):

    return np.pad(

        mask,

        (

            (0,H-mask.shape[0]),

            (0,W-mask.shape[1])

        ),

        constant_values=False

    )

terrain = pad(terrain)

obstacle = pad(obstacle)
walkable = terrain & (~obstacle)

print()

print(

    f"Terrain Size : {H} x {W}"

)

print(

    f"Walkable Cells : {np.sum(walkable):,}"

)
rgb = np.zeros(

    (H,W,3),

    dtype=np.uint8

)

rgb[:] = [120,120,120]

rgb[terrain] = [0,255,0]

rgb[obstacle] = [255,0,0]
start_points = []

goal_points = []

MODE = "START"

drawing_finished = False
# ============================================================
# PART 2
# INTERACTIVE CORRIDOR DRAWING
# ============================================================

print()
print("=" * 60)
print("DRAW START AND GOAL CORRIDORS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12,12))
def redraw():

    ax.clear()

    ax.imshow(rgb)

    # ---------------------------------------------------------
    # START POLYLINE
    # ---------------------------------------------------------

    if len(start_points) > 0:

        pts = np.array(

            [[c,r] for r,c in start_points]

        )

        ax.plot(

            pts[:,0],

            pts[:,1],

            color="cyan",

            linewidth=3,

            label="Start Corridor"

        )

        ax.scatter(

            pts[:,0],

            pts[:,1],

            color="cyan",

            s=40

        )

    # ---------------------------------------------------------
    # GOAL POLYLINE
    # ---------------------------------------------------------

    if len(goal_points) > 0:

        pts = np.array(

            [[c,r] for r,c in goal_points]

        )

        ax.plot(

            pts[:,0],

            pts[:,1],

            color="yellow",

            linewidth=3,

            label="Goal Corridor"

        )

        ax.scatter(

            pts[:,0],

            pts[:,1],

            color="yellow",

            s=40

        )

    ax.set_title(

        f"MODE : {MODE}\n\n"

        "Left Click : Add Waypoint\n"

        "Right Click : Finish Current Corridor"

    )

    ax.axis("off")

    fig.canvas.draw_idle()
def onclick(event):

    global MODE
    global drawing_finished

    if event.xdata is None:

        return

    row = int(event.ydata)
    col = int(event.xdata)

    # ---------------------------------------------------------
    # LEFT CLICK
    # ---------------------------------------------------------

    if event.button == 1:

        if MODE == "START":

            start_points.append(

                (row,col)

            )

            print(

                f"Start Point {len(start_points)} :",

                (row,col)

            )

        else:

            goal_points.append(

                (row,col)

            )

            print(

                f"Goal Point {len(goal_points)} :",

                (row,col)

            )

        redraw()

    # ---------------------------------------------------------
    # RIGHT CLICK
    # ---------------------------------------------------------

    elif event.button == 3:

        if MODE == "START":

            if len(start_points) < 2:

                print(

                    "Need at least 2 start points."

                )

                return

            MODE = "GOAL"

            print()

            print("="*60)

            print("START CORRIDOR COMPLETE")

            print("NOW DRAW GOAL CORRIDOR")

            print("="*60)

            redraw()

        else:

            if len(goal_points) < 2:

                print(

                    "Need at least 2 goal points."

                )

                return

            drawing_finished = True

            plt.close()

fig.canvas.mpl_connect(

    "button_press_event",

    onclick

)
redraw()

plt.show()
if not drawing_finished:

    raise Exception(

        "Drawing cancelled."

    )

print()

print("="*60)

print("DRAWING COMPLETE")

print("="*60)

print()

print(

    "Start Waypoints :",

    len(start_points)

)

print(

    "Goal Waypoints :",

    len(goal_points)

)
selection = {

    "start_points": start_points,

    "goal_points": goal_points

}

with open(

    os.path.join(

        OUT_DIR,

        "selection.json"

    ),

    "w"

) as f:

    json.dump(

        selection,

        f,

        indent=2

    )

print()

print("Saved selection.json")
# ============================================================
# PART 3
# BUILD CORRIDOR BANDS
# ============================================================

print()
print("=" * 60)
print("BUILDING START / GOAL CORRIDORS")
print("=" * 60)

def build_corridor_band(
        polyline_points,
        walkable_mask,
        corridor_width
):
    """
    Convert a user drawn polyline into a corridor band.

    Pipeline
    --------
    Polyline
          ↓
    1-pixel line
          ↓
    Dilation
          ↓
    Corridor
          ↓
    Remove Obstacles
          ↓
    Corridor Mask
    """

    H, W = walkable_mask.shape

    mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    pts = np.array(

        [

            [c, r]

            for r, c in polyline_points

        ],

        dtype=np.int32

    )

    # Draw polyline

    cv2.polylines(

        mask,

        [pts],

        False,

        255,

        thickness=1

    )

    # Corridor Width

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (

            CORRIDOR_WIDTH,

            CORRIDOR_WIDTH

        )

    )

    mask = cv2.dilate(

        mask,

        kernel,

        iterations=1

    )

    mask = mask.astype(bool)

    # Remove Obstacles

    mask &= walkable_mask

    return mask
print()

print("Generating START Corridor...")

start_band = build_corridor_band(

    start_points,

    walkable,

    CORRIDOR_WIDTH

)
print()

print("Generating GOAL Corridor...")

goal_band = build_corridor_band(

    goal_points,

    walkable,

    CORRIDOR_WIDTH

)
print()

print("Extracting Walkable Pixels...")

start_cells = np.argwhere(
    start_band
)

goal_cells = np.argwhere(
    goal_band
)

print()

print(

    f"Start Corridor Pixels : {len(start_cells):,}"

)

print(

    f"Goal Corridor Pixels : {len(goal_cells):,}"

)
np.save(

    os.path.join(

        OUT_DIR,

        "start_band.npy"

    ),

    start_band

)

np.save(

    os.path.join(

        OUT_DIR,

        "goal_band.npy"

    ),

    goal_band

)
preview = rgb.copy()

preview[start_band] = [0,255,255]

preview[goal_band] = [255,255,0]

plt.figure(figsize=(12,12))

plt.imshow(preview)

plt.axis("off")

plt.title("Corridor Bands")

plt.savefig(

    os.path.join(

        OUT_DIR,

        "corridor_preview.png"

    ),

    bbox_inches="tight"

)

plt.close()

print("Saved corridor_preview.png")
print()
print("=" * 60)
print("FARTHEST POINT SAMPLING")
print("=" * 60)

def farthest_point_sampling(
        points,
        k,
        seed=42
):
    """
    FPS Sampling
    """

    if len(points) <= k:

        return points

    rng = np.random.default_rng(seed)

    first = rng.integers(len(points))

    sampled = [

        points[first]

    ]

    distances = np.full(

        len(points),

        np.inf

    )

    selected = np.zeros(

        len(points),

        dtype=bool

    )

    selected[first] = True

    while len(sampled) < k:

        last = sampled[-1]

        d = np.linalg.norm(

            points-last,

            axis=1

        )

        distances = np.minimum(

            distances,

            d

        )

        distances[selected] = -1

        next_idx = np.argmax(

            distances

        )

        sampled.append(

            points[next_idx]

        )

        selected[next_idx] = True

    return np.array(sampled)
print()

print("Sampling START Corridor...")

start_samples = farthest_point_sampling(

    start_cells,

    MAX_START_POINTS,

    RANDOM_SEED

)
print()

print("Sampling GOAL Corridor...")

goal_samples = farthest_point_sampling(

    goal_cells,

    MAX_GOAL_POINTS,

    RANDOM_SEED

)
print()

print(

    f"Start Samples : {len(start_samples)}"

)

print(

    f"Goal Samples : {len(goal_samples)}"

)

print(

    f"Expected A* Runs : "

    f"{len(start_samples)*len(goal_samples):,}"

)
fps_img = preview.copy()

for r,c in start_samples:

    cv2.circle(

        fps_img,

        (int(c),int(r)),

        4,

        (255,255,255),

        -1

    )

for r,c in goal_samples:

    cv2.circle(

        fps_img,

        (int(c),int(r)),

        4,

        (255,128,0),

        -1

    )

plt.figure(figsize=(12,12))

plt.imshow(fps_img)

plt.axis("off")

plt.title("FPS Samples")

plt.savefig(

    os.path.join(

        OUT_DIR,

        "fps_preview.png"

    ),

    bbox_inches="tight"

)

plt.close()

print("Saved fps_preview.png")
# ============================================================
# PART 4
# MULTI A*
# ============================================================

print()
print("="*60)
print("MULTI A* ROUTE GENERATION")
print("="*60)
MOVES = [

    (-1,0),
    (1,0),
    (0,-1),
    (0,1)

]

if ALLOW_DIAGONAL:

    MOVES.extend([

        (-1,-1),

        (-1,1),

        (1,-1),

        (1,1)

    ])

def heuristic(a,b):

    return np.hypot(

        a[0]-b[0],

        a[1]-b[1]

    )
def astar(start,goal):

    pq=[]

    heapq.heappush(

        pq,

        (0,start)

    )

    parent={

        start:None

    }

    gscore={

        start:0

    }

    while pq:

        _,current=heapq.heappop(pq)

        if current==goal:

            path=[]

            node=current

            while node is not None:

                path.append(node)

                node=parent[node]

            return path[::-1]

        for dr,dc in MOVES:

            nr=current[0]+dr
            nc=current[1]+dc

            if not (

                0<=nr<H and
                0<=nc<W

            ):

                continue

            if not walkable[nr,nc]:

                continue

            step=1.0

            if dr!=0 and dc!=0:

                step=DIAGONAL_COST

            neighbour=(nr,nc)

            new_cost=gscore[current]+step

            if (

                neighbour not in gscore

                or

                new_cost<gscore[neighbour]

            ):

                gscore[neighbour]=new_cost

                parent[neighbour]=current

                priority=(

                    new_cost+

                    heuristic(

                        neighbour,

                        goal

                    )

                )

                heapq.heappush(

                    pq,

                    (

                        priority,

                        neighbour

                    )

                )

    return None
# ============================================================
# BUILD ALL START-GOAL PAIRS
# ============================================================

jobs=[]

for start in start_samples:

    for goal in goal_samples:

        jobs.append(

            (

                tuple(start),

                tuple(goal)

            )

        )

print()

print(

    f"Total Jobs : {len(jobs):,}"

)
print()

print("Running Multi A*...")

routes=[]

successful=0

failed=0

total_jobs=len(jobs)
progress_interval=max(

    1,

    total_jobs//100

)
for i,(start,goal) in enumerate(jobs):

    if i%progress_interval==0:

        pct=100*i/total_jobs

        print(

            f"{pct:5.1f}%"

        )

    path=astar(

        start,

        goal

    )

    if path is None:

        failed+=1

        continue

    successful+=1

    routes.append(path)
print()

print("="*60)

print("A* SUMMARY")

print("="*60)

print()

print(

    f"Successful : {successful:,}"

)

print(

    f"Failed : {failed:,}"

)

print(

    f"Routes Generated : {len(routes):,}"

)
lengths=[

    len(route)

    for route in routes

]

print()

print(

    f"Shortest Route : {np.min(lengths)}"

)

print(

    f"Longest Route : {np.max(lengths)}"

)

print(

    f"Average Route : {np.mean(lengths):.2f}"

)
# ============================================================
# PART 5
# REMOVE DUPLICATE ROUTES
# ============================================================

print()
print("="*60)
print("REMOVING DUPLICATE ROUTES")
print("="*60)

unique_routes = {}

for route in routes:

    key = tuple(route)

    if key not in unique_routes:

        unique_routes[key] = route

routes = list(unique_routes.values())

print()

print(f"Unique Routes : {len(routes):,}")
# ============================================================
# CONVERT TO PYTHON JSON FORMAT
# ============================================================

routes_json = [

    [

        [

            int(r),

            int(c)

        ]

        for r,c in route

    ]

    for route in routes

]
# ============================================================
# SAVE PATHS JSON
# ============================================================

json_path = os.path.join(

    OUT_DIR,

    "paths_1.json"

)

with open(

    json_path,

    "w"

) as f:

    json.dump(

        routes_json,

        f,

        indent=2

    )

print()

print("Saved paths_1.json")
# ============================================================
# SAVE NUMPY
# ============================================================

np.save(

    os.path.join(

        OUT_DIR,

        "paths_1.npy"

    ),

    np.array(

        routes_json,

        dtype=object

    ),

    allow_pickle=True

)

print(

    "Saved paths_1.npy"

)
# ============================================================
# ROUTE STATISTICS
# ============================================================

lengths = [

    len(route)

    for route in routes

]

stats = {

    "start_samples":

        int(len(start_samples)),

    "goal_samples":

        int(len(goal_samples)),

    "astar_jobs":

        int(len(jobs)),

    "candidate_routes":

        int(len(routes)),

    "shortest_route":

        int(np.min(lengths)),

    "longest_route":

        int(np.max(lengths)),

    "average_route":

        float(np.mean(lengths))

}
with open(

    os.path.join(

        OUT_DIR,

        "route_statistics.json"

    ),

    "w"

) as f:

    json.dump(

        stats,

        f,

        indent=2

    )

print(

    "Saved route_statistics.json"

)
selection = {

    "start_points":

        [

            [int(r),int(c)]

            for r,c in start_points

        ],

    "goal_points":

        [

            [int(r),int(c)]

            for r,c in goal_points

        ],

    "corridor_width":

        CORRIDOR_WIDTH,

    "fps_start":

        MAX_START_POINTS,

    "fps_goal":

        MAX_GOAL_POINTS

}

with open(

    os.path.join(

        OUT_DIR,

        "selection.json"

    ),

    "w"

) as f:

    json.dump(

        selection,

        f,

        indent=2

    )

print(

    "Saved selection.json"

)
print()

print("Generating Route Visualization...")
canvas = rgb.copy()
for route in routes:

    for r,c in route:

        canvas[r,c] = [0,0,0]
for r,c in start_samples:

    cv2.circle(

        canvas,

        (int(c),int(r)),

        4,

        (0,255,255),

        -1

    )         
for r,c in goal_samples:

    cv2.circle(

        canvas,

        (int(c),int(r)),

        4,

        (255,255,0),

        -1

    )

fig,ax = plt.subplots(

    figsize=(12,12)

)

ax.imshow(canvas)

ax.axis("off")

legend=[

    Patch(

        color="green",

        label="Terrain"

    ),

    Patch(

        color="red",

        label="Obstacle"

    ),

    Patch(

        color="black",

        label="Routes"

    ),

    Patch(

        color="cyan",

        label="Start Samples"

    ),

    Patch(

        color="yellow",

        label="Goal Samples"

    )

]

ax.legend(

    handles=legend,

    loc="lower right"

)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUT_DIR,

        "paths_colored_1.png"

    ),

    bbox_inches="tight"

)

plt.close()
print()

print("="*60)

print("INTERACTIVE MULTI A* COMPLETE")

print("="*60)

print()

print(

    f"Start Samples : {len(start_samples):,}"

)

print(

    f"Goal Samples : {len(goal_samples):,}"

)

print(

    f"A* Jobs : {len(jobs):,}"

)

print(

    f"Routes : {len(routes):,}"

)

print()

print("Outputs")

print("---------------------------")

print("paths_1.json")

print("paths_1.npy")

print("route_statistics.json")

print("selection.json")

print("corridor_preview.png")

print("fps_preview.png")

print("paths_colored_1.png")

print()

print("="*60)   