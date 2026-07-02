#!/usr/bin/env python3

import os
import cv2
import json
import heapq
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

from matplotlib.patches import Circle, Patch

# ============================================================
# CONFIG
# ============================================================

TERRAIN_NPY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\terrain_mask.npy"

OBSTACLE_NPY = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\Static_scripts\step4\obstacle_mask.npy"

OUT_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\inetractive_multi_astar_output"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

MAX_STARTS = 25
MAX_GOALS = 25

ALLOW_DIAG = True

DIAG_COST = 1.414

RANDOM_SEED = 42

# START_RADIUS = 80

# GOAL_RADIUS = 80
print("\n"+"="*60)
print("INTERACTIVE MULTI A*")
print("="*60)

terrain=np.load(TERRAIN_NPY).astype(bool)
obstacle=np.load(OBSTACLE_NPY).astype(bool)

walkable=terrain & (~obstacle)

H,W=walkable.shape

rgb=np.zeros(
    (H,W,3),
    dtype=np.uint8
)

rgb[:]=[120,120,120]

rgb[terrain]=[0,255,0]

rgb[obstacle]=[255,0,0]

clicked_points=[]

# start_radius=START_RADIUS

# goal_radius=GOAL_RADIUS

# selection_stage=0

# preview_circle=None

# preview_text=None
# ============================================================
# DRAWING VARIABLES
# ============================================================

start_points = []

goal_points = []

drawing_goal = False

LINE_WIDTH = 60

# def on_click(event):

#     global selection_stage
#     global preview_circle

#     if event.xdata is None:

#         return

#     x=int(event.xdata)

#     y=int(event.ydata)

#     if selection_stage==0:

#         clicked_points.append(
#             (y,x)
#         )

#         selection_stage=1

#         print(
#             f"Start Selected : ({y},{x})"
#         )

#     elif selection_stage==1:

#         clicked_points.append(
#             (y,x)
#         )

#         selection_stage=2

#         print(
#             f"Goal Selected : ({y},{x})"
#         )

#     redraw()
def on_click(event):

    global drawing_goal

    if event.xdata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    if not drawing_goal:

        start_points.append((y, x))

        print(
            f"Start Point {len(start_points)} : ({y},{x})"
        )

    else:

        goal_points.append((y, x))

        print(
            f"Goal Point {len(goal_points)} : ({y},{x})"
        )

    redraw()

# def on_key(event):

#     global start_radius
#     global goal_radius

#     if selection_stage==1:

#         if event.key=="up":

#             start_radius+=5

#         elif event.key=="down":

#             start_radius=max(
#                 5,
#                 start_radius-5
#             )

#     elif selection_stage==2:

#         if event.key=="up":

#             goal_radius+=5

#         elif event.key=="down":

#             goal_radius=max(
#                 5,
#                 goal_radius-5
#             )

#     redraw()

#     if event.key=="enter":

#         plt.close()
def on_key(event):

    global drawing_goal

    # ---------------------------------------
    # ENTER
    # ---------------------------------------

    if event.key == "enter":

        if not drawing_goal:

            if len(start_points) < 2:

                print(
                    "Select at least two Start points."
                )

                return

            drawing_goal = True

            print()
            print("="*40)
            print("START CORRIDOR COMPLETE")
            print("Now draw GOAL corridor.")
            print("="*40)

        else:

            if len(goal_points) < 2:

                print(
                    "Select at least two Goal points."
                )

                return

            plt.close()

    # ---------------------------------------
    # BACKSPACE
    # ---------------------------------------

    elif event.key == "backspace":

        if not drawing_goal:

            if len(start_points):

                start_points.pop()

        else:

            if len(goal_points):

                goal_points.pop()

        redraw()

# def redraw():

#     ax.clear()

#     ax.imshow(rgb)

#     if len(clicked_points)>=1:

#         r,c=clicked_points[0]

#         ax.scatter(
#             c,
#             r,
#             c="cyan",
#             s=60
#         )

#         ax.add_patch(

#             Circle(

#                 (c,r),

#                 start_radius,

#                 fill=False,

#                 color="cyan",

#                 linewidth=2

#             )

#         )

#     if len(clicked_points)>=2:

#         r,c=clicked_points[1]

#         ax.scatter(

#             c,

#             r,

#             c="yellow",

#             s=60

#         )

#         ax.add_patch(

#             Circle(

#                 (c,r),

#                 goal_radius,

#                 fill=False,

#                 color="yellow",

#                 linewidth=2

#             )

#         )

#     ax.set_title(

#         "Click START then GOAL\n"

#         "UP/DOWN : change radius\n"

#         "ENTER : continue"

#     )

#     ax.axis("off")

#     fig.canvas.draw_idle()
def redraw():

    ax.clear()

    ax.imshow(rgb)

    # ----------------------------------------------------
    # Draw Start Corridor
    # ----------------------------------------------------

    if len(start_points) > 0:

        pts = np.array(

            [

                [c, r]

                for r, c in start_points

            ]

        )

        ax.scatter(

            pts[:,0],

            pts[:,1],

            color="cyan",

            s=50

        )

        if len(start_points) > 1:

            ax.plot(

                pts[:,0],

                pts[:,1],

                color="cyan",

                linewidth=3

            )

    # ----------------------------------------------------
    # Draw Goal Corridor
    # ----------------------------------------------------

    if len(goal_points) > 0:

        pts = np.array(

            [

                [c, r]

                for r, c in goal_points

            ]

        )

        ax.scatter(

            pts[:,0],

            pts[:,1],

            color="yellow",

            s=50

        )

        if len(goal_points) > 1:

            ax.plot(

                pts[:,0],

                pts[:,1],

                color="yellow",

                linewidth=3

            )

    ax.set_title(

        "LEFT CLICK : Add Point\n"

        "ENTER : Finish Corridor\n"

        "BACKSPACE : Remove Last Point"

    )

    ax.axis("off")

    fig.canvas.draw_idle()

fig,ax=plt.subplots(
    figsize=(10,10)
)

cid=fig.canvas.mpl_connect(
    "button_press_event",
    on_click
)

kid=fig.canvas.mpl_connect(
    "key_press_event",
    on_key
)

redraw()

plt.show()

if len(start_points) < 2:

    raise Exception(
        "Need at least 2 Start points."
    )

if len(goal_points) < 2:

    raise Exception(
        "Need at least 2 Goal points."
    )

# start_point=clicked_points[0]

# goal_point=clicked_points[1]

print()

print("Selected Points")

print("----------------")

# print("Start :",start_point)

# print("Goal  :",goal_point)

# print("Start Radius :",start_radius)

# print("Goal Radius :",goal_radius)

# ============================================================
# GENERATE START / GOAL BANDS
# ============================================================

# print("\nGenerating Start/Goal Bands...")

# start_band = np.zeros(
#     (H, W),
#     dtype=bool
# )

# goal_band = np.zeros(
#     (H, W),
#     dtype=bool
# )
# # ------------------------------------------------------------
# # START BAND
# # ------------------------------------------------------------

# sr, sc = start_point

# for r in range(H):

#     for c in range(W):

#         if np.sqrt(
#             (r - sr) ** 2 +
#             (c - sc) ** 2
#         ) <= start_radius:

#             start_band[r, c] = True

# # ------------------------------------------------------------
# # GOAL BAND
# # ------------------------------------------------------------

# gr, gc = goal_point

# for r in range(H):

#     for c in range(W):

#         if np.sqrt(
#             (r - gr) ** 2 +
#             (c - gc) ** 2
#         ) <= goal_radius:

#             goal_band[r, c] = True

# start_band &= walkable

# goal_band &= walkable

# np.save(

#     os.path.join(
#         OUT_DIR,
#         "start_band.npy"
#     ),

#     start_band

# )

# np.save(

#     os.path.join(
#         OUT_DIR,
#         "goal_band.npy"
#     ),

#     goal_band

# )

# print("Saved start_band.npy")
# print("Saved goal_band.npy")

# start_cells = np.array(

#     list(

#         zip(

#             *np.where(
#                 start_band
#             )

#         )

#     )

# )

# goal_cells = np.array(

#     list(

#         zip(

#             *np.where(
#                 goal_band
#             )

#         )

#     )

# )
print()

print("Generating Corridor Bands...")

start_band = np.zeros(
    (H, W),
    dtype=np.uint8
)

goal_band = np.zeros(
    (H, W),
    dtype=np.uint8
)

start_poly = np.array(

    [

        [c, r]

        for r, c in start_points

    ],

    dtype=np.int32

)

goal_poly = np.array(

    [

        [c, r]

        for r, c in goal_points

    ],

    dtype=np.int32

)

cv2.polylines(

    start_band,

    [start_poly],

    False,

    255,

    thickness=LINE_WIDTH

)

cv2.polylines(

    goal_band,

    [goal_poly],

    False,

    255,

    thickness=LINE_WIDTH

)

start_band = start_band.astype(bool)

goal_band = goal_band.astype(bool)

start_band &= walkable

goal_band &= walkable

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

start_cells = np.argwhere(start_band)

goal_cells = np.argwhere(goal_band)

print()

print(
    "Walkable Start Cells :",
    len(start_cells)
)

print(
    "Walkable Goal Cells :",
    len(goal_cells)
)

rng = np.random.default_rng(
    RANDOM_SEED
)

if len(start_cells) > MAX_STARTS:

    idx = rng.choice(

        len(start_cells),

        MAX_STARTS,

        replace=False

    )

    start_cells = start_cells[idx]

if len(goal_cells) > MAX_GOALS:

    idx = rng.choice(

        len(goal_cells),

        MAX_GOALS,

        replace=False

    )

    goal_cells = goal_cells[idx]

print()

print(
    "Sampled Start Points :",
    len(start_cells)
)

print(
    "Sampled Goal Points :",
    len(goal_cells)
)

print(
    "Total Planned A* Runs :",
    len(start_cells) *
    len(goal_cells)
)

preview = np.zeros(
    (H, W, 3),
    dtype=np.uint8
)

preview[:] = [120,120,120]

preview[terrain] = [0,255,0]

preview[obstacle] = [255,0,0]

preview[start_band] = [0,255,255]

preview[goal_band] = [255,255,0]

plt.figure(
    figsize=(10,10)
)

plt.imshow(preview)

# plt.scatter(

#     sc,

#     sr,

#     c="blue",

#     s=70,

#     label="Start"

# )

# plt.scatter(

#     gc,

#     gr,

#     c="red",

#     s=70,

#     label="Goal"

# )

plt.legend()

plt.axis("off")

plt.title(
    "Generated Start & Goal Bands"
)

plt.savefig(

    os.path.join(

        OUT_DIR,

        "start_goal_preview.png"

    ),

    bbox_inches="tight"

)

plt.close()

print(
    "Saved start_goal_preview.png"
)

# ============================================================
# MOVEMENT MODEL
# ============================================================

MOVES = [

    (-1,0),
    (1,0),
    (0,-1),
    (0,1)

]

if ALLOW_DIAG:

    MOVES.extend([

        (-1,-1),
        (-1,1),
        (1,-1),
        (1,1)

    ])

# ============================================================
# HEURISTIC
# ============================================================

def heuristic(a, b):

    return np.hypot(

        a[0]-b[0],

        a[1]-b[1]

    )

# ============================================================
# A*
# ============================================================

def astar(start, goal):

    pq = []

    heapq.heappush(

        pq,

        (0, start)

    )

    parent = {

        start: None

    }

    gscore = {

        start: 0

    }

    while pq:

        _, current = heapq.heappop(pq)

        if current == goal:

            path = []

            node = current

            while node is not None:

                path.append(node)

                node = parent[node]

            return path[::-1]

        for dr, dc in MOVES:

            nr = current[0] + dr
            nc = current[1] + dc

            if not (

                0 <= nr < H and

                0 <= nc < W

            ):

                continue

            if not walkable[nr, nc]:

                continue

            neighbour = (nr, nc)

            step = (

                1.0

                if dr == 0 or dc == 0

                else DIAG_COST

            )

            new_cost = (

                gscore[current]

                + step

            )

            if (

                neighbour not in gscore

                or

                new_cost < gscore[neighbour]

            ):

                gscore[neighbour] = new_cost

                parent[neighbour] = current

                priority = (

                    new_cost +

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

jobs = []

for s in start_cells:

    for g in goal_cells:

        jobs.append(

            (

                tuple(s),

                tuple(g)

            )

        )

print()

print(

    "Total A* Jobs :",

    len(jobs)

)

# ============================================================
# GENERATE ROUTES
# ============================================================

print()

print("Running Multi-A*")

routes = []

for i, (s, g) in enumerate(jobs):

    if i % 25 == 0:

        print(

            f"{i}/{len(jobs)}"

        )

    path = astar(

        s,

        g

    )

    if path is not None:

        routes.append(path)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

unique = {}

for route in routes:

    key = tuple(route)

    if key not in unique:

        unique[key] = route

routes = list(

    unique.values()

)

print()

print(

    "Unique Routes :",

    len(routes)

)

# ============================================================
# ROUTE STATISTICS
# ============================================================

lengths = [

    len(route)

    for route in routes

]

stats = {

    "start_candidates":

        len(start_cells),

    "goal_candidates":

        len(goal_cells),

    "astar_runs":

        len(jobs),

    "valid_routes":

        len(routes),

    "min_length":

        int(np.min(lengths))

        if lengths else 0,

    "max_length":

        int(np.max(lengths))

        if lengths else 0,

    "mean_length":

        float(np.mean(lengths))

        if lengths else 0

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

# ============================================================
# CONVERT ROUTES
# ============================================================

routes_json = [

    [

        [

            int(r),

            int(c)

        ]

        for r, c in route

    ]

    for route in routes

]

# ============================================================
# SAVE JSON
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

print(

    "Saved:",

    json_path

)

# ============================================================
# SAVE NPY
# ============================================================

npy_path = os.path.join(

    OUT_DIR,

    "paths_1.npy"

)

np.save(

    npy_path,

    np.array(

        routes_json,

        dtype=object

    ),

    allow_pickle=True

)

print(

    "Saved:",

    npy_path

)

# ============================================================
# VISUALIZATION
# ============================================================

canvas = np.zeros(

    (H, W, 3),

    dtype=np.uint8

)

canvas[:] = [120,120,120]

canvas[terrain] = [0,255,0]

canvas[obstacle] = [255,0,0]

canvas[start_band] = [0,255,255]

canvas[goal_band] = [255,255,0]
for route in routes_json:

    for r, c in route:

        canvas[

            r,

            c

        ] = [0,0,0]

for r, c in start_cells:

    canvas[

        r,

        c

    ] = [255,255,255]

for r, c in goal_cells:

    canvas[

        r,

        c

    ] = [255,128,0]

plt.figure(

    figsize=(12,12)

)

plt.imshow(

    canvas

)

plt.axis(

    "off"

)

plt.title(

    "Interactive Multi-A*"

)

legend = [

    Patch(

        color="green",

        label="Terrain"

    ),

    Patch(

        color="red",

        label="Obstacle"

    ),

    Patch(

        color="cyan",

        label="Start Band"

    ),

    Patch(

        color="yellow",

        label="Goal Band"

    ),

    Patch(

        color="black",

        label="Routes"

    ),

    Patch(

        color="white",

        label="Sampled Starts"

    ),

    Patch(

        color="orange",

        label="Sampled Goals"

    )

]

plt.legend(

    handles=legend,

    loc="lower right"

)
png_path = os.path.join(

    OUT_DIR,

    "paths_colored_1.png"

)

plt.savefig(

    png_path,

    bbox_inches="tight"

)

plt.close()

print(

    "Saved:",

    png_path

)

# selection = {

#     "start_point": [

#         int(start_point[0]),

#         int(start_point[1])

#     ],

#     "goal_point": [

#         int(goal_point[0]),

#         int(goal_point[1])

#     ],

#     # "start_radius":

#     #     int(start_radius),

#     # "goal_radius":

#     #     int(goal_radius),

#     "sampled_start_cells":

#         int(len(start_cells)),

#     "sampled_goal_cells":

#         int(len(goal_cells))

# }

# with open(

#     os.path.join(

#         OUT_DIR,

#         "selection.json"

#     ),

#     "w"

# ) as f:

#     json.dump(

#         selection,

#         f,

#         indent=2

#     )

print()

print("="*60)

print("INTERACTIVE MULTI A* COMPLETE")

print("="*60)

print()

print(

    f"Start Candidates : {len(start_cells)}"

)

print(

    f"Goal Candidates  : {len(goal_cells)}"

)

print(

    f"Unique Routes    : {len(routes)}"

)

print()

print(

    "Outputs"

)

print("----------------")

print(json_path)

print(npy_path)

print(png_path)

print(

    os.path.join(

        OUT_DIR,

        "selection.json"

    )

)

print(

    os.path.join(

        OUT_DIR,

        "route_statistics.json"

    )

)

print()

print("="*60)
