#!/usr/bin/env python3

import os
import json
import time
import subprocess

# ==========================================================
# CONFIG
# ==========================================================

MAX_CAMERAS = 5

ROUTE_THRESHOLD = 5
COVERAGE_THRESHOLD = 90.0

BASE_DIR = r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai"

RUNTIME_DIR = os.path.join(
    BASE_DIR,
    "runtime_state"
)

# ==========================================================
# SCRIPT PATHS
# ==========================================================

TERRAIN_INTELLIGENCE = os.path.join(
    BASE_DIR,
    "terrain_intelligence.py"
)

AGENT2 = os.path.join(
    BASE_DIR,
    "agent2.py"
)

CANDIDATE_SELECTOR = os.path.join(
    BASE_DIR,
    "candidate_selector.py"
)

CAMERA_SCORING = os.path.join(
    BASE_DIR,
    "camera_scoring.py"
)

VISIBILITY_EVALUATOR = os.path.join(
    BASE_DIR,
    "visibility_evaluator.py"
)

ENVIRONMENT_UPDATE = os.path.join(
    BASE_DIR,
    "environment_update.py"
)

# ==========================================================
# HELPERS
# ==========================================================

def banner(msg):

    print("\n")
    print("=" * 80)
    print(msg)
    print("=" * 80)


def run_script(script_path):

    banner(
        f"RUNNING: {os.path.basename(script_path)}"
    )

    result = subprocess.run(
        ["python", script_path]
    )

    if result.returncode != 0:

        raise Exception(
            f"FAILED: {script_path}"
        )


def latest_environment_state():

    if not os.path.exists(RUNTIME_DIR):
        return None

    iterations = []

    for folder in os.listdir(RUNTIME_DIR):

        if folder.startswith("iteration_"):

            try:

                idx = int(
                    folder.split("_")[1]
                )

                iterations.append(idx)

            except:
                pass

    if len(iterations) == 0:
        return None

    latest_iter = max(iterations)

    state_path = os.path.join(
        RUNTIME_DIR,
        f"iteration_{latest_iter}",
        "environment_state.json"
    )

    if not os.path.exists(state_path):
        return None

    with open(state_path, "r") as f:
        return json.load(f)

# ==========================================================
# MAIN
# ==========================================================

banner(
    "SURVEILLANCE OPTIMIZATION LOOP"
)

for iteration in range(MAX_CAMERAS):

    banner(
        f"ITERATION {iteration+1}"
    )

    try:

        # --------------------------------------------------
        # AGENT 1
        # --------------------------------------------------

        run_script(
            TERRAIN_INTELLIGENCE
        )

        # --------------------------------------------------
        # AGENT 2
        # --------------------------------------------------

        run_script(
            AGENT2
        )

        # --------------------------------------------------
        # CANDIDATE SELECTION
        # --------------------------------------------------

        run_script(
            CANDIDATE_SELECTOR
        )

        # --------------------------------------------------
        # CAMERA SCORING
        # --------------------------------------------------

        run_script(
            CAMERA_SCORING
        )

        # --------------------------------------------------
        # VISIBILITY EVALUATION
        # --------------------------------------------------

        run_script(
            VISIBILITY_EVALUATOR
        )

        # --------------------------------------------------
        # ENVIRONMENT UPDATE
        # --------------------------------------------------

        run_script(
            ENVIRONMENT_UPDATE
        )

        # --------------------------------------------------
        # CHECK STOP CONDITIONS
        # --------------------------------------------------

        env_state = latest_environment_state()

        if env_state is None:

            print(
                "\nNo environment state found."
            )

            continue

        routes_remaining = env_state[
            "routes_remaining"
        ]

        coverage_percent = env_state[
            "coverage_percent"
        ]

        placed_cameras = env_state[
            "placed_cameras"
        ]

        banner(
            "ITERATION SUMMARY"
        )

        print(
            f"Routes Remaining : {routes_remaining}"
        )

        print(
            f"Coverage Percent : {coverage_percent}"
        )

        print(
            f"Placed Cameras   : {placed_cameras}"
        )

        # ----------------------------------------------
        # STOP CONDITION 1
        # ----------------------------------------------

        if routes_remaining <= ROUTE_THRESHOLD:

            banner(
                "STOPPING: ROUTE THRESHOLD REACHED"
            )

            break

        # ----------------------------------------------
        # STOP CONDITION 2
        # ----------------------------------------------

        if coverage_percent >= COVERAGE_THRESHOLD:

            banner(
                "STOPPING: COVERAGE THRESHOLD REACHED"
            )

            break

        # ----------------------------------------------
        # STOP CONDITION 3
        # ----------------------------------------------

        if placed_cameras >= MAX_CAMERAS:

            banner(
                "STOPPING: CAMERA BUDGET EXHAUSTED"
            )

            break

        time.sleep(2)

    except Exception as e:

        banner("PIPELINE FAILED")

        print(e)

        break

banner(
    "OPTIMIZATION COMPLETE"
)
