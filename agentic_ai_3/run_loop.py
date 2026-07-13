#!/usr/bin/env python3
"""
run_loop.py — Agentic Loop Controller
Runs the full pipeline repeatedly until no routes remain or MAX_ITERATIONS reached.

Pipeline per iteration:
  1. terrain_intelligence.py   → summary_iter_N.json
  2. agent2.py (main)          → strategy_iter_N.json
  3. agent3_shortlist.py       → shortlisted_candidates_iter_N.json
  4. agent4_raycaster.py       → fov_results_iter_N.json
  5. agent5_scorer.py          → placement_result_iter_N.json + paths_iter_{N+1}.json
  6. agent6_coverage.py        → coverage_expansion_iter_N.json

Each agent is imported and its top-level code is wrapped as a runnable module.
Alternatively, set RUN_AS_SUBPROCESS = True to call them with subprocess.run().
"""

import os
import json
import subprocess
import sys
import time

# ===========================================================
# CONFIG
# ===========================================================

MAX_ITERATIONS  = 20
STOP_IF_NO_ROUTES = True        # stop early when 0 routes remain
RUN_AS_SUBPROCESS = True        # safer: each agent runs in its own process

RUNTIME_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
    r"\runtime_state"
)

ORIGINAL_PATHS = (
    r"C:\Users\KHUSHI\Documents\deepterrain_internship\poleplacement_codes\DeepTerrain\agentic_ai\multi_astar_v2\paths_1.json"
)

# terrain_intelligence.py and agent2.py live in agentic_ai\
AGENTIC_AI_DIR = (
    r"C:\Users\KHUSHI\Documents"
    r"\deepterrain_internship"
    r"\poleplacement_codes"
    r"\DeepTerrain"
    r"\agentic_ai"
)

# agent3–6 and run_loop.py live in agentic_ai\claude\
CLAUDE_DIR = os.path.join(AGENTIC_AI_DIR, "claude")

AGENTS = [
    os.path.join(AGENTIC_AI_DIR, "terrain_intelligence.py"),
    os.path.join(AGENTIC_AI_DIR, "agent2.py"),
    os.path.join(CLAUDE_DIR,     "agent3_shortlist.py"),
    os.path.join(CLAUDE_DIR,     "agent4_raycaster.py"),
    os.path.join(CLAUDE_DIR,     "agent5_scorer.py"),
    os.path.join(CLAUDE_DIR,     "agent6_coverage.py"),
]

AGENT_NAMES = [
    "Terrain Intelligence",
    "Strategy Generator (LLM)",
    "Candidate Shortlister",
    "Raycasting FOV Engine",
    "Camera Scorer + Placer",
    "Coverage Expansion Planner",
]

os.makedirs(RUNTIME_DIR, exist_ok=True)

# ===========================================================
# HELPERS
# ===========================================================

def count_routes():
    """Return the number of surviving routes in the latest paths file."""
    iters = []
    for fn in os.listdir(RUNTIME_DIR):
        if fn.startswith("paths_iter_") and fn.endswith(".json"):
            try:
                it = int(fn.replace("paths_iter_", "").replace(".json", ""))
                iters.append((it, fn))
            except ValueError:
                pass

    if not iters:
        # Use original file
        if os.path.exists(ORIGINAL_PATHS):
            with open(ORIGINAL_PATHS, "r") as f:
                return len(json.load(f))
        return -1

    iters.sort(key=lambda x: x[0])
    with open(os.path.join(RUNTIME_DIR, iters[-1][1]), "r") as f:
        return len(json.load(f))


def run_agent(script_path: str, name: str):
    """Run a single agent script and check return code."""
    print(f"\n{'─'*60}")
    print(f"  Running: {name}")
    print(f"  Script : {script_path}")
    print(f"{'─'*60}")

    t0 = time.time()

    if RUN_AS_SUBPROCESS:
        result = subprocess.run(
            [sys.executable, script_path],
            check=False
        )
        ok = result.returncode == 0
    else:
        # Import-and-exec approach (shares process state — use carefully)
        try:
            with open(script_path, "r") as f:
                code = f.read()
            exec(compile(code, script_path, "exec"), {"__name__": "__main__"})
            ok = True
        except SystemExit as e:
            ok = (e.code == 0 or e.code is None)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            ok = False

    elapsed = time.time() - t0
    status  = "OK" if ok else "FAILED"
    print(f"\n  [{status}] {name} completed in {elapsed:.1f}s")
    return ok


# ===========================================================
# LOOP
# ===========================================================

print("\n" + "=" * 60)
print("AGENTIC LOOP CONTROLLER")
print("=" * 60)
print(f"Max iterations : {MAX_ITERATIONS}")

for iteration in range(MAX_ITERATIONS):

    routes_before = count_routes()
    if routes_before == 0 and STOP_IF_NO_ROUTES:
        print(f"\nNo routes remaining. Stopping after {iteration} iterations.")
        break

    print(f"\n{'═'*60}")
    print(f"  ITERATION {iteration}  |  routes remaining: {routes_before}")
    print(f"{'═'*60}")

    for script, name in zip(AGENTS, AGENT_NAMES):
        ok = run_agent(script, name)
        if not ok:
            print(f"\n[ABORT] Pipeline failed at: {name} (iteration {iteration})")
            sys.exit(1)

    routes_after = count_routes()
    eliminated   = routes_before - routes_after

    print(f"\n  ── Iteration {iteration} summary ──")
    print(f"     Routes before : {routes_before}")
    print(f"     Routes after  : {routes_after}")
    print(f"     Eliminated    : {eliminated}")

    if routes_after == 0 and STOP_IF_NO_ROUTES:
        print("\nAll routes eliminated. Loop complete.")
        break

print("\n" + "=" * 60)
print("LOOP CONTROLLER DONE")

# Final summary
placed_path = os.path.join(RUNTIME_DIR, "placed_cameras.json")
if os.path.exists(placed_path):
    with open(placed_path, "r") as f:
        cams = json.load(f)
    print(f"Total cameras placed : {len(cams)}")
    print(f"Final routes remaining: {count_routes()}")

print("=" * 60)