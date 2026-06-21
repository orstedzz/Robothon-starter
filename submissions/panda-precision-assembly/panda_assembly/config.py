"""
Panda Precision Assembly - Configuration
Central config for scene paths, peg specs, control params, and benchmark criteria.
"""

from __future__ import annotations
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parent.parent
SCENE_PATH = str(SUBMISSION / "scene.xml")
OUTPUT_DIR = SUBMISSION / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Control Parameters --------------------
# Panda default home position (7 arm joints)
HOME_POS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
GRIPPER_OPEN = 255
GRIPPER_CLOSE = 60  # firm grip on pegs
GRIPPER_RELAX = 20  # release

# IK parameters
IK_DAMPING = 1e-3
IK_TOLERANCE = 0.005      # m - position tolerance
IK_MAX_ITER = 50
IK_DT = 0.01

# Motion parameters
APPROACH_HEIGHT = 0.08    # m above grasp point
MOVE_SPEED = 0.15         # m/s Cartesian speed
LIFT_HEIGHT = 0.20        # m lift height after grasp
INSERT_SPEED = 0.02       # slow insertion speed

# Force thresholds
GRIP_FORCE_THRESHOLD = 3.0   # N - contact detected
INSERT_FORCE_MAX = 15.0      # N - max allowed insertion force

# -------------------- Peg Definitions --------------------
PEGS = [
    {
        "name": "brass",
        "body": "peg_brass_body",
        "grasp_site": "peg_brass_grasp",
        "target_hole": "hole_brass",
        "hole_pos": [0.45, 0.06, 0.115],  # world coords
        "diameter": 0.024,
        "length": 0.06,
        "color": [0.72, 0.65, 0.42],
    },
    {
        "name": "steel",
        "body": "peg_steel_body",
        "grasp_site": "peg_steel_grasp",
        "target_hole": "hole_steel",
        "hole_pos": [0.55, 0.06, 0.115],
        "diameter": 0.024,
        "length": 0.06,
        "color": [0.6, 0.6, 0.65],
    },
    {
        "name": "red",
        "body": "peg_red_body",
        "grasp_site": "peg_red_grasp",
        "target_hole": "hole_red",
        "hole_pos": [0.45, -0.06, 0.115],
        "diameter": 0.016,
        "length": 0.05,
        "color": [0.85, 0.15, 0.12],
    },
    {
        "name": "blue",
        "body": "peg_blue_body",
        "grasp_site": "peg_blue_grasp",
        "target_hole": "hole_blue",
        "hole_pos": [0.55, -0.06, 0.115],
        "diameter": 0.016,
        "length": 0.05,
        "color": [0.12, 0.15, 0.85],
    },
]

# -------------------- Benchmark Criteria --------------------
BENCHMARK = {
    "trials_per_peg": 1,          # deterministic, so 1 trial each
    "grasp_success_force": 1.0,   # N - minimum contact force to confirm grasp
    "insertion_depth": 0.025,     # m - minimum peg insertion into hole
    "max_insertion_time": 15.0,   # s - per peg timeout
    "max_total_time": 90.0,       # s - total timeout
}

# Numerical IDs lookup (populated at runtime)
BODY_IDS = {}
SITE_IDS = {}
SENSOR_IDS = {}
