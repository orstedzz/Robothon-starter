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

HOME_POS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
GRIPPER_OPEN = 255
GRIPPER_CLOSE = 60
INSERT_FORCE_MAX = 15.0

# Peg definitions — positions match the scene layout
PEGS = [
    {"name": "brass", "body": "peg_brass", "hole_body": "hole_brass",
     "peg_xyz": [0.40, 0.14, 0.115], "hole_xyz": [0.25, 0.05, 0.10]},
    {"name": "steel", "body": "peg_steel", "hole_body": "hole_steel",
     "peg_xyz": [0.50, 0.14, 0.115], "hole_xyz": [0.35, 0.05, 0.10]},
    {"name": "red",   "body": "peg_red",   "hole_body": "hole_red",
     "peg_xyz": [0.40, -0.14, 0.105], "hole_xyz": [0.25, -0.05, 0.10]},
    {"name": "blue",  "body": "peg_blue", "hole_body": "hole_blue",
     "peg_xyz": [0.50, -0.14, 0.105], "hole_xyz": [0.35, -0.05, 0.10]},
]

BENCHMARK = {
    "insertion_depth": 0.015,
    "max_insertion_time": 15.0,
}

# IDs populated at runtime
BODY_IDS = {}
SITE_IDS = {}
SENSOR_IDS = {}
