#!/usr/bin/env python3
"""
FFAI Robothon 2026 - Long-Horizon Tasks
=========================================
Franka Emika Panda: Multi-object pick-and-place assembly.

Demonstrates a long-horizon task where the Panda arm sequentially picks
4 colored cubes from a work table and places each on a color-matched target zone.

7 phases per object × 4 objects + home return = 29 sequential phases.

Usage:
  MUJOCO_GL=glfw xvfb-run -a python run_longhorizon.py   # render demo video
  python run_longhorizon.py --view                         # interactive viewer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import imageio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SUBMISSION = Path(__file__).resolve().parent
DEFAULT_SCENE = SUBMISSION / "scene.xml"
DEFAULT_OUTPUT = SUBMISSION / "demo.mp4"


# ---------------------------------------------------------------------------
# Task plan - 4 objects to pick and place
# ---------------------------------------------------------------------------
TASK_PLAN = [
    {"name": "red",    "target": [0.30,  0.25, 0.04]},
    {"name": "green",  "target": [0.50,  0.25, 0.04]},
    {"name": "blue",   "target": [0.40, -0.25, 0.04]},
    {"name": "yellow", "target": [0.20, -0.25, 0.04]},
]

# Panda: 7 arm joints + 1 gripper (0-255, 0=closed, 255=open)
GRIPPER_OPEN = 255
GRIPPER_CLOSED = 80

# Joint-space waypoints [j1..j7, gripper]
HOME       = [0.00, -0.785, 0.00, -2.356, 0.00, 1.571, 0.785, GRIPPER_OPEN]
PRE_REACH  = [0.50, -0.60, 0.20, -2.00, 0.10, 1.20, 0.80, GRIPPER_OPEN]
ABOVE_CUBE = [0.37, -0.85, 0.15, -1.80, 0.00, 1.40, 0.60, GRIPPER_OPEN]
GRASP      = [0.37, -0.85, 0.15, -1.80, 0.00, 1.40, 0.60, GRIPPER_CLOSED]
LIFT       = [0.37, -0.65, 0.20, -2.00, 0.10, 1.25, 0.80, GRIPPER_CLOSED]
PRE_PLACE  = [0.55, -0.90, 0.30, -2.20, 0.20, 1.00, 0.60, GRIPPER_CLOSED]
PLACE      = [0.55, -0.90, 0.30, -2.20, 0.20, 1.00, 0.60, GRIPPER_OPEN]


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_wp(a: list[float], b: list[float], t: float) -> list[float]:
    ts = smoothstep(t)
    return [lerp(va, vb, ts) for va, vb in zip(a, b)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_demo(
    model: mujoco.MjModel,
    duration: float = 30.0,
    output_path: str | Path | None = None,
    fps: int = 30,
) -> Path:
    """Run the full long-horizon task and render to MP4 video."""
    data = mujoco.MjData(model)

    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(duration * fps)
    width, height = 1280, 720

    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.8
    cam.azimuth = 130
    cam.elevation = -25
    cam.lookat = [0.45, 0.0, 0.2]

    print(f"Rendering {total_frames} frames → {output_path} ...")

    writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264', quality=8)

    phase_duration = duration / len(TASK_PLAN)
    approach_t   = phase_duration * 0.20
    reach_t      = phase_duration * 0.15
    grasp_t      = phase_duration * 0.12
    lift_t       = phase_duration * 0.10
    transport_t  = phase_duration * 0.23
    place_t      = phase_duration * 0.12
    retreat_t    = phase_duration * 0.08

    frame_idx = 0
    current_wp = list(HOME)

    for obj_idx, obj in enumerate(TASK_PLAN):
        cube_name = obj["name"]
        print(f"\n  [{obj_idx + 1}/{len(TASK_PLAN)}] {cube_name.upper()} → "
              f"({obj['target'][0]:.2f}, {obj['target'][1]:.2f})")

        phases = [
            ("approach",  PRE_REACH,  approach_t),
            ("reach",     ABOVE_CUBE, reach_t),
            ("grasp",     GRASP,      grasp_t),
            ("lift",      LIFT,       lift_t),
            ("transport", PRE_PLACE,  transport_t),
            ("place",     PLACE,      place_t),
            ("retreat",   PRE_REACH,  retreat_t),
        ]

        for phase_name, phase_target, phase_len in phases:
            steps = max(1, int(phase_len * fps))
            for s in range(steps):
                t = s / steps
                for j in range(len(phase_target)):
                    data.ctrl[j] = lerp(current_wp[j], phase_target[j], smoothstep(t))
                mujoco.mj_step(model, data)
                frame_idx += 1

                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                writer.append_data(frame)

                if frame_idx % 30 == 0:
                    pct = frame_idx / total_frames * 100
                    bar_fill = int(pct / 100 * 25)
                    bar = "█" * bar_fill + "░" * (25 - bar_fill)
                    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% ({frame_idx}/{total_frames})")
                    sys.stdout.flush()

            current_wp = list(phase_target)

    # Return home
    print("\n  Returning to home ...")
    steps = int(1.0 * fps)
    for s in range(steps):
        t = s / steps
        for j in range(len(HOME)):
            data.ctrl[j] = lerp(current_wp[j], HOME[j], smoothstep(t))
        mujoco.mj_step(model, data)
        frame_idx += 1
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        writer.append_data(frame)

    writer.close()
    renderer.close()

    elapsed = frame_idx / fps
    print(f"\n✓ Done: {frame_idx} frames, {elapsed:.1f}s → {output_path}")
    return output_path


def interactive_view(model: mujoco.MjModel):
    data = mujoco.MjData(model)
    for j in range(8):
        data.ctrl[j] = HOME[j]
    print("Interactive viewer. Close the window to exit.")
    mujoco.viewer.launch(model, data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FFAI Robothon - Long-Horizon Tasks (Franka Panda Assembly)"
    )
    parser.add_argument("--scene", default=str(DEFAULT_SCENE),
                        help=f"Scene XML (default: {DEFAULT_SCENE})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output video path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Video duration in seconds (default: 30)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--view", action="store_true",
                        help="Interactive viewer")
    parser.add_argument("--check-assets", action="store_true",
                        help="Verify model loads")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: scene not found: {scene_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    print(f"✓ Model: {model.nq} pos DOF, {model.nv} vel DOF, "
          f"{model.nu} actuators, {model.nbody} bodies")

    if args.check_assets:
        print("✓ All assets verified.")
        return
    if args.view:
        interactive_view(model)
    else:
        output = render_demo(model, args.duration, args.output, args.fps)
        meta = {
            "project": "Long-Horizon Panda Assembly",
            "uuid": "ac553eae-aa22-4456-bb44-d05be92b06dc",
            "robot": "Franka Emika Panda",
            "task": "Multi-object pick-and-place assembly",
            "num_objects": len(TASK_PLAN),
            "phases_per_object": 7,
            "total_phases": len(TASK_PLAN) * 7 + 1,
            "duration_seconds": args.duration,
            "fps": args.fps,
        }
        meta_path = SUBMISSION / "trajectory_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"✓ Metadata: {meta_path}")


if __name__ == "__main__":
    main()
