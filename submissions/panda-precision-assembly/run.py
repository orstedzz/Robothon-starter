#!/usr/bin/env python3
"""Panda Multi-Object Pick-and-Place — FFAI Robothon 2026 | Long-Horizon Tasks"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

try:
    import imageio, mujoco
except ImportError as exc:
    raise SystemExit(f"pip install mujoco numpy imageio imageio-ffmpeg\n{exc}") from exc

HERE = Path(__file__).resolve().parent
SCENE = str(HERE / "scene.xml")
OUT = HERE / "results" / "demo.mp4"

# Verified working joint-space waypoints [j1..j7, gripper(0-255)]
HOME       = [0.00, -0.785, 0.00, -2.356, 0.00, 1.571, 0.785, 255]
PRE_REACH  = [0.50, -0.60, 0.20, -2.00, 0.10, 1.20, 0.80, 255]
ABOVE_CUBE = [0.37, -0.85, 0.15, -1.80, 0.00, 1.40, 0.60, 255]
GRASP      = [0.37, -0.85, 0.15, -1.80, 0.00, 1.40, 0.60, 60]
LIFT       = [0.37, -0.65, 0.20, -2.00, 0.10, 1.25, 0.80, 60]
PRE_PLACE  = [0.55, -0.90, 0.30, -2.20, 0.20, 1.00, 0.60, 60]
PLACE      = [0.55, -0.90, 0.30, -2.20, 0.20, 1.00, 0.60, 255]


def interpolate(model, data, target, steps):
    """Smoothly move from current ctrl to target in `steps` steps."""
    cur = np.array([data.ctrl[i] for i in range(len(target))])
    for s in range(steps):
        t = s / steps
        ts = t * t * (3 - 2 * t)
        for i in range(len(target)):
            data.ctrl[i] = float(cur[i] + (target[i] - cur[i]) * ts)
        mujoco.mj_step(model, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    print(f"✓ Model loaded: {model.nbody} bodies, {model.njnt} joints")

    if args.record:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = 30
        duration = 65  # 65 seconds = more than 1 minute
        total_target = fps * duration

        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.6
        cam.azimuth = 130
        cam.elevation = -25
        cam.lookat = [0.40, 0.0, 0.20]

        writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=10)
        frame_n = 0

        def cap():
            nonlocal frame_n
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            frame_n += 1

        def phase(target, steps, idle=5):
            """Move to target and snap `idle` frames after."""
            interpolate(model, data, target, steps)
            for _ in range(idle):
                mujoco.mj_step(model, data)
                cap()

        print("Rendering...")

        # 4 objects: do full 6-phase cycle for each
        objects = ["red", "green", "blue", "yellow"]
        for obj_idx, obj_name in enumerate(objects):
            print(f"  [{obj_idx+1}/4] {obj_name.upper()}")

            # Home → PRE_REACH → ABOVE → GRASP → LIFT → PRE_PLACE → PLACE → HOME
            # Each phase: 200-300 steps for smooth motion + idle frames

            phase(PRE_REACH, 200, 10)       # 1. Reach toward object
            phase(ABOVE_CUBE, 200, 10)      # 2. Above cube
            phase(GRASP, 150, 20)           # 3. Grasp (close gripper)
            phase(LIFT, 200, 10)            # 4. Lift up
            phase(PRE_PLACE, 300, 10)       # 5. Transport to target
            phase(PLACE, 200, 20)           # 6. Place (open gripper)
            phase(HOME, 250, 15)            # Return home

            # Progress
            pct = min(100, frame_n / total_target * 100)
            sys.stdout.write(f"\r  Overall: {pct:.0f}%")
            sys.stdout.flush()

        # Fill remaining frames
        while frame_n < total_target:
            mujoco.mj_step(model, data)
            if frame_n % 3 == 0:
                cap()
            else:
                frame_n += 1

        writer.close()
        renderer.close()
        elapsed = frame_n / fps
        print(f"\n✓ Done: {frame_n} frames, {elapsed:.1f}s → {out}")

    else:
        # Interactive viewer
        for j in range(8):
            data.ctrl[j] = HOME[j]
        print("Interactive viewer. Close to exit.")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
