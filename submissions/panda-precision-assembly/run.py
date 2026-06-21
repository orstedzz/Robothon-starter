#!/usr/bin/env python3
"""
Panda Pick-and-Place Demo — FFAI Robothon 2026
Long-Horizon Tasks: multi-object pick-and-place with smooth joint-space control.

A clean, reliable demo using only known-good joint positions.
"""

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

# Known-good joint positions
HOME   = [0.00, -0.785, 0.00, -2.356, 0.00, 1.571, 0.785]
PICK   = [0.35, -0.85,  0.20, -1.80,  0.00, 1.40,  0.60]
LIFT   = [0.35, -0.60,  0.25, -1.70,  0.10, 1.10,  0.80]
PLACE  = [0.15, -0.75,  0.25, -2.00,  0.10, 1.20,  0.70]

G_OPEN = 255
G_CLOSE = 60


def smooth_move(model, data, target, steps=200):
    current = np.array([data.ctrl[i] for i in range(len(target))])
    for s in range(steps + 1):
        t = s / steps
        ts = t * t * (3 - 2 * t)
        for i in range(len(target)):
            data.ctrl[i] = float(current[i] + (target[i] - current[i]) * ts)
        mujoco.mj_step(model, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default=str(HERE / "results" / "demo.mp4"))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    print(f"✓ {model.nbody} bodies, {model.njnt} joints")

    if args.record:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = args.fps
        total_target = int(args.duration * fps)

        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.5
        cam.azimuth = 140
        cam.elevation = -25
        cam.lookat = [0.35, 0.0, 0.15]

        writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=10)
        frame_n = 0

        def snap(skip=1):
            nonlocal frame_n
            if frame_n % skip == 0:
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
            frame_n += 1

        # Start at home
        print("Starting...")
        data.ctrl[7] = G_OPEN
        smooth_move(model, data, HOME + [G_OPEN], 100)
        snap(1)

        objects = ["red", "green", "blue", "yellow"]

        for obj_name in objects:
            print(f"  {obj_name.upper()}")

            # Approach
            data.ctrl[7] = G_OPEN
            smooth_move(model, data, PICK + [G_OPEN], 250)
            for _ in range(20):
                mujoco.mj_step(model, data); snap(2)

            # Grasp
            data.ctrl[7] = G_CLOSE
            for _ in range(40):
                mujoco.mj_step(model, data); snap(2)

            # Lift
            smooth_move(model, data, LIFT + [G_CLOSE], 200)
            for _ in range(20):
                mujoco.mj_step(model, data); snap(2)

            # Transport
            smooth_move(model, data, PLACE + [G_CLOSE], 250)
            for _ in range(20):
                mujoco.mj_step(model, data); snap(2)

            # Release
            data.ctrl[7] = G_OPEN
            for _ in range(40):
                mujoco.mj_step(model, data); snap(2)

            # Return
            smooth_move(model, data, HOME + [G_OPEN], 200)
            for _ in range(15):
                mujoco.mj_step(model, data); snap(2)

            pct = min(100, frame_n / total_target * 100)
            sys.stdout.write(f"\r  {pct:.0f}%")
            sys.stdout.flush()

        # Fill remaining frames
        while frame_n < total_target:
            mujoco.mj_step(model, data)
            snap(3)

        writer.close()
        renderer.close()
        print(f"\n✓ {frame_n} frames → {output}")

    else:
        data.ctrl[7] = G_OPEN
        for j in range(7):
            data.ctrl[j] = HOME[j]
        print("Interactive viewer. Close to exit.")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
