#!/usr/bin/env python3
"""Panda Precision Assembly — FFAI Robothon 2026"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import imageio
    import mujoco
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}") from exc

from panda_assembly import config
from panda_assembly.controller import PandaAssemblyController, PEG_WPS, HOME


def render_demo(model, ctrl, output_path, fps=30, duration=15.0):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.8
    cam.azimuth = 135
    cam.elevation = -25
    cam.lookat = [0.45, 0.0, 0.15]

    total_frames = int(duration * fps)
    print(f"Recording {total_frames} frames → {output_path} ...")

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=10)

    def cf():
        renderer.update_scene(ctrl.data, camera=cam)
        writer.append_data(renderer.render())

    frame_idx = 0
    ctrl.home(100)

    for peg in config.PEGS:
        name = peg["name"]
        wp = PEG_WPS[name]
        print(f"  {name.upper()} peg")

        # Approach
        ctrl.smooth_move(wp["approach"], 200)
        for _ in range(15):
            mujoco.mj_step(model, ctrl.data); cf(); frame_idx += 1

        # Grasp
        ctrl.data.ctrl[7] = 60
        for _ in range(20):
            mujoco.mj_step(model, ctrl.data); cf(); frame_idx += 1

        # Lift
        ctrl.smooth_move(wp["lift"], 200)
        for _ in range(15):
            mujoco.mj_step(model, ctrl.data); cf(); frame_idx += 1

        # Transport
        ctrl.smooth_move(wp["hole_above"], 250)
        for _ in range(15):
            mujoco.mj_step(model, ctrl.data); cf(); frame_idx += 1

        # Insert (lower gradually then release)
        current = list(wp["hole_above"])
        for dz in np.linspace(0, -0.08, 60):
            current[2] = wp["hole_above"][2] + dz
            for j in range(8):
                ctrl.data.ctrl[j] = current[j]
            mujoco.mj_step(model, ctrl.data)
            if frame_idx % 3 == 0:
                cf()
            frame_idx += 1

        # Release
        ctrl.data.ctrl[7] = 200
        for _ in range(15):
            mujoco.mj_step(model, ctrl.data); cf(); frame_idx += 1

        # Home
        ctrl.smooth_move(HOME, 150)
        for _ in range(10):
            mujoco.mj_step(model, ctrl.data); cf(); frame_idx += 1

        pct = frame_idx / total_frames * 100
        sys.stdout.write(f"\r  {pct:.0f}%")
        sys.stdout.flush()

    writer.close()
    renderer.close()
    print(f"\n✓ {frame_idx} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default=str(config.OUTPUT_DIR / "demo.mp4"))
    parser.add_argument("--check-assets", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(config.SCENE_PATH)
    data = mujoco.MjData(model)
    print(f"✓ {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")

    if args.check_assets:
        return

    ctrl = PandaAssemblyController(model, data)

    if args.benchmark:
        ctrl.run_benchmark()
    elif args.record:
        render_demo(model, ctrl, args.output)
    else:
        ctrl.home()
        print("Interactive viewer. Close to exit.")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
