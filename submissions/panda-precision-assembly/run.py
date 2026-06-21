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
from panda_assembly.controller import PandaAssemblyController


def render_demo(model, ctrl, output_path, fps=30):
    """Render all 4 peg tasks to demo video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 600  # fixed budget

    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.8
    cam.azimuth = 135
    cam.elevation = -25
    cam.lookat = [0.45, 0.0, 0.15]

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=10)

    def snap():
        renderer.update_scene(ctrl.data, camera=cam)
        writer.append_data(renderer.render())

    frame_idx = 0
    ctrl.home(80)

    for peg in config.PEGS:
        name = peg["name"]
        print(f"  {name.upper()} peg")

        # Approach via IK
        peg_pos = ctrl.get_peg_pos(name)
        if peg_pos is None:
            continue
        target = peg_pos + [0, 0, 0.10]
        ctrl.ik_to(target, 200)
        for _ in range(10):
            mujoco.mj_step(model, ctrl.data)
            snap(); frame_idx += 1

        # Grasp
        ctrl.data.ctrl[7] = 60
        for _ in range(15):
            mujoco.mj_step(model, ctrl.data)
            snap(); frame_idx += 1

        # Lift
        lift_t = peg_pos + [0, 0, 0.25]
        ctrl.ik_to(lift_t, 200)
        for _ in range(10):
            mujoco.mj_step(model, ctrl.data)
            snap(); frame_idx += 1

        # Transport above hole
        hole = np.array(peg["hole_pos"]) + [0, 0, 0.12]
        ctrl.ik_to(hole, 250)
        for _ in range(10):
            mujoco.mj_step(model, ctrl.data)
            snap(); frame_idx += 1

        # Insert (lower into hole)
        for dz in np.linspace(0.12, -0.02, 50):
            t = np.array(peg["hole_pos"]) + [0, 0, dz]
            ctrl.ik_to(t, 40)
            if frame_idx % 2 == 0:
                snap(); frame_idx += 1

        # Release
        ctrl.data.ctrl[7] = 200
        for _ in range(10):
            mujoco.mj_step(model, ctrl.data)
            snap(); frame_idx += 1

        # Home
        ctrl.home(100)
        for _ in range(8):
            mujoco.mj_step(model, ctrl.data)
            snap(); frame_idx += 1

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
