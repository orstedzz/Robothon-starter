#!/usr/bin/env python3
"""Panda Precision Assembly — FFAI Robothon 2026"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

try:
    import imageio, mujoco
except ImportError as exc:
    raise SystemExit(f"Missing: {exc}") from exc

from panda_assembly import config
from panda_assembly.controller import PandaAssemblyController, HOME, KEYS


def render_demo(model, ctrl, output_path, fps=30):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 600

    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.4
    cam.azimuth = 135
    cam.elevation = -25
    cam.lookat = [0.35, 0.0, 0.18]

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=10)

    def frame():
        renderer.update_scene(ctrl.data, camera=cam)
        writer.append_data(renderer.render())

    ctrl.home(80)
    for _ in range(10): mujoco.mj_step(model, ctrl.data); frame()

    idx = 0
    for peg in config.PEGS:
        n = peg["name"]; k = KEYS[n]
        print(f"  {n.upper()}")
        ctrl.smooth_to(k["approach"], 250)
        for _ in range(15): mujoco.mj_step(model, ctrl.data); frame(); idx += 1
        ctrl.smooth_to(k["hold"], 100)
        for _ in range(10): mujoco.mj_step(model, ctrl.data); frame(); idx += 1
        ctrl.smooth_to(k["lift"], 200)
        for _ in range(12): mujoco.mj_step(model, ctrl.data); frame(); idx += 1
        ctrl.smooth_to(k["hole"], 250)
        for _ in range(15): mujoco.mj_step(model, ctrl.data); frame(); idx += 1
        # Insert
        curr = list(k["hole"])
        for dz in np.linspace(0, -0.04, 30):
            curr[2] = k["hole"][2] + dz
            for i in range(8): ctrl.data.ctrl[i] = curr[i]
            mujoco.mj_step(model, ctrl.data)
            if idx % 2 == 0: frame()
            idx += 1
        ctrl.smooth_to(k["release"], 60)
        for _ in range(8): mujoco.mj_step(model, ctrl.data); frame(); idx += 1
        ctrl.smooth_to(HOME, 200)
        for _ in range(10): mujoco.mj_step(model, ctrl.data); frame(); idx += 1
        print(f"    {idx}/{total_frames}")

    writer.close(); renderer.close()
    print(f"✓ {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--output", default=str(config.OUTPUT_DIR / "demo.mp4"))
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(config.SCENE_PATH)
    data = mujoco.MjData(model)
    print(f"✓ {model.nbody} bodies, {model.njnt} joints")

    ctrl = PandaAssemblyController(model, data)
    if args.benchmark: ctrl.run_benchmark()
    elif args.record: render_demo(model, ctrl, args.output)
    else:
        ctrl.home(); mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
