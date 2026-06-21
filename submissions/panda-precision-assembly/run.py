#!/usr/bin/env python3
"""
Panda Precision Assembly — FFAI Robothon 2026
================================================
Long-Horizon Precision Peg-Insertion Task with Franka Emika Panda.

Closes the loop with touch sensor feedback for robust grasping and insertion.

Usage:
  pip install -r requirements.txt

  # Run benchmark (headless) — 4 peg tasks, quantitative results
  MUJOCO_GL=glfw xvfb-run -a python run.py --benchmark

  # Interactive viewer
  python run.py

  # Record demo video
  MUJOCO_GL=glfw xvfb-run -a python run.py --record
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import imageio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\n"
        "  python3 -m pip install -r requirements.txt"
    ) from exc

from panda_assembly import config
from panda_assembly.controller import PandaAssemblyController


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_demo(
    model: mujoco.MjModel,
    ctrl: PandaAssemblyController,
    output_path: str | Path,
    fps: int = 30,
    duration: float = 30.0,
):
    """Render benchmark run to MP4 with telemetry overlay."""
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
    frame_idx = 0
    peg_idx = 0

    ctrl.home(150)

    for peg in config.PEGS:
        peg_idx += 1
        peg_name = peg["name"]
        print(f"  [{peg_idx}/{len(config.PEGS)}] {peg_name.upper()} peg")

        phases = [
            ctrl.approach_peg,
            ctrl.grasp_peg,
            ctrl.lift_peg,
            ctrl.transport_to_hole,
            ctrl.insert_peg,
            ctrl.release_peg,
        ]

        for phase_fn in phases:
            phase_rec = phase_fn(peg_name)
            frames_this_phase = max(5, int(phase_rec.duration / model.opt.timestep))

            for _ in range(min(frames_this_phase, 60)):
                mujoco.mj_step(model, ctrl.data)

                # Telemetry overlay via font rendering disabled for simplicity
                renderer.update_scene(ctrl.data, camera=cam)
                frame = renderer.render()
                writer.append_data(frame)
                frame_idx += 1

                if frame_idx % 30 == 0:
                    pct = frame_idx / total_frames * 100
                    sys.stdout.write(f"\r  {pct:5.1f}% ({frame_idx}/{total_frames})")
                    sys.stdout.flush()

        ctrl.home(100)

    writer.close()
    renderer.close()
    print(f"\n✓ Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Panda Precision Assembly — Long-Horizon Peg Insertion"
    )
    parser.add_argument("--benchmark", action="store_true",
                        help="Run quantitative benchmark (no GUI)")
    parser.add_argument("--record", action="store_true",
                        help="Record demo video")
    parser.add_argument("--output", default=str(config.OUTPUT_DIR / "demo.mp4"),
                        help="Output video path")
    parser.add_argument("--check-assets", action="store_true",
                        help="Verify scene and model load")
    args = parser.parse_args()

    # Load model
    model = mujoco.MjModel.from_xml_path(config.SCENE_PATH)
    data = mujoco.MjData(model)
    print(f"✓ Model loaded: {model.nbody} bodies, {model.njnt} joints, "
          f"{model.nu} actuators, {model.nsensor} sensors")

    if args.check_assets:
        return

    ctrl = PandaAssemblyController(model, data)

    if args.benchmark:
        results = ctrl.run_benchmark()
        n_pass = results["passed"]
        n_total = results["total_tasks"]
        print(f"\n{'='*50}")
        print(f"📊 BENCHMARK: {n_pass}/{n_total} tasks passed")
        print(f"{'='*50}")

    elif args.record:
        render_demo(model, ctrl, args.output)

    else:
        # Interactive viewer
        ctrl.home()
        print("Interactive viewer. Close window to exit.")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
