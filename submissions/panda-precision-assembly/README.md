# Panda Precision Assembly

### Long-Horizon Precision Peg-Insertion Task with Franka Emika Panda

**FFAI Robothon 2026 — Long-Horizon Tasks**
**Participant:** orsted | **Agent:** ccxiex (Claude Code + Codex)
**UUID:** `ac553eae-aa22-4456-bb44-d05be92b06dc`

---

## Robot Platform

**Franka Emika Panda** — 7-DOF collaborative robot arm with parallel-jaw gripper.
Sourced from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).

Custom additions:
- **Touch sensor sites** on both gripper fingertips (`left_finger_touch`, `right_finger_touch`) for closed-loop force feedback
- **7 joint-position sensors** for real-time state tracking
- **Accelerometer** at end-effector

## Task Goal

A **long-horizon precision assembly task** with 6 closed-loop phases per peg:

| Phase | Description | Feedback |
|-------|-------------|----------|
| 1. **Approach** | IK-solved Cartesian move to peg position | Joint-position servoing |
| 2. **Grasp** | Close gripper until contact force detected | Touch-sensor force feedback |
| 3. **Lift** | Raise peg above assembly jig | Position tracking |
| 4. **Transport** | Move peg above target hole | Cartesian IK |
| 5. **Insert** | Slowly lower peg into hole with force monitoring | Force-limited insertion |
| 6. **Release** | Open gripper and retreat | Confirmed release |

**4 pegs** of varying geometry: brass (round, Ø24mm), steel (square, 24mm), red (small round, Ø16mm), blue (small square, 16mm) — each inserted into its color-matched hole.

**6 phases × 4 pegs = 24 sequential closed-loop phases total.**

## Technical Approach

| Component | Implementation |
|-----------|---------------|
| **Physics** | MuJoCo 3.9 simulation with `implicitfast` integrator |
| **Control** | Damped-least-squares IK → joint-space PD tracking |
| **Force feedback** | Touch sensors on gripper pads provide live contact force readings |
| **Grasping** | Gripper closes until touch-sensor force crosses threshold, confirming secure grasp |
| **Insertion** | IK-driven descent with force monitoring; stops on excessive resistance |
| **Benchmark** | Deterministic single-seed run with JSON report |
| **Scene** | Custom MJCF with workbench, 4 free-floating pegs, 4-hole assembly jig |

## Depth of MuJoCo Use

| Feature | Where used |
|---------|-----------|
| **MJCF** | Custom scene with `include` for vendored Panda model |
| **Free joints** | Each peg is a free-floating body (`<freejoint>`) for full 6-DOF physics |
| **Contact dynamics** | Elliptic cone friction model, real contact forces |
| **Touch sensors** | `<touch>` sensors on gripper fingertip sites |
| **Joint-position sensors** | `<jointpos>` on all 7 arm joints |
| **Sites** | Grasp reference points, sensor locations |
| **IK** | `mj_jacSite` Jacobian + damped least-squares |
| **Gravity compensation** | Via `qfrc_bias` in PD controller |
| **Offscreen rendering** | 1280×720 MP4 video via MuJoCo renderer + FFMPEG |

## How to Run

```bash
# From the Robothon-starter repo root
cd submissions/panda-precision-assembly
python3 -m pip install -r requirements.txt

# Run benchmark (headless, 4 peg tasks, JSON report)
MUJOCO_GL=glfw xvfb-run -a python run.py --benchmark

# Record demo video
MUJOCO_GL=glfw xvfb-run -a python run.py --record

# Interactive MuJoCo viewer
python run.py

# Verify assets
python run.py --check-assets
```

### One-Command Reproduce

```bash
python3 -m pip install -r requirements.txt &&
MUJOCO_GL=glfw xvfb-run -a python run.py --benchmark
```

## Core Features

- ✅ **Closed-loop grasp with force feedback** — gripper stops when contact force exceeds threshold
- ✅ **IK-based Cartesian control** — damped least-squares, joint-limit respecting
- ✅ **Force-limited insertion** — stops on excessive resistance to avoid damage
- ✅ **4 distinct peg geometries** — round, square, large, small
- ✅ **Quantitative benchmark** — per-phase duration, force, success/failure
- ✅ **Deterministic** — same seed = same results

## Benchmark Results

*To be generated: run `python run.py --benchmark`*

Example expected output format:
```
Task: BRASS peg → hole
  ✅ approach      1.25s  err=0.002m
  ✅ grasp         0.53s  force=3.42N
  ✅ lift          1.10s  err=0.003m
  ✅ transport     1.05s  err=0.004m
  ✅ insert        2.30s  force=8.15N
  ✅ release       0.81s

📊 BENCHMARK: 4/4 tasks passed
```

## How This Maps to the Rubric

| Dimension | Evidence |
|-----------|----------|
| **Runnability** | `pip install && python run.py --benchmark` — pure CPU, no GPU |
| **Depth of MuJoCo** | Touch sensors, free joints, IK Jacobians, contact forces, offscreen rendering |
| **Task Design** | Precision peg-insertion — a classic long-horizon assembly benchmark |
| **Control** | IK-based Cartesian control + closed-loop force feedback + sensor-gated grasping |
| **Dexterity** | Parallel-jaw gripper with force-controlled grasp |
| **Engineering Quality** | Modular Python package (controller, IK, config, tasks), typed dataclasses |
| **Presentation** | Demo video with benchmark overlay |
| **Innovation** | Force-closed-loop precision assembly with a widely-available cobot arm |

## Project Structure

```
submissions/panda-precision-assembly/
├── run.py                          # Entry point (benchmark / record / viewer)
├── scene.xml                       # Custom MuJoCo scene
├── registration.json               # Competition UUID
├── requirements.txt                # Python dependencies
├── panda_assembly/
│   ├── __init__.py
│   ├── config.py                   # Central configuration
│   └── controller/
│       ├── __init__.py
│       ├── controller.py           # Main controller with closed-loop skills
│       └── ik_solver.py            # DLS inverse kinematics
├── assets/                         # Franka Panda URDF/MJCF model + meshes
├── results/                        # Benchmark output (JSON)
└── docs/
    └── rubric_mapping.md           # Detailed scoring rubric mapping
```

## Current Limitations

- Single-arm manipulation (no bimanual coordination)
- No visual perception (peg positions are assumed known)
- Deterministic benchmark (only 1 seed; stochastic not required)

## Future Improvements

- Add vision-based peg detection (MuJoCo render + OpenCV)
- Implement reinforcement learning policy for adaptive insertion
- Add force-feedback tremor filtering for smoother insertion
- Extend to multi-step sub-assembly (peg → washer → nut)

---

*Built with ccxiex (Claude Code + Codex) for FFAI Robothon 2026*
