"""
Panda Precision Assembly - Main Controller
Closed-loop skills with force feedback and IK-based Cartesian control.
"""

from __future__ import annotations
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import mujoco

from panda_assembly import config
from panda_assembly.controller.ik_solver import solve_ik


# ---------------------------------------------------------------------------
# Phase logging
# ---------------------------------------------------------------------------
@dataclass
class PhaseRecord:
    name: str
    peg: str
    duration: float
    success: bool
    peak_contact_force: float = 0.0
    ik_iters: int = 0
    final_error: float = 0.0
    notes: str = ""


@dataclass
class TrialRecord:
    peg: str
    phases: list[PhaseRecord] = field(default_factory=list)
    overall_success: bool = False
    total_duration: float = 0.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class PandaAssemblyController:
    """Closed-loop controller for precision peg-insertion tasks."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep

        # Cache IDs
        self._cache_ids()
        self._step_count = 0

    def _cache_ids(self):
        m = self.model
        self.gripper_actuator = 7  # actuator 8 = tendon split

        # Body IDs
        for peg in config.PEGS:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, peg["body"])
            config.BODY_IDS[peg["name"]] = bid

        # Site IDs
        for peg in config.PEGS:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, peg["grasp_site"])
            config.SITE_IDS[peg["name"]] = sid

        # Sensor IDs
        for sname in ["left_finger_touch", "right_finger_touch"]:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            config.SENSOR_IDS[sname] = sid

    # ---- Low-level helpers ----

    def home(self, steps: int = 200) -> None:
        """Move to home position."""
        for j in range(7):
            self.data.ctrl[j] = config.HOME_POS[j]
        self.data.ctrl[self.gripper_actuator] = config.GRIPPER_OPEN
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def set_ctrl_smooth(self, ctrl: list[float], steps: int) -> None:
        """Smoothly ramp ctrl from current to target."""
        current = np.array([self.data.ctrl[j] for j in range(len(ctrl))])
        target = np.array(ctrl)
        for s in range(steps):
            t = (s + 1) / steps
            t_s = t * t * (3 - 2 * t)  # smoothstep
            blended = current + (target - current) * t_s
            for j in range(len(ctrl)):
                self.data.ctrl[j] = blended[j]
            mujoco.mj_step(self.model, self.data)

    def ik_move_to(
        self,
        target_pos: np.ndarray,
        max_steps: int = 500,
    ) -> tuple[bool, int, float]:
        """Move end-effector to target position using IK."""
        dq_solution, ik_ok = solve_ik(self.model, self.data, target_pos)

        if not ik_ok:
            # Fallback: still try to move toward target
            pass

        # Smoothly interpolate to IK solution
        current_q = np.array([self.data.qpos[j] for j in range(7)])
        target_q = dq_solution

        for s in range(min(max_steps, 300)):
            t = (s + 1) / 300
            t_s = t * t * (3 - 2 * t)
            for j in range(7):
                self.data.ctrl[j] = current_q[j] + (target_q[j] - current_q[j]) * t_s
            mujoco.mj_step(self.model, self.data)

        # Get final position error
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        if site_id >= 0:
            final_pos = self.data.site_xpos[site_id]
            err = np.linalg.norm(target_pos - final_pos)
        else:
            err = 999.0

        return ik_ok, min(max_steps, 300), float(err)

    def get_contact_force(self) -> float:
        """Read max contact force from finger touch sensors (N)."""
        forces = []
        for sname in ["left_finger_touch", "right_finger_touch"]:
            sid = config.SENSOR_IDS.get(sname, -1)
            if sid >= 0:
                forces.append(float(self.data.sensordata[sid]))
        return max(forces) if forces else 0.0

    def peg_insertion_depth(self, peg_name: str) -> float:
        """Estimate how deep the peg is inserted by checking Z position."""
        bid = config.BODY_IDS.get(peg_name, -1)
        if bid < 0:
            return 0.0
        peg_z = self.data.xpos[bid, 2]
        hole = None
        for p in config.PEGS:
            if p["name"] == peg_name:
                hole = p
                break
        if hole is None:
            return 0.0
        return max(0.0, hole["hole_pos"][2] - peg_z)

    def is_peg_in_hole(self, peg_name: str) -> bool:
        """Check if peg is sufficiently inserted."""
        depth = self.peg_insertion_depth(peg_name)
        return depth >= config.BENCHMARK["insertion_depth"]

    # ---- High-level skills ----

    def approach_peg(self, peg_name: str) -> PhaseRecord:
        """Phase 1: Move above the peg."""
        peg = [p for p in config.PEGS if p["name"] == peg_name][0]
        sitepos = self.data.site_xconfig.SITE_IDS[peg_name]

        # Slightly generous approach: go above
        bid = config.BODY_IDS[peg_name]
        peg_pos = self.data.xpos[bid].copy()
        target = peg_pos + np.array([0, 0, config.APPROACH_HEIGHT])
        target[2] = max(target[2], 0.1)

        ik_ok, n_steps, err = self.ik_move_to(target)
        self.data.ctrl[self.gripper_actuator] = config.GRIPPER_OPEN

        return PhaseRecord(
            name="approach",
            peg=peg_name,
            duration=n_steps * self.dt,
            success=ik_ok,
            final_error=err,
        )

    def grasp_peg(self, peg_name: str) -> PhaseRecord:
        """Phase 2: Close gripper with force feedback."""
        start = time.monotonic()
        peak_force = 0.0

        # Close gripper gradually
        for v in range(config.GRIPPER_OPEN, config.GRIPPER_CLOSE, -5):
            self.data.ctrl[self.gripper_actuator] = v
            for _ in range(3):
                mujoco.mj_step(self.model, self.data)
            cf = self.get_contact_force()
            peak_force = max(peak_force, cf)
            if cf > config.GRIP_FORCE_THRESHOLD:
                break

        # Hold for a few steps
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
            peak_force = max(peak_force, self.get_contact_force())

        success = peak_force >= config.GRIP_FORCE_THRESHOLD
        elapsed = time.monotonic() - start

        return PhaseRecord(
            name="grasp",
            peg=peg_name,
            duration=elapsed,
            success=success,
            peak_contact_force=peak_force,
            notes="contact detected" if success else "no contact",
        )

    def lift_peg(self, peg_name: str) -> PhaseRecord:
        """Phase 3: Lift the peg above the jig."""
        target = [0.45, 0.0, config.LIFT_HEIGHT + 0.1]
        ik_ok, n_steps, err = self.ik_move_to(np.array(target))

        return PhaseRecord(
            name="lift",
            peg=peg_name,
            duration=n_steps * self.dt,
            success=ik_ok,
            final_error=err,
        )

    def transport_to_hole(self, peg_name: str) -> PhaseRecord:
        """Phase 4: Move above the target hole."""
        peg = [p for p in config.PEGS if p["name"] == peg_name][0]
        target = list(peg["hole_pos"])
        target[2] += config.APPROACH_HEIGHT + 0.05

        ik_ok, n_steps, err = self.ik_move_to(np.array(target))

        return PhaseRecord(
            name="transport",
            peg=peg_name,
            duration=n_steps * self.dt,
            success=ik_ok,
            final_error=err,
        )

    def insert_peg(self, peg_name: str) -> PhaseRecord:
        """Phase 5: Slowly insert peg into hole with force monitoring."""
        peg = [p for p in config.PEGS if p["name"] == peg_name][0]
        hole_pos = np.array(peg["hole_pos"])
        start = time.monotonic()
        peak_force = 0.0
        max_time = config.BENCHMARK["max_insertion_time"]
        success = False

        # Step down slowly into the hole
        for dz in np.linspace(0, -0.06, 120):
            target = hole_pos + np.array([0, 0, dz])
            ik_ok, n_steps, err = self.ik_move_to(target, max_steps=50)
            cf = self.get_contact_force()
            peak_force = max(peak_force, cf)

            # Check insertion depth
            if self.is_peg_in_hole(peg_name):
                success = True
                break

            # Check force limit
            if cf > config.INSERT_FORCE_MAX:
                break

            # Check timeout
            if time.monotonic() - start > max_time:
                break

        elapsed = time.monotonic() - start
        return PhaseRecord(
            name="insert",
            peg=peg_name,
            duration=elapsed,
            success=success,
            peak_contact_force=peak_force,
            notes="inserted" if success else "failed",
        )

    def release_peg(self, peg_name: str) -> PhaseRecord:
        """Phase 6: Open gripper and retreat."""
        # Open gripper
        self.data.ctrl[self.gripper_actuator] = config.GRIPPER_OPEN
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        # Retreat upward
        target = [0.45, 0.0, 0.25]
        ik_ok, n_steps, err = self.ik_move_to(np.array(target))

        return PhaseRecord(
            name="release",
            peg=peg_name,
            duration=n_steps * self.dt,
            success=ik_ok,
            notes="released",
        )

    # ---- Full task execution ----

    def execute_peg_task(self, peg_name: str) -> TrialRecord:
        """Execute full long-horizon task for one peg."""
        record = TrialRecord(peg=peg_name)
        t0 = time.monotonic()

        phases = [
            self.approach_peg,
            self.grasp_peg,
            self.lift_peg,
            self.transport_to_hole,
            self.insert_peg,
            self.release_peg,
        ]

        for phase_fn in phases:
            phase_rec = phase_fn(peg_name)
            record.phases.append(phase_rec)

            # If grasp or lift fails, abort
            if phase_rec.name in ("grasp",) and not phase_rec.success:
                break

        record.total_duration = time.monotonic() - t0
        record.overall_success = all(
            p.success for p in record.phases if p.name in ("grasp", "insert", "release")
        )
        return record

    def run_benchmark(self) -> dict[str, Any]:
        """Run all 4 peg tasks and produce benchmark report."""
        self.home(150)
        results = []

        for peg in config.PEGS:
            print(f"\n{'='*50}")
            print(f"Task: {peg['name'].upper()} peg → hole")
            print(f"{'='*50}")

            record = self.execute_peg_task(peg["name"])
            results.append(asdict(record))

            print(f"  Overall: {'✅ PASS' if record.overall_success else '❌ FAIL'}")
            print(f"  Duration: {record.total_duration:.2f}s")
            for p in record.phases:
                icon = "✅" if p.success else "❌"
                print(f"  {icon} {p.name:12s} {p.duration:.2f}s  "
                      f"force={p.peak_contact_force:.2f}N  err={p.final_error:.3f}m")

            # Return to home between pegs
            self.home(100)

        # Summary
        n_total = len(results)
        n_pass = sum(1 for r in results if r["overall_success"])
        summary = {
            "project": "Panda Precision Assembly",
            "uuid": "ac553eae-aa22-4456-bb44-d05be92b06dc",
            "robot": "Franka Emika Panda",
            "task": "Precision Peg-Insertion Assembly (4 pegs)",
            "total_tasks": n_total,
            "passed": n_pass,
            "success_rate": f"{n_pass}/{n_total}",
            "results": results,
            "scoring_notes": {
                "grasp_force_threshold_N": config.GRIP_FORCE_THRESHOLD,
                "insertion_depth_m": config.BENCHMARK["insertion_depth"],
                "max_insertion_force_N": config.INSERT_FORCE_MAX,
            },
        }

        report_path = config.OUTPUT_DIR / "benchmark.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n📊 Benchmark report: {report_path}")

        return summary
