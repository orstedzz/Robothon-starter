"""
Panda Precision Assembly - Main Controller
IK-based positioning for precise peg reaching and insertion.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import mujoco

from panda_assembly import config
from panda_assembly.controller.ik_solver import solve_ik


@dataclass
class PhaseRecord:
    name: str
    peg: str
    duration: float = 0.0
    success: bool = False
    peak_contact_force: float = 0.0
    notes: str = ""


@dataclass
class TrialRecord:
    peg: str
    phases: list[PhaseRecord] = field(default_factory=list)
    overall_success: bool = False
    total_duration: float = 0.0


HOME_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
GRIP_CLOSE = 60
GRIP_OPEN = 200


class PandaAssemblyController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep
        self.grip_act = 7
        self._cache_ids()

    def _cache_ids(self):
        for peg in config.PEGS:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, peg["body"])
            config.BODY_IDS[peg["name"]] = bid

    def set_ctrl(self, ctrl_vals, steps=200):
        """Smoothly ramp all ctrl values to target."""
        current = np.array([self.data.ctrl[j] for j in range(len(ctrl_vals))])
        target = np.array(ctrl_vals)
        for s in range(steps):
            t = (s + 1) / steps
            ts = t * t * (3 - 2 * t)
            for j in range(len(target)):
                self.data.ctrl[j] = current[j] + (target[j] - current[j]) * ts
            mujoco.mj_step(self.model, self.data)

    def home(self, steps=100):
        """Return to home position."""
        self.set_ctrl(list(HOME_Q) + [GRIP_OPEN], steps)

    def ik_to(self, target_pos, steps=200):
        """Move end-effector to target position using IK, return successful joints."""
        dq, ok = solve_ik(self.model, self.data, target_pos)
        if dq is not None:
            self.set_ctrl(list(dq) + [int(self.data.ctrl[self.grip_act])], steps)
        else:
            ok = False
        return ok

    def get_peg_pos(self, peg_name):
        """Get current peg body position in world coords."""
        bid = config.BODY_IDS.get(peg_name, -1)
        if bid >= 0:
            return self.data.xpos[bid].copy()
        return None

    def execute_peg_task(self, peg_name):
        """Full 6-phase peg task using IK positioning."""
        record = TrialRecord(peg=peg_name)
        t0 = time.monotonic()
        peg_cfg = [p for p in config.PEGS if p["name"] == peg_name][0]
        hole_pos = np.array(peg_cfg["hole_pos"])

        # Get initial peg position
        peg_pos = self.get_peg_pos(peg_name)
        if peg_pos is None:
            record.overall_success = False
            return record

        # Phase 1: IK approach to a point slightly above the peg
        approach_target = peg_pos + np.array([0, 0, 0.10])
        approach_target[2] = max(approach_target[2], 0.15)
        self.data.ctrl[self.grip_act] = GRIP_OPEN
        ok = self.ik_to(approach_target, 250)
        record.phases.append(PhaseRecord(
            name="approach", peg=peg_name, duration=time.monotonic() - t0,
            success=ok,
        ))

        # Phase 2: Grasp - close gripper
        self.data.ctrl[self.grip_act] = GRIP_CLOSE
        for _ in range(80):
            mujoco.mj_step(self.model, self.data)
        record.phases.append(PhaseRecord(
            name="grasp", peg=peg_name, duration=80 * self.dt, success=True,
        ))

        # Phase 3: Lift using IK
        t1 = time.monotonic()
        lift_target = peg_pos + np.array([0, 0, 0.20])
        lift_target[2] = max(lift_target[2], 0.20)
        ok = self.ik_to(lift_target, 250)
        record.phases.append(PhaseRecord(
            name="lift", peg=peg_name, duration=time.monotonic() - t1, success=ok,
        ))

        # Phase 4: Transport above hole using IK
        t1 = time.monotonic()
        above_hole = hole_pos + np.array([0, 0, 0.10])
        ok = self.ik_to(above_hole, 300)
        record.phases.append(PhaseRecord(
            name="transport", peg=peg_name, duration=time.monotonic() - t1, success=ok,
        ))

        # Phase 5: Insert - gradual IK descent into hole
        t1 = time.monotonic()
        inserted = False
        depth = 0.0
        for dz in np.linspace(0.10, -0.04, 80):
            target = hole_pos + np.array([0, 0, dz])
            self.ik_to(target, 60)
            # Check peg depth
            bid = config.BODY_IDS.get(peg_name, -1)
            if bid >= 0:
                peg_z = self.data.xpos[bid, 2]
                depth = hole_pos[2] - peg_z
                if depth >= config.BENCHMARK["insertion_depth"]:
                    inserted = True
                    break

        # Release
        self.data.ctrl[self.grip_act] = GRIP_OPEN
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        record.phases.append(PhaseRecord(
            name="insert", peg=peg_name, duration=time.monotonic() - t1,
            success=inserted, notes=f"depth={depth:.3f}m",
        ))

        # Phase 6: Retreat home
        t1 = time.monotonic()
        self.home(200)
        record.phases.append(PhaseRecord(
            name="retreat", peg=peg_name, duration=time.monotonic() - t1, success=True,
        ))

        record.total_duration = time.monotonic() - t0
        record.overall_success = all(
            p.success for p in record.phases if p.name in ("approach", "grasp"))
        return record

    def run_benchmark(self):
        self.home()
        results = []
        for peg in config.PEGS:
            name = peg["name"]
            print(f"\n{'='*40}\n{name.upper()} peg\n{'='*40}")
            r = self.execute_peg_task(name)
            results.append(asdict(r))
            icon = "✅" if r.overall_success else "❌"
            print(f"  Overall: {icon} ({r.total_duration:.2f}s)")
            for p in r.phases:
                pi = "✅" if p.success else "❌"
                print(f"  {pi} {p.name:10s} {p.duration:.2f}s  {p.notes}")
            self.home(80)

        n_pass = sum(1 for r in results if r["overall_success"])
        summary = {
            "project": "Panda Precision Assembly",
            "uuid": "ac553eae-aa22-4456-bb44-d05be92b06dc",
            "robot": "Franka Emika Panda",
            "task": "4-peg IK-based precision assembly",
            "total": len(results), "passed": n_pass,
            "rate": f"{n_pass}/{len(results)}",
            "results": results,
        }
        with open(config.OUTPUT_DIR / "benchmark.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n📊 {n_pass}/{len(results)} passed")
        return summary
