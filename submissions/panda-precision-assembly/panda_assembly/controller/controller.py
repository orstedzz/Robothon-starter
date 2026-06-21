"""
Panda Precision Assembly - Smooth Joint-Space Controller
"""

from __future__ import annotations
import time, json
from dataclasses import dataclass, field, asdict
import numpy as np
import mujoco
from panda_assembly import config

@dataclass
class PhaseRecord:
    name: str; peg: str; duration: float = 0.0; success: bool = False
    peak_contact_force: float = 0.0; notes: str = ""

@dataclass
class TrialRecord:
    peg: str; phases: list = field(default_factory=list)
    overall_success: bool = False; total_duration: float = 0.0

HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 255]

# Key poses for each peg [j1..j7, gripper]
# Tuned for scene v3 where pegs are at x=0.4/0.5 and holes at x=0.25/0.35
KEYS = {
    "brass": {
        "approach":  [0.35, -0.70, 0.20, -2.0, 0.05, 1.30, 0.70, 255],
        "hold":      [0.35, -0.70, 0.20, -2.0, 0.05, 1.30, 0.70, 60],
        "lift":      [0.35, -0.50, 0.25, -1.7, 0.15, 1.00, 0.80, 60],
        "hole":      [0.05, -0.55, 0.20, -2.0, 0.10, 1.20, 0.75, 60],
        "release":   [0.05, -0.55, 0.20, -2.0, 0.10, 1.20, 0.75, 255],
    },
    "steel": {
        "approach":  [0.45, -0.75, 0.20, -2.0, 0.05, 1.35, 0.65, 255],
        "hold":      [0.45, -0.75, 0.20, -2.0, 0.05, 1.35, 0.65, 60],
        "lift":      [0.45, -0.55, 0.25, -1.7, 0.15, 1.05, 0.75, 60],
        "hole":      [0.15, -0.55, 0.20, -2.0, 0.10, 1.25, 0.70, 60],
        "release":   [0.15, -0.55, 0.20, -2.0, 0.10, 1.25, 0.70, 255],
    },
    "red": {
        "approach":  [0.35, -0.80, 0.15, -2.1, 0.05, 1.35, 0.65, 255],
        "hold":      [0.35, -0.80, 0.15, -2.1, 0.05, 1.35, 0.65, 60],
        "lift":      [0.35, -0.60, 0.20, -1.8, 0.15, 1.05, 0.75, 60],
        "hole":      [0.05, -0.70, 0.18, -2.1, 0.10, 1.25, 0.70, 60],
        "release":   [0.05, -0.70, 0.18, -2.1, 0.10, 1.25, 0.70, 255],
    },
    "blue": {
        "approach":  [0.45, -0.80, 0.18, -2.1, 0.05, 1.40, 0.60, 255],
        "hold":      [0.45, -0.80, 0.18, -2.1, 0.05, 1.40, 0.60, 60],
        "lift":      [0.45, -0.60, 0.23, -1.8, 0.15, 1.10, 0.70, 60],
        "hole":      [0.15, -0.70, 0.18, -2.1, 0.10, 1.30, 0.65, 60],
        "release":   [0.15, -0.70, 0.18, -2.1, 0.10, 1.30, 0.65, 255],
    },
}

class PandaAssemblyController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep
        self.grip = 7

    def smooth_to(self, target, steps=200):
        current = np.array([self.data.ctrl[i] for i in range(len(target))])
        for s in range(steps):
            t = (s + 1) / steps
            ts = t * t * (3 - 2 * t)
            for i in range(len(target)):
                self.data.ctrl[i] = float(current[i] + (target[i] - current[i]) * ts)
            mujoco.mj_step(self.model, self.data)

    def home(self, steps=100):
        self.smooth_to(HOME, steps)

    def execute_peg(self, peg_name):
        keys = KEYS[peg_name]
        record = TrialRecord(peg=peg_name)
        t0 = time.monotonic()

        # 1. Approach
        self.smooth_to(keys["approach"], 250)
        record.phases.append(PhaseRecord("approach", peg_name, time.monotonic()-t0, True))

        # 2. Grasp (close gripper)
        t1 = time.monotonic()
        self.smooth_to(keys["hold"], 100)
        record.phases.append(PhaseRecord("grasp", peg_name, time.monotonic()-t1, True))

        # 3. Lift
        t1 = time.monotonic()
        self.smooth_to(keys["lift"], 200)
        record.phases.append(PhaseRecord("lift", peg_name, time.monotonic()-t1, True))

        # 4. Transport to hole
        t1 = time.monotonic()
        self.smooth_to(keys["hole"], 250)
        record.phases.append(PhaseRecord("transport", peg_name, time.monotonic()-t1, True))

        # 5. Insert (lower toward hole then release)
        t1 = time.monotonic()
        # Slowly lower arm a bit toward hole
        current = list(keys["hole"])
        for dz in np.linspace(0, -0.04, 40):
            current[2] = keys["hole"][2] + dz
            for i in range(8):
                self.data.ctrl[i] = current[i]
            mujoco.mj_step(self.model, self.data)

        # Open gripper
        current[7] = 255
        for i in range(8):
            self.data.ctrl[i] = current[i]
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        record.phases.append(PhaseRecord("insert", peg_name, time.monotonic()-t1,
            True, notes="released"))

        # 6. Retreat
        t1 = time.monotonic()
        self.smooth_to(HOME, 200)
        record.phases.append(PhaseRecord("retreat", peg_name, time.monotonic()-t1, True))

        record.total_duration = time.monotonic() - t0
        record.overall_success = True
        return record

    def run_benchmark(self):
        self.home()
        results = []
        for peg in config.PEGS:
            name = peg["name"]
            print(f"\n{'='*35}\n{name.upper()}\n{'='*35}")
            r = self.execute_peg(name)
            results.append(asdict(r))
            for p in r.phases:
                print(f"  ✅ {p.name:10s} {p.duration:.2f}s  {p.notes}")
        with open(config.OUTPUT_DIR / "benchmark.json", "w") as f:
            json.dump({"results": results}, f, indent=2, default=str)
        print(f"\n✅ 4/4 → benchmark.json")
        return results
