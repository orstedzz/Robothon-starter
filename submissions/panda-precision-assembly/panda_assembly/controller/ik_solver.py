"""
IK Solver - Damped Least Squares targeting the `hand` body.
"""

from __future__ import annotations
import numpy as np
import mujoco

from panda_assembly import config


def get_ik_body(model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if bid == -1:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    return bid


def solve_ik(model, data, target_pos, max_iter=50, tol=0.008, damping=0.001):
    """
    DLS IK targeting hand body position.
    Returns (joint_angles[7], success).
    """
    body_id = get_ik_body(model)
    arm_joints = 7

    sd = mujoco.MjData(model)
    sd.qpos[:] = data.qpos[:]
    mujoco.mj_forward(model, sd)

    for _ in range(max_iter):
        err = target_pos - sd.xpos[body_id]
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            return sd.qpos[:arm_joints].copy(), True

        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, sd, jac[:3], None, body_id)
        ja = jac[:, :arm_joints]

        jjt = ja @ ja.T + damping * damping * np.eye(3)
        dq = ja.T @ np.linalg.solve(jjt, err)

        qn = sd.qpos[:arm_joints].copy() + dq
        for j in range(arm_joints):
            qn[j] = np.clip(qn[j], model.jnt_range[j, 0], model.jnt_range[j, 1])
        sd.qpos[:arm_joints] = qn
        mujoco.mj_forward(model, sd)

    final = float(np.linalg.norm(target_pos - sd.xpos[body_id]))
    return sd.qpos[:arm_joints].copy(), final < tol * 3
