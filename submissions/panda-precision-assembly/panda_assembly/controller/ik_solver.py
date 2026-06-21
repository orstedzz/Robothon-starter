"""
IK Solver for Franka Panda - Damped Least Squares Inverse Kinematics.
Targets the `hand` body (closest to gripper fingers).
"""

from __future__ import annotations
import numpy as np
import mujoco

from panda_assembly import config


def get_ik_body_id(model: mujoco.MjModel) -> int:
    """Get the body ID used as IK target (hand body, closest to fingers)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if bid == -1:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    return bid


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    max_iter: int | None = None,
    tol: float | None = None,
    damping: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Damped least-squares IK targeting 'hand' body position.

    Returns (joint_angles[7], success).
    """
    if max_iter is None:
        max_iter = config.IK_MAX_ITER
    if tol is None:
        tol = config.IK_TOLERANCE
    if damping is None:
        damping = config.IK_DAMPING

    body_id = get_ik_body_id(model)
    arm_joints = 7

    sd = mujoco.MjData(model)
    sd.qpos[:] = data.qpos[:]
    mujoco.mj_forward(model, sd)

    for _ in range(max_iter):
        current_pos = sd.xpos[body_id].copy()
        err = target_pos - current_pos
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            return sd.qpos[:arm_joints].copy(), True

        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, sd, jac[:3], None, body_id)
        jac_a = jac[:, :arm_joints]

        jjt = jac_a @ jac_a.T
        jjt += damping * damping * np.eye(3)
        dq = jac_a.T @ np.linalg.solve(jjt, err)

        qn = sd.qpos[:arm_joints].copy() + dq
        for j in range(arm_joints):
            qn[j] = np.clip(qn[j], model.jnt_range[j, 0], model.jnt_range[j, 1])
        sd.qpos[:arm_joints] = qn
        mujoco.mj_forward(model, sd)

    final_err = float(np.linalg.norm(target_pos - sd.xpos[body_id]))
    return sd.qpos[:arm_joints].copy(), final_err < tol * 3
