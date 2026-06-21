"""
IK Solver for Franka Panda - Damped Least Squares Inverse Kinematics.
Solves for joint velocities given a desired end-effector pose.
"""

from __future__ import annotations
import numpy as np
import mujoco

from panda_assembly import config


def get_endeffector_id(model: mujoco.MjModel) -> int:
    """Get the end-effector body ID (link7 = last arm link before gripper)."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
    damping: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Damped least-squares IK.

    Args:
        model: MuJoCo model
        data: MuJoCo data (initial qpos used as seed)
        target_pos: 3D target position (world frame)
        target_quat: Optional target orientation quaternion [w,x,y,z]
        max_iter: Max iterations
        tol: Position tolerance (m)
        damping: DLS damping factor

    Returns:
        (joint_angles[7], success)
    """
    if max_iter is None:
        max_iter = config.IK_MAX_ITER
    if tol is None:
        tol = config.IK_TOLERANCE
    if damping is None:
        damping = config.IK_DAMPING

    body_id = get_endeffector_id(model)
    nq = model.nq  # 7 arm joints + 2 finger = 9
    arm_joints = 7

    # Use a scratch data to avoid modifying the main simulation
    scratch_data = mujoco.MjData(model)
    scratch_data.qpos[:] = data.qpos[:]
    mujoco.mj_forward(model, scratch_data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id == -1:
        # No attachment site, use body position
        site_id = None

    for iteration in range(max_iter):
        if site_id is not None:
            current_pos = scratch_data.site_xpos[site_id].copy()
        else:
            current_pos = scratch_data.xpos[body_id].copy()

        # Position error
        pos_err = target_pos - current_pos
        err_norm = np.linalg.norm(pos_err)

        if err_norm < tol:
            return scratch_data.qpos[:arm_joints].copy(), True

        # Jacobian for end-effector translation
        jac = np.zeros((3, model.nv))
        if site_id is not None:
            mujoco.mj_jacSite(model, scratch_data, jac[:3], None, site_id)
        else:
            mujoco.mj_jacBody(model, scratch_data, jac[:3], None, body_id)

        # Only use the first 7 joints (arm joints, not gripper)
        jac_arm = jac[:, :arm_joints]

        # Damped least squares: Δq = J^T (J J^T + λ² I)⁻¹ e
        jjt = jac_arm @ jac_arm.T
        jjt_reg = jjt + damping * damping * np.eye(3)
        dq = jac_arm.T @ np.linalg.solve(jjt_reg, pos_err)

        # Step forward with joint limits
        q_new = scratch_data.qpos[:arm_joints].copy() + dq

        # Clamp to joint limits
        for j in range(arm_joints):
            q_min = model.jnt_range[j, 0]
            q_max = model.jnt_range[j, 1]
            q_new[j] = np.clip(q_new[j], q_min, q_max)

        scratch_data.qpos[:arm_joints] = q_new
        mujoco.mj_forward(model, scratch_data)

    # Final check
    if site_id is not None:
        final_pos = scratch_data.site_xpos[site_id]
    else:
        final_pos = scratch_data.xpos[body_id]
    success = np.linalg.norm(target_pos - final_pos) < tol * 3

    return scratch_data.qpos[:arm_joints].copy(), success
