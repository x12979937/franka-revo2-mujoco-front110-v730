"""MuJoCo-side helpers for the common Front-110 interface."""

from __future__ import annotations

import numpy as np

from ..planner import UnifiedAction
from ..transforms import revo2_internal_to_active
from .base import SimBackend


class MujocoBackend(SimBackend):
    name = "mujoco"

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def reset_episode(self, episode_spec):
        self.episode_spec = episode_spec

    def get_observation(self):
        raise NotImplementedError("Use extract_low_dim_observation from the task runner with its body/joint ids.")

    def apply_action(self, action: UnifiedAction):
        self.data.ctrl[:7] = action.fr3_joint_target
        hand = action.revo2_internal_target
        if hand is None:
            raise ValueError("MuJoCo converted Revo2 backend expects the 11-DoF internal hand target.")
        self.data.ctrl[7:18] = hand

    def step(self):
        import mujoco

        mujoco.mj_step(self.model, self.data)


def extract_low_dim_observation(model, data, wrist_bid, tool_bids, qadr, qvadr):
    """Return simulator-neutral low-dimensional observation arrays.

    MuJoCo free-joint quaternions are already stored as wxyz, which is the common
    dataset convention. IsaacGym adapters should convert xyzw -> wxyz at export.
    """
    num_tools = len(tool_bids)
    tool_pos = np.zeros((num_tools, 3), dtype=np.float32)
    tool_quat = np.zeros((num_tools, 4), dtype=np.float32)
    tool_linvel = np.zeros((num_tools, 3), dtype=np.float32)
    tool_angvel = np.zeros((num_tools, 3), dtype=np.float32)
    for i in range(num_tools):
        tool_pos[i] = data.qpos[qadr[i] : qadr[i] + 3]
        tool_quat[i] = data.qpos[qadr[i] + 3 : qadr[i] + 7]
        # MuJoCo free-joint qvel layout is angular xyz then linear xyz.
        tool_angvel[i] = data.qvel[qvadr[i] : qvadr[i] + 3]
        tool_linvel[i] = data.qvel[qvadr[i] + 3 : qvadr[i] + 6]
    return {
        "fr3_qpos": np.asarray(data.qpos[:7], dtype=np.float32).copy(),
        "fr3_qvel": np.asarray(data.qvel[:7], dtype=np.float32).copy(),
        "revo2_internal_qpos": np.asarray(data.qpos[7:18], dtype=np.float32).copy(),
        "revo2_internal_qvel": np.asarray(data.qvel[7:18], dtype=np.float32).copy(),
        "revo2_active_qpos": revo2_internal_to_active(data.qpos[7:18]),
        "revo2_active_qvel": revo2_internal_to_active(data.qvel[7:18]),
        "wrist_pos_world": np.asarray(data.xpos[wrist_bid], dtype=np.float32).copy(),
        "wrist_xmat_world": np.asarray(data.xmat[wrist_bid], dtype=np.float32).reshape(3, 3).copy(),
        "tool_pos_world": tool_pos,
        "tool_quat_wxyz": tool_quat,
        "tool_linvel_world": tool_linvel,
        "tool_angvel_world": tool_angvel,
    }
