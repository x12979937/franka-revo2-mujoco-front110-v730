"""IsaacGym action adapter for the common Front-110 planner output."""

from __future__ import annotations

import numpy as np

from ..planner import UnifiedAction
from ..transforms import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, revo2_active_to_internal


class IsaacGymActionAdapter:
    """Small adapter contract for wiring the same planner into IsaacGym.

    The IsaacGym runner owns simulator tensors and DOF ordering. This class keeps
    the convention explicit: common planner actions are joint position targets;
    quaternions crossing the boundary must be converted between common wxyz and
    IsaacGym xyzw.
    """

    simulator = "isaacgym"

    def __init__(self, fr3_dof_indices=None, revo2_active_dof_indices=None, revo2_internal_dof_indices=None):
        self.fr3_dof_indices = fr3_dof_indices
        self.revo2_active_dof_indices = revo2_active_dof_indices
        self.revo2_internal_dof_indices = revo2_internal_dof_indices

    def fill_position_target(self, dof_target, action: UnifiedAction, hand_template=None):
        if self.fr3_dof_indices is None:
            dof_target[..., :7] = action.fr3_joint_target
        else:
            dof_target[..., self.fr3_dof_indices] = action.fr3_joint_target

        if self.revo2_internal_dof_indices is not None:
            hand = action.revo2_internal_target
            if hand is None:
                hand = revo2_active_to_internal(action.revo2_active_target, template=hand_template)
            dof_target[..., self.revo2_internal_dof_indices] = hand
        elif self.revo2_active_dof_indices is not None:
            dof_target[..., self.revo2_active_dof_indices] = action.revo2_active_target
        else:
            dof_target[..., 7:13] = action.revo2_active_target
        return dof_target

    @staticmethod
    def isaac_quat_xyzw_to_common(q_xyzw):
        return quat_xyzw_to_wxyz(np.asarray(q_xyzw, dtype=np.float32))

    @staticmethod
    def common_quat_wxyz_to_isaac(q_wxyz):
        return quat_wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float32))
