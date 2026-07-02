"""Simulator-independent action definitions for the Front-110 task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .constants import REVO2_INTERNAL_DOF
from .transforms import revo2_internal_to_active


@dataclass
class UnifiedAction:
    """Common action emitted by planner/controller code.

    The portable control surface is joint-position target based:
    `fr3_joint_target[7] + revo2_active_target[6]`.
    MuJoCo may additionally carry the exact converted 11-DoF Revo2 target so the
    legacy controller can be replayed without losing finger coupling details.
    IsaacGym can ignore that field or use its own hand expansion adapter.
    """

    fr3_joint_target: np.ndarray
    revo2_active_target: np.ndarray
    revo2_internal_target: np.ndarray | None = None
    phase: str = "unknown"
    metadata: Dict[str, float] | None = None

    @classmethod
    def from_internal_hand(cls, fr3_joint_target, revo2_internal_target, phase="unknown", metadata=None):
        internal = np.asarray(revo2_internal_target, dtype=np.float32)
        if internal.shape[-1] != REVO2_INTERNAL_DOF:
            raise ValueError(f"expected {REVO2_INTERNAL_DOF} hand joints, got {internal.shape}")
        return cls(
            fr3_joint_target=np.asarray(fr3_joint_target, dtype=np.float32),
            revo2_active_target=revo2_internal_to_active(internal),
            revo2_internal_target=internal,
            phase=phase,
            metadata=metadata or {},
        )

    def as_npz_fields(self):
        fields = {
            "action_fr3_joint_target": np.asarray(self.fr3_joint_target, dtype=np.float32),
            "action_revo2_active_target": np.asarray(self.revo2_active_target, dtype=np.float32),
        }
        if self.revo2_internal_target is not None:
            fields["action_revo2_internal_target"] = np.asarray(self.revo2_internal_target, dtype=np.float32)
        return fields
