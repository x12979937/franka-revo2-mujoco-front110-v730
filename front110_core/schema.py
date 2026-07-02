"""Documented dataset schema for Front-110 cross-simulator rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .constants import SCHEMA_VERSION


@dataclass(frozen=True)
class CommonDatasetSchema:
    version: str = SCHEMA_VERSION
    action_space: str = "fr3_joint_target[7] + revo2_active_target[6]"
    quaternion_order: str = "wxyz"
    image_layout: str = "optional uint8 RGB NHWC; optional metric depth NHW"
    required_low_dim_keys: Tuple[str, ...] = (
        "fr3_qpos",
        "fr3_qvel",
        "revo2_active_qpos",
        "revo2_active_qvel",
        "tool_pos_world",
        "tool_quat_wxyz",
        "tool_linvel_world",
        "tool_angvel_world",
        "action_fr3_joint_target",
        "action_revo2_active_target",
        "contact_flags",
        "phase",
        "active_tool_index",
    )


COMMON_DATASET_SCHEMA = CommonDatasetSchema()
