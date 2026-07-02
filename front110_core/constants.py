"""Shared constants for MuJoCo/IsaacGym dataset and controller alignment."""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "front110_common_v1"

FR3_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]
FR3_JOINT_VEL_LIMIT_RAD_S = np.asarray([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], dtype=np.float32)

REVO2_ACTIVE_JOINT_ORDER = ["thumb_flex", "thumb_aux", "index", "middle", "ring", "pinky"]
REVO2_ACTIVE_JOINT_VEL_LIMIT_RAD_S = np.asarray([2.53, 2.62, 2.27, 2.27, 2.27, 2.27], dtype=np.float32)

# The converted MuJoCo hand has 11 position-controlled joints:
# thumb[0:3], index[3:5], middle[5:7], ring[7:9], pinky[9:11].
REVO2_INTERNAL_DOF = 11
REVO2_ACTIVE_DOF = 6

TOOL_LENGTH_M = 0.33
TOOL_DIAMETER_M = 0.03
TOOL_MASS_KG = 0.125
HANDLE_POSITIVE_LENGTH_M = 0.11
FUNCTIONAL_NEGATIVE_LENGTH_M = 0.22
FRONT_SECTOR_DEG = (35.0, 145.0)
NOMINAL_FRONT_ARC_RADIUS_M = 0.50
DROP_XY_JITTER_M = 0.04
YAW_RANDOM_RANGE_DEG = (-10.0, 10.0)

COMMON_QUATERNION_ORDER = "wxyz"
ISAACGYM_QUATERNION_ORDER = "xyzw"
