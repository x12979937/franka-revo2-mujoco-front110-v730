"""Common NPZ recorder/validator for MuJoCo and IsaacGym Front-110 rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import (
    DROP_XY_JITTER_M,
    FRONT_SECTOR_DEG,
    HANDLE_POSITIVE_LENGTH_M,
    NOMINAL_FRONT_ARC_RADIUS_M,
    REVO2_ACTIVE_JOINT_ORDER,
    SCHEMA_VERSION,
    TOOL_DIAMETER_M,
    TOOL_LENGTH_M,
    TOOL_MASS_KG,
)
from .schema import COMMON_DATASET_SCHEMA


def contact_flags_array(thumb_contact=False, opposing_finger_contact=False, bad_functional_contact=False):
    return np.asarray([thumb_contact, opposing_finger_contact, bad_functional_contact], dtype=np.bool_)


class EpisodeRecorder:
    def __init__(self, num_tools, schema_version=SCHEMA_VERSION):
        self.num_tools = int(num_tools)
        self.schema_version = schema_version
        self.rows = []

    def append(self, obs, action, phase, active_tool_index, contact_flags, sim_time_s):
        row = {k: np.asarray(v).copy() for k, v in obs.items()}
        row.update(action.as_npz_fields())
        row["phase"] = str(phase)
        row["active_tool_index"] = int(active_tool_index)
        row["contact_flags"] = np.asarray(contact_flags, dtype=np.bool_).copy()
        row["sim_time_s"] = float(sim_time_s)
        self.rows.append(row)

    def save_npz(self, path):
        if not self.rows:
            raise RuntimeError("cannot save empty episode")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.rows[0].keys())
        arrays = {}
        for key in keys:
            values = [r[key] for r in self.rows]
            if key == "phase":
                arrays[key] = np.asarray(values, dtype="U32")
            elif key == "active_tool_index":
                arrays[key] = np.asarray(values, dtype=np.int32)
            else:
                arrays[key] = np.stack(values)
        arrays["schema_version"] = np.asarray(self.schema_version)
        arrays["revo2_active_joint_order"] = np.asarray(REVO2_ACTIVE_JOINT_ORDER, dtype="U32")
        arrays["contact_flags_order"] = np.asarray(["thumb_contact", "opposing_finger_contact", "bad_functional_contact"], dtype="U32")
        np.savez_compressed(path, **arrays)
        return path


def build_manifest(summary, episode_npz, simulator):
    return {
        "schema_version": SCHEMA_VERSION,
        "simulator": simulator,
        "episode_npz": str(episode_npz),
        "action_space": COMMON_DATASET_SCHEMA.action_space,
        "quaternion_order": COMMON_DATASET_SCHEMA.quaternion_order,
        "front_sector_deg": list(FRONT_SECTOR_DEG),
        "nominal_front_arc_radius_m": NOMINAL_FRONT_ARC_RADIUS_M,
        "drop_xy_jitter_m": DROP_XY_JITTER_M,
        "tool": {
            "length_m": TOOL_LENGTH_M,
            "diameter_m": TOOL_DIAMETER_M,
            "mass_kg": TOOL_MASS_KG,
            "handle_positive_length_m": HANDLE_POSITIVE_LENGTH_M,
        },
        "summary": {
            "seed": summary.get("seed"),
            "passed": summary.get("passed"),
            "total": summary.get("total"),
            "release_order_angles_deg": summary.get("release_order_angles_deg"),
            "robot_mjcf": summary.get("robot_mjcf"),
        },
    }


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def validate_npz(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in COMMON_DATASET_SCHEMA.required_low_dim_keys if k not in data.files]
        if missing:
            raise ValueError(f"{path} missing required keys: {missing}")
        t = int(data["fr3_qpos"].shape[0])
        checks = {
            "fr3_qpos": (t, 7),
            "fr3_qvel": (t, 7),
            "revo2_active_qpos": (t, 6),
            "revo2_active_qvel": (t, 6),
            "action_fr3_joint_target": (t, 7),
            "action_revo2_active_target": (t, 6),
            "contact_flags": (t, 3),
            "active_tool_index": (t,),
        }
        for key, shape in checks.items():
            if data[key].shape != shape:
                raise ValueError(f"{key} shape {data[key].shape} != {shape}")
        if data["tool_pos_world"].shape[:2] != (t, data["tool_quat_wxyz"].shape[1]):
            raise ValueError("tool position/quaternion tool dimensions disagree")
    return True
