"""Lightweight hooks for recording Dynamic_Gym IsaacGym rollouts in the common schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends.isaacgym_backend import IsaacGymTensorAdapter
from .dataset_adapter import EpisodeRecorder, build_manifest, contact_flags_array, validate_npz, write_manifest
from .planner import UnifiedAction
from .transforms import revo2_isaac_internal_to_active


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_env_vector(value, width: int):
    arr = _to_numpy(value)
    if arr.ndim == 2:
        arr = arr[0]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] != width:
        raise ValueError(f"expected width {width}, got shape {arr.shape}")
    return arr


def _resolve_output_dir(dataset_out_dir, runner_out_dir: Path) -> Path:
    out = Path(dataset_out_dir)
    if not out.is_absolute():
        out = Path(runner_out_dir) / out
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class IsaacGymCommonDatasetHook:
    """Owns one common-schema recorder for a v490/v699 IsaacGym rollout."""

    env: Any
    out_dir: Path
    stride: int = 1
    validate: bool = False

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, int(self.stride))
        self.adapter = IsaacGymTensorAdapter.from_env(self.env)
        self.recorder: EpisodeRecorder | None = None
        self.ep = None
        self.steps_recorded = 0

    def start_episode(self, ep: int, num_tools: int):
        self.ep = int(ep)
        self.steps_recorded = 0
        self.recorder = EpisodeRecorder(num_tools=num_tools)

    def record_step(
        self,
        *,
        step: int,
        sim_time_s: float,
        arm_target,
        hand_internal_target,
        phase: str,
        active_tool_index: int,
        thumb_contact=False,
        opposing_finger_contact=False,
        bad_functional_contact=False,
        metadata: dict[str, float] | None = None,
    ):
        if self.recorder is None:
            raise RuntimeError("start_episode must be called before record_step")
        if int(step) % self.stride != 0:
            return

        arm = _first_env_vector(arm_target, 7)
        hand_internal = _first_env_vector(hand_internal_target, 11)
        action = UnifiedAction(
            fr3_joint_target=arm,
            revo2_active_target=revo2_isaac_internal_to_active(hand_internal),
            revo2_internal_target=hand_internal,
            phase=str(phase),
            metadata=metadata or {},
        )
        obs = self.adapter.get_observation(env_id=0, refresh=True, as_numpy=True)
        contacts = contact_flags_array(
            thumb_contact=bool(thumb_contact),
            opposing_finger_contact=bool(opposing_finger_contact),
            bad_functional_contact=bool(bad_functional_contact),
        )
        self.recorder.append(
            obs=obs,
            action=action,
            phase=str(phase),
            active_tool_index=int(active_tool_index),
            contact_flags=contacts,
            sim_time_s=float(sim_time_s),
        )
        self.steps_recorded += 1

    def save_episode(self, summary: dict[str, Any] | None = None) -> Path | None:
        if self.recorder is None or not self.recorder.rows:
            return None
        ep = 0 if self.ep is None else int(self.ep)
        npz_path = self.out_dir / f"isaacgym_common_episode_ep{ep:03d}.npz"
        self.recorder.save_npz(npz_path)
        if self.validate:
            validate_npz(npz_path)

        manifest_summary = dict(summary or {})
        manifest_summary.setdefault("seed", manifest_summary.get("seed"))
        manifest_summary.setdefault("passed", manifest_summary.get("success"))
        manifest_summary.setdefault("total", 1)
        manifest = build_manifest(manifest_summary, npz_path, simulator="isaacgym")
        manifest["runner"] = "run_isaacgym_front110_common_dataset_v490.py"
        manifest["steps_recorded"] = self.steps_recorded
        write_manifest(self.out_dir / f"manifest_ep{ep:03d}.json", manifest)
        return npz_path


def maybe_create_common_dataset_hook(args, env):
    dataset_out = getattr(args, "common_dataset_out_dir", None)
    if not dataset_out:
        return None
    out_dir = _resolve_output_dir(dataset_out, Path(args.out_dir))
    return IsaacGymCommonDatasetHook(
        env=env,
        out_dir=out_dir,
        stride=getattr(args, "common_dataset_stride", 1),
        validate=getattr(args, "common_dataset_validate", False),
    )
