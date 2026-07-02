#!/usr/bin/env python3
"""Smoke-test the IsaacGym tensor adapter without requiring IsaacGym itself."""

from __future__ import annotations

import numpy as np
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from front110_core.backends.isaacgym_backend import IsaacGymTensorAdapter
from front110_core.dataset_adapter import validate_npz
from front110_core.isaacgym_runner_bridge import IsaacGymCommonDatasetHook
from front110_core.planner import UnifiedAction
from front110_core.transforms import revo2_active_to_isaac_internal, revo2_isaac_internal_to_active


class _FakeGym:
    def refresh_actor_root_state_tensor(self, sim):
        return None

    def refresh_rigid_body_state_tensor(self, sim):
        return None

    def refresh_dof_state_tensor(self, sim):
        return None


class _FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.num_arm_dofs = 7
        self.num_hand_arm_dofs = 18
        self.num_actions = 18
        self.device = "cpu"
        self.sim = object()
        self.gym = _FakeGym()
        self.cur_targets = np.zeros((2, 18), dtype=np.float32)

        self.arm_hand_dof_pos = np.arange(36, dtype=np.float32).reshape(2, 18) * 0.01
        self.arm_hand_dof_vel = np.arange(36, dtype=np.float32).reshape(2, 18) * 0.001

        self.root_state_tensor = np.zeros((8, 13), dtype=np.float32)
        self.object_indices = np.asarray([[2, 3], [4, 5]], dtype=np.int64)
        for actor in range(2, 6):
            self.root_state_tensor[actor, 0:3] = [actor, actor + 0.1, actor + 0.2]
            self.root_state_tensor[actor, 3:7] = [0.1, 0.2, 0.3, 0.9]  # Isaac xyzw
            self.root_state_tensor[actor, 7:10] = [0.0, 0.0, -1.5]
            self.root_state_tensor[actor, 10:13] = [0.0, 0.0, 0.2]

        self.palm_handle = 1
        self.fingertip_handles = [2, 3, 4]
        self.rigid_body_name_to_idx = {"palm": 1}
        self.rigid_body_states = np.zeros((2, 6, 13), dtype=np.float32)
        self.rigid_body_states[:, 1, 0:3] = [[0.3, 0.0, 0.5], [0.4, 0.0, 0.5]]
        self.rigid_body_states[:, 1, 3:7] = [0.0, 0.0, 0.0, 1.0]
        self.rigid_body_states[:, 2:5, 0:3] = 0.25


def main():
    active = np.asarray([0.8, 0.4, 0.31, 0.42, 0.53, 0.64], dtype=np.float32)
    isaac_internal = revo2_active_to_isaac_internal(active)
    roundtrip = revo2_isaac_internal_to_active(isaac_internal)
    np.testing.assert_allclose(roundtrip, active, atol=1e-6)

    env = _FakeEnv()
    adapter = IsaacGymTensorAdapter.from_env(env)
    action = UnifiedAction(
        fr3_joint_target=np.arange(7, dtype=np.float32) + 1.0,
        revo2_active_target=active,
        phase="smoke",
    )
    adapter.apply_action_to_targets(action)
    np.testing.assert_allclose(env.cur_targets[:, :7], np.broadcast_to(action.fr3_joint_target, (2, 7)))
    np.testing.assert_allclose(env.cur_targets[:, 7:18], np.broadcast_to(isaac_internal, (2, 11)))

    obs = adapter.get_observation(env_id=0)
    assert obs["fr3_qpos"].shape == (7,)
    assert obs["revo2_active_qpos"].shape == (6,)
    assert obs["tool_pos_world"].shape == (2, 3)
    assert obs["tool_quat_wxyz"].shape == (2, 4)
    np.testing.assert_allclose(obs["tool_quat_wxyz"][0], [0.9, 0.1, 0.2, 0.3])
    assert obs["wrist_pos_world"].shape == (3,)
    assert obs["fingertip_pos_world"].shape == (3, 3)

    obs_all = adapter.get_observation(env_id=None)
    assert obs_all["fr3_qpos"].shape == (2, 7)
    assert obs_all["tool_pos_world"].shape == (2, 2, 3)

    with tempfile.TemporaryDirectory() as tmp:
        hook = IsaacGymCommonDatasetHook(env=env, out_dir=Path(tmp), stride=1, validate=True)
        hook.start_episode(ep=0, num_tools=2)
        hook.record_step(
            step=0,
            sim_time_s=0.0,
            arm_target=action.fr3_joint_target,
            hand_internal_target=isaac_internal,
            phase="smoke",
            active_tool_index=1,
            thumb_contact=True,
            opposing_finger_contact=True,
        )
        npz = hook.save_episode({"seed": 1, "success": True, "total": 1, "release_order_angles_deg": [90.0]})
        assert npz is not None and npz.exists()
        assert (Path(tmp) / "manifest_ep000.json").exists()
        validate_npz(npz)

    print("isaacgym adapter and runner bridge smoke passed")


if __name__ == "__main__":
    main()
