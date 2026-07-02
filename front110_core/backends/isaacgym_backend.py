"""IsaacGym tensor/action adapter for the common Front-110 planner output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants import REVO2_INTERNAL_DOF
from ..planner import UnifiedAction
from ..transforms import (
    quat_xyzw_to_wxyz,
    quat_wxyz_to_xyzw,
    revo2_active_to_isaac_internal,
    revo2_isaac_internal_to_active,
)


@dataclass
class IsaacGymIndexMap:
    """Tensor indices used to bridge a Dynamic_Gym IsaacGym env to common data."""

    fr3_dof_indices: Any
    revo2_internal_dof_indices: Any
    tool_actor_indices: Any | None = None
    wrist_body_index: int | None = None
    fingertip_body_indices: Any | None = None
    palm_body_index: int | None = None


def _is_torch(value) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device")


def _to_numpy(value):
    if value is None:
        return None
    if _is_torch(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_like(value, like):
    if _is_torch(like):
        import torch

        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return np.asarray(value, dtype=np.asarray(like).dtype)


def _long_indices(value, like=None):
    if _is_torch(like) or _is_torch(value):
        import torch

        device = like.device if _is_torch(like) else value.device
        return torch.as_tensor(value, dtype=torch.long, device=device)
    return np.asarray(value, dtype=np.int64)


def _stack_last(parts, like):
    if _is_torch(like):
        import torch

        return torch.stack(parts, dim=-1)
    return np.stack(parts, axis=-1).astype(np.float32)


def _quat_xyzw_to_wxyz_backend(q):
    return _stack_last([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], q)


def _revo2_isaac_internal_to_active_backend(q):
    return _stack_last(
        [
            q[..., 8],
            0.5 * (q[..., 9] + q[..., 10]),
            0.5 * (q[..., 0] + q[..., 1]),
            0.5 * (q[..., 2] + q[..., 3]),
            0.5 * (q[..., 6] + q[..., 7]),
            0.5 * (q[..., 4] + q[..., 5]),
        ],
        q,
    )


def _revo2_active_to_isaac_internal_backend(q_active, template=None, like=None):
    if _is_torch(q_active) or _is_torch(like):
        import torch

        ref = q_active if _is_torch(q_active) else like
        q = torch.as_tensor(q_active, dtype=ref.dtype, device=ref.device)
        if template is None:
            out = torch.zeros(q.shape[:-1] + (REVO2_INTERNAL_DOF,), dtype=ref.dtype, device=ref.device)
            out[..., 0] = q[..., 2]
            out[..., 1] = q[..., 2]
            out[..., 2] = q[..., 3]
            out[..., 3] = q[..., 3]
            out[..., 4] = q[..., 5]
            out[..., 5] = q[..., 5]
            out[..., 6] = q[..., 4]
            out[..., 7] = q[..., 4]
            out[..., 8] = q[..., 0]
            out[..., 9] = q[..., 1]
            out[..., 10] = q[..., 1]
            return out
        t = torch.as_tensor(template, dtype=ref.dtype, device=ref.device)
        prefix = torch.broadcast_shapes(q.shape[:-1], t.shape[:-1])
        if q.shape[:-1] != prefix:
            q = torch.broadcast_to(q, prefix + (q.shape[-1],))
        if t.shape[:-1] != prefix:
            t = torch.broadcast_to(t, prefix + (t.shape[-1],))
        out = torch.zeros(prefix + (REVO2_INTERNAL_DOF,), dtype=ref.dtype, device=ref.device)
        for a, b, src in [(0, 1, 2), (2, 3, 3), (4, 5, 5), (6, 7, 4), (9, 10, 1)]:
            raw = torch.abs(t[..., a]) + torch.abs(t[..., b])
            denom = torch.clamp(raw, min=1e-6)
            wa = torch.where(raw > 1e-6, torch.abs(t[..., a]) / denom, torch.full_like(raw, 0.5))
            wb = torch.where(raw > 1e-6, torch.abs(t[..., b]) / denom, torch.full_like(raw, 0.5))
            out[..., a] = q[..., src] * (2.0 * wa)
            out[..., b] = q[..., src] * (2.0 * wb)
        out[..., 8] = q[..., 0]
        return out
    return revo2_active_to_isaac_internal(q_active, template=template)


def _select_env_rows(value, env_ids):
    if env_ids is None:
        return value
    return value[_long_indices(env_ids, value)]


def _maybe_squeeze_single(value, squeeze):
    return value[0] if squeeze and getattr(value, "shape", ())[0] == 1 else value


class IsaacGymTensorAdapter:
    """Bridge Dynamic_Gym/IsaacGym tensors to the common Front-110 schema.

    The adapter assumes the current Dynamic_Gym FR3+Revo2 layout:
    FR3 arm first, then Revo2 physical hand order
    index(2), middle(2), pinky(2), ring(2), thumb(3).
    """

    simulator = "isaacgym"

    def __init__(self, index_map: IsaacGymIndexMap | None = None, env=None):
        self.index_map = index_map or IsaacGymIndexMap(
            fr3_dof_indices=list(range(7)),
            revo2_internal_dof_indices=list(range(7, 18)),
        )
        self.env = env

    @classmethod
    def from_env(
        cls,
        env,
        tool_actor_indices=None,
        wrist_body_index=None,
        fingertip_body_indices=None,
        palm_body_index=None,
    ):
        num_arm = int(getattr(env, "num_arm_dofs", 7))
        num_hand_arm = int(getattr(env, "num_hand_arm_dofs", num_arm + REVO2_INTERNAL_DOF))
        hand_dof = max(0, min(REVO2_INTERNAL_DOF, num_hand_arm - num_arm))
        if hand_dof != REVO2_INTERNAL_DOF:
            raise ValueError(f"expected 11 Revo2 hand DOFs after FR3 arm, got {hand_dof}")

        if tool_actor_indices is None:
            tool_actor_indices = getattr(env, "object_indices", None)
        if palm_body_index is None:
            palm_body_index = getattr(env, "palm_handle", None)
        if fingertip_body_indices is None:
            fingertip_body_indices = getattr(env, "fingertip_handles", None)
        if wrist_body_index is None:
            wrist_body_index = palm_body_index
            name_to_idx = getattr(env, "rigid_body_name_to_idx", {}) or {}
            for key in ("panda_hand", "fr3_hand", "hand", "palm"):
                if key in name_to_idx:
                    wrist_body_index = int(name_to_idx[key])
                    break

        return cls(
            IsaacGymIndexMap(
                fr3_dof_indices=list(range(num_arm)),
                revo2_internal_dof_indices=list(range(num_arm, num_arm + REVO2_INTERNAL_DOF)),
                tool_actor_indices=tool_actor_indices,
                wrist_body_index=None if wrist_body_index is None else int(wrist_body_index),
                fingertip_body_indices=fingertip_body_indices,
                palm_body_index=None if palm_body_index is None else int(palm_body_index),
            ),
            env=env,
        )

    def refresh(self, env=None):
        env = env or self.env
        if env is None or not hasattr(env, "gym"):
            return
        for name in ("refresh_actor_root_state_tensor", "refresh_rigid_body_state_tensor", "refresh_dof_state_tensor"):
            fn = getattr(env.gym, name, None)
            if fn is not None:
                fn(env.sim)

    def fill_position_target(self, dof_target, action: UnifiedAction, env_ids=None, hand_template=None):
        rows = slice(None) if env_ids is None else _long_indices(env_ids, dof_target)
        fr3_idx = _long_indices(self.index_map.fr3_dof_indices, dof_target)
        hand_idx = _long_indices(self.index_map.revo2_internal_dof_indices, dof_target)

        if env_ids is None:
            dof_target[..., fr3_idx] = _as_like(action.fr3_joint_target, dof_target)
            target_view = dof_target[..., hand_idx]
        else:
            dof_target[rows[:, None] if _is_torch(rows) else np.asarray(rows)[:, None], fr3_idx] = _as_like(
                action.fr3_joint_target, dof_target
            )
            target_view = dof_target[rows[:, None] if _is_torch(rows) else np.asarray(rows)[:, None], hand_idx]

        if hand_template is None:
            hand_template = target_view
        hand = _revo2_active_to_isaac_internal_backend(
            action.revo2_active_target,
            template=hand_template,
            like=dof_target,
        )
        if env_ids is None:
            dof_target[..., hand_idx] = hand
        else:
            dof_target[rows[:, None] if _is_torch(rows) else np.asarray(rows)[:, None], hand_idx] = hand
        return dof_target

    def apply_action_to_targets(self, action: UnifiedAction, env=None, env_ids=None, submit=False, robot_indices=None):
        env = env or self.env
        if env is None:
            raise ValueError("env is required")
        self.fill_position_target(env.cur_targets, action, env_ids=env_ids)
        if submit:
            self.submit_targets(env, env_ids=env_ids, robot_indices=robot_indices)
        return env.cur_targets

    def apply_to_env_cur_targets(self, env, action: UnifiedAction, env_ids=None, submit=False, robot_indices=None):
        return self.apply_action_to_targets(action, env=env, env_ids=env_ids, submit=submit, robot_indices=robot_indices)

    def submit_targets(self, env=None, env_ids=None, robot_indices=None):
        env = env or self.env
        if env is None:
            raise ValueError("env is required")
        if robot_indices is None:
            robot_indices = getattr(env, "robot_indices", None)
        if robot_indices is None:
            if env_ids is None:
                robot_indices = np.arange(int(getattr(env, "num_envs", env.cur_targets.shape[0])), dtype=np.int32)
            else:
                robot_indices = env_ids
        try:
            from isaacgym import gymtorch
        except Exception as exc:  # pragma: no cover - only available inside IsaacGym.
            raise RuntimeError("isaacgym.gymtorch is required to submit targets to the simulator") from exc

        if _is_torch(env.cur_targets):
            import torch

            idx = torch.as_tensor(robot_indices, dtype=torch.int32, device=env.cur_targets.device)
            count = int(idx.numel())
        else:
            idx = np.asarray(robot_indices, dtype=np.int32)
            count = int(idx.size)
        env.gym.set_dof_position_target_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.cur_targets),
            gymtorch.unwrap_tensor(idx),
            count,
        )

    def _dof_pos_vel(self, env):
        if hasattr(env, "arm_hand_dof_pos") and hasattr(env, "arm_hand_dof_vel"):
            return env.arm_hand_dof_pos, env.arm_hand_dof_vel
        dof_state = getattr(env, "dof_state", None)
        if dof_state is None:
            raise ValueError("env must expose arm_hand_dof_pos/vel or dof_state")
        dof_per_env = int(getattr(env, "num_hand_arm_dofs", env.cur_targets.shape[-1]))
        state = dof_state.reshape(int(getattr(env, "num_envs", 1)), dof_per_env, 2)
        return state[..., 0], state[..., 1]

    def _tool_root_states(self, env, env_ids):
        root = getattr(env, "root_state_tensor", None)
        actor_indices = self.index_map.tool_actor_indices
        if root is None or actor_indices is None:
            raise ValueError("env.root_state_tensor and tool actor indices are required for tool observations")

        indices = actor_indices
        if env_ids is not None:
            indices = _select_env_rows(indices, env_ids)
        indices = _long_indices(indices, root)
        states = root[indices]
        if len(getattr(states, "shape", ())) == 1:
            states = states.reshape((1, states.shape[0]))
        if len(states.shape) == 2:
            states = states.reshape(states.shape[:-1] + (1, states.shape[-1]))
        return states

    def get_observation(self, env=None, env_id=0, env_ids=None, refresh=True, as_numpy=True):
        env = env or self.env
        if env is None:
            raise ValueError("env is required")
        if refresh:
            self.refresh(env)

        squeeze_single = False
        if env_ids is None and env_id is not None:
            env_ids = [int(env_id)]
            squeeze_single = True

        qpos, qvel = self._dof_pos_vel(env)
        qpos = _select_env_rows(qpos, env_ids)
        qvel = _select_env_rows(qvel, env_ids)
        fr3_idx = _long_indices(self.index_map.fr3_dof_indices, qpos)
        hand_idx = _long_indices(self.index_map.revo2_internal_dof_indices, qpos)
        hand_qpos = qpos[..., hand_idx]
        hand_qvel = qvel[..., hand_idx]

        tool_state = self._tool_root_states(env, env_ids)
        obs = {
            "fr3_qpos": qpos[..., fr3_idx],
            "fr3_qvel": qvel[..., fr3_idx],
            "revo2_internal_qpos_isaac": hand_qpos,
            "revo2_internal_qvel_isaac": hand_qvel,
            "revo2_active_qpos": _revo2_isaac_internal_to_active_backend(hand_qpos),
            "revo2_active_qvel": _revo2_isaac_internal_to_active_backend(hand_qvel),
            "tool_pos_world": tool_state[..., 0:3],
            "tool_quat_wxyz": _quat_xyzw_to_wxyz_backend(tool_state[..., 3:7]),
            "tool_linvel_world": tool_state[..., 7:10],
            "tool_angvel_world": tool_state[..., 10:13],
        }

        rb = getattr(env, "rigid_body_states", None)
        wrist_idx = self.index_map.wrist_body_index
        if rb is not None and wrist_idx is not None:
            rb_sel = _select_env_rows(rb, env_ids)
            wrist = rb_sel[..., wrist_idx, :]
            obs["wrist_pos_world"] = wrist[..., 0:3]
            obs["wrist_quat_wxyz"] = _quat_xyzw_to_wxyz_backend(wrist[..., 3:7])
        if rb is not None and self.index_map.fingertip_body_indices is not None:
            rb_sel = _select_env_rows(rb, env_ids)
            tips = rb_sel[..., _long_indices(self.index_map.fingertip_body_indices, rb_sel), :]
            obs["fingertip_pos_world"] = tips[..., 0:3]

        if squeeze_single:
            obs = {k: _maybe_squeeze_single(v, True) for k, v in obs.items()}
        if as_numpy:
            obs = {k: _to_numpy(v).astype(np.float32, copy=False) for k, v in obs.items()}
        return obs

    @staticmethod
    def isaac_quat_xyzw_to_common(q_xyzw):
        return quat_xyzw_to_wxyz(np.asarray(q_xyzw, dtype=np.float32))

    @staticmethod
    def common_quat_wxyz_to_isaac(q_wxyz):
        return quat_wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float32))


class IsaacGymActionAdapter(IsaacGymTensorAdapter):
    """Backward-compatible alias for older code importing the action adapter."""

    def __init__(self, fr3_dof_indices=None, revo2_active_dof_indices=None, revo2_internal_dof_indices=None):
        if revo2_active_dof_indices is not None:
            raise ValueError("IsaacGym uses the 11-DoF physical Revo2 order; pass revo2_internal_dof_indices")
        super().__init__(
            IsaacGymIndexMap(
                fr3_dof_indices=list(range(7)) if fr3_dof_indices is None else fr3_dof_indices,
                revo2_internal_dof_indices=(
                    list(range(7, 18)) if revo2_internal_dof_indices is None else revo2_internal_dof_indices
                ),
            )
        )
