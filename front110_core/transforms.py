"""Coordinate and joint-layout transforms shared by dataset adapters."""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float32)
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1).astype(np.float32)


def quat_xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float32)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1).astype(np.float32)


def revo2_internal_to_active(q_internal):
    """Map converted 11-DoF Revo2 command/state to the user-facing 6-DoF order."""
    q = np.asarray(q_internal, dtype=np.float32)
    if q.shape[-1] != 11:
        raise ValueError(f"expected 11 internal Revo2 joints, got shape {q.shape}")
    return np.stack(
        [
            q[..., 0],
            0.5 * (q[..., 1] + q[..., 2]),
            0.5 * (q[..., 3] + q[..., 4]),
            0.5 * (q[..., 5] + q[..., 6]),
            0.5 * (q[..., 7] + q[..., 8]),
            0.5 * (q[..., 9] + q[..., 10]),
        ],
        axis=-1,
    ).astype(np.float32)


def revo2_active_to_internal(q_active, template=None):
    """Expand the 6-DoF active Revo2 layout to the converted 11-DoF MuJoCo layout.

    If a template is supplied, thumb metacarpal/proximal split and each two-link
    finger pair keep the template's local ratio where possible. This lets the
    simulator-independent action remain 6-DoF while MuJoCo receives 11 targets.
    """
    q = np.asarray(q_active, dtype=np.float32)
    if q.shape[-1] != 6:
        raise ValueError(f"expected 6 active Revo2 joints, got shape {q.shape}")
    out = np.zeros(q.shape[:-1] + (11,), dtype=np.float32)
    out[..., 0] = q[..., 0]

    if template is None:
        out[..., 1] = q[..., 1]
        out[..., 2] = q[..., 1]
        out[..., 3] = q[..., 2]
        out[..., 4] = q[..., 2]
        out[..., 5] = q[..., 3]
        out[..., 6] = q[..., 3]
        out[..., 7] = q[..., 4]
        out[..., 8] = q[..., 4]
        out[..., 9] = q[..., 5]
        out[..., 10] = q[..., 5]
        return out

    t = np.asarray(template, dtype=np.float32)
    if t.shape[-1] != 11:
        raise ValueError(f"expected 11-DoF template, got shape {t.shape}")
    pairs = [(1, 2, 1), (3, 4, 2), (5, 6, 3), (7, 8, 4), (9, 10, 5)]
    for a, b, src in pairs:
        raw = np.abs(t[..., a]) + np.abs(t[..., b])
        denom = np.maximum(1e-6, raw)
        wa = np.where(raw > 1e-6, np.abs(t[..., a]) / denom, 0.5)
        wb = np.where(raw > 1e-6, np.abs(t[..., b]) / denom, 0.5)
        out[..., a] = q[..., src] * (2.0 * wa)
        out[..., b] = q[..., src] * (2.0 * wb)
    return out.astype(np.float32)


def revo2_isaac_internal_to_active(q_internal):
    """Map IsaacGym Revo2 11-DoF order to the common 6-DoF active order.

    IsaacGym task scripts use physical hand order:
    index(2), middle(2), pinky(2), ring(2), thumb(3).
    The common active order remains:
    thumb_flex, thumb_aux, index, middle, ring, pinky.
    """
    q = np.asarray(q_internal, dtype=np.float32)
    if q.shape[-1] != 11:
        raise ValueError(f"expected 11 IsaacGym Revo2 joints, got shape {q.shape}")
    return np.stack(
        [
            q[..., 8],
            0.5 * (q[..., 9] + q[..., 10]),
            0.5 * (q[..., 0] + q[..., 1]),
            0.5 * (q[..., 2] + q[..., 3]),
            0.5 * (q[..., 6] + q[..., 7]),
            0.5 * (q[..., 4] + q[..., 5]),
        ],
        axis=-1,
    ).astype(np.float32)


def revo2_active_to_isaac_internal(q_active, template=None):
    """Expand common Revo2 active order to IsaacGym's 11-DoF physical order."""
    q = np.asarray(q_active, dtype=np.float32)
    if q.shape[-1] != 6:
        raise ValueError(f"expected 6 active Revo2 joints, got shape {q.shape}")
    if template is None:
        out = np.zeros(q.shape[:-1] + (11,), dtype=np.float32)
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

    t = np.asarray(template, dtype=np.float32)
    if t.shape[-1] != 11:
        raise ValueError(f"expected 11-DoF IsaacGym template, got shape {t.shape}")
    prefix = np.broadcast_shapes(q.shape[:-1], t.shape[:-1])
    if q.shape[:-1] != prefix:
        q = np.broadcast_to(q, prefix + (6,))
    if t.shape[:-1] != prefix:
        t = np.broadcast_to(t, prefix + (11,))
    out = np.zeros(prefix + (11,), dtype=np.float32)
    pairs = [(0, 1, 2), (2, 3, 3), (4, 5, 5), (6, 7, 4), (9, 10, 1)]
    for a, b, src in pairs:
        raw = np.abs(t[..., a]) + np.abs(t[..., b])
        denom = np.maximum(1e-6, raw)
        wa = np.where(raw > 1e-6, np.abs(t[..., a]) / denom, 0.5)
        wb = np.where(raw > 1e-6, np.abs(t[..., b]) / denom, 0.5)
        out[..., a] = q[..., src] * (2.0 * wa)
        out[..., b] = q[..., src] * (2.0 * wb)
    out[..., 8] = q[..., 0]
    return out.astype(np.float32)


def ensure_float32(value):
    return np.asarray(value, dtype=np.float32)
