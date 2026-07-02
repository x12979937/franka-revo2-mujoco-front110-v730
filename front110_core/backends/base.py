"""Abstract backend contract used by MuJoCo and IsaacGym runners."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SimBackend(ABC):
    name = "base"

    @abstractmethod
    def reset_episode(self, episode_spec):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError
