"""Backend-independent pieces shared by every fighting-game environment.

Every backend (retro / diambra / dummy) returns observations as a Dict with
  "state":  float32 vector of game-state features (RAM), roughly in [-1, 1]
  "frame":  optional uint8 (H, W, 1) grayscale image (only if cfg.env.use_frame)
and fills `info` with the same keys, so training / eval code never needs to
know which game is underneath:

  info["own_hp"], info["opp_hp"]   health as a fraction of max health
  info["hp_lost"]                  True if the agent lost health during this step
                                   (label source for the auxiliary head)
  info["round_done"]               True on the step a round ends
  info["round_won"]                True/False on that step (None otherwise)
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


def damage_reward(prev_own: float, own: float, prev_opp: float, opp: float) -> float:
    """r_t = damage dealt - damage taken, as fractions of max health.

    Health *increases* (new round refill) are ignored via max(0, .)."""
    dealt = max(0.0, prev_opp - opp)
    taken = max(0.0, prev_own - own)
    return float(dealt - taken)


class ActionHistory(gym.Wrapper):
    """Adds key "actions": one-hot encoding of the last `n` actions (flattened).

    Gives even the memory-less MLP baseline a short context of its own moves,
    and is cheap. Works for Discrete action spaces only."""

    def __init__(self, env: gym.Env, n: int):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.n = n
        self.n_actions = int(env.action_space.n)
        self.hist: deque[int] = deque(maxlen=n)
        spaces = dict(env.observation_space.spaces)
        spaces["actions"] = gym.spaces.Box(0.0, 1.0, (n * self.n_actions,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def _encode(self) -> np.ndarray:
        out = np.zeros((self.n, self.n_actions), np.float32)
        for i, a in enumerate(self.hist):
            out[i, a] = 1.0
        return out.reshape(-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.hist.clear()
        obs = dict(obs)
        obs["actions"] = self._encode()
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.hist.append(int(action))
        obs = dict(obs)
        obs["actions"] = self._encode()
        return obs, r, term, trunc, info


class RoundStats(gym.Wrapper):
    """Accumulates per-episode round statistics into info["episode_rounds"]
    at episode end (won, lost) so the eval callback / Monitor can log them."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.won = self.lost = 0

    def reset(self, **kwargs):
        self.won = self.lost = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if info.get("round_done"):
            if info.get("round_won"):
                self.won += 1
            else:
                self.lost += 1
        if term or trunc:
            info["episode_rounds"] = {"won": self.won, "lost": self.lost}
        return obs, r, term, trunc, info


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate (better than +-1.96 sigma for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
