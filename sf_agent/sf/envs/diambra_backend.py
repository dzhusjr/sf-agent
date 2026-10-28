"""Street Fighter III: 3rd Strike (sfiii3n) via DIAMBRA Arena.

Needs: DIAMBRA account, the sfiii3n.zip ROM, and DIAMBRA engine servers.
Normally launched as `diambra run -s=N python train.py ...`, which starts N engine
containers (Docker) and exports DIAMBRA_ENVS="host:port host:port ...".
On the VSC (no Docker) see vsc/start_engines_apptainer.sh, which sets DIAMBRA_ENVS itself.

IMPORTANT: sfiii3n exposes NO positions in RAM (checked in diambra-arena 2.2.7
integratedGames.json): only health, side, stun, super bar, wins, character, timer, stage.
A RAM-only agent therefore cannot see the distance to the opponent. Hence the
`use_frame` option (small grayscale frame + CNN) and the planned ablation
"RAM only vs RAM + frame".
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .common import damage_reward

GAME = "sfiii3n"
HP_LO, HP_HI = -1.0, 160.0

RAM_KEYS = [
    "own_health", "opp_health", "own_side", "opp_side",
    "own_stun_bar", "opp_stun_bar", "own_stunned", "opp_stunned",
    "own_super_bar", "opp_super_bar", "own_super_count", "opp_super_count",
    "timer",
]


def _gymnasium1_compat():
    """diambra-arena 2.2.7 passes numpy ints to MultiBinary/Discrete, which gymnasium>=1.0
    rejects. stable-retro needs gymnasium>=1.0, so instead of pinning an old gymnasium we
    cast to int here. Harmless if diambra fixes it upstream."""
    if getattr(gym.spaces.MultiBinary, "_sf_patched", False):
        return
    orig = gym.spaces.MultiBinary.__init__

    def __init__(self, n, *a, **kw):
        if isinstance(n, np.integer):
            n = int(n)
        orig(self, n, *a, **kw)

    gym.spaces.MultiBinary.__init__ = __init__
    gym.spaces.MultiBinary._sf_patched = True


_gymnasium1_compat()


def enable_mock():
    """Replace the gRPC engine with DIAMBRA's own mock (random game dynamics).
    Lets us test the full pipeline without account / ROM / Docker."""
    from unittest import mock
    from diambra.arena.utils.engine_mock import DiambraEngineMock
    m = DiambraEngineMock(fps=100000)
    base = "diambra.arena.engine.interface.DiambraEngine."
    for name, fn in [("__init__", m.mock__init__), ("env_init", m.mock_env_init),
                     ("reset", m.mock_reset), ("step", m.mock_step), ("close", m.mock_close)]:
        mock.patch(base + name, fn).start()


class DiambraAdapter(gym.Wrapper):
    """Converts DIAMBRA's flattened dict obs into the common {"state", "frame"} format,
    computes our own (backend-independent) reward and the common info keys."""

    def __init__(self, env: gym.Env, use_frame: bool, win_bonus: float):
        super().__init__(env)
        self.use_frame = use_frame
        self.win_bonus = win_bonus
        self.ram_space = gym.spaces.Dict({k: env.observation_space[k] for k in RAM_KEYS})
        dim = gym.spaces.flatdim(self.ram_space)
        spaces = {"state": gym.spaces.Box(-1.0, 1.0, (dim,), np.float32)}
        if use_frame:
            spaces["frame"] = env.observation_space["frame"]
        self.observation_space = gym.spaces.Dict(spaces)

    @staticmethod
    def _hp(scaled) -> float:
        """scale=True maps health [-1, 160] -> [0, 1]; convert back to fraction of max."""
        raw = float(np.asarray(scaled).reshape(-1)[0]) * (HP_HI - HP_LO) + HP_LO
        return max(raw, 0.0) / HP_HI

    def _convert(self, obs):
        ram = {k: obs[k] for k in RAM_KEYS}
        out = {"state": gym.spaces.flatten(self.ram_space, ram).astype(np.float32)}
        if self.use_frame:
            out["frame"] = obs["frame"]
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.own, self.opp = self._hp(obs["own_health"]), self._hp(obs["opp_health"])
        info.update(own_hp=self.own, opp_hp=self.opp, hp_lost=False, round_done=False, round_won=None)
        return self._convert(obs), info

    def step(self, action):
        obs, _native_r, term, trunc, info = self.env.step(action)
        own, opp = self._hp(obs["own_health"]), self._hp(obs["opp_health"])
        r = damage_reward(self.own, own, self.opp, opp)
        round_done = bool(info.get("round_done", False))
        won = None
        if round_done:
            won = own > opp
            r += self.win_bonus if won else -self.win_bonus
        info.update(own_hp=own, opp_hp=opp, hp_lost=own < self.own - 1e-6,
                    round_done=round_done, round_won=won)
        self.own, self.opp = own, opp
        return self._convert(obs), float(r), term, trunc, info

    def action_names(self) -> list[str]:
        n = self.action_space.n
        return [f"a{i}" for i in range(n)]


def make_diambra_env(rank: int, characters: str | None = "Ryu", difficulty: int | None = 3,
                     step_ratio: int = 6, use_frame: bool = False, win_bonus: float = 1.0,
                     role: str | None = None, render_mode: str | None = None,
                     mock: bool = False) -> gym.Env:
    import diambra.arena
    from diambra.arena import EnvironmentSettings, Roles, SpaceTypes, WrappersSettings

    s = EnvironmentSettings()
    s.step_ratio = step_ratio
    s.difficulty = difficulty
    s.characters = characters
    s.action_space = SpaceTypes.DISCRETE
    s.continue_game = 0.0
    s.frame_shape = (84, 84, 1) if use_frame else (32, 32, 1)   # frame is tiny if unused
    if role is not None:
        s.role = Roles.P1 if role == "P1" else Roles.P2

    w = WrappersSettings()
    w.scale = True
    w.role_relative = True
    w.flatten = True
    w.filter_keys = list(RAM_KEYS)

    kw = {"env_addresses": [f"localhost:{50051 + i}" for i in range(rank + 1)]} if mock else {}
    env = diambra.arena.make(GAME, s, w, render_mode=render_mode, rank=rank, **kw)
    return DiambraAdapter(env, use_frame=use_frame, win_bonus=win_bonus)
