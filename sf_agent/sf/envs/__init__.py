"""Environment factory: one place that builds identical envs for training (VSC),
evaluation and the laptop demo. Never construct envs any other way -- train/demo
mismatches (action set, frame-skip, wrappers) silently ruin a trained agent."""
from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecEnv,
                                              VecFrameStack, VecTransposeImage)

from .common import ActionHistory, DropKeys, RoundStats

BACKENDS = ("retro", "dummy")


def _make_base(env_cfg: dict, rank: int, render_mode: str | None) -> gym.Env:
    b = env_cfg["backend"]
    use_frame = env_cfg.get("use_frame", False)
    if b == "dummy":                     # toy game for tests, no ROM needed
        from .dummy_backend import DummyFightEnv
        return DummyFightEnv(difficulty=env_cfg.get("difficulty", 3), use_frame=use_frame,
                             render_mode=render_mode, seed=rank)
    if b == "retro":                     # SF2 on stable-retro
        from .retro_backend import SF2RetroEnv
        r = env_cfg.get("retro", {})
        return SF2RetroEnv(state_path=r.get("state_path"), frame_skip=r.get("frame_skip", 4),
                           use_frame=use_frame, use_macros=r.get("use_macros", True),
                           episode=r.get("episode", "match"), win_bonus=env_cfg.get("win_bonus", 1.0),
                           intro_frames=r.get("intro_frames", 120), render_mode=render_mode,
                           state_paths=r.get("state_paths"), seed=rank)
    raise ValueError(f"unknown backend {b!r}, choose from {BACKENDS}")


def make_env_fn(env_cfg: dict, rank: int, render_mode: str | None = None,
                monitor_dir: str | None = None) -> Callable[[], gym.Env]:
    def _init():
        env = _make_base(env_cfg, rank, render_mode)
        if not env_cfg.get("use_state", True):       # pixel-only baseline
            assert env_cfg.get("use_frame", False), "env.use_state=false needs env.use_frame=true"
            env = DropKeys(env, ["state"])
        if env_cfg.get("action_history", 0) > 0:
            env = ActionHistory(env, env_cfg["action_history"])
        env = RoundStats(env)
        if monitor_dir:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, os.path.join(monitor_dir, str(rank)) if monitor_dir else None)
        return env
    return _init


def make_vec_env(env_cfg: dict, n_envs: int, start_rank: int = 0, render_mode: str | None = None,
                 monitor_dir: str | None = None, subprocess: bool | None = None) -> VecEnv:
    """retro allows ONE emulator per process -> subprocesses for retro, in-process otherwise."""
    fns = [make_env_fn(env_cfg, start_rank + i, render_mode, monitor_dir) for i in range(n_envs)]
    if subprocess is None:
        subprocess = env_cfg["backend"] == "retro"
    venv = SubprocVecEnv(fns) if subprocess else DummyVecEnv(fns)
    return wrap_vec(venv, env_cfg)


def wrap_vec(venv: VecEnv, env_cfg: dict) -> VecEnv:
    if env_cfg.get("frame_stack", 1) > 1:
        venv = VecFrameStack(venv, env_cfg["frame_stack"])
    if env_cfg.get("use_frame", False):
        venv = VecTransposeImage(venv)     # (H, W, C) -> (C, H, W) for the CNN
    return venv
