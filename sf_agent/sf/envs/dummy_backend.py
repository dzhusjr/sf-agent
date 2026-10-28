"""Tiny 1-D fighting game with no ROM, no emulator, no account.

Purpose: smoke-test the whole pipeline (training, aux head, eval, checkpoints,
Slurm) on any machine in seconds. The scripted opponent *telegraphs* its attacks
with a 2-step wind-up, so an agent with memory / an aux head that predicts
"will I be hit soon" has something real to learn -- a sanity check that the
aux head works before spending cluster hours on the real game.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .common import damage_reward

ACTIONS = ["noop", "left", "right", "block", "punch", "kick"]
ARENA = 10.0
MAX_HP = 100.0


class DummyFightEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, difficulty: int = 3, max_round_steps: int = 300, use_frame: bool = False,
                 rounds_to_win: int = 2, render_mode=None, seed: int | None = None):
        self.difficulty = difficulty
        self.max_round_steps = max_round_steps
        self.use_frame = use_frame
        self.rounds_to_win = rounds_to_win
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        spaces = {"state": gym.spaces.Box(-1.0, 1.0, (7,), np.float32)}
        if use_frame:
            spaces["frame"] = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    def _new_round(self):
        self.own_x, self.opp_x = 3.0, 7.0
        self.own_hp, self.opp_hp = MAX_HP, MAX_HP
        self.windup = 0          # opponent wind-up counter (telegraph)
        self.own_cd = 0          # own attack cooldown
        self.t = 0

    def _obs(self):
        dx = (self.opp_x - self.own_x) / ARENA
        s = np.array([self.own_hp / MAX_HP, self.opp_hp / MAX_HP, dx, abs(dx),
                      float(self.windup > 0), self.own_cd / 3.0, self.t / self.max_round_steps],
                     np.float32)
        obs = {"state": s}
        if self.use_frame:
            obs["frame"] = self._frame()
        return obs

    def _frame(self):
        img = np.zeros((84, 84, 1), np.uint8)
        for x, val in ((self.own_x, 255), (self.opp_x, 128)):
            c = int(np.clip(x / ARENA * 83, 0, 83))
            img[40:70, max(0, c - 2):c + 3, 0] = val
        return img

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.rounds = {"won": 0, "lost": 0}
        self._new_round()
        return self._obs(), {"own_hp": 1.0, "opp_hp": 1.0, "hp_lost": False,
                             "round_done": False, "round_won": None}

    def step(self, action):
        a = ACTIONS[int(action)]
        prev_own, prev_opp = self.own_hp, self.opp_hp
        self.t += 1
        self.own_cd = max(0, self.own_cd - 1)

        # --- agent
        if a == "left":
            self.own_x = max(0.0, self.own_x - 0.5)
        elif a == "right":
            self.own_x = min(self.opp_x - 0.5, self.own_x + 0.5)
        elif a in ("punch", "kick") and self.own_cd == 0:
            reach, dmg = (1.0, 6.0) if a == "punch" else (1.6, 9.0)
            if abs(self.opp_x - self.own_x) <= reach:
                self.opp_hp -= dmg
                self.windup = 0      # hit interrupts opponent's attack
            self.own_cd = 1 if a == "punch" else 3

        # --- scripted opponent: approach, wind-up 2 steps, strike
        dist = abs(self.opp_x - self.own_x)
        aggression = 0.15 + 0.08 * self.difficulty
        if self.windup > 0:
            self.windup -= 1
            if self.windup == 0 and dist <= 1.6 and a != "block":
                self.own_hp -= 8.0 + self.difficulty
        elif dist > 1.2:
            self.opp_x = max(self.own_x + 0.5, self.opp_x - 0.4)
        elif self.rng.random() < aggression:
            self.windup = 2

        own, opp = max(self.own_hp, 0) / MAX_HP, max(self.opp_hp, 0) / MAX_HP
        r = damage_reward(prev_own / MAX_HP, own, prev_opp / MAX_HP, opp)
        info = {"own_hp": own, "opp_hp": opp, "hp_lost": self.own_hp < prev_own,
                "round_done": False, "round_won": None}

        ko = self.own_hp <= 0 or self.opp_hp <= 0
        if ko or self.t >= self.max_round_steps:
            won = self.opp_hp < self.own_hp
            info["round_done"], info["round_won"] = True, bool(won)
            self.rounds["won" if won else "lost"] += 1
            r += 1.0 if won else -1.0
            self._new_round()
        terminated = max(self.rounds.values()) >= self.rounds_to_win
        return self._obs(), r, terminated, False, info

    def render(self):
        return np.repeat(self._frame(), 3, axis=2)
