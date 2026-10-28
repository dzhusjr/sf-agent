"""Street Fighter II' Special Champion Edition (Sega Genesis) via stable-retro.

No account, no Docker, no servers: `pip install stable-retro` + your own ROM.
Runs headless on the VSC (one emulator per *process* -> always use SubprocVecEnv).

Observation "state" vector (all roughly in [-1, 1]):
  0 own_hp            health / 176
  1 opp_hp
  2 dx                (enemy_x - agent_x) / 256   signed -> also encodes facing
  3 |dx|
  4 own_y, 5 opp_y    / 256   (jumping)
  6 own_status/65535, 7 opp_status/65535   raw animation/state codes (scaled)
  8 round_progress    steps in round / max_round_steps

RAM map (sf2_genesis_data.json): the first six variables are the stock stable-retro
integration. agent_*/enemy_* x, y, status, character are ADDED by us: player structs
start at 0xFF8000 (agent) and 0xFF8280 (enemy), health sits at +0x42 in both (matches
the stock addresses), x/y/status offsets come from community RAM maps.
-> VERIFY them with tools/probe_ram.py (walk left/right, jump) before trusting them.

Actions: a Discrete set of meaningful moves + optional special-move macros
(Hadouken, Shoryuken, Tatsumaki), executed as short frame sequences relative
to the direction the agent is facing.
"""
from __future__ import annotations

import gzip
import os

import gymnasium as gym
import numpy as np

from .common import damage_reward

GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"
MAX_HP = 176.0
DATA_JSON = os.path.join(os.path.dirname(__file__), "sf2_genesis_data.json")

# 6-button Genesis pad in SF2: X/Y/Z = light/medium/heavy punch, A/B/C = light/medium/heavy kick
ATTACKS = {"LP": "X", "MP": "Y", "HP": "Z", "LK": "A", "MK": "B", "HK": "C"}

# Basic actions: (direction tokens, attack) ; "F"/"B" = forward/back relative to facing
BASIC = [
    ((), None), (("B",), None), (("F",), None), (("UP",), None), (("DOWN",), None),
    (("UP", "B"), None), (("UP", "F"), None), (("DOWN", "B"), None), (("DOWN", "F"), None),
] + [((), k) for k in ATTACKS] + [(("DOWN",), k) for k in ATTACKS] + [(("UP", "F"), k) for k in ("HP", "HK")]

# Special moves as frame sequences of (tokens, n_frames)
MACROS = {
    "hadouken":  [(("DOWN",), 2), (("DOWN", "F"), 2), (("F", "MP"), 3)],
    "shoryuken": [(("F",), 2), (("DOWN",), 2), (("DOWN", "F", "HP"), 3)],
    "tatsumaki": [(("DOWN",), 2), (("DOWN", "B"), 2), (("B", "MK"), 3)],
}


class SF2RetroEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, state_path: str | None = None, frame_skip: int = 4,
                 use_frame: bool = False, use_macros: bool = True,
                 episode: str = "match", max_round_steps: int = 1500,
                 win_bonus: float = 1.0, render_mode: str | None = None):
        import stable_retro  # imported lazily so other backends work without it

        self.frame_skip = frame_skip
        self.use_frame = use_frame
        self.episode = episode                 # "round" or "match" (best of 3)
        self.max_round_steps = max_round_steps
        self.win_bonus = win_bonus
        self.render_mode = render_mode

        self.env = stable_retro.make(
            GAME, info=DATA_JSON,
            use_restricted_actions=stable_retro.Actions.ALL,
            obs_type=stable_retro.Observations.IMAGE,
            render_mode="rgb_array",
        )
        if state_path:  # custom save state (other opponent / difficulty), see tools/make_state.py
            with gzip.open(state_path, "rb") as fh:
                self.env.initial_state = fh.read()
        self.buttons = list(self.env.buttons)

        self.actions = [("basic", a) for a in BASIC]
        if use_macros:
            self.actions += [("macro", m) for m in MACROS]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        spaces = {"state": gym.spaces.Box(-1.0, 1.0, (9,), np.float32)}
        if use_frame:
            spaces["frame"] = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    # ------------------------------------------------------------------ helpers
    def action_names(self) -> list[str]:
        out = []
        for kind, a in self.actions:
            if kind == "macro":
                out.append(a)
            else:
                dirs, atk = a
                out.append("+".join(list(dirs) + ([atk] if atk else [])) or "noop")
        return out

    def _mask(self, tokens) -> np.ndarray:
        m = np.zeros(len(self.buttons), np.int8)
        facing_right = self.info.get("enemy_x", 0) >= self.info.get("agent_x", 0)
        for t in tokens:
            if t == "F":
                t = "RIGHT" if facing_right else "LEFT"
            elif t == "B":
                t = "LEFT" if facing_right else "RIGHT"
            elif t in ATTACKS:
                t = ATTACKS[t]
            m[self.buttons.index(t)] = 1
        return m

    def _sequence(self, action: int):
        kind, a = self.actions[int(action)]
        if kind == "macro":
            return MACROS[a]
        dirs, atk = a
        return [(tuple(dirs) + ((atk,) if atk else ()), self.frame_skip)]

    def _hp(self, info):
        return (max(info["health"], 0) / MAX_HP, max(info["enemy_health"], 0) / MAX_HP)

    def _obs(self, img):
        i = self.info
        dx = (i.get("enemy_x", 0) - i.get("agent_x", 0)) / 256.0
        own, opp = self._hp(i)
        s = np.array([own, opp, dx, abs(dx), i.get("agent_y", 0) / 256.0, i.get("enemy_y", 0) / 256.0,
                      i.get("agent_status", 0) / 65535.0, i.get("enemy_status", 0) / 65535.0,
                      self.t / self.max_round_steps], np.float32)
        obs = {"state": np.clip(s, -1.0, 1.0)}
        if self.use_frame:
            import cv2
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            obs["frame"] = cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA)[..., None]
        return obs

    def _wait_next_round(self, max_frames=600):
        """After a KO the game plays an animation; skip it with no-ops until both
        health bars are refilled (next round) or the match/scene ends."""
        noop = np.zeros(len(self.buttons), np.int8)
        img = None
        for _ in range(max_frames):
            img, _, done, _, self.info = self.env.step(noop)
            if done or (self.info["health"] == MAX_HP and self.info["enemy_health"] == MAX_HP):
                break
        return img

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        img, self.info = self.env.reset(seed=seed)
        self.info = dict(self.info) if self.info else {}
        noop = np.zeros(len(self.buttons), np.int8)
        img, _, _, _, self.info = self.env.step(noop)   # populate RAM variables
        self.t = 0
        self.rounds = {"won": 0, "lost": 0}
        self._last_img = img
        own, opp = self._hp(self.info)
        return self._obs(img), {"own_hp": own, "opp_hp": opp, "hp_lost": False,
                                "round_done": False, "round_won": None}

    def step(self, action):
        prev_own, prev_opp = self._hp(self.info)
        prev_raw_own, prev_raw_opp = self.info["health"], self.info["enemy_health"]
        img, emu_done, round_done, won = self._last_img, False, False, None

        for tokens, n in self._sequence(action):
            mask = self._mask(tokens)
            for _ in range(n):
                img, _, emu_done, _, self.info = self.env.step(mask)
                h, eh = self.info["health"], self.info["enemy_health"]
                if h < 0 or eh < 0:                      # KO (health goes to -1)
                    round_done, won = True, eh < 0 and h >= 0
                elif h > prev_raw_own and eh > prev_raw_opp:  # both refilled -> time-over round change
                    round_done, won = True, prev_raw_opp < prev_raw_own
                if round_done or emu_done:
                    break
            if round_done or emu_done:
                break
        self.t += 1
        own, opp = self._hp(self.info)
        reward = damage_reward(prev_own, own, prev_opp, opp)
        info = {"own_hp": own, "opp_hp": opp, "hp_lost": own < prev_own - 1e-6,
                "round_done": False, "round_won": None}

        if not round_done and self.t >= self.max_round_steps:   # safety net
            round_done, won = True, opp < own
        terminated = bool(emu_done)
        if round_done:
            info["round_done"], info["round_won"] = True, bool(won)
            reward += self.win_bonus if won else -self.win_bonus
            self.rounds["won" if won else "lost"] += 1
            self.t = 0
            if self.episode == "round" or max(self.rounds.values()) >= 2:
                terminated = True
            else:
                img = self._wait_next_round()
        self._last_img = img
        return self._obs(img), float(reward), terminated, False, info

    def render(self):
        return self._last_img

    def close(self):
        self.env.close()
