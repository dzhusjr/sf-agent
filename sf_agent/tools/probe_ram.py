"""Verify the extra SF2 RAM addresses (x, y, status) before trusting them.

Plays a scripted sequence (walk right, walk left, jump, crouch) and prints the
variables every few frames. Expected: agent_x grows while walking right, shrinks
walking left; agent_y changes during the jump; health values match the bars.
Saves a few frames to probe_frames/ so you can eyeball positions.

  python tools/probe_ram.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import stable_retro  # noqa: E402

from sf.envs.retro_backend import DATA_JSON, GAME  # noqa: E402

env = stable_retro.make(GAME, info=DATA_JSON, use_restricted_actions=stable_retro.Actions.ALL,
                        render_mode="rgb_array")
buttons = list(env.buttons)
env.reset()
os.makedirs("probe_frames", exist_ok=True)


def press(*names):
    m = np.zeros(len(buttons), np.int8)
    for n in names:
        m[buttons.index(n)] = 1
    return m


script = [("idle", press(), 60), ("right", press("RIGHT"), 60), ("left", press("LEFT"), 60),
          ("jump", press("UP"), 40), ("crouch", press("DOWN"), 30), ("punch", press("Z"), 30)]
keys = ["health", "enemy_health", "agent_x", "agent_y", "agent_status",
        "enemy_x", "enemy_y", "enemy_status", "enemy_character"]
f = 0
for name, mask, n in script:
    for i in range(n):
        img, _, done, _, info = env.step(mask)
        f += 1
        if i % 10 == 0:
            print(f"{name:7s} f={f:4d} " + " ".join(f"{k}={info.get(k)}" for k in keys))
        if i == n // 2:
            cv2.imwrite(f"probe_frames/{f:04d}_{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
env.close()
print("frames written to probe_frames/")
