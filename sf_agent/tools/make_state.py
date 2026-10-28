"""Create extra SF2 save states (other opponents / difficulty) by playing with the keyboard.
Used for the generalisation experiment: train on the stock state, test on new ones.

  python tools/make_state.py states/vs_ken.state

Keys (window must be focused):  arrows/WASD = d-pad, u i o = punches (X Y Z), j k l = kicks (A B C),
enter = START, m = MODE, p = save state to the given path, q = quit.
Each key press holds the button for 6 frames (cv2 has no key-up events).
Tip: go to OPTIONS to change difficulty, pick characters, and save right when the round starts.
"""
import gzip
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import stable_retro  # noqa: E402

from sf.envs.retro_backend import DATA_JSON, GAME  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "states/custom.state"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
env = stable_retro.make(GAME, state=stable_retro.State.NONE, info=DATA_JSON,
                        use_restricted_actions=stable_retro.Actions.ALL, render_mode="rgb_array")
buttons = list(env.buttons)
env.reset()
KEYMAP = {ord("w"): "UP", ord("s"): "DOWN", ord("a"): "LEFT", ord("d"): "RIGHT",
          82: "UP", 84: "DOWN", 81: "LEFT", 83: "RIGHT",
          ord("u"): "X", ord("i"): "Y", ord("o"): "Z", ord("j"): "A", ord("k"): "B", ord("l"): "C",
          13: "START", ord("m"): "MODE"}
held, hold = None, 0
img = env.render()
while True:
    m = np.zeros(len(buttons), np.int8)
    if hold > 0:
        m[buttons.index(held)] = 1
        hold -= 1
    img, *_ = env.step(m)
    cv2.imshow("make_state (p=save, q=quit)", cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), None, fx=3, fy=3))
    k = cv2.waitKey(16) & 0xFF
    if k == ord("q"):
        break
    if k == ord("p"):
        with gzip.open(out, "wb") as fh:
            fh.write(env.em.get_state())
        print("saved", out)
    elif k in KEYMAP:
        held, hold = KEYMAP[k], 6
env.close()
