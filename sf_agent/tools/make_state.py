"""Create extra SF2 save states (other opponents / difficulty) by playing with the keyboard.
Used for the generalisation experiment: train on the stock state, test on new ones.

  python tools/make_state.py states/vs_ken.state       # saves exactly there
  python tools/make_state.py states/ryu_vs_             # prefix: saves states/ryu_vs_<opponent>.state, named from RAM

Ladder trick (Ryu vs every opponent): x instantly KOs the CPU opponent (writes its health to -1), so you can
walk through the arcade ladder in a few minutes; press p at "ROUND 1" of each new opponent, keep playing, q at the end.

Keys (window must be focused):  arrows/WASD = d-pad, u i o = punches (X Y Z), j k l = kicks (A B C),
e or enter = START, m = MODE, x = KO the opponent, p = save state, q = quit.
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
          0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT",   # macOS arrow codes
          13: "START", 10: "START", ord("e"): "START", ord("m"): "MODE"}
# ids 0-7 and 10 are verified; 8, 9, 11 are Vega/Sagat/Bison in an unknown order -> rename those by the screenshot
CHARS = ["ryu", "honda", "blanka", "guile", "ken", "chunli", "zangief", "dhalsim", "boss8", "boss9", "balrog", "boss11"]
held, hold, info = None, 0, {}
img = env.render()
while True:
    m = np.zeros(len(buttons), np.int8)
    if hold > 0:
        m[buttons.index(held)] = 1
        hold -= 1
    img, _, _, _, info = env.step(m)
    cv2.imshow("make_state (p=save, q=quit)", cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), None, fx=3, fy=3))
    k = cv2.waitKey(16) & 0xFF
    if k == ord("q"):
        break
    if k == ord("x"):
        env.unwrapped.data.set_value("enemy_health", -1)
        print("opponent KO")
    elif k == ord("p"):
        cid = int(info.get("enemy_character", -1))
        name = CHARS[cid] if 0 <= cid < len(CHARS) else f"id{cid}"
        path = out if out.endswith(".state") else f"{out}{name}.state"
        with gzip.open(path, "wb") as fh:
            fh.write(env.em.get_state())
        print("saved", path, f"(enemy_character={cid} -> {name}; check the screen)")
    elif k in KEYMAP:
        held, hold = KEYMAP[k], 6
        print("pressed", KEYMAP[k])
    elif k != 255:
        print(f"unmapped key code {k}")
env.close()
