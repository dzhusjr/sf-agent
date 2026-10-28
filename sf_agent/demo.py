"""Laptop demo: watch a trained agent fight the built-in CPU.

  python demo.py runs/lstm_aux/seed0                     # retro backend (no account)
  diambra run python demo.py runs/lstm_aux/seed0         # diambra backend (1 engine)

Copy the whole run folder from the VSC (model.zip + config.yaml) to your laptop.
Same code, same config -> same environment as in training. Window: OpenCV, press q to quit.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from sf.agent import is_recurrent, load_model
from sf.config import load_config
from sf.envs import make_vec_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default="model.zip")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=float, default=15.0, help="agent decisions per second")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(os.path.join(args.run_dir, "config.yaml"), args.set)
    venv = make_vec_env(cfg["env"], 1, render_mode="rgb_array", subprocess=False)
    model = load_model(os.path.join(args.run_dir, args.model), cfg["algo"])
    rec = is_recurrent(cfg["algo"])

    obs = venv.reset()
    state, start = None, np.ones((1,), dtype=bool)
    won = lost = 0
    while True:
        t0 = time.time()
        if rec:
            action, state = model.predict(obs, state=state, episode_start=start, deterministic=args.deterministic)
        else:
            action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, start, infos = venv.step(action)
        info = infos[0]
        if info.get("round_done"):
            won += int(bool(info["round_won"]))
            lost += int(not info["round_won"])
        frame = venv.env_method("render")[0] if hasattr(venv, "env_method") else None
        if frame is not None:
            img = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, f"AI rounds won {won} lost {lost}  hp {info.get('own_hp', 0):.2f} vs {info.get('opp_hp', 0):.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("SF agent", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        time.sleep(max(0.0, 1.0 / args.fps - (time.time() - t0)))
    venv.close()


if __name__ == "__main__":
    main()
