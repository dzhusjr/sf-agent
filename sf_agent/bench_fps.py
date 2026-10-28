"""Measure environment throughput (agent steps / second) for different numbers of
parallel envs, with random actions. Run this FIRST on a VSC node: it decides how
many timesteps you can afford per experiment.

  python bench_fps.py --backend retro --n 1 8 16 32 --seconds 30
  diambra run -s=32 python bench_fps.py --backend diambra --n 1 8 32
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from sf.config import load_config
from sf.envs import make_vec_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="retro")
    ap.add_argument("--n", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("-c", "--config", default=os.path.join(os.path.dirname(__file__), "configs", "base.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config, [f"env.backend={args.backend}"])

    print(f"backend={args.backend}  cpus={os.cpu_count()}")
    for n in args.n:
        venv = make_vec_env(cfg["env"], n)
        venv.reset()
        steps, t0 = 0, time.time()
        while time.time() - t0 < args.seconds:
            venv.step(np.array([venv.action_space.sample() for _ in range(n)]))
            steps += n
        dt = time.time() - t0
        venv.close()
        sps = steps / dt
        print(f"n_envs={n:3d}  {sps:8.0f} agent-steps/s  -> 10M steps in {1e7 / sps / 3600:5.1f} h")


if __name__ == "__main__":
    main()
