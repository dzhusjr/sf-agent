"""Evaluate a trained agent: win rate (+95% Wilson CI), health difference, move distribution.

  python eval.py runs/lstm_aux/seed0 --rounds 100
  python eval.py runs/lstm_aux/seed0 --rounds 100 --set env.difficulty=6      # generalisation test
  python eval.py runs/lstm_aux/seed0 --set env.retro.state_path=states/vs_ken.state

Reads the run's own config.yaml so the environment is identical to training.
Writes <run>/eval_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sf.agent import is_recurrent, load_model, play_rounds
from sf.config import load_config
from sf.envs import make_vec_env
from sf.envs.common import wilson_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default="model.zip", help="model.zip | best_model.zip | checkpoints/...")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(os.path.join(args.run_dir, "config.yaml"), args.set)
    venv = make_vec_env(cfg["env"], 1, start_rank=0)
    model = load_model(os.path.join(args.run_dir, args.model), cfg["algo"])
    r = play_rounds(model, venv, args.rounds, is_recurrent(cfg["algo"]), args.deterministic)
    venv.close()

    lo, hi = wilson_ci(r["won"], r["rounds"])
    r["win_rate_ci95"] = [lo, hi]
    counts = np.asarray(r["action_counts"], float)
    print(f"win rate {r['win_rate']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  over {r['rounds']} rounds")
    print(f"mean health difference at round end {r['mean_hp_diff']:+.3f}")
    print("move distribution (top 8):")
    for a in np.argsort(-counts)[:8]:
        print(f"  action {a:3d}: {counts[a] / counts.sum():6.1%}")
    out = os.path.join(args.run_dir, f"eval_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(r, f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
